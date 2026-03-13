"""Fused CUDA kernel for log-mel spectrogram computation.

Replaces faster-whisper's pure-NumPy FeatureExtractor with a GPU implementation.
Fuses: frame extraction → windowing → FFT → power spectrum → mel filterbank → log scaling.

Integration: output is numpy array (n_mels, n_frames) compatible with
faster_whisper.WhisperModel.encode(features).

Usage:
    from audioserve.cuda.mel_spectrogram import patch_whisper_model
    model = WhisperModel("large-v3", device="cuda")
    patch_whisper_model(model)  # Now uses CUDA mel spectrogram
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.cpp_extension import load_inline

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>

// ============================================================
// Kernel 1: Frame extraction + windowing
//
// Each thread block handles one frame.
// Reads hop_length-strided windows from padded audio,
// multiplies by Hanning window, writes to output buffer.
// ============================================================
__global__ void frame_and_window_kernel(
    const float* __restrict__ audio,      // padded audio, length = audio_len
    const float* __restrict__ window,     // Hanning window, length = n_fft
    float* __restrict__ frames,           // output: (n_frames, n_fft)
    int n_fft,
    int hop_length,
    int n_frames,
    int audio_len
) {
    int frame_idx = blockIdx.x;
    int bin_idx = threadIdx.x;

    if (frame_idx >= n_frames || bin_idx >= n_fft) return;

    // Center-padded: frame starts at frame_idx * hop_length
    // (audio is already padded with n_fft/2 on each side)
    int audio_idx = frame_idx * hop_length + bin_idx;

    float sample = (audio_idx < audio_len) ? audio[audio_idx] : 0.0f;
    frames[frame_idx * n_fft + bin_idx] = sample * window[bin_idx];
}


// ============================================================
// Kernel 2: Power spectrum + mel filterbank + log scaling
//
// After cuFFT produces complex output (n_frames, n_fft/2+1),
// this kernel computes:
//   magnitudes = |complex|^2  (drop last frame)
//   mel_spec = mel_filters @ magnitudes
//   log_spec = log10(clamp(mel_spec, 1e-10))
//
// Each thread computes one (mel_bin, frame) output element.
// ============================================================
__global__ void mel_log_kernel(
    const cufftComplex* __restrict__ fft_out,  // (n_frames, n_freq) complex, row-major
    const float* __restrict__ mel_filters,     // (n_mels, n_freq) row-major
    float* __restrict__ output,                // (n_mels, n_frames_use) row-major
    int n_frames,         // total frames from FFT
    int n_frames_use,     // n_frames - 1 (drop last frame, matching stft[..., :-1])
    int n_freq,           // n_fft/2 + 1 = 201
    int n_mels
) {
    // Each thread: one (mel_bin, frame) pair
    int mel_bin = blockIdx.y * blockDim.y + threadIdx.y;
    int frame = blockIdx.x * blockDim.x + threadIdx.x;

    if (mel_bin >= n_mels || frame >= n_frames_use) return;

    // Dot product: mel_filters[mel_bin, :] . magnitudes[:, frame]
    // magnitudes shape is (n_freq, n_frames_use) — transposed from FFT output
    // mel_filters is (n_mels, n_freq), so mel_spec = mel_filters @ magnitudes
    float sum = 0.0f;

    for (int k = 0; k < n_freq; k++) {
        // FFT output is (n_frames, n_freq) row-major
        cufftComplex c = fft_out[frame * n_freq + k];
        float mag_sq = c.x * c.x + c.y * c.y;  // |z|^2
        sum += mel_filters[mel_bin * n_freq + k] * mag_sq;
    }

    // log10 with clamp
    float log_val = log10f(fmaxf(sum, 1e-10f));

    // Store raw log value; normalization (max-8, +4/4) done after finding global max
    output[mel_bin * n_frames_use + frame] = log_val;
}


// ============================================================
// Kernel 3: Log normalization
//
// log_spec = max(log_spec, global_max - 8.0)
// log_spec = (log_spec + 4.0) / 4.0
// ============================================================
__global__ void log_normalize_kernel(
    float* __restrict__ data,
    int n,
    float global_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = data[idx];
    val = fmaxf(val, global_max - 8.0f);
    val = (val + 4.0f) / 4.0f;
    data[idx] = val;
}


// ============================================================
// Host orchestration
// ============================================================
torch::Tensor mel_spectrogram_cuda(
    torch::Tensor audio,         // 1D float32 CUDA tensor (raw waveform)
    torch::Tensor mel_filters,   // (n_mels, n_fft/2) float32 CUDA tensor
    torch::Tensor hann_window,   // (n_fft,) float32 CUDA tensor
    int n_fft,
    int hop_length,
    int padding
) {
    TORCH_CHECK(audio.is_cuda(), "audio must be CUDA");
    TORCH_CHECK(mel_filters.is_cuda(), "mel_filters must be CUDA");
    TORCH_CHECK(hann_window.is_cuda(), "hann_window must be CUDA");
    TORCH_CHECK(audio.dtype() == torch::kFloat32, "audio must be float32");

    int n_mels = mel_filters.size(0);

    // Step 1: Pad audio (padding zeros at end + n_fft/2 center padding on each side)
    int pad_end = padding;
    int center_pad = n_fft / 2;
    int orig_len = audio.size(0);
    int padded_len = center_pad + orig_len + pad_end + center_pad;

    auto padded = torch::zeros({padded_len}, audio.options());
    // reflect padding for center (matching numpy reflect mode)
    // For simplicity, use zero padding — the difference is negligible for Whisper
    // Actually, faster-whisper uses reflect padding in stft. Let's match it.
    // Copy audio to center
    padded.slice(0, center_pad, center_pad + orig_len + pad_end).slice(0, 0, orig_len).copy_(audio);
    // Reflect left
    if (center_pad > 0) {
        auto left_reflect = audio.slice(0, 1, center_pad + 1).flip(0);
        padded.slice(0, 0, center_pad).copy_(left_reflect);
    }
    // Reflect right
    int right_start = center_pad + orig_len + pad_end;
    int right_len = center_pad;
    if (right_len > 0 && orig_len + pad_end > 1) {
        int src_end = orig_len + pad_end;
        int reflect_avail = src_end - 1;
        int actual_reflect = (right_len < reflect_avail) ? right_len : reflect_avail;
        if (actual_reflect > 0) {
            // Source for reflection: padded[center_pad + src_end - 2, ..., center_pad + src_end - 1 - actual_reflect]
            auto right_src = padded.slice(0, right_start - 1 - actual_reflect, right_start - 1).flip(0);
            padded.slice(0, right_start, right_start + actual_reflect).copy_(right_src);
        }
    }

    int audio_len = padded_len;
    int n_frames = 1 + (audio_len - n_fft) / hop_length;

    // Step 2: Frame extraction + windowing
    auto frames = torch::zeros({n_frames, n_fft}, audio.options());

    frame_and_window_kernel<<<n_frames, n_fft>>>(
        padded.data_ptr<float>(),
        hann_window.data_ptr<float>(),
        frames.data_ptr<float>(),
        n_fft,
        hop_length,
        n_frames,
        audio_len
    );

    // Step 3: cuFFT (real-to-complex, batched)
    int n_freq = n_fft / 2 + 1;  // 201 for n_fft=400
    auto fft_out = torch::zeros({n_frames, n_freq, 2}, audio.options());  // complex as (re, im)

    cufftHandle plan;
    cufftPlan1d(&plan, n_fft, CUFFT_R2C, n_frames);
    cufftExecR2C(plan, frames.data_ptr<float>(), (cufftComplex*)fft_out.data_ptr<float>());
    cufftDestroy(plan);

    // Step 4: Power spectrum + mel filterbank + log10
    int n_frames_use = n_frames - 1;  // drop last frame (matching stft[..., :-1])
    auto output = torch::zeros({n_mels, n_frames_use}, audio.options());

    // 2D grid: (frames, mels)
    dim3 block(16, 16);
    dim3 grid(
        (n_frames_use + block.x - 1) / block.x,
        (n_mels + block.y - 1) / block.y
    );

    mel_log_kernel<<<grid, block>>>(
        (cufftComplex*)fft_out.data_ptr<float>(),
        mel_filters.data_ptr<float>(),
        output.data_ptr<float>(),
        n_frames,
        n_frames_use,
        n_freq,
        n_mels
    );

    // Step 5: Find global max and normalize
    float global_max = output.max().item<float>();

    int total = n_mels * n_frames_use;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    log_normalize_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        total,
        global_max
    );

    return output;
}
"""

cpp_source = r"""
#include <torch/extension.h>

torch::Tensor mel_spectrogram_cuda(
    torch::Tensor audio,
    torch::Tensor mel_filters,
    torch::Tensor hann_window,
    int n_fft,
    int hop_length,
    int padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mel_spectrogram", &mel_spectrogram_cuda, "Fused mel spectrogram (CUDA)");
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="mel_spectrogram_cuda",
            cpp_sources=[cpp_source],
            cuda_sources=[cuda_source],
            extra_ldflags=["-lcufft"],
            verbose=False,
        )
    return _module


def mel_spectrogram_cuda(
    waveform: np.ndarray,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
    padding: int = 160,
    mel_filters: np.ndarray | None = None,
) -> np.ndarray:
    """Compute log-mel spectrogram on GPU using fused CUDA kernels.

    Args:
        waveform: 1D float32 numpy array of audio samples (16kHz mono).
        n_fft: FFT window size.
        hop_length: Hop between frames.
        n_mels: Number of mel bands.
        padding: Zero-padding at end of audio.
        mel_filters: Precomputed mel filterbank (n_mels, n_fft//2+1-1).
                     If None, computed from scratch.

    Returns:
        Log-mel spectrogram as numpy array, shape (n_mels, n_frames).
        Compatible with faster_whisper.WhisperModel.encode().
    """
    mod = _get_module()

    if mel_filters is None:
        from faster_whisper.feature_extractor import FeatureExtractor
        mel_filters = FeatureExtractor.get_mel_filters(16000, n_fft, n_mels=n_mels).astype(np.float32)

    waveform_t = torch.from_numpy(waveform.astype(np.float32)).cuda()
    mel_filters_t = torch.from_numpy(mel_filters).cuda()
    hann_window = torch.hann_window(n_fft, device="cuda", dtype=torch.float32)

    result = mod.mel_spectrogram(
        waveform_t, mel_filters_t, hann_window,
        n_fft, hop_length, padding,
    )

    return result.cpu().numpy()


class CUDAFeatureExtractor:
    """Drop-in replacement for faster_whisper.FeatureExtractor using CUDA kernels.

    Preserves all attributes of the original so faster-whisper's internals
    (chunk_length, nb_max_frames, time_per_frame, etc.) continue to work.
    Only __call__ is overridden to use the GPU.
    """

    def __init__(self, original):
        self._original = original
        # Copy all attributes so faster-whisper code can access them
        self.n_fft = original.n_fft
        self.hop_length = original.hop_length
        self.chunk_length = original.chunk_length
        self.n_samples = original.n_samples
        self.nb_max_frames = original.nb_max_frames
        self.time_per_frame = original.time_per_frame
        self.sampling_rate = original.sampling_rate
        self.mel_filters = original.mel_filters

        # Pre-upload mel filters and window to GPU (reused across calls)
        self._mel_filters_cuda = torch.from_numpy(original.mel_filters).cuda()
        self._hann_window = torch.hann_window(self.n_fft, device="cuda", dtype=torch.float32)

    def __call__(self, waveform: np.ndarray, padding: int = 160, chunk_length=None):
        if chunk_length is not None:
            self.n_samples = chunk_length * self.sampling_rate
            self.nb_max_frames = self.n_samples // self.hop_length

        mod = _get_module()

        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)

        waveform_t = torch.from_numpy(waveform).cuda()

        result = mod.mel_spectrogram(
            waveform_t, self._mel_filters_cuda, self._hann_window,
            self.n_fft, self.hop_length, padding,
        )

        return result.cpu().numpy()

    # Delegate stft and get_mel_filters to original (used in some code paths)
    @staticmethod
    def get_mel_filters(*args, **kwargs):
        from faster_whisper.feature_extractor import FeatureExtractor
        return FeatureExtractor.get_mel_filters(*args, **kwargs)

    @staticmethod
    def stft(*args, **kwargs):
        from faster_whisper.feature_extractor import FeatureExtractor
        return FeatureExtractor.stft(*args, **kwargs)


def patch_whisper_model(model: WhisperModel) -> None:
    """Replace a WhisperModel's feature extractor with the CUDA version.

    Args:
        model: A loaded faster_whisper.WhisperModel instance.
    """
    # Eagerly compile the kernel on first patch
    _get_module()

    original_fe = model.feature_extractor
    cuda_fe = CUDAFeatureExtractor(original_fe)
    model.feature_extractor = cuda_fe
    logger.info("Patched WhisperModel with CUDA mel spectrogram kernel")
