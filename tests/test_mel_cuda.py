"""Tests for CUDA mel spectrogram kernel."""

import numpy as np
import pytest
import torch


@pytest.mark.gpu
class TestMelSpectrogramCUDA:
    """Validate CUDA kernel output against faster-whisper CPU reference."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from faster_whisper.feature_extractor import FeatureExtractor
        self.fe = FeatureExtractor()

    def _compare(self, audio: np.ndarray, atol: float = 1e-4):
        from audioserve.cuda.mel_spectrogram import mel_spectrogram_cuda

        cpu = self.fe(audio, padding=160)
        gpu = mel_spectrogram_cuda(audio, mel_filters=self.fe.mel_filters)
        assert cpu.shape == gpu.shape, f"Shape mismatch: {cpu.shape} vs {gpu.shape}"
        np.testing.assert_allclose(gpu, cpu, atol=atol, rtol=1e-4)

    def test_sine_wave(self):
        t = np.linspace(0, 2, 32000, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        self._compare(audio)

    def test_speech_like(self):
        rng = np.random.default_rng(42)
        t = np.linspace(0, 2, 32000, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 300 * t) + 0.1 * rng.standard_normal(32000).astype(np.float32)
        self._compare(audio)

    def test_silence(self):
        audio = np.zeros(16000, dtype=np.float32)
        self._compare(audio)

    def test_impulse(self):
        audio = np.zeros(16000, dtype=np.float32)
        audio[8000] = 1.0
        self._compare(audio)

    def test_short_audio(self):
        audio = np.random.default_rng(0).standard_normal(1600).astype(np.float32)
        self._compare(audio)

    def test_long_audio_30s(self):
        audio = np.random.default_rng(1).standard_normal(480000).astype(np.float32) * 0.1
        self._compare(audio, atol=2e-4)

    def test_real_audio(self):
        from audioserve.core.preprocessing import load_audio
        audio_data = load_audio("/audioserve/audio.mp3")
        self._compare(audio_data.waveform, atol=2e-4)

    def test_output_dtype(self):
        from audioserve.cuda.mel_spectrogram import mel_spectrogram_cuda
        audio = np.zeros(16000, dtype=np.float32)
        result = mel_spectrogram_cuda(audio, mel_filters=self.fe.mel_filters)
        assert result.dtype == np.float32

    def test_speedup(self):
        """Verify GPU is at least 5x faster than CPU on a 30s clip."""
        import time

        from audioserve.cuda.mel_spectrogram import mel_spectrogram_cuda

        audio = np.random.default_rng(2).standard_normal(480000).astype(np.float32) * 0.3

        # CPU
        times_cpu = []
        for _ in range(5):
            t0 = time.perf_counter()
            self.fe(audio, padding=160)
            times_cpu.append(time.perf_counter() - t0)

        # GPU warmup
        mel_spectrogram_cuda(audio, mel_filters=self.fe.mel_filters)
        torch.cuda.synchronize()

        times_gpu = []
        for _ in range(5):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            mel_spectrogram_cuda(audio, mel_filters=self.fe.mel_filters)
            torch.cuda.synchronize()
            times_gpu.append(time.perf_counter() - t0)

        speedup = np.median(times_cpu) / np.median(times_gpu)
        assert speedup > 5, f"Expected >5x speedup, got {speedup:.1f}x"
