"""Audio preprocessing — loading, resampling, normalization."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np


TARGET_SAMPLE_RATE = 16000


@dataclass
class AudioData:
    """Preprocessed audio ready for model inference."""

    waveform: np.ndarray  # float32, mono, 16kHz, shape (num_samples,)
    sample_rate: int
    duration: float  # seconds
    original_sample_rate: int

    @property
    def num_samples(self) -> int:
        return self.waveform.shape[0]


def load_audio(source: str | bytes | np.ndarray, sample_rate: int | None = None) -> AudioData:
    """Load audio from file path, bytes, or numpy array.

    Always returns mono float32 at 16kHz.
    Supports wav, flac, ogg, mp3, and other formats via librosa/soundfile.
    """
    if isinstance(source, np.ndarray):
        waveform = source.astype(np.float32)
        sr = sample_rate or TARGET_SAMPLE_RATE
    elif isinstance(source, (bytes, str)):
        waveform, sr = _read_audio(source)
    else:
        raise ValueError(f"Unsupported audio source type: {type(source)}")

    # Convert stereo to mono
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1).astype(np.float32)

    original_sr = sr

    # Resample to 16kHz if needed
    if sr != TARGET_SAMPLE_RATE:
        waveform = _resample(waveform, sr, TARGET_SAMPLE_RATE)
        sr = TARGET_SAMPLE_RATE

    # Ensure float32 (critical — ONNX VAD and CTranslate2 require it)
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)

    duration = len(waveform) / sr

    return AudioData(
        waveform=waveform,
        sample_rate=sr,
        duration=duration,
        original_sample_rate=original_sr,
    )


def _read_audio(source: str | bytes) -> tuple[np.ndarray, int]:
    """Read audio from file path or bytes. Handles mp3, wav, flac, ogg, etc."""
    import soundfile as sf

    try:
        if isinstance(source, bytes):
            waveform, sr = sf.read(io.BytesIO(source), dtype="float32")
        else:
            waveform, sr = sf.read(source, dtype="float32")
        return waveform, sr
    except Exception:
        # soundfile can't handle mp3 and some other formats — fall back to librosa
        pass

    import librosa

    if isinstance(source, bytes):
        waveform, sr = librosa.load(io.BytesIO(source), sr=None, mono=False)
    else:
        waveform, sr = librosa.load(source, sr=None, mono=False)

    # librosa returns (channels, samples) for multi-channel, transpose to (samples, channels)
    if waveform.ndim > 1:
        waveform = waveform.T

    return waveform.astype(np.float32), sr


def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio. Uses soxr if available, falls back to linear interpolation."""
    if orig_sr == target_sr:
        return waveform

    try:
        import soxr

        return soxr.resample(waveform, orig_sr, target_sr).astype(np.float32)
    except ImportError:
        pass

    # Linear interpolation fallback
    ratio = target_sr / orig_sr
    new_length = int(len(waveform) * ratio)
    indices = np.arange(new_length, dtype=np.float32) / np.float32(ratio)
    left = np.floor(indices).astype(np.int64)
    right = np.minimum(left + 1, len(waveform) - 1)
    frac = indices - left.astype(np.float32)
    return (waveform[left] * (1 - frac) + waveform[right] * frac).astype(np.float32)
