"""Audio preprocessing — loading, resampling, normalization."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf


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
    """
    if isinstance(source, np.ndarray):
        waveform = source.astype(np.float32)
        sr = sample_rate or TARGET_SAMPLE_RATE
    elif isinstance(source, bytes):
        waveform, sr = sf.read(io.BytesIO(source), dtype="float32")
    elif isinstance(source, str):
        waveform, sr = sf.read(source, dtype="float32")
    else:
        raise ValueError(f"Unsupported audio source type: {type(source)}")

    # Convert stereo to mono
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    original_sr = sr

    # Resample to 16kHz if needed
    if sr != TARGET_SAMPLE_RATE:
        waveform = _resample(waveform, sr, TARGET_SAMPLE_RATE)
        sr = TARGET_SAMPLE_RATE

    duration = len(waveform) / sr

    return AudioData(
        waveform=waveform,
        sample_rate=sr,
        duration=duration,
        original_sample_rate=original_sr,
    )


def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using linear interpolation (fast, good enough for inference).

    For production, consider torchaudio or librosa resampling.
    """
    if orig_sr == target_sr:
        return waveform

    ratio = target_sr / orig_sr
    new_length = int(len(waveform) * ratio)
    indices = np.arange(new_length) / ratio
    left = np.floor(indices).astype(np.int64)
    right = np.minimum(left + 1, len(waveform) - 1)
    frac = indices - left
    return waveform[left] * (1 - frac) + waveform[right] * frac
