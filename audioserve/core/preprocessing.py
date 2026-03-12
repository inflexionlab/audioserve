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
    """Read audio from file path or bytes. Handles mp3, wav, flac, ogg, etc.

    Tries soundfile first (fast, supports wav/flac/ogg).
    Falls back to ffmpeg subprocess for mp3 and other formats.
    """
    import soundfile as sf

    buf = io.BytesIO(source) if isinstance(source, bytes) else None

    try:
        if buf is not None:
            waveform, sr = sf.read(buf, dtype="float32")
        else:
            waveform, sr = sf.read(source, dtype="float32")
        return waveform, sr
    except Exception:
        pass

    # Fallback: decode via ffmpeg (handles mp3, aac, wma, opus, etc.)
    return _read_audio_ffmpeg(source)


def _read_audio_ffmpeg(source: str | bytes) -> tuple[np.ndarray, int]:
    """Decode any audio format to float32 PCM via ffmpeg subprocess."""
    import subprocess
    import tempfile

    if isinstance(source, bytes):
        # Write bytes to a temp file — ffmpeg reads from file, outputs to pipe
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
            tmp.write(source)
            tmp_path = tmp.name
        input_path = tmp_path
    else:
        input_path = source
        tmp_path = None

    try:
        # ffmpeg: decode to 16kHz mono float32 PCM on stdout
        cmd = [
            "ffmpeg", "-i", input_path,
            "-f", "f32le",       # raw float32 little-endian
            "-acodec", "pcm_f32le",
            "-ar", "16000",      # resample to 16kHz
            "-ac", "1",          # mono
            "-v", "error",       # suppress noisy output
            "pipe:1",            # output to stdout
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=300)

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed to decode audio: {stderr}")

        waveform = np.frombuffer(result.stdout, dtype=np.float32)
        return waveform, 16000

    finally:
        if tmp_path:
            import os
            os.unlink(tmp_path)


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
