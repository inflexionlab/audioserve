"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def mono_silence_1s() -> np.ndarray:
    """1 second of silence at 16kHz."""
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def mono_tone_3s() -> np.ndarray:
    """3 seconds of 440Hz sine wave at 16kHz."""
    t = np.linspace(0, 3, 3 * 16000, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def stereo_tone_1s() -> np.ndarray:
    """1 second stereo 440Hz tone at 16kHz."""
    t = np.linspace(0, 1, 16000, dtype=np.float32)
    mono = 0.5 * np.sin(2 * np.pi * 440 * t)
    return np.stack([mono, mono], axis=1)  # shape (16000, 2)


@pytest.fixture
def audio_44100_1s() -> np.ndarray:
    """1 second of tone at 44100Hz (needs resampling)."""
    t = np.linspace(0, 1, 44100, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t)
