"""Tests for audio preprocessing pipeline."""

from __future__ import annotations

import tempfile
import subprocess

import numpy as np
import pytest

from audioserve.core.preprocessing import AudioData, load_audio, _resample


class TestLoadAudioNumpy:
    def test_float32_passthrough(self, mono_silence_1s):
        result = load_audio(mono_silence_1s)
        assert result.waveform.dtype == np.float32
        assert result.sample_rate == 16000
        assert result.num_samples == 16000
        assert abs(result.duration - 1.0) < 0.01

    def test_stereo_to_mono(self, stereo_tone_1s):
        result = load_audio(stereo_tone_1s)
        assert result.waveform.ndim == 1
        assert result.num_samples == 16000

    def test_float64_cast_to_float32(self):
        audio = np.zeros(16000, dtype=np.float64)
        result = load_audio(audio)
        assert result.waveform.dtype == np.float32

    def test_custom_sample_rate(self, mono_silence_1s):
        result = load_audio(mono_silence_1s, sample_rate=16000)
        assert result.sample_rate == 16000


class TestLoadAudioFile:
    def test_wav_file(self, mono_tone_3s):
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, mono_tone_3s, 16000)
            result = load_audio(f.name)

        assert result.waveform.dtype == np.float32
        assert abs(result.duration - 3.0) < 0.05
        assert result.sample_rate == 16000

    def test_wav_bytes(self, mono_tone_3s):
        import soundfile as sf
        import io

        buf = io.BytesIO()
        sf.write(buf, mono_tone_3s, 16000, format="WAV")
        result = load_audio(buf.getvalue())

        assert result.waveform.dtype == np.float32
        assert abs(result.duration - 3.0) < 0.05

    def test_invalid_source_type(self):
        with pytest.raises(ValueError, match="Unsupported audio source type"):
            load_audio(12345)


class TestResample:
    def test_same_rate_noop(self):
        audio = np.ones(16000, dtype=np.float32)
        result = _resample(audio, 16000, 16000)
        np.testing.assert_array_equal(result, audio)

    def test_downsample(self):
        audio = np.ones(44100, dtype=np.float32)
        result = _resample(audio, 44100, 16000)
        expected_len = int(44100 * 16000 / 44100)
        assert abs(len(result) - expected_len) <= 1
        assert result.dtype == np.float32

    def test_upsample(self):
        audio = np.ones(8000, dtype=np.float32)
        result = _resample(audio, 8000, 16000)
        expected_len = int(8000 * 16000 / 8000)
        assert abs(len(result) - expected_len) <= 1


class TestFfmpegFallback:
    def test_mp3_via_ffmpeg(self, mono_tone_3s):
        """Generate a real mp3 via ffmpeg and verify load_audio handles it."""
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_f:
            sf.write(wav_f.name, mono_tone_3s, 16000)
            wav_path = wav_f.name

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_f:
            mp3_path = mp3_f.name

        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-ar", "16000", mp3_path],
            capture_output=True,
        )
        result = load_audio(mp3_path)
        assert result.waveform.dtype == np.float32
        assert abs(result.duration - 3.0) < 0.2  # mp3 encoding adds slight padding
