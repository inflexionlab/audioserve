"""Tests for NVIDIA Parakeet model runner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from audioserve.config import ModelConfig
from audioserve.models.parakeet import ParakeetRunner


class TestParakeetRunner:
    """Unit tests for ParakeetRunner (no GPU required)."""

    def test_init(self):
        config = ModelConfig(model_id="nvidia/parakeet-ctc-0.6b")
        runner = ParakeetRunner(config)
        assert runner.model_id == "nvidia/parakeet-ctc-0.6b"
        assert not runner.is_loaded

    def test_not_loaded_raises(self):
        config = ModelConfig(model_id="nvidia/parakeet-ctc-0.6b")
        runner = ParakeetRunner(config)
        with pytest.raises(RuntimeError, match="not loaded"):
            runner.transcribe_batch([np.zeros(16000)], [{}])

    def test_merge_no_overlap(self):
        from audioserve.models.base import Segment

        config = ModelConfig(model_id="nvidia/parakeet-ctc-0.6b")
        runner = ParakeetRunner(config)

        segments = [
            Segment(text="hello world", start=0.0, end=1.0),
            Segment(text="foo bar", start=1.0, end=2.0),
        ]
        merged = runner._merge_overlapping_segments(segments)
        assert len(merged) == 2

    def test_merge_with_overlap(self):
        from audioserve.models.base import Segment

        config = ModelConfig(model_id="nvidia/parakeet-ctc-0.6b")
        runner = ParakeetRunner(config)

        segments = [
            Segment(text="hello world foo", start=0.0, end=1.0),
            Segment(text="world foo bar baz", start=0.8, end=2.0),
        ]
        merged = runner._merge_overlapping_segments(segments)
        assert len(merged) == 2
        assert merged[1].text == "bar baz"

    def test_merge_single_segment(self):
        from audioserve.models.base import Segment

        config = ModelConfig(model_id="nvidia/parakeet-ctc-0.6b")
        runner = ParakeetRunner(config)

        segments = [Segment(text="hello", start=0.0, end=1.0)]
        merged = runner._merge_overlapping_segments(segments)
        assert len(merged) == 1


class TestParakeetRegistry:
    """Test that Parakeet models are resolved by the registry."""

    def test_parakeet_ctc_06b(self):
        from audioserve.models.registry import create_runner

        config = ModelConfig(model_id="nvidia/parakeet-ctc-0.6b")
        runner = create_runner(config)
        assert isinstance(runner, ParakeetRunner)

    def test_parakeet_ctc_11b(self):
        from audioserve.models.registry import create_runner

        config = ModelConfig(model_id="nvidia/parakeet-ctc-1.1b")
        runner = create_runner(config)
        assert isinstance(runner, ParakeetRunner)

    def test_parakeet_case_insensitive(self):
        from audioserve.models.registry import create_runner

        config = ModelConfig(model_id="NVIDIA/Parakeet-CTC-0.6b")
        runner = create_runner(config)
        assert isinstance(runner, ParakeetRunner)


@pytest.mark.gpu
class TestParakeetIntegration:
    """GPU integration tests with real Parakeet model."""

    @pytest.fixture(autouse=True)
    def setup_model(self):
        config = ModelConfig(model_id="nvidia/parakeet-ctc-0.6b", dtype="float32")
        self.runner = ParakeetRunner(config)
        self.runner.load()
        yield
        self.runner.unload()

    def test_is_loaded(self):
        assert self.runner.is_loaded

    def test_transcribe_silence(self):
        audio = np.zeros(16000 * 2, dtype=np.float32)
        result = self.runner.transcribe_batch([audio], [{}])[0]
        assert isinstance(result.text, str)
        assert result.duration == pytest.approx(2.0, abs=0.1)
        assert result.processing_time > 0

    def test_transcribe_tone(self):
        audio = np.sin(2 * np.pi * 440 * np.arange(16000 * 3) / 16000).astype(np.float32)
        result = self.runner.transcribe_batch([audio], [{}])[0]
        assert isinstance(result.text, str)

    def test_transcribe_real_audio(self):
        """Test with real audio file if available."""
        import os
        audio_path = "/audioserve/audio.mp3"
        if not os.path.exists(audio_path):
            pytest.skip("audio.mp3 not available")

        from audioserve.core.preprocessing import load_audio
        audio_data = load_audio(audio_path)
        result = self.runner.transcribe_batch([audio_data.waveform], [{}])[0]
        assert len(result.text) > 100
        assert result.duration > 300
        assert len(result.segments) >= 1
