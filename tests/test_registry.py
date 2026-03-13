"""Tests for model registry."""

from __future__ import annotations

import pytest

from audioserve.config import ModelConfig
from audioserve.models.registry import create_runner


class TestCreateRunner:
    def test_whisper_tiny(self):
        config = ModelConfig(model_id="openai/whisper-tiny")
        runner = create_runner(config)
        from audioserve.models.whisper import WhisperRunner
        assert isinstance(runner, WhisperRunner)

    def test_whisper_large_v3(self):
        config = ModelConfig(model_id="openai/whisper-large-v3")
        runner = create_runner(config)
        from audioserve.models.whisper import WhisperRunner
        assert isinstance(runner, WhisperRunner)

    def test_distil_whisper(self):
        config = ModelConfig(model_id="distil-whisper/distil-large-v3")
        runner = create_runner(config)
        from audioserve.models.whisper import WhisperRunner
        assert isinstance(runner, WhisperRunner)

    def test_systran_faster_whisper(self):
        config = ModelConfig(model_id="Systran/faster-whisper-large-v3")
        runner = create_runner(config)
        from audioserve.models.whisper import WhisperRunner
        assert isinstance(runner, WhisperRunner)

    def test_wav2vec2(self):
        config = ModelConfig(model_id="facebook/wav2vec2-base-960h")
        runner = create_runner(config)
        from audioserve.models.wav2vec2 import Wav2Vec2Runner
        assert isinstance(runner, Wav2Vec2Runner)

    def test_hubert(self):
        config = ModelConfig(model_id="facebook/hubert-large-ls960-ft")
        runner = create_runner(config)
        from audioserve.models.wav2vec2 import Wav2Vec2Runner
        assert isinstance(runner, Wav2Vec2Runner)

    def test_unknown_model_raises(self):
        config = ModelConfig(model_id="some/random-model")
        with pytest.raises(ValueError, match="Unknown model architecture"):
            create_runner(config)

    def test_case_insensitive(self):
        config = ModelConfig(model_id="OpenAI/Whisper-Tiny")
        runner = create_runner(config)
        from audioserve.models.whisper import WhisperRunner
        assert isinstance(runner, WhisperRunner)
