"""Model registry — resolves model IDs to the correct runner."""

from __future__ import annotations

import logging

from audioserve.config import ModelConfig
from audioserve.models.base import BaseModelRunner

logger = logging.getLogger(__name__)

# Patterns to identify model architecture from model ID
_WHISPER_PATTERNS = [
    "whisper",
    "distil-whisper",
    "Systran/faster-whisper",
]

_WAV2VEC2_PATTERNS = [
    "wav2vec2",
    "hubert",
    "data2vec-audio",
]


def create_runner(config: ModelConfig) -> BaseModelRunner:
    """Create the appropriate model runner based on model ID."""
    model_id = config.model_id.lower()

    if any(p.lower() in model_id for p in _WHISPER_PATTERNS):
        from audioserve.models.whisper import WhisperRunner

        logger.info("Resolved %s → WhisperRunner", config.model_id)
        return WhisperRunner(config)

    if any(p.lower() in model_id for p in _WAV2VEC2_PATTERNS):
        from audioserve.models.wav2vec2 import Wav2Vec2Runner

        logger.info("Resolved %s → Wav2Vec2Runner", config.model_id)
        return Wav2Vec2Runner(config)

    raise ValueError(
        f"Unknown model architecture for '{config.model_id}'. "
        f"Supported patterns: Whisper (openai/whisper-*, distil-whisper/*), "
        f"Wav2Vec2/HuBERT (facebook/wav2vec2-*, facebook/hubert-*)"
    )
