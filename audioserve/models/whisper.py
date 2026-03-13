"""Whisper model runner using faster-whisper (CTranslate2) backend."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from audioserve.config import ModelConfig
from audioserve.models.base import (
    BaseModelRunner,
    Segment,
    TranscriptionResult,
    WordInfo,
)

logger = logging.getLogger(__name__)


# Maps HuggingFace model IDs to faster-whisper model sizes
_HF_TO_FASTER_WHISPER = {
    "openai/whisper-tiny": "tiny",
    "openai/whisper-tiny.en": "tiny.en",
    "openai/whisper-base": "base",
    "openai/whisper-base.en": "base.en",
    "openai/whisper-small": "small",
    "openai/whisper-small.en": "small.en",
    "openai/whisper-medium": "medium",
    "openai/whisper-medium.en": "medium.en",
    "openai/whisper-large-v2": "large-v2",
    "openai/whisper-large-v3": "large-v3",
    "openai/whisper-large-v3-turbo": "large-v3-turbo",
    "distil-whisper/distil-large-v3": "distil-large-v3",
    "distil-whisper/distil-large-v2": "distil-large-v2",
    "distil-whisper/distil-medium.en": "distil-medium.en",
    "distil-whisper/distil-small.en": "distil-small.en",
}

# faster-whisper compute type mapping
_DTYPE_TO_COMPUTE_TYPE = {
    "float16": "float16",
    "float32": "float32",
    "int8": "int8",
    "int8_float16": "int8_float16",
}


class WhisperRunner(BaseModelRunner):
    """Whisper inference using faster-whisper (CTranslate2).

    Supports all Whisper model sizes and distil-whisper variants.
    Uses faster-whisper's built-in batched inference with dynamic padding.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._model = None
        self._batched_model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel, BatchedInferencePipeline

        model_size = self._resolve_model_size()
        compute_type = _DTYPE_TO_COMPUTE_TYPE.get(self.config.dtype, "float16")

        logger.info(
            "Loading Whisper model: %s (compute_type=%s, device=%s:%d)",
            model_size,
            compute_type,
            self.config.device.value,
            self.config.device_index,
        )

        t0 = time.monotonic()
        self._model = WhisperModel(
            model_size,
            device=self.config.device.value,
            device_index=self.config.device_index,
            compute_type=compute_type,
        )
        self._batched_model = BatchedInferencePipeline(model=self._model)

        # Patch feature extractor with CUDA kernel if available
        try:
            from audioserve.cuda.mel_spectrogram import patch_whisper_model
            patch_whisper_model(self._model)
        except Exception as e:
            logger.warning("CUDA mel spectrogram unavailable, using CPU: %s", e)

        load_time = time.monotonic() - t0
        logger.info("Whisper model loaded in %.1fs", load_time)

    def unload(self) -> None:
        del self._batched_model
        del self._model
        self._batched_model = None
        self._model = None

        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Whisper model unloaded")

    def transcribe_batch(
        self,
        audio_arrays: list[Any],
        params: list[dict],
    ) -> list[TranscriptionResult]:
        """Transcribe a batch of audio arrays.

        For a single input, uses BatchedInferencePipeline which internally
        splits long audio into VAD segments and processes them in batches.
        For multiple inputs, processes each sequentially through the batched
        pipeline (faster-whisper doesn't support cross-input batching, but
        each input benefits from internal VAD-based batching).
        """
        if not self._batched_model:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = []
        for audio, p in zip(audio_arrays, params):
            result = self._transcribe_single(audio, p)
            results.append(result)

        return results

    def transcribe_single(
        self,
        audio: np.ndarray,
        params: dict | None = None,
    ) -> TranscriptionResult:
        """Transcribe a single audio array. Convenience method."""
        return self._transcribe_single(audio, params or {})

    def _transcribe_single(
        self,
        audio: np.ndarray,
        params: dict,
    ) -> TranscriptionResult:
        """Internal single-audio transcription."""
        if self._batched_model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        t0 = time.monotonic()

        language = params.get("language")
        beam_size = params.get("beam_size", 5)
        word_timestamps = params.get("word_timestamps", True)
        batch_size = params.get("batch_size", self.config.max_batch_size)

        segments_iter, info = self._batched_model.transcribe(
            audio,
            language=language,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
            batch_size=batch_size,
        )

        segments = []
        full_text_parts = []

        for seg in segments_iter:
            words = []
            if seg.words:
                words = [
                    WordInfo(
                        word=w.word.strip(),
                        start=w.start,
                        end=w.end,
                        confidence=w.probability,
                    )
                    for w in seg.words
                ]

            segment = Segment(
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
                words=words,
                confidence=seg.avg_logprob,
            )
            segments.append(segment)
            full_text_parts.append(seg.text.strip())

        processing_time = time.monotonic() - t0
        duration = len(audio) / 16000.0

        return TranscriptionResult(
            text=" ".join(full_text_parts),
            segments=segments,
            language=info.language,
            language_confidence=info.language_probability,
            duration=duration,
            processing_time=processing_time,
        )

    def _resolve_model_size(self) -> str:
        """Resolve model ID to faster-whisper model size string."""
        model_id = self.config.model_id

        # Check direct mapping
        if model_id in _HF_TO_FASTER_WHISPER:
            return _HF_TO_FASTER_WHISPER[model_id]

        # Check if it's already a faster-whisper size name
        valid_sizes = set(_HF_TO_FASTER_WHISPER.values())
        if model_id in valid_sizes:
            return model_id

        # Check Systran/faster-whisper-* models (pass through as HF ID)
        if model_id.startswith("Systran/faster-whisper-"):
            return model_id

        # Assume it's a HuggingFace model ID or local path — pass through
        return model_id

    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
