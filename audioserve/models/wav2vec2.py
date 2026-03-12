"""Wav2Vec2 / HuBERT model runner using PyTorch + torch.compile."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch

from audioserve.config import ModelConfig
from audioserve.models.base import (
    BaseModelRunner,
    Segment,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)


class Wav2Vec2Runner(BaseModelRunner):
    """Wav2Vec2 / HuBERT inference using HuggingFace Transformers + torch.compile.

    Supports any CTC-based encoder model from HuggingFace:
    - facebook/wav2vec2-base-960h
    - facebook/wav2vec2-large-xlsr-53-*
    - facebook/hubert-large-ls960-ft
    - jonatasgrosman/wav2vec2-large-xlsr-53-russian
    - etc.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._model = None
        self._processor = None
        self._device = None

    def load(self) -> None:
        from transformers import AutoModelForCTC, AutoProcessor

        device_str = f"{self.config.device.value}:{self.config.device_index}"
        self._device = torch.device(device_str)

        dtype = torch.float16 if self.config.dtype == "float16" else torch.float32

        logger.info("Loading Wav2Vec2 model: %s (dtype=%s)", self.config.model_id, self.config.dtype)
        t0 = time.monotonic()

        self._processor = AutoProcessor.from_pretrained(self.config.model_id)
        self._model = AutoModelForCTC.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
        ).to(self._device)
        self._model.eval()

        # torch.compile for inference optimization
        try:
            self._model = torch.compile(self._model, mode="reduce-overhead")
            logger.info("torch.compile applied successfully")
        except Exception as e:
            logger.warning("torch.compile failed, falling back to eager mode: %s", e)

        load_time = time.monotonic() - t0
        logger.info("Wav2Vec2 model loaded in %.1fs", load_time)

    def unload(self) -> None:
        import gc

        del self._model
        del self._processor
        self._model = None
        self._processor = None

        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Wav2Vec2 model unloaded")

    def transcribe_batch(
        self,
        audio_arrays: list[Any],
        params: list[dict],
    ) -> list[TranscriptionResult]:
        """Batched CTC inference with dynamic padding."""
        if not self._model or not self._processor:
            raise RuntimeError("Model not loaded. Call load() first.")

        t0 = time.monotonic()

        # Tokenize with padding
        inputs = self._processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        # Inference
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=self.config.dtype == "float16"):
                logits = self._model(**inputs).logits

        # CTC greedy decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcriptions = self._processor.batch_decode(predicted_ids)

        batch_time = time.monotonic() - t0

        # Build results
        results = []
        for i, (text, audio) in enumerate(zip(transcriptions, audio_arrays)):
            duration = len(audio) / 16000.0
            # Generate approximate segment (CTC doesn't give natural segments)
            segments = [Segment(text=text.strip(), start=0.0, end=duration)] if text.strip() else []

            results.append(
                TranscriptionResult(
                    text=text.strip(),
                    segments=segments,
                    duration=duration,
                    processing_time=batch_time / len(audio_arrays),
                )
            )

        logger.debug(
            "Wav2Vec2 batch inference: %d items in %.3fs (%.1fx realtime)",
            len(audio_arrays),
            batch_time,
            sum(len(a) / 16000 for a in audio_arrays) / batch_time if batch_time > 0 else 0,
        )

        return results

    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
