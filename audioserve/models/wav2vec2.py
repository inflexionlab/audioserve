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

# Max chunk duration in seconds for CTC models (avoids OOM on long audio)
MAX_CHUNK_SECONDS = 30
# Overlap between chunks in seconds (avoids cutting words at boundaries)
CHUNK_OVERLAP_SECONDS = 2


class Wav2Vec2Runner(BaseModelRunner):
    """Wav2Vec2 / HuBERT inference using HuggingFace Transformers + torch.compile.

    Supports any CTC-based encoder model from HuggingFace:
    - facebook/wav2vec2-base-960h
    - facebook/wav2vec2-large-xlsr-53-*
    - facebook/hubert-large-ls960-ft
    - jonatasgrosman/wav2vec2-large-xlsr-53-russian
    - etc.

    Long audio is automatically chunked (30s windows with 2s overlap)
    to avoid GPU OOM.
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
        """Transcribe a batch of audio arrays, chunking long audio to avoid OOM."""
        if not self._model or not self._processor:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = []
        for audio, p in zip(audio_arrays, params):
            result = self._transcribe_single_chunked(audio)
            results.append(result)

        return results

    def _transcribe_single_chunked(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe a single audio array, splitting into chunks if needed."""
        t0 = time.monotonic()
        duration = len(audio) / 16000.0
        chunk_samples = MAX_CHUNK_SECONDS * 16000
        overlap_samples = CHUNK_OVERLAP_SECONDS * 16000
        step = chunk_samples - overlap_samples

        if len(audio) <= chunk_samples:
            # Short audio — process directly
            text = self._infer_chunk(audio)
            processing_time = time.monotonic() - t0
            segments = [Segment(text=text, start=0.0, end=duration)] if text else []
            return TranscriptionResult(
                text=text,
                segments=segments,
                duration=duration,
                processing_time=processing_time,
            )

        # Long audio — chunk with overlap
        logger.info(
            "Chunking %.1fs audio into %ds windows with %ds overlap",
            duration, MAX_CHUNK_SECONDS, CHUNK_OVERLAP_SECONDS,
        )

        segments = []
        offset = 0

        while offset < len(audio):
            end = min(offset + chunk_samples, len(audio))
            chunk = audio[offset:end]

            chunk_text = self._infer_chunk(chunk)

            if chunk_text:
                chunk_start = offset / 16000.0
                chunk_end = end / 16000.0
                segments.append(Segment(
                    text=chunk_text,
                    start=chunk_start,
                    end=chunk_end,
                ))

            offset += step

        # Deduplicate overlapping text between consecutive chunks
        merged_segments = self._merge_overlapping_segments(segments)
        full_text = " ".join(s.text for s in merged_segments)

        processing_time = time.monotonic() - t0
        logger.info(
            "Wav2Vec2 chunked inference: %.1fs audio in %.1fs (%.1fx realtime)",
            duration, processing_time,
            duration / processing_time if processing_time > 0 else 0,
        )

        return TranscriptionResult(
            text=full_text,
            segments=merged_segments,
            duration=duration,
            processing_time=processing_time,
        )

    def _infer_chunk(self, audio: np.ndarray) -> str:
        """Run CTC inference on a single chunk."""
        inputs = self._processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=self.config.dtype == "float16"):
                logits = self._model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        text = self._processor.batch_decode(predicted_ids)[0]
        return text.strip()

    def _merge_overlapping_segments(self, segments: list[Segment]) -> list[Segment]:
        """Simple dedup: for overlapping chunks, remove duplicate words at boundaries."""
        if len(segments) <= 1:
            return segments

        merged = [segments[0]]

        for seg in segments[1:]:
            prev_words = merged[-1].text.split()
            curr_words = seg.text.split()

            # Find overlap: check if last N words of prev match first N words of curr
            best_overlap = 0
            max_check = min(len(prev_words), len(curr_words), 10)

            for n in range(1, max_check + 1):
                if prev_words[-n:] == curr_words[:n]:
                    best_overlap = n

            if best_overlap > 0:
                # Remove overlapping words from current segment
                deduped_text = " ".join(curr_words[best_overlap:])
                if deduped_text:
                    merged.append(Segment(
                        text=deduped_text,
                        start=seg.start,
                        end=seg.end,
                    ))
            else:
                merged.append(seg)

        return merged

    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
