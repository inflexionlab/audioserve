"""NVIDIA Parakeet model runner using HuggingFace Transformers.

Supports FastConformer-CTC models:
- nvidia/parakeet-ctc-0.6b
- nvidia/parakeet-ctc-1.1b

Uses AutoModelForCTC + AutoProcessor from transformers (no NeMo dependency).
BPE tokenizer produces word-piece output (much better than character-level CTC).
Long audio is chunked like Wav2Vec2 to avoid OOM.
"""

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

# Parakeet handles longer context than Wav2Vec2 due to efficient attention,
# but we still chunk to avoid OOM on very long audio
MAX_CHUNK_SECONDS = 60
CHUNK_OVERLAP_SECONDS = 2


class ParakeetRunner(BaseModelRunner):
    """NVIDIA Parakeet FastConformer-CTC inference via HuggingFace Transformers.

    Architecture: FastConformer encoder (24 Conformer blocks, 8x downsampling)
    with CTC head and BPE tokenizer (~1024 tokens).

    Compared to Wav2Vec2:
    - BPE tokenizer → word-piece output (no character-level gibberish)
    - FastConformer → more efficient attention for longer sequences
    - Trained on 64K hours of speech → better out-of-box accuracy
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._model = None
        self._processor = None
        self._device = None
        self._beam_search_available = False
        self._blank_id = 0

    def load(self) -> None:
        from transformers import AutoModelForCTC, AutoProcessor

        device_str = f"{self.config.device.value}:{self.config.device_index}"
        self._device = torch.device(device_str)

        dtype = torch.float16 if self.config.dtype == "float16" else torch.float32

        logger.info(
            "Loading Parakeet model: %s (dtype=%s)", self.config.model_id, self.config.dtype
        )
        t0 = time.monotonic()

        self._processor = AutoProcessor.from_pretrained(self.config.model_id)
        self._model = AutoModelForCTC.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
        ).to(self._device)
        self._model.eval()

        try:
            self._model = torch.compile(self._model, mode="default")
            logger.info("torch.compile applied successfully")
        except Exception as e:
            logger.warning("torch.compile failed, falling back to eager mode: %s", e)

        # Try to compile CUDA CTC beam search kernel
        try:
            from audioserve.cuda.ctc_beam_search import _get_module
            _get_module()
            self._beam_search_available = True
            self._blank_id = self._processor.tokenizer.pad_token_id or 0
            logger.info("CUDA CTC beam search kernel ready (blank_id=%d)", self._blank_id)
        except Exception as e:
            logger.warning("CUDA CTC beam search unavailable, using greedy: %s", e)
            self._beam_search_available = False

        load_time = time.monotonic() - t0
        logger.info("Parakeet model loaded in %.1fs", load_time)

    def unload(self) -> None:
        import gc

        del self._model
        del self._processor
        self._model = None
        self._processor = None

        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Parakeet model unloaded")

    def transcribe_batch(
        self,
        audio_arrays: list[Any],
        params: list[dict],
    ) -> list[TranscriptionResult]:
        if not self._model or not self._processor:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = []
        for audio, p in zip(audio_arrays, params):
            beam_width = p.get("beam_size", 1) if p else 1
            result = self._transcribe_single(audio, beam_width=beam_width)
            results.append(result)
        return results

    def _transcribe_single(
        self, audio: np.ndarray, beam_width: int = 1
    ) -> TranscriptionResult:
        t0 = time.monotonic()
        duration = len(audio) / 16000.0
        chunk_samples = MAX_CHUNK_SECONDS * 16000
        overlap_samples = CHUNK_OVERLAP_SECONDS * 16000
        step = chunk_samples - overlap_samples

        if len(audio) <= chunk_samples:
            text = self._infer_chunk(audio, beam_width=beam_width)
            processing_time = time.monotonic() - t0
            segments = [Segment(text=text, start=0.0, end=duration)] if text else []
            return TranscriptionResult(
                text=text,
                segments=segments,
                duration=duration,
                processing_time=processing_time,
            )

        logger.info(
            "Chunking %.1fs audio into %ds windows with %ds overlap",
            duration, MAX_CHUNK_SECONDS, CHUNK_OVERLAP_SECONDS,
        )

        segments = []
        offset = 0
        while offset < len(audio):
            end = min(offset + chunk_samples, len(audio))
            chunk = audio[offset:end]
            chunk_text = self._infer_chunk(chunk, beam_width=beam_width)
            if chunk_text:
                segments.append(Segment(
                    text=chunk_text,
                    start=offset / 16000.0,
                    end=end / 16000.0,
                ))
            offset += step

        merged = self._merge_overlapping_segments(segments)
        full_text = " ".join(s.text for s in merged)

        processing_time = time.monotonic() - t0
        logger.info(
            "Parakeet chunked inference: %.1fs audio in %.1fs (%.1fx realtime)",
            duration, processing_time,
            duration / processing_time if processing_time > 0 else 0,
        )

        return TranscriptionResult(
            text=full_text,
            segments=merged,
            duration=duration,
            processing_time=processing_time,
        )

    def _infer_chunk(self, audio: np.ndarray, beam_width: int = 1) -> str:
        inputs = self._processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=self.config.dtype == "float16"):
                logits = self._model(**inputs).logits

        if beam_width > 1 and self._beam_search_available:
            from audioserve.cuda.ctc_beam_search import ctc_beam_search_decode

            log_probs = torch.log_softmax(logits.float(), dim=-1).squeeze(0)
            text = ctc_beam_search_decode(
                log_probs,
                self._processor,
                blank_id=self._blank_id,
                beam_width=beam_width,
            )
        else:
            predicted_ids = torch.argmax(logits, dim=-1)
            text = self._processor.batch_decode(predicted_ids)[0]

        return text.strip()

    def _merge_overlapping_segments(self, segments: list[Segment]) -> list[Segment]:
        if len(segments) <= 1:
            return segments

        merged = [segments[0]]
        for seg in segments[1:]:
            prev_words = merged[-1].text.split()
            curr_words = seg.text.split()

            best_overlap = 0
            max_check = min(len(prev_words), len(curr_words), 10)
            for n in range(1, max_check + 1):
                if prev_words[-n:] == curr_words[:n]:
                    best_overlap = n

            if best_overlap > 0:
                deduped_text = " ".join(curr_words[best_overlap:])
                if deduped_text:
                    merged.append(Segment(text=deduped_text, start=seg.start, end=seg.end))
            else:
                merged.append(seg)

        return merged

    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
