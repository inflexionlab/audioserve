"""Speaker diarization using pyannote.audio."""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from audioserve.config import DiarizationConfig
from audioserve.models.base import DiarizationSegment, DiarizedTranscriptionResult, Segment, TranscriptionResult

logger = logging.getLogger(__name__)


class DiarizationRunner:
    """Speaker diarization using pyannote.audio.

    Provides "who spoke when" and merges with ASR results
    to produce speaker-attributed transcriptions.
    """

    def __init__(self, config: DiarizationConfig) -> None:
        self.config = config
        self._pipeline = None

    def load(self) -> None:
        from pyannote.audio import Pipeline

        logger.info("Loading diarization pipeline: %s", self.config.model_id)
        t0 = time.monotonic()

        self._pipeline = Pipeline.from_pretrained(
            self.config.model_id,
            use_auth_token=self.config.auth_token,
        )

        if torch.cuda.is_available():
            self._pipeline.to(torch.device("cuda"))

        load_time = time.monotonic() - t0
        logger.info("Diarization pipeline loaded in %.1fs", load_time)

    def unload(self) -> None:
        import gc

        del self._pipeline
        self._pipeline = None
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def diarize(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[DiarizationSegment]:
        """Run diarization on audio, return speaker segments."""
        if not self._pipeline:
            raise RuntimeError("Diarization pipeline not loaded. Call load() first.")

        t0 = time.monotonic()

        # pyannote expects {"waveform": tensor(1, T), "sample_rate": int}
        waveform_tensor = torch.from_numpy(waveform).unsqueeze(0).float()
        audio_input = {"waveform": waveform_tensor, "sample_rate": sample_rate}

        min_sp = min_speakers or self.config.min_speakers
        max_sp = max_speakers or self.config.max_speakers

        kwargs = {}
        if min_sp is not None:
            kwargs["min_speakers"] = min_sp
        if max_sp is not None:
            kwargs["max_speakers"] = max_sp

        diarization = self._pipeline(audio_input, **kwargs)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                DiarizationSegment(
                    speaker=speaker,
                    start=turn.start,
                    end=turn.end,
                )
            )

        elapsed = time.monotonic() - t0
        n_speakers = len(set(s.speaker for s in segments))
        logger.info(
            "Diarization complete: %d segments, %d speakers in %.1fs",
            len(segments),
            n_speakers,
            elapsed,
        )

        return segments

    def merge_transcription_with_diarization(
        self,
        transcription: TranscriptionResult,
        diarization_segments: list[DiarizationSegment],
    ) -> DiarizedTranscriptionResult:
        """Merge ASR word/segment timestamps with diarization speaker labels.

        For each ASR segment (or word), find the overlapping diarization segment
        and assign the speaker label.
        """
        asr_segments = transcription.segments
        merged_segments = []

        for seg in asr_segments:
            # If we have word-level timestamps, assign speakers per word
            if seg.words:
                speaker = self._find_speaker_for_interval(
                    seg.start, seg.end, diarization_segments
                )
                merged_seg = Segment(
                    text=seg.text,
                    start=seg.start,
                    end=seg.end,
                    words=seg.words,
                    speaker=speaker,
                    confidence=seg.confidence,
                )
                merged_segments.append(merged_seg)
            else:
                speaker = self._find_speaker_for_interval(
                    seg.start, seg.end, diarization_segments
                )
                merged_seg = Segment(
                    text=seg.text,
                    start=seg.start,
                    end=seg.end,
                    speaker=speaker,
                    confidence=seg.confidence,
                )
                merged_segments.append(merged_seg)

        all_speakers = sorted(set(s.speaker for s in merged_segments if s.speaker))

        # Build full text with speaker labels
        text_parts = []
        current_speaker = None
        for seg in merged_segments:
            if seg.speaker != current_speaker:
                current_speaker = seg.speaker
                label = current_speaker or "UNKNOWN"
                text_parts.append(f"\n[{label}]: {seg.text}")
            else:
                text_parts.append(seg.text)

        return DiarizedTranscriptionResult(
            text=" ".join(text_parts).strip(),
            segments=merged_segments,
            speakers=all_speakers,
            language=transcription.language,
            duration=transcription.duration,
            processing_time=transcription.processing_time,
        )

    def _find_speaker_for_interval(
        self,
        start: float,
        end: float,
        diarization_segments: list[DiarizationSegment],
    ) -> str | None:
        """Find the speaker with maximum overlap for a given time interval."""
        best_speaker = None
        best_overlap = 0.0

        for dseg in diarization_segments:
            overlap_start = max(start, dseg.start)
            overlap_end = min(end, dseg.end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dseg.speaker

        return best_speaker
