"""Abstract base class for all model runners."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Segment:
    """A transcription segment with timing info."""

    text: str
    start: float  # seconds
    end: float  # seconds
    words: list[WordInfo] = field(default_factory=list)
    speaker: str | None = None  # filled by diarization
    confidence: float | None = None


@dataclass
class WordInfo:
    """Word-level timing and confidence."""

    word: str
    start: float
    end: float
    confidence: float | None = None


@dataclass
class TranscriptionResult:
    """Result of an ASR inference."""

    text: str
    segments: list[Segment] = field(default_factory=list)
    language: str | None = None
    language_confidence: float | None = None
    duration: float = 0.0  # audio duration in seconds
    processing_time: float = 0.0  # inference time in seconds


@dataclass
class DiarizationSegment:
    """A speaker segment from diarization."""

    speaker: str
    start: float
    end: float


@dataclass
class DiarizedTranscriptionResult:
    """ASR result with speaker labels."""

    text: str
    segments: list[Segment] = field(default_factory=list)  # segments with .speaker filled
    speakers: list[str] = field(default_factory=list)  # unique speaker IDs
    language: str | None = None
    duration: float = 0.0
    processing_time: float = 0.0


class BaseModelRunner(abc.ABC):
    """Abstract interface for all model runners.

    Every model backend (Whisper, Wav2Vec2, Parakeet, etc.) implements this.
    """

    @abc.abstractmethod
    def load(self) -> None:
        """Load model weights into GPU memory."""

    @abc.abstractmethod
    def unload(self) -> None:
        """Release GPU memory."""

    @abc.abstractmethod
    def transcribe_batch(
        self,
        audio_arrays: list[Any],
        params: list[dict],
    ) -> list[TranscriptionResult]:
        """Run batched inference on a list of audio inputs.

        Args:
            audio_arrays: List of preprocessed audio (format depends on runner).
            params: Per-request parameters (language, beam_size, etc.).

        Returns:
            List of TranscriptionResult, one per input.
        """

    @property
    @abc.abstractmethod
    def model_id(self) -> str:
        """The model identifier (e.g. 'openai/whisper-large-v3')."""

    @property
    @abc.abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded in memory."""

    @property
    def supports_batching(self) -> bool:
        """Whether this runner supports true batched inference."""
        return True
