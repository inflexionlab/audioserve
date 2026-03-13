"""API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TranscribeRequest(BaseModel):
    """Request body for transcription (used with JSON + base64 audio)."""

    language: str | None = Field(None, description="Language code (e.g. 'en', 'ru'). None for auto-detect.")
    beam_size: int = Field(5, ge=1, le=20)
    word_timestamps: bool = True
    model: str | None = Field(None, description="Model ID to use. None for default.")


class WordResponse(BaseModel):
    word: str
    start: float
    end: float
    confidence: float | None = None


class SegmentResponse(BaseModel):
    text: str
    start: float
    end: float
    words: list[WordResponse] = []
    speaker: str | None = None
    confidence: float | None = None


class TranscribeResponse(BaseModel):
    text: str
    segments: list[SegmentResponse] = []
    language: str | None = None
    language_confidence: float | None = None
    duration: float = 0.0
    processing_time: float = 0.0


class DiarizeRequest(BaseModel):
    """Request body for transcription with diarization."""

    language: str | None = None
    beam_size: int = Field(5, ge=1, le=20)
    word_timestamps: bool = True
    model: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


class DiarizedTranscribeResponse(BaseModel):
    text: str
    segments: list[SegmentResponse] = []
    speakers: list[str] = []
    language: str | None = None
    duration: float = 0.0
    processing_time: float = 0.0


class ModelInfoResponse(BaseModel):
    model_id: str
    backend: str
    is_loaded: bool


class HealthResponse(BaseModel):
    status: str
    models: list[ModelInfoResponse] = []
    diarization_available: bool = False
    streaming_available: bool = False
