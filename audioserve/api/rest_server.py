"""FastAPI REST server for AudioServe."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from audioserve.api.schemas import (
    DiarizedTranscribeResponse,
    DiarizeRequest,
    HealthResponse,
    ModelInfoResponse,
    SegmentResponse,
    TranscribeResponse,
    WordResponse,
)

if TYPE_CHECKING:
    from audioserve.engine import AudioServeEngine

logger = logging.getLogger(__name__)


def create_app(engine: AudioServeEngine) -> FastAPI:
    """Create the FastAPI application with all routes."""

    app = FastAPI(
        title="AudioServe",
        description="Optimized inference server for audio models",
        version="0.1.0",
    )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        models = []
        for model_id, runner in engine._runners.items():
            models.append(
                ModelInfoResponse(
                    model_id=model_id,
                    backend=runner.__class__.__name__,
                    is_loaded=runner.is_loaded,
                )
            )
        return HealthResponse(
            status="ok",
            models=models,
            diarization_available=engine.has_diarization,
        )

    @app.post("/v1/transcribe", response_model=TranscribeResponse)
    async def transcribe(
        audio: UploadFile = File(..., description="Audio file (wav, mp3, flac, ogg, etc.)"),
        language: str | None = Form(None),
        beam_size: int = Form(5),
        word_timestamps: bool = Form(True),
        model: str | None = Form(None),
    ):
        """Transcribe an audio file.

        Upload any audio file and get back transcription with timestamps.
        """
        t0 = time.monotonic()

        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        try:
            result = await engine.transcribe(
                audio_bytes,
                model=model,
                language=language,
                beam_size=beam_size,
                word_timestamps=word_timestamps,
            )
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return _to_transcribe_response(result)

    @app.post("/v1/transcribe+diarize", response_model=DiarizedTranscribeResponse)
    async def transcribe_with_diarization(
        audio: UploadFile = File(..., description="Audio file"),
        language: str | None = Form(None),
        beam_size: int = Form(5),
        word_timestamps: bool = Form(True),
        model: str | None = Form(None),
        min_speakers: int | None = Form(None),
        max_speakers: int | None = Form(None),
    ):
        """Transcribe audio with speaker diarization.

        Returns transcription with speaker labels assigned to each segment.
        """
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        if not engine.has_diarization:
            raise HTTPException(
                status_code=503,
                detail="Diarization not enabled. Start engine with diarization=True.",
            )

        try:
            result = await engine.transcribe_with_diarization(
                audio_bytes,
                model=model,
                language=language,
                beam_size=beam_size,
                word_timestamps=word_timestamps,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return DiarizedTranscribeResponse(
            text=result.text,
            segments=[
                SegmentResponse(
                    text=s.text,
                    start=s.start,
                    end=s.end,
                    words=[
                        WordResponse(word=w.word, start=w.start, end=w.end, confidence=w.confidence)
                        for w in s.words
                    ],
                    speaker=s.speaker,
                    confidence=s.confidence,
                )
                for s in result.segments
            ],
            speakers=result.speakers,
            language=result.language,
            duration=result.duration,
            processing_time=result.processing_time,
        )

    @app.get("/v1/models")
    async def list_models():
        """List all loaded models."""
        return {
            "models": [
                {
                    "model_id": model_id,
                    "backend": runner.__class__.__name__,
                    "is_loaded": runner.is_loaded,
                }
                for model_id, runner in engine._runners.items()
            ]
        }

    return app


def _to_transcribe_response(result) -> TranscribeResponse:
    return TranscribeResponse(
        text=result.text,
        segments=[
            SegmentResponse(
                text=s.text,
                start=s.start,
                end=s.end,
                words=[
                    WordResponse(word=w.word, start=w.start, end=w.end, confidence=w.confidence)
                    for w in s.words
                ],
                speaker=s.speaker,
                confidence=s.confidence,
            )
            for s in result.segments
        ],
        language=result.language,
        language_confidence=result.language_confidence,
        duration=result.duration,
        processing_time=result.processing_time,
    )
