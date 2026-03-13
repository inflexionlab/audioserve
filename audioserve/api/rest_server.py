"""FastAPI REST server for AudioServe."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

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

# Valid Whisper language codes
_VALID_LANGUAGES = {
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs",
    "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu",
    "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jw", "ka",
    "kk", "km", "kn", "ko", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml",
    "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt",
    "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw",
    "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo",
    "yue", "zh",
}


def _sanitize_language(language: str | None) -> str | None:
    """Validate and sanitize language parameter."""
    if language is None or language.strip() == "":
        return None
    lang = language.strip().lower()
    if lang == "string" or lang == "null" or lang == "none":
        return None
    if lang not in _VALID_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid language code '{language}'. Use ISO 639-1 codes like 'en', 'ru', 'de'.",
        )
    return lang


def create_app(engine: AudioServeEngine) -> FastAPI:
    """Create the FastAPI application with all routes."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await engine.start_batch_workers()
        logger.info("Batch workers started")
        yield
        # Workers are cleaned up by engine.stop()

    app = FastAPI(
        title="AudioServe",
        description="Optimized inference server for audio models",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {type(exc).__name__}: {exc}"},
        )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        status = "ok" if engine.is_ready else "loading"
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
            status=status,
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

        lang = _sanitize_language(language)

        try:
            result = await engine.transcribe(
                audio_bytes,
                model=model,
                language=lang,
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

        lang = _sanitize_language(language)

        try:
            result = await engine.transcribe_with_diarization(
                audio_bytes,
                model=model,
                language=lang,
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
