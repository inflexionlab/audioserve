"""gRPC server for AudioServe."""

from __future__ import annotations

import asyncio
import logging
from concurrent import futures
from typing import TYPE_CHECKING

import grpc

from audioserve.proto import audioserve_pb2, audioserve_pb2_grpc

if TYPE_CHECKING:
    from audioserve.engine import AudioServeEngine

logger = logging.getLogger(__name__)


class AudioServeServicer(audioserve_pb2_grpc.AudioServeServicer):
    """gRPC service implementation for AudioServe."""

    def __init__(self, engine: AudioServeEngine) -> None:
        self.engine = engine
        self._loop = asyncio.new_event_loop()

    def Transcribe(self, request, context):
        """Transcribe audio."""
        try:
            language = request.language if request.HasField("language") else None
            model = request.model if request.HasField("model") else None
            beam_size = request.beam_size or 5

            result = self._loop.run_until_complete(
                self.engine.transcribe(
                    audio=request.audio,
                    model=model,
                    language=language,
                    beam_size=beam_size,
                    word_timestamps=request.word_timestamps,
                )
            )

            return _to_transcribe_response(result)

        except Exception as e:
            logger.exception("gRPC Transcribe error")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return audioserve_pb2.TranscribeResponse()

    def TranscribeWithDiarization(self, request, context):
        """Transcribe audio with speaker diarization."""
        if not self.engine.has_diarization:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Diarization not enabled.")
            return audioserve_pb2.DiarizedTranscribeResponse()

        try:
            language = request.language if request.HasField("language") else None
            model = request.model if request.HasField("model") else None
            beam_size = request.beam_size or 5
            min_speakers = request.min_speakers if request.HasField("min_speakers") else None
            max_speakers = request.max_speakers if request.HasField("max_speakers") else None

            # Treat 0 as "not set"
            if min_speakers == 0:
                min_speakers = None
            if max_speakers == 0:
                max_speakers = None

            result = self._loop.run_until_complete(
                self.engine.transcribe_with_diarization(
                    audio=request.audio,
                    model=model,
                    language=language,
                    beam_size=beam_size,
                    word_timestamps=request.word_timestamps,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
            )

            segments = [
                audioserve_pb2.Segment(
                    text=s.text,
                    start=s.start,
                    end=s.end,
                    words=[
                        audioserve_pb2.WordInfo(
                            word=w.word, start=w.start, end=w.end,
                            confidence=w.confidence or 0.0,
                        )
                        for w in s.words
                    ],
                    speaker=s.speaker or "",
                    confidence=s.confidence or 0.0,
                )
                for s in result.segments
            ]

            return audioserve_pb2.DiarizedTranscribeResponse(
                text=result.text,
                segments=segments,
                speakers=result.speakers,
                language=result.language or "",
                duration=result.duration,
                processing_time=result.processing_time,
            )

        except Exception as e:
            logger.exception("gRPC TranscribeWithDiarization error")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return audioserve_pb2.DiarizedTranscribeResponse()

    def ListModels(self, request, context):
        models = [
            audioserve_pb2.ModelInfo(
                model_id=model_id,
                backend=runner.__class__.__name__,
                is_loaded=runner.is_loaded,
            )
            for model_id, runner in self.engine._runners.items()
        ]
        return audioserve_pb2.ListModelsResponse(models=models)

    def HealthCheck(self, request, context):
        models = [
            audioserve_pb2.ModelInfo(
                model_id=model_id,
                backend=runner.__class__.__name__,
                is_loaded=runner.is_loaded,
            )
            for model_id, runner in self.engine._runners.items()
        ]
        status = "ok" if self.engine.is_ready else "loading"
        return audioserve_pb2.HealthCheckResponse(
            status=status,
            models=models,
            diarization_available=self.engine.has_diarization,
        )


def _to_transcribe_response(result) -> audioserve_pb2.TranscribeResponse:
    segments = [
        audioserve_pb2.Segment(
            text=s.text,
            start=s.start,
            end=s.end,
            words=[
                audioserve_pb2.WordInfo(
                    word=w.word, start=w.start, end=w.end,
                    confidence=w.confidence or 0.0,
                )
                for w in s.words
            ],
            speaker=s.speaker or "",
            confidence=s.confidence or 0.0,
        )
        for s in result.segments
    ]

    return audioserve_pb2.TranscribeResponse(
        text=result.text,
        segments=segments,
        language=result.language or "",
        language_confidence=result.language_confidence or 0.0,
        duration=result.duration,
        processing_time=result.processing_time,
    )


def start_grpc_server(engine: AudioServeEngine, port: int = 50051) -> grpc.Server:
    """Start the gRPC server in a background thread."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = AudioServeServicer(engine)
    audioserve_pb2_grpc.add_AudioServeServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("gRPC server started on port %d", port)
    return server
