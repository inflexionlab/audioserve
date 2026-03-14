"""AudioServe Engine — the core orchestrator."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import numpy as np

from audioserve.config import (
    DiarizationConfig,
    EngineConfig,
    ModelConfig,
    SchedulerConfig,
    ServerConfig,
    StreamingConfig,
    TaskType,
)
from audioserve.core.preprocessing import AudioData, load_audio
from audioserve.core.scheduler import DynamicBatchScheduler, InferenceRequest
from audioserve.models.base import (
    BaseModelRunner,
    DiarizedTranscriptionResult,
    TranscriptionResult,
)
from audioserve.core.streaming import StreamingSession
from audioserve.core.vad import SileroVAD
from audioserve.models.diarization import DiarizationRunner
from audioserve.models.registry import create_runner

logger = logging.getLogger(__name__)


class AudioServeEngine:
    """Main engine that manages models, scheduling, and inference.

    Requests are routed through a DynamicBatchScheduler which accumulates
    concurrent requests, forms optimal batches (sorted by duration), and
    dispatches them to the model runner's transcribe_batch().

    Usage:
        engine = AudioServeEngine(model="openai/whisper-large-v3")
        engine.start()
        result = await engine.transcribe("audio.wav")
        engine.serve(port=8000)
    """

    def __init__(
        self,
        model: str | None = None,
        models: list[dict] | None = None,
        dtype: str = "float16",
        max_batch_size: int = 32,
        gpu_memory_utilization: float = 0.9,
        diarization: bool = False,
        hf_token: str | None = None,
    ) -> None:
        # Build config from convenience args
        model_configs = []

        if model:
            model_configs.append(
                ModelConfig(model_id=model, dtype=dtype, max_batch_size=max_batch_size)
            )
        if models:
            for m in models:
                model_configs.append(
                    ModelConfig(
                        model_id=m["model"],
                        task=TaskType(m.get("task", "asr")),
                        dtype=m.get("dtype", dtype),
                        max_batch_size=m.get("max_batch_size", max_batch_size),
                    )
                )

        diarization_config = None
        if diarization:
            diarization_config = DiarizationConfig(auth_token=hf_token)

        self.config = EngineConfig(
            models=model_configs,
            diarization=diarization_config,
            scheduler=SchedulerConfig(max_batch_size=max_batch_size),
            gpu_memory_utilization=gpu_memory_utilization,
        )

        self._runners: dict[str, BaseModelRunner] = {}
        self._diarization_runner: DiarizationRunner | None = None
        self._vad: SileroVAD | None = None
        self._scheduler = DynamicBatchScheduler(self.config.scheduler)
        self._batch_workers: list[asyncio.Task] = []
        self._running = False
        self._ready = False

    def start(self) -> None:
        """Load all models into GPU memory."""
        logger.info("Starting AudioServe engine...")

        for model_config in self.config.models:
            runner = create_runner(model_config)
            runner.load()
            self._runners[model_config.model_id] = runner
            self._scheduler.set_model_max_batch_size(
                model_config.model_id, model_config.max_batch_size
            )
            logger.info("Model ready: %s", model_config.model_id)

        if self.config.diarization:
            self._diarization_runner = DiarizationRunner(self.config.diarization)
            self._diarization_runner.load()
            logger.info("Diarization pipeline ready")

        self._running = True
        self._ready = True
        logger.info(
            "AudioServe engine started with %d model(s)", len(self._runners)
        )

    async def start_batch_workers(self) -> None:
        """Start background batch worker loops for each loaded model.

        Must be called from an async context (e.g. FastAPI lifespan).
        """
        for model_id in self._runners:
            task = asyncio.create_task(self._batch_worker_loop(model_id))
            self._batch_workers.append(task)
            logger.info("Batch worker started for %s", model_id)

    def stop(self) -> None:
        """Unload all models and stop workers."""
        self._running = False
        self._ready = False

        for task in self._batch_workers:
            task.cancel()
        self._batch_workers.clear()

        for runner in self._runners.values():
            runner.unload()
        self._runners.clear()

        if self._diarization_runner:
            self._diarization_runner.unload()
            self._diarization_runner = None

        self._vad = None

        logger.info("AudioServe engine stopped")

    def start_vad(self) -> None:
        """Load the Silero VAD model for streaming support."""
        if self._vad is None:
            self._vad = SileroVAD(
                threshold=self.config.streaming.vad_threshold,
                min_speech_ms=self.config.streaming.min_speech_ms,
                min_silence_ms=self.config.streaming.min_silence_ms,
            )
        if not self._vad.is_loaded:
            self._vad.load()
            logger.info("VAD ready for streaming")

    def create_streaming_session(
        self,
        model: str | None = None,
        language: str | None = None,
        beam_size: int = 5,
    ) -> StreamingSession:
        """Create a new streaming ASR session.

        Args:
            model: Model ID to use. Defaults to first loaded model.
            language: Language code. None for auto-detect.
            beam_size: Beam search width.

        Returns:
            StreamingSession that the WebSocket handler feeds audio into.
        """
        if self._vad is None or not self._vad.is_loaded:
            raise RuntimeError("VAD not loaded. Call start_vad() first.")

        runner = self._get_runner(model)

        # Each session gets its own VAD instance with fresh state
        session_vad = SileroVAD(
            threshold=self.config.streaming.vad_threshold,
            min_speech_ms=self.config.streaming.min_speech_ms,
            min_silence_ms=self.config.streaming.min_silence_ms,
        )
        session_vad.model = self._vad.model  # Share the model weights
        session_vad.reset()

        session = StreamingSession(
            runner=runner,
            vad=session_vad,
            language=language,
            beam_size=beam_size,
            inference_interval_ms=self.config.streaming.inference_interval_ms,
            max_buffer_seconds=self.config.streaming.max_buffer_seconds,
        )
        return session

    @property
    def has_streaming(self) -> bool:
        """True when VAD is loaded and streaming is available."""
        return self._vad is not None and self._vad.is_loaded

    async def transcribe(
        self,
        audio: str | bytes | np.ndarray,
        model: str | None = None,
        language: str | None = None,
        beam_size: int = 5,
        word_timestamps: bool = True,
    ) -> TranscriptionResult:
        """Transcribe audio using the specified (or default) model.

        When batch workers are running (server mode), enqueues the request
        into the scheduler for optimal batching. Otherwise falls back to
        direct single-request inference.

        Args:
            audio: File path, bytes, or numpy array.
            model: Model ID to use. Defaults to first loaded model.
            language: Language code (e.g. 'en', 'ru'). None for auto-detect.
            beam_size: Beam search width.
            word_timestamps: Whether to compute word-level timestamps.
        """
        runner = self._get_runner(model)
        audio_data = load_audio(audio)

        params = {
            "language": language,
            "beam_size": beam_size,
            "word_timestamps": word_timestamps,
        }

        # If batch workers are active, route through scheduler
        if self._batch_workers:
            request = InferenceRequest(
                audio_duration=audio_data.duration,
                payload=audio_data.waveform,
                model_id=runner.model_id,
                params=params,
            )
            future = await self._scheduler.enqueue(request)
            return await future

        # Direct inference for CLI / non-server usage
        results = runner.transcribe_batch(
            [audio_data.waveform],
            [params],
        )
        return results[0]

    async def transcribe_with_diarization(
        self,
        audio: str | bytes | np.ndarray,
        model: str | None = None,
        language: str | None = None,
        beam_size: int = 5,
        word_timestamps: bool = True,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> DiarizedTranscriptionResult:
        """Transcribe audio with speaker diarization.

        Runs ASR and diarization in parallel, then merges results.
        """
        if not self._diarization_runner or not self._diarization_runner.is_loaded:
            raise RuntimeError(
                "Diarization not enabled. Pass diarization=True when creating the engine."
            )

        audio_data = load_audio(audio)

        # Run ASR
        transcription = await self.transcribe(
            audio_data.waveform,
            model=model,
            language=language,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
        )

        # Run diarization
        diarization_segments = self._diarization_runner.diarize(
            audio_data.waveform,
            sample_rate=audio_data.sample_rate,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        # Merge
        result = self._diarization_runner.merge_transcription_with_diarization(
            transcription, diarization_segments
        )

        return result

    async def _batch_worker_loop(self, model_id: str) -> None:
        """Background loop: pull batches from the scheduler and run inference."""
        runner = self._runners[model_id]

        while self._running:
            try:
                batch = await self._scheduler.get_batch(model_id)
            except asyncio.CancelledError:
                break

            try:
                audio_arrays = [r.payload for r in batch.requests]
                params = [r.params for r in batch.requests]

                results = await asyncio.get_event_loop().run_in_executor(
                    None, runner.transcribe_batch, audio_arrays, params,
                )

                for request, result in zip(batch.requests, results):
                    if request.future and not request.future.done():
                        request.future.set_result(result)

            except Exception as e:
                logger.exception("Batch inference failed for %s", model_id)
                for request in batch.requests:
                    if request.future and not request.future.done():
                        request.future.set_exception(e)

    def serve(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        grpc_port: int | None = 50051,
    ) -> None:
        """Start the HTTP and gRPC servers (blocking).

        Args:
            host: Bind address for REST API.
            port: Port for REST API.
            grpc_port: Port for gRPC API. None to disable gRPC.
        """
        import uvicorn

        from audioserve.api.rest_server import create_app

        self.config.server.host = host
        self.config.server.port = port

        # Start gRPC in background thread
        grpc_server = None
        if grpc_port:
            from audioserve.api.grpc_server import start_grpc_server

            grpc_server = start_grpc_server(self, port=grpc_port)
            self.config.server.grpc_port = grpc_port

        try:
            app = create_app(self)
            uvicorn.run(app, host=host, port=port, log_level="info")
        finally:
            if grpc_server:
                grpc_server.stop(grace=5)

    def _get_runner(self, model_id: str | None = None) -> BaseModelRunner:
        """Get a model runner by ID, or the default (first) one."""
        if not self._runners:
            raise RuntimeError("No models loaded. Call start() first.")

        if model_id is None:
            return next(iter(self._runners.values()))

        if model_id not in self._runners:
            available = list(self._runners.keys())
            raise ValueError(
                f"Model '{model_id}' not loaded. Available: {available}"
            )

        return self._runners[model_id]

    @property
    def loaded_models(self) -> list[str]:
        return list(self._runners.keys())

    @property
    def is_ready(self) -> bool:
        """True when all models are loaded and engine is accepting requests."""
        return self._ready

    @property
    def has_diarization(self) -> bool:
        return self._diarization_runner is not None and self._diarization_runner.is_loaded
