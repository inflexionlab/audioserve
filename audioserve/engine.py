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
    TaskType,
)
from audioserve.core.preprocessing import AudioData, load_audio
from audioserve.core.scheduler import DynamicBatchScheduler, InferenceRequest
from audioserve.models.base import (
    BaseModelRunner,
    DiarizedTranscriptionResult,
    TranscriptionResult,
)
from audioserve.models.diarization import DiarizationRunner
from audioserve.models.registry import create_runner

logger = logging.getLogger(__name__)


class AudioServeEngine:
    """Main engine that manages models, scheduling, and inference.

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
        self._scheduler = DynamicBatchScheduler(self.config.scheduler)
        self._batch_workers: list[asyncio.Task] = []
        self._running = False

    def start(self) -> None:
        """Load all models into GPU memory."""
        logger.info("Starting AudioServe engine...")

        for model_config in self.config.models:
            runner = create_runner(model_config)
            runner.load()
            self._runners[model_config.model_id] = runner
            logger.info("Model ready: %s", model_config.model_id)

        if self.config.diarization:
            self._diarization_runner = DiarizationRunner(self.config.diarization)
            self._diarization_runner.load()
            logger.info("Diarization pipeline ready")

        self._running = True
        logger.info(
            "AudioServe engine started with %d model(s)", len(self._runners)
        )

    def stop(self) -> None:
        """Unload all models and stop workers."""
        self._running = False

        for task in self._batch_workers:
            task.cancel()
        self._batch_workers.clear()

        for runner in self._runners.values():
            runner.unload()
        self._runners.clear()

        if self._diarization_runner:
            self._diarization_runner.unload()
            self._diarization_runner = None

        logger.info("AudioServe engine stopped")

    async def transcribe(
        self,
        audio: str | bytes | np.ndarray,
        model: str | None = None,
        language: str | None = None,
        beam_size: int = 5,
        word_timestamps: bool = True,
    ) -> TranscriptionResult:
        """Transcribe audio using the specified (or default) model.

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

        # Direct inference (bypass scheduler for single requests)
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

    def serve(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the HTTP/gRPC server (blocking)."""
        import uvicorn

        from audioserve.api.rest_server import create_app

        self.config.server.host = host
        self.config.server.port = port

        app = create_app(self)
        uvicorn.run(app, host=host, port=port, log_level="info")

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
    def has_diarization(self) -> bool:
        return self._diarization_runner is not None and self._diarization_runner.is_loaded
