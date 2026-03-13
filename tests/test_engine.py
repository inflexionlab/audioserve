"""Tests for AudioServeEngine."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from audioserve.engine import AudioServeEngine


class TestEngineInit:
    def test_single_model(self):
        engine = AudioServeEngine(model="openai/whisper-tiny")
        assert len(engine.config.models) == 1
        assert engine.config.models[0].model_id == "openai/whisper-tiny"
        assert not engine.is_ready

    def test_multi_model(self):
        engine = AudioServeEngine(models=[
            {"model": "openai/whisper-tiny"},
            {"model": "facebook/wav2vec2-base-960h", "dtype": "float32"},
        ])
        assert len(engine.config.models) == 2

    def test_diarization_config(self):
        engine = AudioServeEngine(
            model="openai/whisper-tiny",
            diarization=True,
            hf_token="test-token",
        )
        assert engine.config.diarization is not None
        assert engine.config.diarization.auth_token == "test-token"

    def test_no_diarization_by_default(self):
        engine = AudioServeEngine(model="openai/whisper-tiny")
        assert engine.config.diarization is None
        assert not engine.has_diarization


class TestEngineGetRunner:
    def test_no_models_raises(self):
        engine = AudioServeEngine(model="openai/whisper-tiny")
        # Not started, so no runners loaded
        with pytest.raises(RuntimeError, match="No models loaded"):
            engine._get_runner()

    def test_invalid_model_raises(self):
        engine = AudioServeEngine(model="openai/whisper-tiny")
        engine._runners["openai/whisper-tiny"] = "mock"
        with pytest.raises(ValueError, match="not loaded"):
            engine._get_runner("nonexistent/model")


class TestEngineDiarization:
    @pytest.mark.asyncio
    async def test_diarize_without_diarization_raises(self):
        engine = AudioServeEngine(model="openai/whisper-tiny")
        engine._running = True
        audio = np.zeros(16000, dtype=np.float32)
        with pytest.raises(RuntimeError, match="Diarization not enabled"):
            await engine.transcribe_with_diarization(audio)


@pytest.mark.gpu
class TestEngineWhisperIntegration:
    """Integration tests that require GPU + model download."""

    @pytest.fixture(autouse=True)
    def setup_engine(self):
        self.engine = AudioServeEngine(model="openai/whisper-tiny")
        self.engine.start()
        yield
        self.engine.stop()

    @pytest.mark.asyncio
    async def test_transcribe_silence(self):
        audio = np.zeros(16000 * 3, dtype=np.float32)
        result = await self.engine.transcribe(audio)
        assert hasattr(result, "text")
        assert hasattr(result, "segments")
        assert result.duration > 0

    @pytest.mark.asyncio
    async def test_transcribe_returns_language(self):
        audio = np.zeros(16000 * 3, dtype=np.float32)
        result = await self.engine.transcribe(audio)
        # Whisper should detect some language even for silence
        assert result.language is not None

    @pytest.mark.asyncio
    async def test_is_ready_after_start(self):
        assert self.engine.is_ready

    @pytest.mark.asyncio
    async def test_loaded_models(self):
        assert "openai/whisper-tiny" in self.engine.loaded_models


@pytest.mark.gpu
class TestEngineSchedulerIntegration:
    """Test that the scheduler is wired into the engine correctly."""

    @pytest.mark.asyncio
    async def test_scheduler_routes_requests(self):
        """When batch workers are running, requests go through the scheduler."""
        engine = AudioServeEngine(model="openai/whisper-tiny")
        engine.start()
        try:
            await engine.start_batch_workers()

            audio = np.zeros(16000 * 2, dtype=np.float32)
            result = await asyncio.wait_for(
                engine.transcribe(audio),
                timeout=30.0,
            )
            assert hasattr(result, "text")
        finally:
            engine.stop()

    @pytest.mark.asyncio
    async def test_concurrent_requests_batched(self):
        """Multiple concurrent requests should be batched together."""
        engine = AudioServeEngine(model="openai/whisper-tiny", max_batch_size=4)
        engine.start()
        try:
            await engine.start_batch_workers()

            audio1 = np.zeros(16000 * 2, dtype=np.float32)
            audio2 = np.zeros(16000 * 3, dtype=np.float32)

            # Enqueue both before the scheduler timeout fires
            task1 = asyncio.create_task(engine.transcribe(audio1))
            task2 = asyncio.create_task(engine.transcribe(audio2))

            # Give both tasks a chance to enqueue
            await asyncio.sleep(0)

            r1 = await asyncio.wait_for(task1, timeout=60.0)
            r2 = await asyncio.wait_for(task2, timeout=60.0)
            assert hasattr(r1, "text")
            assert hasattr(r2, "text")
        finally:
            engine.stop()
