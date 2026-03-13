"""Tests for multi-model serving."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from audioserve.cli import _parse_model_specs
from audioserve.config import SchedulerConfig
from audioserve.core.scheduler import DynamicBatchScheduler, InferenceRequest
from audioserve.engine import AudioServeEngine


# --- CLI model spec parsing ---


class TestParseModelSpecs:
    def test_single_model(self):
        specs = _parse_model_specs(("openai/whisper-tiny",), "float16")
        assert specs == [{"model": "openai/whisper-tiny", "dtype": "float16"}]

    def test_model_with_dtype(self):
        specs = _parse_model_specs(("facebook/wav2vec2-base-960h:float32",), "float16")
        assert specs == [{"model": "facebook/wav2vec2-base-960h", "dtype": "float32"}]

    def test_multiple_models(self):
        specs = _parse_model_specs(
            ("openai/whisper-tiny", "facebook/wav2vec2-base-960h:float32"),
            "float16",
        )
        assert len(specs) == 2
        assert specs[0] == {"model": "openai/whisper-tiny", "dtype": "float16"}
        assert specs[1] == {"model": "facebook/wav2vec2-base-960h", "dtype": "float32"}

    def test_invalid_dtype_treated_as_model_id(self):
        specs = _parse_model_specs(("org/model:notadtype",), "float16")
        assert specs == [{"model": "org/model:notadtype", "dtype": "float16"}]

    def test_default_dtype_applied(self):
        specs = _parse_model_specs(("openai/whisper-tiny",), "int8")
        assert specs[0]["dtype"] == "int8"


# --- Engine multi-model init ---


class TestEngineMultiModel:
    def test_multi_model_config(self):
        engine = AudioServeEngine(models=[
            {"model": "openai/whisper-tiny"},
            {"model": "openai/whisper-base"},
        ])
        assert len(engine.config.models) == 2
        assert engine.config.models[0].model_id == "openai/whisper-tiny"
        assert engine.config.models[1].model_id == "openai/whisper-base"

    def test_multi_model_different_dtypes(self):
        engine = AudioServeEngine(models=[
            {"model": "openai/whisper-tiny", "dtype": "float16"},
            {"model": "facebook/wav2vec2-base-960h", "dtype": "float32"},
        ])
        assert engine.config.models[0].dtype == "float16"
        assert engine.config.models[1].dtype == "float32"

    def test_get_runner_default_returns_first(self):
        engine = AudioServeEngine(models=[
            {"model": "openai/whisper-tiny"},
            {"model": "openai/whisper-base"},
        ])
        # Mock runners
        runner_a = MagicMock()
        runner_b = MagicMock()
        engine._runners["openai/whisper-tiny"] = runner_a
        engine._runners["openai/whisper-base"] = runner_b

        assert engine._get_runner() is runner_a

    def test_get_runner_by_id(self):
        engine = AudioServeEngine(models=[
            {"model": "openai/whisper-tiny"},
            {"model": "openai/whisper-base"},
        ])
        runner_a = MagicMock()
        runner_b = MagicMock()
        engine._runners["openai/whisper-tiny"] = runner_a
        engine._runners["openai/whisper-base"] = runner_b

        assert engine._get_runner("openai/whisper-base") is runner_b

    def test_get_runner_invalid_lists_available(self):
        engine = AudioServeEngine(models=[
            {"model": "openai/whisper-tiny"},
            {"model": "openai/whisper-base"},
        ])
        engine._runners["openai/whisper-tiny"] = MagicMock()
        engine._runners["openai/whisper-base"] = MagicMock()

        with pytest.raises(ValueError, match="whisper-tiny"):
            engine._get_runner("nonexistent/model")

    def test_loaded_models_lists_all(self):
        engine = AudioServeEngine(models=[
            {"model": "openai/whisper-tiny"},
            {"model": "openai/whisper-base"},
        ])
        engine._runners["openai/whisper-tiny"] = MagicMock()
        engine._runners["openai/whisper-base"] = MagicMock()

        assert engine.loaded_models == ["openai/whisper-tiny", "openai/whisper-base"]


# --- Scheduler per-model batch size ---


class TestSchedulerPerModelBatchSize:
    @pytest.fixture
    def scheduler(self):
        config = SchedulerConfig(max_batch_size=8, max_wait_time_ms=50)
        return DynamicBatchScheduler(config)

    def _make_request(self, model_id: str, duration: float = 1.0) -> InferenceRequest:
        return InferenceRequest(
            audio_duration=duration,
            payload=f"audio_{duration}s",
            model_id=model_id,
        )

    @pytest.mark.asyncio
    async def test_per_model_limit_respected(self, scheduler):
        scheduler.set_model_max_batch_size("small-model", 2)

        for i in range(5):
            await scheduler.enqueue(self._make_request("small-model", float(i)))

        batch = await asyncio.wait_for(
            scheduler.get_batch("small-model"), timeout=1.0
        )
        assert batch.size == 2
        assert scheduler.pending_count("small-model") == 3

    @pytest.mark.asyncio
    async def test_global_limit_used_when_no_override(self, scheduler):
        for i in range(10):
            await scheduler.enqueue(self._make_request("default-model", float(i)))

        batch = await asyncio.wait_for(
            scheduler.get_batch("default-model"), timeout=1.0
        )
        assert batch.size == 8  # global max_batch_size

    @pytest.mark.asyncio
    async def test_per_model_capped_by_global(self, scheduler):
        # Per-model is higher than global — should use global
        scheduler.set_model_max_batch_size("big-model", 100)

        for i in range(12):
            await scheduler.enqueue(self._make_request("big-model", float(i)))

        batch = await asyncio.wait_for(
            scheduler.get_batch("big-model"), timeout=1.0
        )
        assert batch.size == 8  # capped at global

    @pytest.mark.asyncio
    async def test_different_models_different_limits(self, scheduler):
        scheduler.set_model_max_batch_size("model-a", 2)
        scheduler.set_model_max_batch_size("model-b", 4)

        for i in range(6):
            await scheduler.enqueue(self._make_request("model-a", float(i)))
            await scheduler.enqueue(self._make_request("model-b", float(i)))

        batch_a = await asyncio.wait_for(
            scheduler.get_batch("model-a"), timeout=1.0
        )
        batch_b = await asyncio.wait_for(
            scheduler.get_batch("model-b"), timeout=1.0
        )
        assert batch_a.size == 2
        assert batch_b.size == 4


# --- Integration: multi-model with GPU ---


@pytest.mark.gpu
class TestMultiModelIntegration:
    """Integration tests that require GPU + model downloads."""

    @pytest.fixture(autouse=True)
    def setup_engine(self):
        self.engine = AudioServeEngine(models=[
            {"model": "openai/whisper-tiny"},
            {"model": "openai/whisper-base"},
        ])
        self.engine.start()
        yield
        self.engine.stop()

    @pytest.mark.asyncio
    async def test_both_models_loaded(self):
        assert "openai/whisper-tiny" in self.engine.loaded_models
        assert "openai/whisper-base" in self.engine.loaded_models
        assert self.engine.is_ready

    @pytest.mark.asyncio
    async def test_route_to_specific_model(self):
        audio = np.zeros(16000 * 2, dtype=np.float32)
        result = await self.engine.transcribe(audio, model="openai/whisper-base")
        assert hasattr(result, "text")

    @pytest.mark.asyncio
    async def test_default_routes_to_first(self):
        audio = np.zeros(16000 * 2, dtype=np.float32)
        result = await self.engine.transcribe(audio)
        assert hasattr(result, "text")

    @pytest.mark.asyncio
    async def test_concurrent_different_models(self):
        await self.engine.start_batch_workers()

        audio = np.zeros(16000 * 2, dtype=np.float32)
        task_a = asyncio.create_task(
            self.engine.transcribe(audio, model="openai/whisper-tiny")
        )
        task_b = asyncio.create_task(
            self.engine.transcribe(audio, model="openai/whisper-base")
        )

        r_a = await asyncio.wait_for(task_a, timeout=30.0)
        r_b = await asyncio.wait_for(task_b, timeout=30.0)
        assert hasattr(r_a, "text")
        assert hasattr(r_b, "text")

    @pytest.mark.asyncio
    async def test_invalid_model_returns_error(self):
        with pytest.raises(ValueError, match="not loaded"):
            await self.engine.transcribe(
                np.zeros(16000, dtype=np.float32),
                model="nonexistent/model",
            )
