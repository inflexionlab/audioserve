"""Tests for configuration dataclasses."""

from __future__ import annotations

from audioserve.config import (
    DeviceType,
    DiarizationConfig,
    EngineConfig,
    ModelConfig,
    SchedulerConfig,
    ServerConfig,
    TaskType,
)


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig(model_id="openai/whisper-tiny")
        assert cfg.task == TaskType.ASR
        assert cfg.device == DeviceType.CUDA
        assert cfg.device_index == 0
        assert cfg.dtype == "float16"
        assert cfg.max_batch_size == 32

    def test_compute_type_auto(self):
        cfg = ModelConfig(model_id="test", dtype="int8")
        assert cfg.compute_type == "int8"

    def test_compute_type_override(self):
        cfg = ModelConfig(model_id="test", compute_type="int8_float16")
        assert cfg.compute_type == "int8_float16"


class TestSchedulerConfig:
    def test_defaults(self):
        cfg = SchedulerConfig()
        assert cfg.max_batch_size == 32
        assert cfg.max_wait_time_ms == 100.0
        assert cfg.sort_by_duration is True
        assert cfg.max_audio_duration == 600.0


class TestEngineConfig:
    def test_defaults(self):
        cfg = EngineConfig()
        assert cfg.models == []
        assert cfg.diarization is None
        assert cfg.gpu_memory_utilization == 0.9

    def test_with_diarization(self):
        cfg = EngineConfig(
            diarization=DiarizationConfig(auth_token="test-token"),
        )
        assert cfg.diarization is not None
        assert cfg.diarization.auth_token == "test-token"


class TestDiarizationConfig:
    def test_defaults(self):
        cfg = DiarizationConfig()
        assert cfg.model_id == "pyannote/speaker-diarization-3.1"
        assert cfg.auth_token is None
