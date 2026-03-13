"""Engine and model configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class TaskType(str, Enum):
    ASR = "asr"
    DIARIZATION = "diarization"


class DeviceType(str, Enum):
    CUDA = "cuda"
    CPU = "cpu"


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    model_id: str
    task: TaskType = TaskType.ASR
    device: DeviceType = DeviceType.CUDA
    device_index: int = 0
    dtype: Literal["float16", "float32", "int8", "int8_float16"] = "float16"
    max_batch_size: int = 32
    # Whisper-specific
    compute_type: str | None = None  # CTranslate2 compute type override

    def __post_init__(self) -> None:
        if self.compute_type is None:
            self.compute_type = self.dtype


@dataclass
class DiarizationConfig:
    """Configuration for the diarization pipeline."""

    model_id: str = "pyannote/speaker-diarization-3.1"
    segmentation_model: str = "pyannote/segmentation-3.0"
    min_speakers: int | None = None
    max_speakers: int | None = None
    auth_token: str | None = None  # HuggingFace token (pyannote requires it)


@dataclass
class ServerConfig:
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    grpc_port: int = 50051
    max_concurrent_requests: int = 128
    request_timeout: float = 300.0  # seconds


@dataclass
class SchedulerConfig:
    """Scheduler / dynamic batching configuration."""

    max_batch_size: int = 32
    max_wait_time_ms: float = 100.0  # max time to wait for batch to fill
    sort_by_duration: bool = True  # sort requests by audio length to minimize padding
    max_audio_duration: float = 600.0  # max audio duration in seconds (10 min)


@dataclass
class StreamingConfig:
    """Configuration for streaming ASR sessions."""

    vad_threshold: float = 0.5  # Silero VAD speech probability threshold
    min_speech_ms: int = 250  # minimum speech duration to trigger inference
    min_silence_ms: int = 700  # silence duration to confirm + trim buffer
    inference_interval_ms: int = 1000  # run Whisper every N ms of new audio
    max_buffer_seconds: float = 30.0  # max audio buffered per session


@dataclass
class EngineConfig:
    """Top-level engine configuration."""

    models: list[ModelConfig] = field(default_factory=list)
    diarization: DiarizationConfig | None = None
    server: ServerConfig = field(default_factory=ServerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    gpu_memory_utilization: float = 0.9
