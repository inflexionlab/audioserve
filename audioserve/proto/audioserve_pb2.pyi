from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TranscribeRequest(_message.Message):
    __slots__ = ("audio", "language", "beam_size", "word_timestamps", "model")
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    BEAM_SIZE_FIELD_NUMBER: _ClassVar[int]
    WORD_TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    audio: bytes
    language: str
    beam_size: int
    word_timestamps: bool
    model: str
    def __init__(self, audio: _Optional[bytes] = ..., language: _Optional[str] = ..., beam_size: _Optional[int] = ..., word_timestamps: bool = ..., model: _Optional[str] = ...) -> None: ...

class DiarizeRequest(_message.Message):
    __slots__ = ("audio", "language", "beam_size", "word_timestamps", "model", "min_speakers", "max_speakers")
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    BEAM_SIZE_FIELD_NUMBER: _ClassVar[int]
    WORD_TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    MIN_SPEAKERS_FIELD_NUMBER: _ClassVar[int]
    MAX_SPEAKERS_FIELD_NUMBER: _ClassVar[int]
    audio: bytes
    language: str
    beam_size: int
    word_timestamps: bool
    model: str
    min_speakers: int
    max_speakers: int
    def __init__(self, audio: _Optional[bytes] = ..., language: _Optional[str] = ..., beam_size: _Optional[int] = ..., word_timestamps: bool = ..., model: _Optional[str] = ..., min_speakers: _Optional[int] = ..., max_speakers: _Optional[int] = ...) -> None: ...

class WordInfo(_message.Message):
    __slots__ = ("word", "start", "end", "confidence")
    WORD_FIELD_NUMBER: _ClassVar[int]
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    word: str
    start: float
    end: float
    confidence: float
    def __init__(self, word: _Optional[str] = ..., start: _Optional[float] = ..., end: _Optional[float] = ..., confidence: _Optional[float] = ...) -> None: ...

class Segment(_message.Message):
    __slots__ = ("text", "start", "end", "words", "speaker", "confidence")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    WORDS_FIELD_NUMBER: _ClassVar[int]
    SPEAKER_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    text: str
    start: float
    end: float
    words: _containers.RepeatedCompositeFieldContainer[WordInfo]
    speaker: str
    confidence: float
    def __init__(self, text: _Optional[str] = ..., start: _Optional[float] = ..., end: _Optional[float] = ..., words: _Optional[_Iterable[_Union[WordInfo, _Mapping]]] = ..., speaker: _Optional[str] = ..., confidence: _Optional[float] = ...) -> None: ...

class TranscribeResponse(_message.Message):
    __slots__ = ("text", "segments", "language", "language_confidence", "duration", "processing_time")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_TIME_FIELD_NUMBER: _ClassVar[int]
    text: str
    segments: _containers.RepeatedCompositeFieldContainer[Segment]
    language: str
    language_confidence: float
    duration: float
    processing_time: float
    def __init__(self, text: _Optional[str] = ..., segments: _Optional[_Iterable[_Union[Segment, _Mapping]]] = ..., language: _Optional[str] = ..., language_confidence: _Optional[float] = ..., duration: _Optional[float] = ..., processing_time: _Optional[float] = ...) -> None: ...

class DiarizedTranscribeResponse(_message.Message):
    __slots__ = ("text", "segments", "speakers", "language", "duration", "processing_time")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    SPEAKERS_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_TIME_FIELD_NUMBER: _ClassVar[int]
    text: str
    segments: _containers.RepeatedCompositeFieldContainer[Segment]
    speakers: _containers.RepeatedScalarFieldContainer[str]
    language: str
    duration: float
    processing_time: float
    def __init__(self, text: _Optional[str] = ..., segments: _Optional[_Iterable[_Union[Segment, _Mapping]]] = ..., speakers: _Optional[_Iterable[str]] = ..., language: _Optional[str] = ..., duration: _Optional[float] = ..., processing_time: _Optional[float] = ...) -> None: ...

class ListModelsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListModelsResponse(_message.Message):
    __slots__ = ("models",)
    MODELS_FIELD_NUMBER: _ClassVar[int]
    models: _containers.RepeatedCompositeFieldContainer[ModelInfo]
    def __init__(self, models: _Optional[_Iterable[_Union[ModelInfo, _Mapping]]] = ...) -> None: ...

class ModelInfo(_message.Message):
    __slots__ = ("model_id", "backend", "is_loaded")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    IS_LOADED_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    backend: str
    is_loaded: bool
    def __init__(self, model_id: _Optional[str] = ..., backend: _Optional[str] = ..., is_loaded: bool = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("status", "models", "diarization_available")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    DIARIZATION_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    status: str
    models: _containers.RepeatedCompositeFieldContainer[ModelInfo]
    diarization_available: bool
    def __init__(self, status: _Optional[str] = ..., models: _Optional[_Iterable[_Union[ModelInfo, _Mapping]]] = ..., diarization_available: bool = ...) -> None: ...
