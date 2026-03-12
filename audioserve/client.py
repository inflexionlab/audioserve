"""AudioServe Python client."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import httpx


@dataclass
class ClientTranscriptionResult:
    text: str
    segments: list[dict] = field(default_factory=list)
    language: str | None = None
    language_confidence: float | None = None
    duration: float = 0.0
    processing_time: float = 0.0
    speakers: list[str] | None = None


class AudioServeClient:
    """Client for the AudioServe REST API."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def transcribe(
        self,
        audio: str | bytes | Path,
        language: str | None = None,
        beam_size: int = 5,
        word_timestamps: bool = True,
        model: str | None = None,
    ) -> ClientTranscriptionResult:
        """Transcribe audio via the server."""
        files, data = self._prepare_request(audio, language, beam_size, word_timestamps, model)

        resp = self._client.post(f"{self.base_url}/v1/transcribe", files=files, data=data)
        resp.raise_for_status()
        return self._parse_response(resp.json())

    def diarize(
        self,
        audio: str | bytes | Path,
        language: str | None = None,
        beam_size: int = 5,
        word_timestamps: bool = True,
        model: str | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> ClientTranscriptionResult:
        """Transcribe audio with speaker diarization."""
        files, data = self._prepare_request(audio, language, beam_size, word_timestamps, model)
        if min_speakers is not None:
            data["min_speakers"] = str(min_speakers)
        if max_speakers is not None:
            data["max_speakers"] = str(max_speakers)

        resp = self._client.post(f"{self.base_url}/v1/transcribe+diarize", files=files, data=data)
        resp.raise_for_status()
        result = self._parse_response(resp.json())
        result.speakers = resp.json().get("speakers", [])
        return result

    def health(self) -> dict:
        resp = self._client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def models(self) -> list[dict]:
        resp = self._client.get(f"{self.base_url}/v1/models")
        resp.raise_for_status()
        return resp.json()["models"]

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _prepare_request(self, audio, language, beam_size, word_timestamps, model):
        if isinstance(audio, (str, Path)):
            path = Path(audio)
            file_obj = open(path, "rb")
            filename = path.name
        elif isinstance(audio, bytes):
            file_obj = io.BytesIO(audio)
            filename = "audio.wav"
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")

        files = {"audio": (filename, file_obj)}
        data = {
            "beam_size": str(beam_size),
            "word_timestamps": str(word_timestamps).lower(),
        }
        if language:
            data["language"] = language
        if model:
            data["model"] = model

        return files, data

    def _parse_response(self, data: dict) -> ClientTranscriptionResult:
        return ClientTranscriptionResult(
            text=data["text"],
            segments=data.get("segments", []),
            language=data.get("language"),
            language_confidence=data.get("language_confidence"),
            duration=data.get("duration", 0.0),
            processing_time=data.get("processing_time", 0.0),
        )
