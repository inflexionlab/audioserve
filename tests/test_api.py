"""Tests for the REST API."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from httpx import ASGITransport, AsyncClient

from audioserve.engine import AudioServeEngine
from audioserve.api.rest_server import create_app, _sanitize_language


def _make_wav_bytes(duration: float = 1.0, sr: int = 16000) -> bytes:
    """Create WAV bytes of silence."""
    audio = np.zeros(int(sr * duration), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


class TestSanitizeLanguage:
    def test_none(self):
        assert _sanitize_language(None) is None

    def test_empty(self):
        assert _sanitize_language("") is None
        assert _sanitize_language("  ") is None

    def test_valid(self):
        assert _sanitize_language("en") == "en"
        assert _sanitize_language("  EN  ") == "en"

    def test_null_strings(self):
        assert _sanitize_language("null") is None
        assert _sanitize_language("none") is None
        assert _sanitize_language("string") is None

    def test_invalid_raises(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _sanitize_language("xx")
        assert exc_info.value.status_code == 400


@pytest.mark.gpu
class TestRESTEndpoints:
    """Integration tests for REST API endpoints (require GPU)."""

    @pytest.fixture(autouse=True)
    async def setup_client(self):
        self.engine = AudioServeEngine(model="openai/whisper-tiny")
        self.engine.start()
        app = create_app(self.engine)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            self.client = client
            yield
        self.engine.stop()

    @pytest.mark.asyncio
    async def test_health(self):
        resp = await self.client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "loading")
        assert len(data["models"]) == 1

    @pytest.mark.asyncio
    async def test_list_models(self):
        resp = await self.client.get("/v1/models")
        assert resp.status_code == 200
        models = resp.json()["models"]
        assert len(models) == 1
        assert models[0]["model_id"] == "openai/whisper-tiny"

    @pytest.mark.asyncio
    async def test_transcribe(self):
        wav_bytes = _make_wav_bytes(2.0)
        resp = await self.client.post(
            "/v1/transcribe",
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data
        assert "segments" in data
        assert "duration" in data

    @pytest.mark.asyncio
    async def test_transcribe_empty_file(self):
        resp = await self.client.post(
            "/v1/transcribe",
            files={"audio": ("test.wav", b"", "audio/wav")},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_transcribe_invalid_language(self):
        wav_bytes = _make_wav_bytes()
        resp = await self.client.post(
            "/v1/transcribe",
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            data={"language": "xx"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_diarize_without_diarization(self):
        wav_bytes = _make_wav_bytes()
        resp = await self.client.post(
            "/v1/transcribe+diarize",
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 503
