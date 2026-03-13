"""Tests for streaming ASR components."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

from audioserve.config import StreamingConfig
from audioserve.core.vad import SileroVAD, VAD_CHUNK_SAMPLES


class TestSileroVAD:
    """Test VAD loading, processing, and state management."""

    @pytest.fixture
    def vad(self):
        v = SileroVAD(threshold=0.5, min_speech_ms=250, min_silence_ms=500)
        v.load()
        yield v

    def test_load(self, vad):
        assert vad.is_loaded

    def test_not_loaded_raises(self):
        v = SileroVAD()
        with pytest.raises(RuntimeError, match="not loaded"):
            v.process_chunk(np.zeros(512, dtype=np.float32))

    def test_silence_no_segments(self, vad):
        silence = np.zeros(16000 * 2, dtype=np.float32)
        segments = vad.process_chunk(silence)
        assert segments == []

    def test_reset_clears_state(self, vad):
        audio = np.zeros(16000, dtype=np.float32)
        vad.process_chunk(audio)
        assert vad._sample_offset > 0

        vad.reset()
        assert vad._sample_offset == 0
        assert not vad._is_speaking
        assert vad._silence_counter == 0
        assert vad._speech_counter == 0

    def test_flush_no_speech(self, vad):
        result = vad.flush()
        assert result == []

    def test_process_speech_then_silence(self, vad):
        t = np.linspace(0, 2, 16000 * 2, dtype=np.float32)
        speech = 0.8 * np.sin(2 * np.pi * 300 * t)
        rng = np.random.default_rng(42)
        speech += 0.1 * rng.standard_normal(len(speech), dtype=np.float32)

        segments = vad.process_chunk(speech)
        silence = np.zeros(16000 * 2, dtype=np.float32)
        segments.extend(vad.process_chunk(silence))

        if not segments:
            segments = vad.flush()

        assert isinstance(segments, list)

    def test_multiple_chunks_accumulate_offset(self, vad):
        chunk = np.zeros(1024, dtype=np.float32)
        vad.process_chunk(chunk)
        assert vad._sample_offset == 1024

        vad.process_chunk(chunk)
        assert vad._sample_offset == 2048


class TestStreamingConfig:
    def test_defaults(self):
        config = StreamingConfig()
        assert config.vad_threshold == 0.5
        assert config.min_speech_ms == 250
        assert config.min_silence_ms == 700
        assert config.inference_interval_ms == 1000
        assert config.max_buffer_seconds == 30.0

    def test_custom(self):
        config = StreamingConfig(
            vad_threshold=0.3,
            min_silence_ms=500,
            inference_interval_ms=500,
            max_buffer_seconds=60.0,
        )
        assert config.vad_threshold == 0.3
        assert config.min_silence_ms == 500
        assert config.inference_interval_ms == 500


class TestStreamingSession:
    """Unit tests for StreamingSession. Tests that call start() must be async."""

    @pytest.mark.asyncio
    async def test_session_creation(self):
        from audioserve.core.streaming import StreamingSession

        runner = MagicMock()
        vad = SileroVAD()
        vad.load()

        session = StreamingSession(runner=runner, vad=vad)
        assert not session.is_active

        session.start()
        assert session.is_active

        # Clean up
        session._active = False
        session._inference_loop_task.cancel()
        try:
            await session._inference_loop_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_feed_audio_bytes(self):
        from audioserve.core.streaming import StreamingSession

        runner = MagicMock()
        vad = SileroVAD()
        vad.load()

        session = StreamingSession(runner=runner, vad=vad)
        session.start()

        audio = np.zeros(1024, dtype=np.float32)
        session.feed_audio(audio.tobytes())

        assert len(session._audio_buffer) == 1024

        session._active = False
        session._inference_loop_task.cancel()
        try:
            await session._inference_loop_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_feed_audio_numpy(self):
        from audioserve.core.streaming import StreamingSession

        runner = MagicMock()
        vad = SileroVAD()
        vad.load()

        session = StreamingSession(runner=runner, vad=vad)
        session.start()

        audio = np.zeros(1024, dtype=np.float32)
        session.feed_audio(audio)

        assert len(session._audio_buffer) == 1024

        session._active = False
        session._inference_loop_task.cancel()
        try:
            await session._inference_loop_task
        except asyncio.CancelledError:
            pass

    def test_feed_audio_when_inactive_is_noop(self):
        from audioserve.core.streaming import StreamingSession

        runner = MagicMock()
        vad = SileroVAD()
        vad.load()

        session = StreamingSession(runner=runner, vad=vad)
        audio = np.zeros(1024, dtype=np.float32)
        session.feed_audio(audio)
        assert len(session._audio_buffer) == 0

    @pytest.mark.asyncio
    async def test_buffer_accumulates(self):
        from audioserve.core.streaming import StreamingSession

        runner = MagicMock()
        vad = SileroVAD()
        vad.load()

        session = StreamingSession(runner=runner, vad=vad)
        session.start()

        for _ in range(5):
            session.feed_audio(np.zeros(1600, dtype=np.float32))

        assert len(session._audio_buffer) == 8000
        assert session._new_samples_since_inference == 8000

        session._active = False
        session._inference_loop_task.cancel()
        try:
            await session._inference_loop_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_end_stream_marks_inactive(self):
        from audioserve.core.streaming import StreamingSession

        runner = MagicMock()
        runner.transcribe_batch = MagicMock(return_value=[MagicMock(text="")])
        vad = SileroVAD()
        vad.load()

        session = StreamingSession(runner=runner, vad=vad)
        session.start()
        assert session.is_active

        await session.end_stream()
        assert not session.is_active


class TestStreamingResult:
    def test_partial_result(self):
        from audioserve.core.streaming import StreamingResult

        r = StreamingResult(text="hello wor", is_final=False)
        assert not r.is_final
        assert r.text == "hello wor"

    def test_final_result(self):
        from audioserve.core.streaming import StreamingResult

        r = StreamingResult(text="hello world", is_final=True)
        assert r.is_final


class TestEngineStreaming:
    """Test engine streaming integration without GPU models."""

    def test_has_streaming_false_by_default(self):
        from audioserve.engine import AudioServeEngine
        engine = AudioServeEngine(model="openai/whisper-tiny")
        assert not engine.has_streaming

    def test_start_vad(self):
        from audioserve.engine import AudioServeEngine
        engine = AudioServeEngine(model="openai/whisper-tiny")
        engine.start_vad()
        assert engine.has_streaming

    def test_create_session_without_vad_raises(self):
        from audioserve.engine import AudioServeEngine
        engine = AudioServeEngine(model="openai/whisper-tiny")
        with pytest.raises(RuntimeError, match="VAD not loaded"):
            engine.create_streaming_session()

    def test_create_session_without_runners_raises(self):
        from audioserve.engine import AudioServeEngine
        engine = AudioServeEngine(model="openai/whisper-tiny")
        engine.start_vad()
        with pytest.raises(RuntimeError, match="No models loaded"):
            engine.create_streaming_session()

    @pytest.mark.asyncio
    async def test_create_session_with_mock_runner(self):
        from audioserve.engine import AudioServeEngine

        engine = AudioServeEngine(model="openai/whisper-tiny")
        engine._runners["openai/whisper-tiny"] = MagicMock()
        engine.start_vad()

        session = engine.create_streaming_session()
        assert session is not None
        assert not session.is_active
        session.start()
        assert session.is_active

        session._active = False
        session._inference_loop_task.cancel()
        try:
            await session._inference_loop_task
        except asyncio.CancelledError:
            pass

    def test_create_session_with_specific_model(self):
        from audioserve.engine import AudioServeEngine

        engine = AudioServeEngine(models=[
            {"model": "openai/whisper-tiny"},
            {"model": "openai/whisper-base"},
        ])
        engine._runners["openai/whisper-tiny"] = MagicMock()
        engine._runners["openai/whisper-base"] = MagicMock()
        engine.start_vad()

        session = engine.create_streaming_session(model="openai/whisper-base")
        assert session._runner is engine._runners["openai/whisper-base"]


@pytest.mark.gpu
class TestStreamingIntegration:
    """Integration tests that require GPU + model downloads."""

    @pytest.fixture(autouse=True)
    def setup_engine(self):
        from audioserve.engine import AudioServeEngine
        self.engine = AudioServeEngine(model="openai/whisper-tiny")
        self.engine.start()
        self.engine.start_vad()
        yield
        self.engine.stop()

    @pytest.mark.asyncio
    async def test_streaming_end_to_end(self):
        session = self.engine.create_streaming_session()
        session.start()

        t = np.linspace(0, 2, 16000 * 2, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        chunk_size = 4096
        for i in range(0, len(audio), chunk_size):
            session.feed_audio(audio[i:i + chunk_size])
            await asyncio.sleep(0)

        await session.end_stream()

        results = []
        async for result in session.results():
            results.append(result)

        assert not session.is_active
