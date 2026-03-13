"""Streaming ASR session — growing buffer with re-transcription for real-time output."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncIterator

import numpy as np

from audioserve.core.vad import SileroVAD

if TYPE_CHECKING:
    from audioserve.models.base import BaseModelRunner

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000


@dataclass
class StreamingResult:
    """A transcription update from the streaming session."""

    text: str  # new text to append (for final) or full current text (for partial)
    is_final: bool  # True = confirmed, won't change. False = interim, may update.


class StreamingSession:
    """Manages a single streaming ASR connection.

    Uses a growing-buffer re-transcription approach:
    1. Audio chunks arrive and are appended to a buffer
    2. Periodically (every inference_interval_ms), Whisper runs on the full buffer
    3. New transcription is diffed against previous — new text is emitted
    4. VAD detects silence → confirms current text as final, trims buffer
    5. Buffer is capped at max_buffer_seconds (30s = Whisper's context window)

    This gives near-realtime output: text appears within ~1s of speech.
    """

    def __init__(
        self,
        runner: BaseModelRunner,
        vad: SileroVAD,
        language: str | None = None,
        beam_size: int = 5,
        inference_interval_ms: int = 1000,
        max_buffer_seconds: float = 30.0,
    ) -> None:
        self._runner = runner
        self._vad = vad
        self._language = language
        self._beam_size = beam_size
        self._inference_interval_s = inference_interval_ms / 1000.0
        self._max_buffer_samples = int(max_buffer_seconds * _SAMPLE_RATE)

        # Audio state
        self._audio_buffer = np.array([], dtype=np.float32)
        self._new_samples_since_inference = 0

        # Transcription state
        self._confirmed_text = ""  # text that's been finalized (won't change)
        self._prev_transcript = ""  # last full transcript from Whisper on current buffer

        # Async plumbing
        self._result_queue: asyncio.Queue[StreamingResult | None] = asyncio.Queue()
        self._inference_lock = asyncio.Lock()
        self._inference_loop_task: asyncio.Task | None = None
        self._active = False

    def start(self) -> None:
        """Initialize the session and start the inference loop."""
        self._vad.reset()
        self._active = True
        self._audio_buffer = np.array([], dtype=np.float32)
        self._new_samples_since_inference = 0
        self._confirmed_text = ""
        self._prev_transcript = ""
        self._inference_loop_task = asyncio.create_task(self._inference_loop())

    def feed_audio(self, pcm_data: bytes | np.ndarray) -> None:
        """Feed a chunk of PCM audio (float32, 16kHz, mono)."""
        if not self._active:
            return

        if isinstance(pcm_data, bytes):
            chunk = np.frombuffer(pcm_data, dtype=np.float32).copy()
        else:
            chunk = pcm_data.astype(np.float32)

        self._audio_buffer = np.concatenate([self._audio_buffer, chunk])
        self._new_samples_since_inference += len(chunk)

        # Run VAD to detect silence boundaries
        self._vad.process_chunk(chunk)

    async def _inference_loop(self) -> None:
        """Background loop: periodically transcribe the buffer and emit results."""
        while self._active:
            await asyncio.sleep(self._inference_interval_s)

            if not self._active:
                break

            # Only run if we have meaningful new audio
            min_new = int(0.3 * _SAMPLE_RATE)  # at least 300ms new audio
            if self._new_samples_since_inference < min_new:
                # Check if VAD detected end-of-speech — if so, confirm
                if self._vad._silence_counter >= self._vad.min_silence_samples:
                    await self._confirm_and_trim()
                continue

            if len(self._audio_buffer) < _SAMPLE_RATE // 4:  # less than 250ms total
                continue

            await self._run_inference()

    async def _run_inference(self) -> None:
        """Transcribe the current buffer and emit new text."""
        async with self._inference_lock:
            audio = self._audio_buffer.copy()
            self._new_samples_since_inference = 0

        if len(audio) == 0:
            return

        # Trim to max 30s (Whisper's context window)
        if len(audio) > self._max_buffer_samples:
            audio = audio[-self._max_buffer_samples:]

        loop = asyncio.get_event_loop()
        params = {
            "language": self._language,
            "beam_size": self._beam_size,
            "word_timestamps": False,
        }

        try:
            results = await loop.run_in_executor(
                None, self._runner.transcribe_batch, [audio], [params]
            )
            new_transcript = results[0].text.strip()
        except Exception as e:
            logger.exception("Streaming inference error: %s", e)
            return

        if not new_transcript:
            return

        # Diff: find what's new compared to previous transcript
        if new_transcript != self._prev_transcript:
            self._prev_transcript = new_transcript

            # Emit the full current text as a partial result
            # Client replaces their display with this
            await self._result_queue.put(StreamingResult(
                text=self._confirmed_text + new_transcript,
                is_final=False,
            ))

        # Check if VAD says speech ended — confirm the text
        if self._vad._silence_counter >= self._vad.min_silence_samples:
            await self._confirm_and_trim()

    async def _confirm_and_trim(self) -> None:
        """Confirm current transcript as final and reset buffer."""
        if not self._prev_transcript:
            return

        full_text = self._confirmed_text + self._prev_transcript
        self._confirmed_text = full_text + " "
        self._prev_transcript = ""

        # Emit as final
        await self._result_queue.put(StreamingResult(
            text=full_text,
            is_final=True,
        ))

        # Trim buffer — keep a small tail for context continuity
        keep_samples = int(0.5 * _SAMPLE_RATE)  # 500ms overlap
        if len(self._audio_buffer) > keep_samples:
            self._audio_buffer = self._audio_buffer[-keep_samples:]
        self._new_samples_since_inference = 0

        # Reset VAD state for next utterance
        self._vad.reset()

    async def end_stream(self) -> None:
        """Finalize the stream — run final inference and close."""
        if not self._active:
            return

        self._active = False

        # Cancel the inference loop
        if self._inference_loop_task:
            self._inference_loop_task.cancel()
            try:
                await self._inference_loop_task
            except asyncio.CancelledError:
                pass

        # Run one final inference on remaining buffer
        if len(self._audio_buffer) > _SAMPLE_RATE // 4:  # at least 250ms
            loop = asyncio.get_event_loop()
            params = {
                "language": self._language,
                "beam_size": self._beam_size,
                "word_timestamps": False,
            }
            try:
                results = await loop.run_in_executor(
                    None, self._runner.transcribe_batch,
                    [self._audio_buffer], [params]
                )
                final_text = results[0].text.strip()
                if final_text:
                    full_text = self._confirmed_text + final_text
                    await self._result_queue.put(StreamingResult(
                        text=full_text,
                        is_final=True,
                    ))
            except Exception as e:
                logger.exception("Final streaming inference error: %s", e)

        # Signal end
        await self._result_queue.put(None)

    async def results(self) -> AsyncIterator[StreamingResult]:
        """Async iterator over transcription results."""
        while True:
            result = await self._result_queue.get()
            if result is None:
                break
            yield result

    @property
    def is_active(self) -> bool:
        return self._active
