"""Voice Activity Detection using Silero VAD for streaming endpoint detection."""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Silero VAD expects 16kHz audio
_SAMPLE_RATE = 16000

# Silero VAD accepts specific chunk sizes (in samples at 16kHz)
# 512 (32ms), 1024 (64ms), or 1536 (96ms)
VAD_CHUNK_SAMPLES = 512


class SileroVAD:
    """Silero VAD wrapper for streaming speech endpoint detection.

    Processes audio in small chunks and detects speech start/end boundaries.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 700,
    ) -> None:
        self.threshold = threshold
        self.min_speech_samples = int(min_speech_ms * _SAMPLE_RATE / 1000)
        self.min_silence_samples = int(min_silence_ms * _SAMPLE_RATE / 1000)

        self._model = None
        self._is_speaking = False
        self._speech_start = 0
        self._silence_counter = 0
        self._speech_counter = 0
        self._sample_offset = 0

    def load(self) -> None:
        """Load the Silero VAD model."""
        self._model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        self._model.eval()
        logger.info("Silero VAD loaded")

    def reset(self) -> None:
        """Reset state for a new stream."""
        if self._model is not None:
            self._model.reset_states()
        self._is_speaking = False
        self._speech_start = 0
        self._silence_counter = 0
        self._speech_counter = 0
        self._sample_offset = 0

    def process_chunk(self, audio: np.ndarray) -> list[tuple[int, int]]:
        """Process an audio chunk and return completed speech segments.

        Args:
            audio: float32 numpy array at 16kHz.

        Returns:
            List of (start_sample, end_sample) for each completed speech segment.
        """
        if self._model is None:
            raise RuntimeError("VAD not loaded. Call load() first.")

        completed = []
        offset = 0

        while offset < len(audio):
            end = min(offset + VAD_CHUNK_SAMPLES, len(audio))
            chunk = audio[offset:end]

            # Silero requires exact chunk sizes — pad last chunk
            if len(chunk) < VAD_CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, VAD_CHUNK_SAMPLES - len(chunk)))

            tensor = torch.from_numpy(chunk)
            prob = self._model(tensor, _SAMPLE_RATE).item()

            abs_pos = self._sample_offset + end

            if prob >= self.threshold:
                self._silence_counter = 0
                self._speech_counter += end - offset
                if not self._is_speaking and self._speech_counter >= self.min_speech_samples:
                    self._is_speaking = True
                    self._speech_start = self._sample_offset + offset - self._speech_counter + (end - offset)
            else:
                self._speech_counter = 0
                if self._is_speaking:
                    self._silence_counter += end - offset
                    if self._silence_counter >= self.min_silence_samples:
                        # Speech segment complete
                        speech_end = abs_pos - self._silence_counter
                        completed.append((self._speech_start, speech_end))
                        self._is_speaking = False
                        self._silence_counter = 0

            offset = end

        self._sample_offset += len(audio)
        return completed

    def flush(self) -> list[tuple[int, int]]:
        """Flush any remaining speech at end of stream.

        Returns:
            List with final speech segment if one was in progress.
        """
        if self._is_speaking:
            segment = (self._speech_start, self._sample_offset)
            self._is_speaking = False
            return [segment]
        return []

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self):
        """The underlying Silero model (for sharing weights across sessions)."""
        return self._model

    @model.setter
    def model(self, value) -> None:
        self._model = value

    @property
    def silence_counter(self) -> int:
        """Current accumulated silence in samples."""
        return self._silence_counter
