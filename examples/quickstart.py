"""Quick test: transcribe a generated audio sample with Whisper."""

import asyncio
import numpy as np
import time


async def main():
    from audioserve.engine import AudioServeEngine

    # Use whisper-tiny for fast testing
    print("=== AudioServe Quick Test ===\n")

    engine = AudioServeEngine(model="openai/whisper-tiny", dtype="float16")
    print("Loading model...")
    engine.start()
    print(f"Loaded models: {engine.loaded_models}\n")

    # Generate a simple test tone (silence + noise — Whisper should detect no speech)
    # For a real test, use an actual audio file
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    # Sine wave at 440Hz, very low amplitude (whisper should handle this gracefully)
    audio = 0.01 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    print("Transcribing synthetic audio (sine wave — expect empty/noise)...")
    t0 = time.monotonic()
    result = await engine.transcribe(audio, word_timestamps=False)
    elapsed = time.monotonic() - t0

    print(f"Text: '{result.text}'")
    print(f"Language: {result.language} (confidence: {result.language_confidence:.2f})")
    print(f"Duration: {result.duration:.1f}s")
    print(f"Processing time: {result.processing_time:.3f}s")
    print(f"Total time (incl. preprocessing): {elapsed:.3f}s")
    print(f"Realtime factor: {result.duration / result.processing_time:.1f}x")

    print("\n✓ AudioServe is working!")
    engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
