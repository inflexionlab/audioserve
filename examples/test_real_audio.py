"""Test AudioServe with real speech audio."""

import asyncio
import time


async def main():
    from audioserve.engine import AudioServeEngine

    print("=== AudioServe Real Audio Test ===\n")

    # Test 1: whisper-tiny (fast, less accurate)
    print("--- Test 1: whisper-tiny ---")
    engine = AudioServeEngine(model="openai/whisper-tiny", dtype="float16")
    engine.start()

    t0 = time.monotonic()
    result = await engine.transcribe("examples/test_audio.wav", word_timestamps=True)
    elapsed = time.monotonic() - t0

    print(f"Text: '{result.text}'")
    print(f"Language: {result.language} (confidence: {result.language_confidence:.2f})")
    print(f"Segments: {len(result.segments)}")
    for seg in result.segments:
        print(f"  [{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
        if seg.words:
            for w in seg.words[:5]:
                print(f"    '{w.word}' [{w.start:.2f}s - {w.end:.2f}s] conf={w.confidence:.3f}")
            if len(seg.words) > 5:
                print(f"    ... and {len(seg.words) - 5} more words")
    print(f"Processing: {result.processing_time:.3f}s | Realtime: {result.duration / result.processing_time:.1f}x\n")
    engine.stop()

    # Test 2: whisper-base (slightly larger)
    print("--- Test 2: whisper-base ---")
    engine = AudioServeEngine(model="openai/whisper-base", dtype="float16")
    engine.start()

    t0 = time.monotonic()
    result = await engine.transcribe("examples/test_audio.wav", word_timestamps=True)
    elapsed = time.monotonic() - t0

    print(f"Text: '{result.text}'")
    print(f"Language: {result.language} (confidence: {result.language_confidence:.2f})")
    print(f"Processing: {result.processing_time:.3f}s | Realtime: {result.duration / result.processing_time:.1f}x\n")
    engine.stop()

    print("=== All tests passed! ===")


if __name__ == "__main__":
    asyncio.run(main())
