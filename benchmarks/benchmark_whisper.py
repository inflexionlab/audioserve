"""Benchmark Whisper models via AudioServe vs direct faster-whisper."""

import asyncio
import time
import numpy as np


async def benchmark_model(model_id: str, audio_path: str, n_runs: int = 5):
    """Benchmark a single model."""
    from audioserve.engine import AudioServeEngine

    print(f"\n{'='*60}")
    print(f"Model: {model_id}")
    print(f"{'='*60}")

    engine = AudioServeEngine(model=model_id, dtype="float16")
    engine.start()

    # Warmup
    await engine.transcribe(audio_path, word_timestamps=True)

    # Benchmark
    times = []
    for i in range(n_runs):
        t0 = time.monotonic()
        result = await engine.transcribe(audio_path, word_timestamps=True)
        elapsed = time.monotonic() - t0
        times.append(elapsed)

    engine.stop()

    avg = np.mean(times)
    std = np.std(times)
    audio_dur = result.duration
    rtf = audio_dur / avg

    print(f"  Text: '{result.text}'")
    print(f"  Audio duration: {audio_dur:.1f}s")
    print(f"  Avg inference: {avg*1000:.1f}ms ± {std*1000:.1f}ms")
    print(f"  Min inference: {min(times)*1000:.1f}ms")
    print(f"  Max inference: {max(times)*1000:.1f}ms")
    print(f"  Realtime factor: {rtf:.1f}x")
    print(f"  Language: {result.language}")

    return {"model": model_id, "avg_ms": avg * 1000, "rtf": rtf, "text": result.text}


async def main():
    audio_path = "examples/test_audio.wav"
    n_runs = 5

    print("=== AudioServe Whisper Benchmark ===")
    print(f"Audio: {audio_path}")
    print(f"Runs per model: {n_runs}")

    models = [
        "openai/whisper-tiny",
        "openai/whisper-base",
        "openai/whisper-small",
    ]

    results = []
    for model_id in models:
        r = await benchmark_model(model_id, audio_path, n_runs)
        results.append(r)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Model':<35} {'Avg (ms)':>10} {'RTF':>8}")
    print(f"{'-'*35} {'-'*10} {'-'*8}")
    for r in results:
        print(f"{r['model']:<35} {r['avg_ms']:>10.1f} {r['rtf']:>7.1f}x")


if __name__ == "__main__":
    asyncio.run(main())
