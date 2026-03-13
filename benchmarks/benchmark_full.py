"""Full benchmark: Whisper models + diarization on RTX A5000."""

import asyncio
import os
import time
import json
import numpy as np


async def benchmark_model(model_id, audio_path, dtype="float16", n_runs=3):
    from audioserve.engine import AudioServeEngine

    engine = AudioServeEngine(model=model_id, dtype=dtype)
    engine.start()

    # Warmup
    await engine.transcribe(audio_path, word_timestamps=True)

    times = []
    result = None
    for i in range(n_runs):
        t0 = time.monotonic()
        result = await engine.transcribe(audio_path, word_timestamps=True)
        times.append(time.monotonic() - t0)

    engine.stop()

    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    avg = np.mean(times)
    return {
        "model": model_id,
        "dtype": dtype,
        "audio_duration": result.duration,
        "avg_ms": avg * 1000,
        "min_ms": min(times) * 1000,
        "rtf": result.duration / avg,
        "text_preview": result.text[:100],
        "language": result.language,
    }


async def benchmark_diarization(audio_path, hf_token):
    from audioserve.engine import AudioServeEngine

    engine = AudioServeEngine(
        model="openai/whisper-large-v3",
        dtype="float16",
        diarization=True,
        hf_token=hf_token,
    )
    engine.start()

    # Warmup
    await engine.transcribe(audio_path, word_timestamps=True)

    # Benchmark ASR only
    t0 = time.monotonic()
    asr_result = await engine.transcribe(audio_path, word_timestamps=True)
    asr_time = time.monotonic() - t0

    # Benchmark ASR + diarization
    t0 = time.monotonic()
    diar_result = await engine.transcribe_with_diarization(audio_path, word_timestamps=True)
    diar_time = time.monotonic() - t0

    engine.stop()

    return {
        "audio_duration": asr_result.duration,
        "asr_only_ms": asr_time * 1000,
        "asr_diarization_ms": diar_time * 1000,
        "diarization_overhead_ms": (diar_time - asr_time) * 1000,
        "speakers": diar_result.speakers,
        "num_segments": len(diar_result.segments),
    }


async def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    audio_path = os.path.join(repo_root, "audio.mp3")
    hf_token = os.environ.get("HF_TOKEN", "")

    print("=" * 70)
    print("AudioServe Benchmark — RTX A5000 24GB")
    print("=" * 70)

    from audioserve.core.preprocessing import load_audio
    audio = load_audio(audio_path)
    print(f"Audio: {audio_path} ({audio.duration:.1f}s)")
    print()

    # Whisper models
    models = [
        ("openai/whisper-tiny", "float16"),
        ("openai/whisper-base", "float16"),
        ("openai/whisper-small", "float16"),
        ("openai/whisper-medium", "float16"),
        ("openai/whisper-large-v3", "float16"),
        ("openai/whisper-large-v3-turbo", "float16"),
    ]

    results = []
    for model_id, dtype in models:
        print(f"Benchmarking {model_id}...")
        try:
            r = await benchmark_model(model_id, audio_path, dtype)
            results.append(r)
            print(f"  {r['avg_ms']:.0f}ms avg | {r['rtf']:.1f}x realtime")
        except Exception as e:
            print(f"  FAILED: {e}")

    # Wav2Vec2
    print(f"Benchmarking facebook/wav2vec2-base-960h...")
    try:
        r = await benchmark_model("facebook/wav2vec2-base-960h", audio_path, "float32")
        results.append(r)
        print(f"  {r['avg_ms']:.0f}ms avg | {r['rtf']:.1f}x realtime")
    except Exception as e:
        print(f"  FAILED: {e}")

    # Diarization
    print(f"\nBenchmarking diarization (whisper-large-v3 + pyannote)...")
    try:
        diar = await benchmark_diarization(audio_path, hf_token)
        print(f"  ASR only: {diar['asr_only_ms']:.0f}ms")
        print(f"  ASR + diarization: {diar['asr_diarization_ms']:.0f}ms")
        print(f"  Diarization overhead: {diar['diarization_overhead_ms']:.0f}ms")
        print(f"  Speakers: {diar['speakers']}")
    except Exception as e:
        print(f"  FAILED: {e}")
        diar = None

    # Summary table
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Audio: {audio.duration:.1f}s | GPU: RTX A5000 24GB | CUDA 13.0")
    print(f"\n{'Model':<40} {'Dtype':<10} {'Avg (ms)':>10} {'RTF':>8}")
    print(f"{'-'*40} {'-'*10} {'-'*10} {'-'*8}")
    for r in results:
        print(f"{r['model']:<40} {r['dtype']:<10} {r['avg_ms']:>10.0f} {r['rtf']:>7.1f}x")

    if diar:
        print(f"\nDiarization (whisper-large-v3 + pyannote):")
        print(f"  ASR only:          {diar['asr_only_ms']:>8.0f}ms")
        print(f"  ASR + diarization: {diar['asr_diarization_ms']:>8.0f}ms")
        print(f"  Overhead:          {diar['diarization_overhead_ms']:>8.0f}ms")

    # Save results
    results_path = os.path.join(script_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({"models": results, "diarization": diar}, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    asyncio.run(main())
