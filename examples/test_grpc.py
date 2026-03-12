"""Test gRPC API."""

import subprocess
import sys
import time

import grpc


def main():
    print("=== AudioServe gRPC Test ===\n")

    # Start server
    print("Starting server (REST:8321 + gRPC:50061)...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "audioserve.cli", "serve",
         "-m", "openai/whisper-tiny", "-p", "8321", "--grpc-port", "50061"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for server
    import httpx
    for i in range(30):
        try:
            r = httpx.get("http://localhost:8321/health", timeout=2)
            if r.status_code == 200:
                print(f"Server ready after {i + 1}s\n")
                break
        except Exception:
            time.sleep(1)
    else:
        print("Server failed to start")
        proc.kill()
        return

    try:
        from audioserve.proto import audioserve_pb2, audioserve_pb2_grpc

        channel = grpc.insecure_channel("localhost:50061")
        stub = audioserve_pb2_grpc.AudioServeStub(channel)

        # Wait for gRPC to be ready
        for i in range(10):
            try:
                health = stub.HealthCheck(audioserve_pb2.HealthCheckRequest())
                break
            except grpc.RpcError:
                time.sleep(1)

        # Test HealthCheck
        print("--- HealthCheck ---")
        health = stub.HealthCheck(audioserve_pb2.HealthCheckRequest())
        print(f"  Status: {health.status}")
        print(f"  Models: {[m.model_id for m in health.models]}")
        print(f"  Diarization: {health.diarization_available}")

        # Test ListModels
        print("\n--- ListModels ---")
        models = stub.ListModels(audioserve_pb2.ListModelsRequest())
        for m in models.models:
            print(f"  {m.model_id} ({m.backend}) loaded={m.is_loaded}")

        # Test Transcribe
        print("\n--- Transcribe ---")
        with open("examples/test_audio.wav", "rb") as f:
            audio_bytes = f.read()

        t0 = time.monotonic()
        result = stub.Transcribe(audioserve_pb2.TranscribeRequest(
            audio=audio_bytes,
            beam_size=5,
            word_timestamps=True,
        ))
        elapsed = time.monotonic() - t0

        print(f"  Text: '{result.text}'")
        print(f"  Language: {result.language} (confidence: {result.language_confidence:.2f})")
        print(f"  Duration: {result.duration:.1f}s")
        print(f"  Processing: {result.processing_time:.3f}s")
        print(f"  gRPC round-trip: {elapsed:.3f}s")
        print(f"  Segments: {len(result.segments)}")
        if result.segments:
            seg = result.segments[0]
            print(f"  First segment: [{seg.start:.1f}s-{seg.end:.1f}s] {seg.text}")
            print(f"  Words: {len(seg.words)}")

        channel.close()
        print("\n=== All gRPC tests passed! ===")

    finally:
        proc.kill()
        proc.wait()


if __name__ == "__main__":
    main()
