"""Test: start server and hit it with the client."""

import subprocess
import sys
import time

import httpx


def main():
    print("=== AudioServe Server Test ===\n")

    # Start server in background
    print("Starting server with whisper-tiny...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "audioserve.cli", "serve", "-m", "openai/whisper-tiny", "-p", "8321"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for server to be ready
    for i in range(60):
        try:
            r = httpx.get("http://localhost:8321/health", timeout=2)
            if r.status_code == 200:
                print(f"Server ready after {i + 1}s")
                break
        except Exception:
            time.sleep(1)
    else:
        print("Server failed to start")
        proc.kill()
        return

    try:
        # Test health endpoint
        health = httpx.get("http://localhost:8321/health").json()
        print(f"Health: {health}\n")

        # Test transcribe endpoint
        print("Testing /v1/transcribe ...")
        with open("examples/test_audio.wav", "rb") as f:
            r = httpx.post(
                "http://localhost:8321/v1/transcribe",
                files={"audio": ("test.wav", f)},
                data={"word_timestamps": "true"},
                timeout=60,
            )
        result = r.json()
        print(f"  Status: {r.status_code}")
        print(f"  Text: '{result['text']}'")
        print(f"  Language: {result['language']}")
        print(f"  Segments: {len(result['segments'])}")
        print(f"  Processing time: {result['processing_time']:.3f}s")
        print(f"  Duration: {result['duration']:.1f}s")

        # Test with Python client
        print("\nTesting Python client...")
        from audioserve.client import AudioServeClient

        with AudioServeClient("http://localhost:8321") as client:
            result = client.transcribe("examples/test_audio.wav", language="en")
            print(f"  Client result: '{result.text}'")
            print(f"  Processing: {result.processing_time:.3f}s")

        # Test models endpoint
        models = httpx.get("http://localhost:8321/v1/models").json()
        print(f"\nModels: {models}")

        print("\n=== All server tests passed! ===")

    finally:
        proc.kill()
        proc.wait()


if __name__ == "__main__":
    main()
