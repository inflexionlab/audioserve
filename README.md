# AudioServe

Optimized inference server for audio models — like vLLM, but for ASR.

**One command to deploy any speech recognition model as a production API.**

## Features

- **Multi-model support** — Whisper (all sizes), Wav2Vec2, HuBERT, distil-whisper
- **Optimized Whisper inference** — powered by faster-whisper (CTranslate2), up to 87x realtime
- **Speaker diarization** — integrated pyannote.audio pipeline with per-segment speaker labels
- **Dynamic batching** — sort-by-duration padding minimization, configurable batch scheduler
- **REST + gRPC APIs** — both protocols, same server
- **Two endpoints** — pure ASR (`/v1/transcribe`) and ASR+diarization (`/v1/transcribe+diarize`)
- **Word-level timestamps** — precise timing for every word
- **Language auto-detection** — 99 languages via Whisper
- **Docker ready** — single command deployment
- **Python client + CLI** — programmatic access or command line

## Quickstart

### Docker

```bash
docker build -t audioserve .
docker run --gpus all -p 8000:8000 -p 50051:50051 audioserve \
  serve -m openai/whisper-large-v3
```

### pip

```bash
pip install audioserve
audioserve serve -m openai/whisper-large-v3
```

### Python

```python
from audioserve import AudioServeEngine

engine = AudioServeEngine(model="openai/whisper-large-v3")
engine.start()
engine.serve(port=8000)  # REST on 8000, gRPC on 50051
```

## API

### REST — Transcribe

```bash
curl -X POST http://localhost:8000/v1/transcribe \
  -F "audio=@meeting.wav" \
  -F "language=en"
```

### REST — Transcribe + Diarization

```bash
curl -X POST http://localhost:8000/v1/transcribe+diarize \
  -F "audio=@meeting.wav" \
  -F "min_speakers=2" \
  -F "max_speakers=5"
```

### gRPC

```python
import grpc
from audioserve.proto import audioserve_pb2, audioserve_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = audioserve_pb2_grpc.AudioServeStub(channel)

with open("meeting.wav", "rb") as f:
    response = stub.Transcribe(audioserve_pb2.TranscribeRequest(
        audio=f.read(),
        beam_size=5,
        word_timestamps=True,
    ))
print(response.text)
```

### Python Client

```python
from audioserve import AudioServeClient

client = AudioServeClient("http://localhost:8000")
result = client.transcribe("meeting.wav", language="en")
print(result.text)
print(result.segments)  # word-level timestamps
```

### Diarization

```bash
audioserve serve -m openai/whisper-large-v3 --diarization --hf-token YOUR_TOKEN
```

Requires accepting the pyannote model license on HuggingFace:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

## Supported Models

| Model | Architecture | Backend | Languages |
|-------|-------------|---------|-----------|
| openai/whisper-* (tiny → large-v3-turbo) | Encoder-Decoder | faster-whisper (CTranslate2) | 99 |
| distil-whisper/* | Distilled Encoder-Decoder | faster-whisper (CTranslate2) | EN |
| facebook/wav2vec2-* | Encoder + CTC | PyTorch + torch.compile | per-finetune |
| facebook/hubert-* | Encoder + CTC | PyTorch + torch.compile | per-finetune |
| pyannote/speaker-diarization-3.1 | Segmentation + clustering | pyannote.audio | any |

## Benchmarks

RTX A5000 24GB, CUDA 13.0, 6.5 min audio file:

| Model | Avg Latency | Realtime Factor |
|-------|-------------|-----------------|
| openai/whisper-tiny | 4.5s | 87x |
| openai/whisper-base | 4.7s | 83x |
| openai/whisper-small | 5.0s | 78x |
| openai/whisper-medium | 6.9s | 56x |
| openai/whisper-large-v3 | 9.0s | 43x |
| openai/whisper-large-v3-turbo | 6.6s | 59x |
| facebook/wav2vec2-base-960h | 2.9s | 133x |

**Diarization overhead** (whisper-large-v3 + pyannote): +6.6s for speaker identification.

## Architecture

```
Client (REST/gRPC)
    |
    v
Request Queue → Dynamic Batch Scheduler → Model Runner → Response
                      |
              Sort by duration,
              pad minimally
                      |
        WhisperRunner | Wav2Vec2Runner | PyAnnoteRunner
        (CTranslate2)   (torch.compile)   (pyannote.audio)
```

## Roadmap

- **v0.1** (current) — Whisper, Wav2Vec2, diarization, REST + gRPC API, dynamic batching, Docker
- **v0.2** — Streaming ASR (WebSocket), NVIDIA Parakeet support, custom CUDA mel spectrogram kernel
- **v0.3** — Multi-GPU, INT8 quantization, Prometheus metrics, Kubernetes deployment

## Development

```bash
git clone https://github.com/inflexionlab/audioserve.git
cd audioserve
pip install -e ".[dev]"
pytest
```

## License

Apache 2.0
