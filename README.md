# AudioServe

Optimized inference server for audio models — like vLLM, but for ASR.

**One command to deploy any speech recognition model as a production API.**

## Features

- **Multi-model support** — Whisper (all sizes), Wav2Vec2, HuBERT, distil-whisper
- **Optimized Whisper inference** — powered by faster-whisper (CTranslate2), up to 25x realtime
- **Speaker diarization** — integrated pyannote.audio pipeline with per-segment speaker labels
- **Dynamic batching** — sort-by-duration padding minimization, configurable batch scheduler
- **Two endpoints** — pure ASR (`/v1/transcribe`) and ASR+diarization (`/v1/transcribe+diarize`)
- **Word-level timestamps** — precise timing for every word
- **Language auto-detection** — 99 languages via Whisper
- **Python client** — simple API for programmatic access
- **CLI** — serve models or transcribe files from the command line

## Quickstart

```bash
pip install audioserve
```

### Start a server

```bash
audioserve serve -m openai/whisper-large-v3
```

### Or use Python

```python
from audioserve import AudioServeEngine

engine = AudioServeEngine(model="openai/whisper-large-v3")
engine.start()
engine.serve(port=8000)
```

### Transcribe via API

```bash
curl -X POST http://localhost:8000/v1/transcribe \
  -F "audio=@meeting.wav" \
  -F "language=en"
```

### Python client

```python
from audioserve import AudioServeClient

client = AudioServeClient("http://localhost:8000")
result = client.transcribe("meeting.wav", language="en")
print(result.text)
print(result.segments)  # with timestamps and word-level detail
```

### ASR + Speaker Diarization

```python
engine = AudioServeEngine(
    model="openai/whisper-large-v3",
    diarization=True,
    hf_token="your_token",  # required for pyannote
)
engine.start()
engine.serve()
```

```bash
curl -X POST http://localhost:8000/v1/transcribe+diarize \
  -F "audio=@meeting.wav" \
  -F "min_speakers=2" \
  -F "max_speakers=5"
```

### CLI transcription (no server needed)

```bash
audioserve transcribe recording.wav -m openai/whisper-large-v3 -l en
```

## Supported Models

| Model | Architecture | Backend | Languages |
|-------|-------------|---------|-----------|
| openai/whisper-* (tiny → large-v3-turbo) | Encoder-Decoder | faster-whisper (CTranslate2) | 99 |
| distil-whisper/* | Distilled Encoder-Decoder | faster-whisper (CTranslate2) | EN |
| facebook/wav2vec2-* | Encoder + CTC | PyTorch + torch.compile | per-finetune |
| facebook/hubert-* | Encoder + CTC | PyTorch + torch.compile | per-finetune |
| pyannote/speaker-diarization-3.1 | Segmentation + clustering | pyannote.audio | any |

## Benchmarks (RTX A5000, 24GB)

| Model | Avg Latency (3.4s audio) | Realtime Factor |
|-------|--------------------------|-----------------|
| whisper-tiny | 134ms | 25x |
| whisper-base | 142ms | 24x |
| whisper-small | 195ms | 17x |

## Architecture

```
Client (REST/gRPC) → Request Queue → Dynamic Batch Scheduler → Model Runner → Response
                                          ↓
                               Sort by duration, pad minimally
                                          ↓
                              WhisperRunner | Wav2Vec2Runner | PyAnnoteRunner
```

## Roadmap

- **v0.1** (current) — Whisper, Wav2Vec2, diarization, REST API, dynamic batching
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
