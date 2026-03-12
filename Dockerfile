FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps: ffmpeg for audio decoding, python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch first (large layer, cache separately)
RUN pip3 install --no-cache-dir --break-system-packages \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install audioserve
COPY pyproject.toml README.md ./
COPY audioserve/ audioserve/
RUN pip3 install --no-cache-dir --break-system-packages .

EXPOSE 8000 50051

ENTRYPOINT ["audioserve"]
CMD ["serve", "-m", "openai/whisper-large-v3"]
