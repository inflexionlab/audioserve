"""AudioServe — Optimized inference server for audio models."""

__version__ = "0.1.0"

from audioserve.engine import AudioServeEngine
from audioserve.client import AudioServeClient

__all__ = ["AudioServeEngine", "AudioServeClient", "__version__"]
