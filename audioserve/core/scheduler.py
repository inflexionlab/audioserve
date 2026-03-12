"""Dynamic batching scheduler for audio inference requests."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from audioserve.config import SchedulerConfig

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """A single inference request in the queue."""

    request_id: str = field(default_factory=lambda: uuid4().hex[:12])
    audio_duration: float = 0.0  # seconds — used for sorting/batching
    payload: Any = None  # model-specific data (AudioData, file path, etc.)
    model_id: str = ""
    task: str = "asr"
    params: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
    future: asyncio.Future | None = None


@dataclass
class Batch:
    """A batch of requests to be processed together."""

    requests: list[InferenceRequest]
    created_at: float = field(default_factory=time.monotonic)

    @property
    def size(self) -> int:
        return len(self.requests)

    @property
    def max_duration(self) -> float:
        return max(r.audio_duration for r in self.requests) if self.requests else 0.0


class DynamicBatchScheduler:
    """Accumulates requests and forms optimal batches.

    Key optimizations:
    - Sorts by audio duration to minimize padding waste
    - Respects max wait time to bound latency
    - Groups requests by model to avoid cross-model batching
    """

    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self._queues: dict[str, list[InferenceRequest]] = {}  # model_id -> queue
        self._lock = asyncio.Lock()
        self._notify: dict[str, asyncio.Event] = {}

    async def enqueue(self, request: InferenceRequest) -> asyncio.Future:
        """Add a request to the queue. Returns a future for the result."""
        loop = asyncio.get_running_loop()
        request.future = loop.create_future()

        async with self._lock:
            model_id = request.model_id
            if model_id not in self._queues:
                self._queues[model_id] = []
                self._notify[model_id] = asyncio.Event()

            self._queues[model_id].append(request)
            self._notify[model_id].set()

        logger.debug(
            "Enqueued request %s for model %s (duration=%.1fs, queue_size=%d)",
            request.request_id,
            model_id,
            request.audio_duration,
            len(self._queues[model_id]),
        )

        return request.future

    async def get_batch(self, model_id: str) -> Batch:
        """Wait for and return the next batch for a given model.

        Waits until either:
        - max_batch_size requests are accumulated, or
        - max_wait_time_ms has elapsed since the first request arrived
        """
        while True:
            # Wait for at least one request
            if model_id not in self._queues or not self._queues[model_id]:
                if model_id not in self._notify:
                    self._notify[model_id] = asyncio.Event()
                self._notify[model_id].clear()
                await self._notify[model_id].wait()

            # Wait for batch to fill or timeout
            deadline = time.monotonic() + self.config.max_wait_time_ms / 1000.0

            while True:
                async with self._lock:
                    queue = self._queues.get(model_id, [])
                    if len(queue) >= self.config.max_batch_size:
                        break

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break

                try:
                    await asyncio.wait_for(
                        self._wait_for_more(model_id),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    break

            # Extract batch
            async with self._lock:
                queue = self._queues.get(model_id, [])
                if not queue:
                    continue

                batch_requests = queue[: self.config.max_batch_size]
                self._queues[model_id] = queue[self.config.max_batch_size :]

                # Sort by duration to minimize padding
                if self.config.sort_by_duration:
                    batch_requests.sort(key=lambda r: r.audio_duration)

                batch = Batch(requests=batch_requests)
                logger.info(
                    "Formed batch for %s: size=%d, max_duration=%.1fs",
                    model_id,
                    batch.size,
                    batch.max_duration,
                )
                return batch

    async def _wait_for_more(self, model_id: str) -> None:
        """Wait until more requests arrive for this model."""
        event = self._notify.get(model_id)
        if event:
            event.clear()
            await event.wait()

    def pending_count(self, model_id: str) -> int:
        return len(self._queues.get(model_id, []))

    def total_pending(self) -> int:
        return sum(len(q) for q in self._queues.values())
