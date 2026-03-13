"""Tests for the dynamic batch scheduler."""

from __future__ import annotations

import asyncio

import pytest

from audioserve.config import SchedulerConfig
from audioserve.core.scheduler import Batch, DynamicBatchScheduler, InferenceRequest


@pytest.fixture
def scheduler():
    config = SchedulerConfig(max_batch_size=4, max_wait_time_ms=50, sort_by_duration=True)
    return DynamicBatchScheduler(config)


def _make_request(model_id: str = "test-model", duration: float = 1.0) -> InferenceRequest:
    return InferenceRequest(
        audio_duration=duration,
        payload=f"audio_{duration}s",
        model_id=model_id,
        params={"language": "en"},
    )


class TestEnqueue:
    @pytest.mark.asyncio
    async def test_enqueue_returns_future(self, scheduler):
        req = _make_request()
        future = await scheduler.enqueue(req)
        assert isinstance(future, asyncio.Future)
        assert not future.done()

    @pytest.mark.asyncio
    async def test_pending_count(self, scheduler):
        assert scheduler.total_pending() == 0

        await scheduler.enqueue(_make_request())
        assert scheduler.pending_count("test-model") == 1
        assert scheduler.total_pending() == 1

        await scheduler.enqueue(_make_request())
        assert scheduler.pending_count("test-model") == 2

    @pytest.mark.asyncio
    async def test_separate_model_queues(self, scheduler):
        await scheduler.enqueue(_make_request("model-a"))
        await scheduler.enqueue(_make_request("model-b"))

        assert scheduler.pending_count("model-a") == 1
        assert scheduler.pending_count("model-b") == 1
        assert scheduler.total_pending() == 2


class TestGetBatch:
    @pytest.mark.asyncio
    async def test_batch_on_timeout(self, scheduler):
        """Batch should form after max_wait_time_ms even if not full."""
        await scheduler.enqueue(_make_request(duration=2.0))
        await scheduler.enqueue(_make_request(duration=1.0))

        batch = await asyncio.wait_for(
            scheduler.get_batch("test-model"),
            timeout=1.0,
        )
        assert isinstance(batch, Batch)
        assert batch.size == 2

    @pytest.mark.asyncio
    async def test_batch_sorted_by_duration(self, scheduler):
        """Requests should be sorted by audio duration."""
        await scheduler.enqueue(_make_request(duration=5.0))
        await scheduler.enqueue(_make_request(duration=1.0))
        await scheduler.enqueue(_make_request(duration=3.0))

        batch = await asyncio.wait_for(
            scheduler.get_batch("test-model"),
            timeout=1.0,
        )
        durations = [r.audio_duration for r in batch.requests]
        assert durations == sorted(durations)

    @pytest.mark.asyncio
    async def test_batch_respects_max_size(self, scheduler):
        """Should not exceed max_batch_size."""
        for i in range(6):
            await scheduler.enqueue(_make_request(duration=float(i)))

        batch = await asyncio.wait_for(
            scheduler.get_batch("test-model"),
            timeout=1.0,
        )
        assert batch.size == 4  # max_batch_size

        # Remaining 2 should still be in queue
        assert scheduler.pending_count("test-model") == 2

    @pytest.mark.asyncio
    async def test_batch_max_duration(self, scheduler):
        await scheduler.enqueue(_make_request(duration=1.0))
        await scheduler.enqueue(_make_request(duration=10.0))

        batch = await asyncio.wait_for(
            scheduler.get_batch("test-model"),
            timeout=1.0,
        )
        assert batch.max_duration == 10.0


class TestBatchDataclass:
    def test_size(self):
        reqs = [_make_request() for _ in range(3)]
        batch = Batch(requests=reqs)
        assert batch.size == 3

    def test_max_duration(self):
        reqs = [_make_request(duration=d) for d in [1.0, 5.0, 3.0]]
        batch = Batch(requests=reqs)
        assert batch.max_duration == 5.0

    def test_empty_batch(self):
        batch = Batch(requests=[])
        assert batch.size == 0
        assert batch.max_duration == 0.0
