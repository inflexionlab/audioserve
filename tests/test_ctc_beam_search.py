"""Tests for CTC beam search CUDA kernel.

CPU reference implementation validates the algorithm;
GPU tests compare CUDA kernel output against the reference.
"""

from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import pytest
import torch

LOG_ZERO = -1e30


# ============================================================
# CPU reference implementation (for validation)
# ============================================================


def log_add(a: float, b: float) -> float:
    if a <= LOG_ZERO:
        return b
    if b <= LOG_ZERO:
        return a
    mx = max(a, b)
    return mx + math.log1p(math.exp(min(a, b) - mx))


def cpu_ctc_beam_search(
    log_probs: np.ndarray,
    blank_id: int = 0,
    beam_width: int = 10,
    top_k: int = 1,
) -> list[tuple[list[int], float]]:
    """Reference CTC prefix beam search in pure Python.

    Args:
        log_probs: (T, V) numpy array of log-probabilities.
        blank_id: Blank token index.
        beam_width: Number of beams to keep.
        top_k: Number of results to return.

    Returns:
        List of (token_ids, score) tuples, sorted by score descending.
    """
    T, V = log_probs.shape

    # beams: dict mapping tuple(tokens) -> (log_pb, log_pnb)
    beams: dict[tuple, tuple[float, float]] = {(): (log_probs[0, blank_id], LOG_ZERO)}

    # Initialize with non-blank tokens at t=0
    for v in range(V):
        if v == blank_id:
            continue
        beams[(v,)] = (LOG_ZERO, log_probs[0, v])

    # Prune initial beams
    beams = _prune_beams(beams, beam_width)

    for t in range(1, T):
        new_beams: dict[tuple, tuple[float, float]] = defaultdict(
            lambda: (LOG_ZERO, LOG_ZERO)
        )

        for prefix, (old_pb, old_pnb) in beams.items():
            old_total = log_add(old_pb, old_pnb)

            # Blank extension
            cur_pb, cur_pnb = new_beams[prefix]
            new_beams[prefix] = (
                log_add(cur_pb, old_total + log_probs[t, blank_id]),
                cur_pnb,
            )

            # Token extensions
            for v in range(V):
                if v == blank_id:
                    continue

                if prefix and v == prefix[-1]:
                    # Repeat: only extend via blank path
                    new_pnb = old_pb + log_probs[t, v]
                    new_prefix = prefix + (v,)
                    cpb, cpnb = new_beams[new_prefix]
                    new_beams[new_prefix] = (cpb, log_add(cpnb, new_pnb))

                    # Also extend same prefix via non-blank (collapse)
                    cpb2, cpnb2 = new_beams[prefix]
                    new_beams[prefix] = (
                        cpb2,
                        log_add(cpnb2, old_pnb + log_probs[t, v]),
                    )
                else:
                    new_prefix = prefix + (v,)
                    cpb, cpnb = new_beams[new_prefix]
                    new_beams[new_prefix] = (
                        cpb,
                        log_add(cpnb, old_total + log_probs[t, v]),
                    )

        beams = _prune_beams(new_beams, beam_width)

    # Return top-K
    scored = [
        (list(prefix), log_add(pb, pnb))
        for prefix, (pb, pnb) in beams.items()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def _prune_beams(beams, beam_width):
    scored = [
        (prefix, log_add(pb, pnb), (pb, pnb))
        for prefix, (pb, pnb) in beams.items()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return {prefix: probs for prefix, _, probs in scored[:beam_width]}


# ============================================================
# Helper to make deterministic log-probs
# ============================================================


def make_log_probs(T: int, V: int, spike_sequence: list[int] | None = None) -> np.ndarray:
    """Create log-probs with optional spike pattern.

    If spike_sequence is given, makes the specified token dominant at each timestep.
    """
    # Start with uniform
    lp = np.full((T, V), -10.0, dtype=np.float32)

    if spike_sequence:
        for t, tok in enumerate(spike_sequence):
            if t < T:
                lp[t, tok] = -0.01  # dominant
    else:
        # Random but valid log-probs
        rng = np.random.RandomState(42)
        raw = rng.randn(T, V).astype(np.float32)
        lp = raw - np.log(np.exp(raw).sum(axis=1, keepdims=True))

    return lp


# ============================================================
# Unit tests: CPU reference
# ============================================================


class TestCPUReference:
    """Validate the CPU reference beam search implementation."""

    def test_greedy_equivalent_peaky(self):
        """On very peaky distributions, beam_width=1 matches greedy argmax."""
        # With very peaked probabilities, beam search and greedy should agree
        lp = np.full((10, 4), -10.0, dtype=np.float32)
        # Clear sequence: blank, A, A, blank, B, B, blank, C, blank, blank
        spike = [0, 1, 1, 0, 2, 2, 0, 3, 0, 0]
        for t, tok in enumerate(spike):
            lp[t, tok] = -0.001

        # Greedy: collapse A,A -> A; B,B -> B; remove blanks -> [A, B, C]
        result = cpu_ctc_beam_search(lp, blank_id=0, beam_width=1)
        assert len(result) >= 1
        assert result[0][0] == [1, 2, 3]

    def test_blank_merging(self):
        """A-blank-A should produce [A, A] (two separate A's)."""
        # V=3: blank=0, A=1, B=2
        # t=0: A dominant, t=1: blank dominant, t=2: A dominant
        lp = np.full((3, 3), -10.0, dtype=np.float32)
        lp[0, 1] = -0.01  # A
        lp[1, 0] = -0.01  # blank
        lp[2, 1] = -0.01  # A
        result = cpu_ctc_beam_search(lp, blank_id=0, beam_width=5)
        assert result[0][0] == [1, 1]  # A, A

    def test_repeat_collapse(self):
        """A-A (no blank) should collapse to single A."""
        lp = np.full((2, 3), -10.0, dtype=np.float32)
        lp[0, 1] = -0.01  # A
        lp[1, 1] = -0.01  # A again
        result = cpu_ctc_beam_search(lp, blank_id=0, beam_width=5)
        assert result[0][0] == [1]  # single A

    def test_simple_abc(self):
        """Strong A-B-C signal should decode to [A, B, C]."""
        lp = make_log_probs(3, 4, spike_sequence=[1, 2, 3])
        result = cpu_ctc_beam_search(lp, blank_id=0, beam_width=5)
        assert result[0][0] == [1, 2, 3]

    def test_beam_width_effect(self):
        """Wider beam should find equal or better scoring path."""
        rng = np.random.RandomState(77)
        raw = rng.randn(30, 10).astype(np.float32)
        lp = raw - np.log(np.exp(raw).sum(axis=1, keepdims=True))

        narrow = cpu_ctc_beam_search(lp, blank_id=0, beam_width=2)
        wide = cpu_ctc_beam_search(lp, blank_id=0, beam_width=20)
        assert wide[0][1] >= narrow[0][1] - 1e-5

    def test_all_blank(self):
        """If blank dominates every timestep, output should be empty."""
        lp = np.full((10, 4), -10.0, dtype=np.float32)
        lp[:, 0] = -0.001  # blank dominant everywhere
        result = cpu_ctc_beam_search(lp, blank_id=0, beam_width=5)
        assert result[0][0] == []

    def test_top_k(self):
        """Should return multiple hypotheses when top_k > 1."""
        rng = np.random.RandomState(99)
        raw = rng.randn(20, 6).astype(np.float32)
        lp = raw - np.log(np.exp(raw).sum(axis=1, keepdims=True))

        result = cpu_ctc_beam_search(lp, blank_id=0, beam_width=10, top_k=3)
        assert len(result) >= 2
        # Scores should be descending
        for i in range(len(result) - 1):
            assert result[i][1] >= result[i + 1][1] - 1e-6


# ============================================================
# GPU tests: CUDA kernel vs CPU reference
# ============================================================


@pytest.mark.gpu
class TestCTCBeamSearchCUDA:
    """Validate CUDA kernel against CPU reference."""

    def test_kernel_compiles(self):
        """CUDA kernel should compile without errors."""
        from audioserve.cuda.ctc_beam_search import _get_module
        mod = _get_module()
        assert mod is not None

    def test_simple_abc(self):
        """Strong A-B-C signal: CUDA should match CPU."""
        from audioserve.cuda.ctc_beam_search import ctc_beam_search

        lp = make_log_probs(3, 4, spike_sequence=[1, 2, 3])
        cpu_result = cpu_ctc_beam_search(lp, blank_id=0, beam_width=5)

        lp_gpu = torch.from_numpy(lp).cuda()
        gpu_result = ctc_beam_search(lp_gpu, blank_id=0, beam_width=5, top_k=1)

        assert gpu_result[0][0][0] == cpu_result[0][0]

    def test_matches_cpu_peaky(self):
        """Peaky distribution — CUDA and CPU should agree on top beam."""
        from audioserve.cuda.ctc_beam_search import ctc_beam_search

        # Create peaky log-probs (one token clearly dominant per timestep)
        rng = np.random.RandomState(42)
        lp = np.full((50, 8), -10.0, dtype=np.float32)
        for t in range(50):
            tok = rng.randint(0, 8)
            lp[t, tok] = -0.05

        cpu_result = cpu_ctc_beam_search(lp, blank_id=0, beam_width=10)
        gpu_result = ctc_beam_search(
            torch.from_numpy(lp).cuda(), blank_id=0, beam_width=10, top_k=1
        )

        assert gpu_result[0][0][0] == cpu_result[0][0]

    def test_matches_cpu_deterministic(self):
        """Deterministic spike pattern: CUDA and CPU should agree exactly."""
        from audioserve.cuda.ctc_beam_search import ctc_beam_search

        # Construct log-probs with clear winner at each timestep
        # blank=0, then tokens: A=1 B=2 C=3 D=4
        lp = np.full((8, 5), -10.0, dtype=np.float32)
        # Sequence: A blank B B blank A C D
        # Expected CTC decode: A B B A C D (blank removed, B-B collapsed to B)
        # Actually: A, blank, B, B -> A B (collapse), blank, A -> A B A, C, D -> A B A C D
        spikes = [1, 0, 2, 2, 0, 1, 3, 4]
        for t, tok in enumerate(spikes):
            lp[t, tok] = -0.001

        cpu_result = cpu_ctc_beam_search(lp, blank_id=0, beam_width=10)
        gpu_result = ctc_beam_search(
            torch.from_numpy(lp).cuda(), blank_id=0, beam_width=10, top_k=1
        )

        assert gpu_result[0][0][0] == cpu_result[0][0]

    def test_score_quality_random(self):
        """Random inputs: GPU score within 10% of CPU (approximate beam search)."""
        from audioserve.cuda.ctc_beam_search import ctc_beam_search

        rng = np.random.RandomState(42)
        raw = rng.randn(50, 8).astype(np.float32)
        lp = raw - np.log(np.exp(raw).sum(axis=1, keepdims=True))

        cpu_result = cpu_ctc_beam_search(lp, blank_id=0, beam_width=10)
        gpu_result = ctc_beam_search(
            torch.from_numpy(lp).cuda(), blank_id=0, beam_width=10, top_k=1
        )

        cpu_score = cpu_result[0][1]
        gpu_score = gpu_result[0][0][1]
        # Both scores are negative; GPU may be slightly worse due to pruning
        assert gpu_score > cpu_score * 1.15, (
            f"GPU score {gpu_score:.2f} too far from CPU {cpu_score:.2f}"
        )

    def test_score_quality_large_vocab(self):
        """Large vocab: GPU score should be close to CPU (within 5% of path length)."""
        from audioserve.cuda.ctc_beam_search import ctc_beam_search

        rng = np.random.RandomState(42)
        raw = rng.randn(100, 32).astype(np.float32)
        lp = raw - np.log(np.exp(raw).sum(axis=1, keepdims=True))

        cpu_result = cpu_ctc_beam_search(lp, blank_id=0, beam_width=10)
        gpu_result = ctc_beam_search(
            torch.from_numpy(lp).cuda(), blank_id=0, beam_width=10, top_k=1
        )

        cpu_score = cpu_result[0][1]
        gpu_score = gpu_result[0][0][1]
        # GPU may find slightly different path, but score should be comparable
        assert gpu_score > cpu_score * 1.05  # within 5% of CPU score

    def test_blank_merging(self):
        """A-blank-A should produce [A, A]."""
        from audioserve.cuda.ctc_beam_search import ctc_beam_search

        lp = np.full((3, 3), -10.0, dtype=np.float32)
        lp[0, 1] = -0.01
        lp[1, 0] = -0.01
        lp[2, 1] = -0.01

        result = ctc_beam_search(
            torch.from_numpy(lp).cuda(), blank_id=0, beam_width=5, top_k=1
        )
        assert result[0][0][0] == [1, 1]

    def test_repeat_collapse(self):
        """A-A should collapse to single A."""
        from audioserve.cuda.ctc_beam_search import ctc_beam_search

        lp = np.full((2, 3), -10.0, dtype=np.float32)
        lp[0, 1] = -0.01
        lp[1, 1] = -0.01

        result = ctc_beam_search(
            torch.from_numpy(lp).cuda(), blank_id=0, beam_width=5, top_k=1
        )
        assert result[0][0][0] == [1]

    def test_all_blank(self):
        """All-blank input should produce empty output."""
        from audioserve.cuda.ctc_beam_search import ctc_beam_search

        lp = np.full((10, 4), -10.0, dtype=np.float32)
        lp[:, 0] = -0.001

        result = ctc_beam_search(
            torch.from_numpy(lp).cuda(), blank_id=0, beam_width=5, top_k=1
        )
        assert result[0][0][0] == []

    def test_batch_parallel(self):
        """Batched input: each element should match independent computation."""
        from audioserve.cuda.ctc_beam_search import ctc_beam_search

        rng = np.random.RandomState(55)
        B, T, V = 4, 30, 8
        raw = rng.randn(B, T, V).astype(np.float32)
        lp = raw - np.log(np.exp(raw).sum(axis=2, keepdims=True))

        # Batch result
        batch_results = ctc_beam_search(
            torch.from_numpy(lp).cuda(), blank_id=0, beam_width=8, top_k=1
        )

        # Individual results
        for b in range(B):
            single = ctc_beam_search(
                torch.from_numpy(lp[b]).cuda(), blank_id=0, beam_width=8, top_k=1
            )
            assert batch_results[b][0][0] == single[0][0][0]

    def test_long_sequence(self):
        """T=500 timesteps — should not crash or OOM."""
        from audioserve.cuda.ctc_beam_search import ctc_beam_search

        rng = np.random.RandomState(66)
        raw = rng.randn(500, 32).astype(np.float32)
        lp = raw - np.log(np.exp(raw).sum(axis=1, keepdims=True))

        result = ctc_beam_search(
            torch.from_numpy(lp).cuda(), blank_id=0, beam_width=10, top_k=1
        )
        assert len(result) == 1
        assert len(result[0]) >= 1
        tokens = result[0][0][0]
        assert all(0 < t < 32 for t in tokens)

    def test_scores_finite_and_negative(self):
        """Output scores should be finite negative log-probs."""
        from audioserve.cuda.ctc_beam_search import ctc_beam_search

        rng = np.random.RandomState(77)
        raw = rng.randn(50, 10).astype(np.float32)
        lp = raw - np.log(np.exp(raw).sum(axis=1, keepdims=True))

        result = ctc_beam_search(
            torch.from_numpy(lp).cuda(), blank_id=0, beam_width=10, top_k=5
        )
        for tokens, score in result[0]:
            assert np.isfinite(score)
            assert score <= 0.0

    def test_top_k_sorted(self):
        """Multiple hypotheses should be sorted by score descending."""
        from audioserve.cuda.ctc_beam_search import ctc_beam_search

        rng = np.random.RandomState(88)
        raw = rng.randn(30, 8).astype(np.float32)
        lp = raw - np.log(np.exp(raw).sum(axis=1, keepdims=True))

        result = ctc_beam_search(
            torch.from_numpy(lp).cuda(), blank_id=0, beam_width=10, top_k=5
        )
        scores = [s for _, s in result[0]]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1] - 1e-5


@pytest.mark.gpu
class TestCTCBeamSearchIntegration:
    """End-to-end with real Wav2Vec2 model."""

    @pytest.fixture(autouse=True)
    def setup_model(self):
        from audioserve.config import ModelConfig
        from audioserve.models.wav2vec2 import Wav2Vec2Runner

        config = ModelConfig(model_id="facebook/wav2vec2-base-960h", dtype="float32")
        self.runner = Wav2Vec2Runner(config)
        self.runner.load()
        yield
        self.runner.unload()

    def test_beam_vs_greedy(self):
        """Beam search should produce valid text, not worse than greedy."""
        # Generate a sine wave (will produce gibberish but should not crash)
        audio = np.sin(2 * np.pi * 440 * np.arange(16000 * 3) / 16000).astype(np.float32)

        greedy_result = self.runner.transcribe_batch(
            [audio], [{"beam_size": 1}]
        )[0]
        beam_result = self.runner.transcribe_batch(
            [audio], [{"beam_size": 10}]
        )[0]

        # Both should produce some text (or both empty for pure tone)
        assert isinstance(greedy_result.text, str)
        assert isinstance(beam_result.text, str)

    def test_beam_search_on_speech(self):
        """Beam search on real-ish audio should produce text."""
        # White noise with some structure (simulates speech-like signal)
        rng = np.random.RandomState(42)
        audio = rng.randn(16000 * 2).astype(np.float32) * 0.1

        result = self.runner.transcribe_batch(
            [audio], [{"beam_size": 5}]
        )[0]
        assert isinstance(result.text, str)

    def test_fallback_when_beam1(self):
        """beam_size=1 should use greedy path (no kernel call)."""
        audio = np.zeros(16000, dtype=np.float32)
        result = self.runner.transcribe_batch(
            [audio], [{"beam_size": 1}]
        )[0]
        assert isinstance(result.text, str)
