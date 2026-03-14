"""CTC beam search decoding via custom CUDA kernel.

Replaces torch.argmax greedy CTC decoding with proper beam search on GPU.
One thread block per batch element; timestep loop runs inside the kernel
to avoid per-step kernel launch overhead.

Algorithm:
    For each timestep t:
    1. Expand: each beam produces candidates (blank ext, repeat collapse, new tokens).
       Same-prefix candidates use sentinel tokens (-1=blank, -2=collapse).
    2. Prune: top-K candidates by score.
    3. Materialize: build new beams from selected candidates.
    4. Dedup: merge beams with identical token sequences.

Log-space arithmetic throughout (log_add via log-sum-exp) to avoid underflow.

Usage:
    from audioserve.cuda.ctc_beam_search import ctc_beam_search
    # log_probs: (T, V) or (B, T, V) float32 tensor on CUDA
    results = ctc_beam_search(log_probs, blank_id=0, beam_width=10)
    # results: list of list of (token_ids, score) tuples
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.cpp_extension import load_inline

logger = logging.getLogger(__name__)

MAX_BEAM_WIDTH = 16
MAX_SEQ_LEN = 512
MAX_VOCAB_SIZE = 64

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define LOG_ZERO -1e30f
#define MAX_BEAM_WIDTH 16
#define MAX_SEQ_LEN 512
#define MAX_VOCAB_SIZE 64
#define MAX_CANDIDATES (MAX_BEAM_WIDTH * (MAX_VOCAB_SIZE + 2))
#define BLOCK_SIZE 128

__device__ __forceinline__ float log_add(float a, float b) {
    if (a <= LOG_ZERO) return b;
    if (b <= LOG_ZERO) return a;
    float mx = fmaxf(a, b);
    return mx + log1pf(expf(fminf(a, b) - mx));
}

// ============================================================
// Kernel: CTC beam search — one block per batch element
// ============================================================
__global__ void ctc_beam_search_kernel(
    const float* __restrict__ log_probs,    // (B, T, V)
    int*         __restrict__ out_tokens,    // (B, K, MAX_SEQ_LEN)
    int*         __restrict__ out_lengths,   // (B, K)
    float*       __restrict__ out_scores,    // (B, K)
    int*         __restrict__ beam_tok_buf,  // (B, MAX_BEAM_WIDTH, MAX_SEQ_LEN)
    int*         __restrict__ new_tok_buf,   // (B, MAX_BEAM_WIDTH, MAX_SEQ_LEN)
    float*       __restrict__ g_cand_score,  // (B, MAX_CANDIDATES)
    float*       __restrict__ g_cand_pb,
    float*       __restrict__ g_cand_pnb,
    int*         __restrict__ g_cand_parent,
    int*         __restrict__ g_cand_token,
    int blank_id, int beam_width, int top_k,
    int B, int T, int V
) {
    int batch = blockIdx.x;
    if (batch >= B) return;
    int tid = threadIdx.x;

    __shared__ float s_log_pb[MAX_BEAM_WIDTH];
    __shared__ float s_log_pnb[MAX_BEAM_WIDTH];
    __shared__ int   s_last_token[MAX_BEAM_WIDTH];
    __shared__ int   s_length[MAX_BEAM_WIDTH];
    __shared__ int   s_num_beams;
    __shared__ float s_old_pb[MAX_BEAM_WIDTH];
    __shared__ float s_old_pnb[MAX_BEAM_WIDTH];
    __shared__ int   s_old_last[MAX_BEAM_WIDTH];
    __shared__ int   s_old_len[MAX_BEAM_WIDTH];
    __shared__ int   s_num_cands;

    const float* my_lp = log_probs + (long long)batch * T * V;
    int* my_beam = beam_tok_buf + (long long)batch * MAX_BEAM_WIDTH * MAX_SEQ_LEN;
    int* my_new  = new_tok_buf  + (long long)batch * MAX_BEAM_WIDTH * MAX_SEQ_LEN;

    long long coff = (long long)batch * MAX_CANDIDATES;
    float* c_score  = g_cand_score  + coff;
    float* c_pb     = g_cand_pb     + coff;
    float* c_pnb    = g_cand_pnb    + coff;
    int*   c_parent = g_cand_parent + coff;
    int*   c_token  = g_cand_token  + coff;

    // ---- Initialize ----
    if (tid == 0) {
        s_log_pb[0] = my_lp[blank_id];
        s_log_pnb[0] = LOG_ZERO;
        s_last_token[0] = -1;
        s_length[0] = 0;
        s_num_beams = 1;

        int nb = 1;
        for (int v = 0; v < V && nb < beam_width; v++) {
            if (v == blank_id) continue;
            s_log_pb[nb] = LOG_ZERO;
            s_log_pnb[nb] = my_lp[v];
            s_last_token[nb] = v;
            s_length[nb] = 1;
            my_beam[nb * MAX_SEQ_LEN] = v;
            nb++;
        }
        s_num_beams = nb;
    }
    __syncthreads();

    // ---- Main loop ----
    for (int t = 1; t < T; t++) {
        const float* lp_t = my_lp + (long long)t * V;

        if (tid < s_num_beams) {
            s_old_pb[tid]   = s_log_pb[tid];
            s_old_pnb[tid]  = s_log_pnb[tid];
            s_old_last[tid] = s_last_token[tid];
            s_old_len[tid]  = s_length[tid];
        }
        if (tid == 0) s_num_cands = 0;
        __syncthreads();

        int num_beams = s_num_beams;

        // PHASE 1: Expand — generate all candidates
        // For each beam, produce:
        //   - blank extension (token=-1): same prefix, blank prob
        //   - repeat collapse (token=-2): same prefix if last_token matches, non-blank prob
        //   - new token (token>=0, token!=blank, token!=last): new prefix
        //   - repeat via blank (token>=0, token==last): new prefix (distinct repeat)
        int total_work = num_beams * (V + 2);  // V tokens + blank + collapse
        for (int w = tid; w < total_work; w += BLOCK_SIZE) {
            int bi = w / (V + 2);
            int vi = w % (V + 2);

            float old_pb  = s_old_pb[bi];
            float old_pnb = s_old_pnb[bi];
            float old_total = log_add(old_pb, old_pnb);
            int old_last = s_old_last[bi];

            if (vi == V) {
                // Blank extension
                float npb = old_total + lp_t[blank_id];
                int idx = atomicAdd(&s_num_cands, 1);
                if (idx < MAX_CANDIDATES) {
                    c_score[idx] = npb; c_pb[idx] = npb; c_pnb[idx] = LOG_ZERO;
                    c_parent[idx] = bi; c_token[idx] = -1;
                }
            } else if (vi == V + 1) {
                // Repeat collapse: same prefix, non-blank prob
                if (old_last >= 0 && old_last != blank_id) {
                    float npnb = old_pnb + lp_t[old_last];
                    int idx = atomicAdd(&s_num_cands, 1);
                    if (idx < MAX_CANDIDATES) {
                        c_score[idx] = npnb; c_pb[idx] = LOG_ZERO; c_pnb[idx] = npnb;
                        c_parent[idx] = bi; c_token[idx] = -2;
                    }
                }
            } else {
                int token = vi;
                if (token == blank_id) continue;

                if (token == old_last) {
                    // Repeat via blank path -> distinct repeat (new prefix)
                    float npnb = old_pb + lp_t[token];
                    int idx = atomicAdd(&s_num_cands, 1);
                    if (idx < MAX_CANDIDATES) {
                        c_score[idx] = npnb; c_pb[idx] = LOG_ZERO; c_pnb[idx] = npnb;
                        c_parent[idx] = bi; c_token[idx] = token;
                    }
                } else {
                    // Different token -> new prefix
                    float npnb = old_total + lp_t[token];
                    int idx = atomicAdd(&s_num_cands, 1);
                    if (idx < MAX_CANDIDATES) {
                        c_score[idx] = npnb; c_pb[idx] = LOG_ZERO; c_pnb[idx] = npnb;
                        c_parent[idx] = bi; c_token[idx] = token;
                    }
                }
            }
        }
        __syncthreads();

        // PHASE 2: Top-K selection (serial, thread 0)
        if (tid == 0) {
            int nc = min(s_num_cands, (int)MAX_CANDIDATES);
            int select = min(beam_width * 3, nc);  // keep extra for dedup merging

            for (int i = 0; i < select; i++) {
                int best = i;
                float bs = c_score[i];
                for (int j = i + 1; j < nc; j++) {
                    if (c_score[j] > bs) { best = j; bs = c_score[j]; }
                }
                if (best != i) {
                    float tmp; int itmp;
                    #define SWAP_F(a,b) tmp=a; a=b; b=tmp
                    #define SWAP_I(a,b) itmp=a; a=b; b=itmp
                    SWAP_F(c_score[i], c_score[best]);
                    SWAP_F(c_pb[i], c_pb[best]);
                    SWAP_F(c_pnb[i], c_pnb[best]);
                    SWAP_I(c_parent[i], c_parent[best]);
                    SWAP_I(c_token[i], c_token[best]);
                    #undef SWAP_F
                    #undef SWAP_I
                }
            }
            s_num_cands = select;
        }
        __syncthreads();

        // PHASE 3: Materialize beams from top candidates
        int mat_count = min(beam_width, s_num_cands);
        if (tid < mat_count) {
            int parent = c_parent[tid];
            int token  = c_token[tid];
            int plen   = s_old_len[parent];

            for (int i = 0; i < plen; i++)
                my_new[tid * MAX_SEQ_LEN + i] = my_beam[parent * MAX_SEQ_LEN + i];

            if (token == -1) {
                // Blank extension
                s_log_pb[tid]     = c_pb[tid];
                s_log_pnb[tid]    = LOG_ZERO;
                s_last_token[tid] = s_old_last[parent];
                s_length[tid]     = plen;
            } else if (token == -2) {
                // Repeat collapse
                s_log_pb[tid]     = LOG_ZERO;
                s_log_pnb[tid]    = c_pnb[tid];
                s_last_token[tid] = s_old_last[parent];
                s_length[tid]     = plen;
            } else {
                // New token
                my_new[tid * MAX_SEQ_LEN + plen] = token;
                s_log_pb[tid]     = LOG_ZERO;
                s_log_pnb[tid]    = c_pnb[tid];
                s_last_token[tid] = token;
                s_length[tid]     = plen + 1;
            }
        }
        if (tid == 0) s_num_beams = mat_count;
        __syncthreads();

        // Copy new -> beam
        for (int bi = tid; bi < mat_count; bi += BLOCK_SIZE) {
            for (int i = 0; i < s_length[bi]; i++)
                my_beam[bi * MAX_SEQ_LEN + i] = my_new[bi * MAX_SEQ_LEN + i];
        }
        __syncthreads();

        // PHASE 4: Prefix deduplication
        if (tid == 0) {
            for (int i = 0; i < s_num_beams; i++) {
                if (s_log_pb[i] <= LOG_ZERO && s_log_pnb[i] <= LOG_ZERO) continue;
                for (int j = i + 1; j < s_num_beams; j++) {
                    if (s_log_pb[j] <= LOG_ZERO && s_log_pnb[j] <= LOG_ZERO) continue;
                    if (s_length[i] != s_length[j]) continue;
                    if (s_last_token[i] != s_last_token[j]) continue;

                    bool same = true;
                    for (int k = 0; k < s_length[i]; k++) {
                        if (my_beam[i * MAX_SEQ_LEN + k] != my_beam[j * MAX_SEQ_LEN + k]) {
                            same = false; break;
                        }
                    }
                    if (same) {
                        s_log_pb[i]  = log_add(s_log_pb[i],  s_log_pb[j]);
                        s_log_pnb[i] = log_add(s_log_pnb[i], s_log_pnb[j]);
                        s_log_pb[j]  = LOG_ZERO;
                        s_log_pnb[j] = LOG_ZERO;
                    }
                }
            }
            // Compact
            int w = 0;
            for (int r = 0; r < s_num_beams; r++) {
                if (s_log_pb[r] > LOG_ZERO || s_log_pnb[r] > LOG_ZERO) {
                    if (w != r) {
                        s_log_pb[w] = s_log_pb[r]; s_log_pnb[w] = s_log_pnb[r];
                        s_last_token[w] = s_last_token[r]; s_length[w] = s_length[r];
                        for (int k = 0; k < s_length[r]; k++)
                            my_beam[w * MAX_SEQ_LEN + k] = my_beam[r * MAX_SEQ_LEN + k];
                    }
                    w++;
                }
            }
            s_num_beams = w;
        }
        __syncthreads();
    }

    // ---- Final sort by total score ----
    if (tid == 0) {
        for (int i = 0; i < s_num_beams; i++)
            c_score[i] = log_add(s_log_pb[i], s_log_pnb[i]);

        // Selection sort
        for (int i = 0; i < s_num_beams - 1; i++) {
            int best = i;
            float bs = c_score[i];
            for (int j = i + 1; j < s_num_beams; j++) {
                if (c_score[j] > bs) { best = j; bs = c_score[j]; }
            }
            if (best != i) {
                float tmp; int itmp;
                #define SW(a,b,t) t=a; a=b; b=t
                SW(c_score[i], c_score[best], tmp);
                SW(s_log_pb[i], s_log_pb[best], tmp);
                SW(s_log_pnb[i], s_log_pnb[best], tmp);
                SW(s_last_token[i], s_last_token[best], itmp);
                SW(s_length[i], s_length[best], itmp);
                #undef SW
                int mx = max(s_length[i], s_length[best]);
                for (int k = 0; k < mx; k++) {
                    int a = my_beam[i * MAX_SEQ_LEN + k];
                    my_beam[i * MAX_SEQ_LEN + k] = my_beam[best * MAX_SEQ_LEN + k];
                    my_beam[best * MAX_SEQ_LEN + k] = a;
                }
            }
        }
    }
    __syncthreads();

    // Write output
    int ok = min(top_k, s_num_beams);
    if (tid < ok) {
        long long bt = ((long long)batch * top_k + tid) * MAX_SEQ_LEN;
        long long bk = (long long)batch * top_k + tid;
        out_scores[bk]  = log_add(s_log_pb[tid], s_log_pnb[tid]);
        out_lengths[bk] = s_length[tid];
        for (int i = 0; i < s_length[tid]; i++)
            out_tokens[bt + i] = my_beam[tid * MAX_SEQ_LEN + i];
    }
}

// Host entry point
std::vector<torch::Tensor> ctc_beam_search(
    torch::Tensor log_probs, int blank_id, int beam_width, int top_k
) {
    TORCH_CHECK(log_probs.is_cuda(), "log_probs must be on CUDA");
    TORCH_CHECK(log_probs.dim() == 3, "log_probs must be 3D (B, T, V)");
    TORCH_CHECK(log_probs.scalar_type() == torch::kFloat32, "log_probs must be float32");
    TORCH_CHECK(beam_width <= MAX_BEAM_WIDTH, "beam_width exceeds MAX_BEAM_WIDTH");

    auto lp = log_probs.contiguous();
    int B = lp.size(0), T = lp.size(1), V = lp.size(2);
    TORCH_CHECK(V <= MAX_VOCAB_SIZE, "vocab_size exceeds MAX_VOCAB_SIZE");

    auto opts = torch::TensorOptions().device(log_probs.device());
    auto out_tokens  = torch::zeros({B, top_k, MAX_SEQ_LEN}, opts.dtype(torch::kInt32));
    auto out_lengths = torch::zeros({B, top_k}, opts.dtype(torch::kInt32));
    auto out_scores  = torch::full({B, top_k}, -1e30f, opts.dtype(torch::kFloat32));
    auto beam_buf = torch::zeros({B, MAX_BEAM_WIDTH, MAX_SEQ_LEN}, opts.dtype(torch::kInt32));
    auto new_buf  = torch::zeros({B, MAX_BEAM_WIDTH, MAX_SEQ_LEN}, opts.dtype(torch::kInt32));

    long long cs = (long long)B * MAX_CANDIDATES;
    auto gs  = torch::empty({cs}, opts.dtype(torch::kFloat32));
    auto gpb = torch::empty({cs}, opts.dtype(torch::kFloat32));
    auto gpnb= torch::empty({cs}, opts.dtype(torch::kFloat32));
    auto gp  = torch::empty({cs}, opts.dtype(torch::kInt32));
    auto gt  = torch::empty({cs}, opts.dtype(torch::kInt32));

    ctc_beam_search_kernel<<<B, BLOCK_SIZE>>>(
        lp.data_ptr<float>(),
        out_tokens.data_ptr<int>(), out_lengths.data_ptr<int>(), out_scores.data_ptr<float>(),
        beam_buf.data_ptr<int>(), new_buf.data_ptr<int>(),
        gs.data_ptr<float>(), gpb.data_ptr<float>(), gpnb.data_ptr<float>(),
        gp.data_ptr<int>(), gt.data_ptr<int>(),
        blank_id, beam_width, top_k, B, T, V
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    return {out_tokens, out_lengths, out_scores};
}
"""

cpp_source = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> ctc_beam_search(torch::Tensor, int, int, int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ctc_beam_search", &ctc_beam_search, "CTC beam search decoding (CUDA)");
}
"""

_module = None


def _get_module():
    """Lazy-compile and cache the CUDA extension."""
    global _module
    if _module is None:
        logger.info("Compiling CTC beam search CUDA kernel...")
        _module = load_inline(
            name="ctc_beam_search_cuda",
            cpp_sources=[cpp_source],
            cuda_sources=[cuda_source],
            verbose=False,
        )
        logger.info("CTC beam search CUDA kernel compiled")
    return _module


def ctc_beam_search(
    log_probs: torch.Tensor,
    blank_id: int = 0,
    beam_width: int = 10,
    top_k: int = 1,
) -> list[list[tuple[list[int], float]]]:
    """Run CTC beam search decoding on GPU.

    Args:
        log_probs: (T, V) or (B, T, V) float32 log-probabilities on CUDA.
        blank_id: Index of the CTC blank token.
        beam_width: Number of beams to maintain at each timestep.
        top_k: Number of top hypotheses to return per batch element.

    Returns:
        List (per batch element) of lists of (token_ids, score) tuples,
        sorted by score descending.
    """
    mod = _get_module()

    if log_probs.dim() == 2:
        log_probs = log_probs.unsqueeze(0)

    log_probs = log_probs.contiguous().float()
    if not log_probs.is_cuda:
        log_probs = log_probs.cuda()

    out_tokens, out_lengths, out_scores = mod.ctc_beam_search(
        log_probs, blank_id, beam_width, top_k
    )

    tokens_np = out_tokens.cpu().numpy()
    lengths_np = out_lengths.cpu().numpy()
    scores_np = out_scores.cpu().numpy()

    B, K = tokens_np.shape[0], tokens_np.shape[1]
    results = []
    for b in range(B):
        beams = []
        for k in range(K):
            length = int(lengths_np[b, k])
            score = float(scores_np[b, k])
            if score <= -1e29:
                continue
            beams.append((tokens_np[b, k, :length].tolist(), score))
        results.append(beams)
    return results


def ctc_beam_search_decode(
    log_probs: torch.Tensor,
    processor,
    blank_id: int = 0,
    beam_width: int = 10,
) -> str:
    """Beam search + decode to text string.

    Drop-in replacement for greedy argmax decoding in Wav2Vec2Runner.
    """
    results = ctc_beam_search(log_probs, blank_id, beam_width, top_k=1)
    if not results or not results[0]:
        return ""

    token_ids = results[0][0][0]
    if not token_ids:
        return ""

    tokenizer = processor.tokenizer
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return text.strip()
