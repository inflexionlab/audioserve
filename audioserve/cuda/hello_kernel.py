"""Hello-world CUDA kernel — verifies toolchain works end-to-end.

Implements a simple audio normalization: divide all samples by the max absolute value.
This exercises: global memory read/write, thread indexing, reduction (finding max), kernel launch.
"""

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel 1: Find max absolute value using parallel reduction
__global__ void find_max_abs_kernel(const float* input, float* block_maxes, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory
    sdata[tid] = (idx < n) ? fabsf(input[idx]) : 0.0f;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        block_maxes[blockIdx.x] = sdata[0];
    }
}

// Kernel 2: Normalize all elements by max value
__global__ void normalize_kernel(const float* input, float* output, float max_val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] / max_val;
    }
}

// Host function that orchestrates both kernels
torch::Tensor normalize_audio_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    int n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Step 1: Find max absolute value
    auto block_maxes = torch::zeros({blocks}, input.options());

    find_max_abs_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        input.data_ptr<float>(),
        block_maxes.data_ptr<float>(),
        n
    );

    // Reduce block maxes on CPU (few values, not worth another kernel)
    float max_val = block_maxes.max().item<float>();

    if (max_val < 1e-10f) {
        return torch::zeros_like(input);
    }

    // Step 2: Normalize
    auto output = torch::empty_like(input);

    normalize_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        max_val,
        n
    );

    return output;
}
"""

cpp_source = r"""
#include <torch/extension.h>

torch::Tensor normalize_audio_cuda(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("normalize_audio", &normalize_audio_cuda, "Normalize audio by max abs value (CUDA)");
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="hello_cuda",
            cpp_sources=[cpp_source],
            cuda_sources=[cuda_source],
            verbose=False,
        )
    return _module


def normalize_audio_cuda(audio_tensor: torch.Tensor) -> torch.Tensor:
    """Normalize audio to [-1, 1] range using a custom CUDA kernel.

    Args:
        audio_tensor: 1D float32 CUDA tensor of audio samples.

    Returns:
        Normalized tensor on CUDA.
    """
    mod = _get_module()
    return mod.normalize_audio(audio_tensor)
