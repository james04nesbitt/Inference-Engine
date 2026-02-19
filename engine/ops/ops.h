#pragma once

#include "engine/tensor/tensor.h"

namespace ie {
namespace ops {

// ============================================================================
// Math Kernels — YOUR SECOND MAJOR LEARNING OBJECTIVE
//
// Implement these operations one at a time. Start with add() and matmul(),
// then move to the normalization and activation functions.
//
// Performance journey:
//   1. Naive loops (get it correct)
//   2. Cache-friendly access patterns (understand memory hierarchy)
//   3. SIMD intrinsics (SSE/AVX on x86) for 4-8x speedup
//   4. Multi-threaded with std::jthread or a thread pool
//
// Each function should validate shapes and throw on mismatch.
// ============================================================================

// --- Element-wise Operations ---

// Element-wise addition: c = a + b (with broadcasting)
Tensor add(const Tensor& a, const Tensor& b);

// Element-wise multiplication: c = a * b (with broadcasting)
Tensor mul(const Tensor& a, const Tensor& b);

// --- Matrix Operations ---

// Matrix multiply: C[m,n] = A[m,k] @ B[k,n]
// This is THE most important kernel for transformer inference.
// Start naive, then optimize with tiling and SIMD.
Tensor matmul(const Tensor& a, const Tensor& b);

// --- Normalization ---

// RMS Normalization (used by Gemma instead of LayerNorm)
//   output = (x / sqrt(mean(x^2) + eps)) * weight
Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps = 1e-6f);

// --- Activation Functions ---

// SiLU (Sigmoid Linear Unit) — used in Gemma's FFN
//   silu(x) = x * sigmoid(x)
Tensor silu(const Tensor& x);

// GeLU (Gaussian Error Linear Unit)
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
Tensor gelu(const Tensor& x);

// Standard softmax along the last dimension
//   softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
Tensor softmax(const Tensor& x, int64_t dim = -1);

// --- Positional Encoding ---

// Rotary Positional Embedding (RoPE)
// Applies rotation to query/key tensors based on position.
// This is how Gemma encodes positional information.
//
// Parameters:
//   x: input tensor [batch, seq_len, n_heads, head_dim]
//   positions: position indices [batch, seq_len]
//   freq_base: base frequency (typically 10000.0)
Tensor rope(const Tensor& x, const Tensor& positions,
            float freq_base = 10000.0f);

// --- Embedding ---

// Table lookup: output[i] = table[indices[i]]
Tensor embedding(const Tensor& table, const Tensor& indices);

}  // namespace ops
}  // namespace ie
