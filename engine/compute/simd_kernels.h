#pragma once

#include <cstddef>
#include <cstdint>

#include "engine/tensor/tensor.h"

namespace ie {
namespace compute {

// ============================================================================
// SIMD-Accelerated Kernels — Google Highway
//
// Highway provides portable SIMD intrinsics that compile to the best
// available instruction set on the target CPU (AVX2, AVX-512, NEON, etc.).
//
// Start by implementing the scalar (naive) version of each kernel, then
// replace the inner loop with Highway ops for a 4-16x speedup.
//
// Key Highway concepts:
//   - HWY_NAMESPACE: Highway puts code in a per-target namespace
//   - hn::ScalableTag<float> d: declares a "descriptor" for float vectors
//   - hn::Lanes(d): how many floats fit in one SIMD register
//   - hn::Load(d, ptr): load a SIMD register from memory
//   - hn::MulAdd(a, b, c): fused multiply-add: a*b + c
//   - hn::Store(result, d, out_ptr): store a SIMD register to memory
//
// Reference: https://github.com/google/highway
// ============================================================================

// --- GEMM (General Matrix Multiply) ---
// C[M, N] = A[M, K] * B[K, N]
//
// This is THE critical kernel. Transformer inference is ~90% matmul.
//
// Optimization roadmap:
//   1. Naive i,j,k triple loop (correct but slow)
//   2. Reorder to i,k,j (cache-friendly — B accessed sequentially)
//   3. Tiling: break into blocks that fit in L2 cache (~256KB)
//      - Typical tile: 64x64 or 128x128 depending on cache size
//      - Each tile of A stays in L2 while you sweep across tiles of B
//   4. Prefetching: __builtin_prefetch or HWY prefetch for next tile
//   5. Highway SIMD: vectorize the inner loop with hn::MulAdd
//   6. Multi-thread: partition M dimension across threads
//
// Performance target: P99 latency reduction vs naive FP16 baseline
//
void SimdGemm(const float* a, const float* b, float* c,
              int64_t M, int64_t N, int64_t K);

// --- GEMV (General Matrix-Vector Multiply) ---
// y[M] = A[M, K] * x[K]
//
// Used during single-token decode (sequence length = 1).
// Simpler than GEMM — only needs vectorization along K dimension.
//
void SimdGemv(const float* a, const float* x, float* y,
              int64_t M, int64_t K);

// --- Dot Product ---
// Innermost kernel used by both GEMM and GEMV.
// Vectorize with Highway hn::MulAdd + hn::ReduceSum.
//
float SimdDotProduct(const float* a, const float* b, int64_t n);

// --- Softmax ---
// Numerically stable softmax:
//   1. Find max (hn::MaxOfLanes)
//   2. Subtract max and exponentiate (hn::Exp or table lookup)
//   3. Sum (hn::ReduceSum)
//   4. Divide
//
void SimdSoftmax(const float* input, float* output, int64_t n);

// --- RMS Norm ---
// Compute x / sqrt(mean(x^2) + eps) * weight
//
void SimdRmsNorm(const float* input, const float* weight, float* output,
                 int64_t n, float eps);

// --- SiLU Activation ---
// x * sigmoid(x) = x / (1 + exp(-x))
//
void SimdSilu(const float* input, float* output, int64_t n);

// --- Element-wise Multiply ---
// c[i] = a[i] * b[i]
//
void SimdMul(const float* a, const float* b, float* c, int64_t n);

// --- Element-wise Add ---
// c[i] = a[i] + b[i]
//
void SimdAdd(const float* a, const float* b, float* c, int64_t n);

}  // namespace compute
}  // namespace ie
