#include "engine/compute/simd_kernels.h"

#include <cmath>
#include <stdexcept>

// ============================================================================
// Highway SIMD integration pattern:
//
// When you're ready to add SIMD, replace the scalar loops below with
// Highway operations. The typical pattern is:
//
//   #include "hwy/highway.h"
//   HWY_BEFORE_NAMESPACE();
//   namespace ie {
//   namespace compute {
//   namespace HWY_NAMESPACE {
//
//     namespace hn = hwy::HWY_NAMESPACE;
//
//     void SimdDotProductImpl(const float* a, const float* b, float* out,
//                             int64_t n) {
//       const hn::ScalableTag<float> d;
//       const int64_t lanes = hn::Lanes(d);
//       auto sum = hn::Zero(d);
//       int64_t i = 0;
//       for (; i + lanes <= n; i += lanes) {
//         auto va = hn::Load(d, a + i);
//         auto vb = hn::Load(d, b + i);
//         sum = hn::MulAdd(va, vb, sum);
//       }
//       *out = hn::ReduceSum(d, sum);
//       // Handle remainder with scalar loop
//       for (; i < n; ++i) *out += a[i] * b[i];
//     }
//
//   }  // namespace HWY_NAMESPACE
//   }  // namespace compute
//   }  // namespace ie
//   HWY_AFTER_NAMESPACE();
//
//   // Dispatch to the best available target at runtime:
//   #if HWY_ONCE
//   namespace ie { namespace compute {
//   HWY_EXPORT(SimdDotProductImpl);
//   float SimdDotProduct(const float* a, const float* b, int64_t n) {
//     float result = 0;
//     HWY_DYNAMIC_DISPATCH(SimdDotProductImpl)(a, b, &result, n);
//     return result;
//   }
//   }}
//   #endif
//
// ============================================================================

namespace ie {
namespace compute {

void SimdGemm(const float* a, const float* b, float* c,
              int64_t M, int64_t N, int64_t K) {
  // TODO: Implement GEMM
  //
  // Phase 1 — Naive (get correct output):
  //   for (i = 0..M)
  //     for (j = 0..N)
  //       c[i*N + j] = 0
  //       for (k = 0..K)
  //         c[i*N + j] += a[i*K + k] * b[k*N + j]
  //
  // Phase 2 — Cache-friendly (reorder to i,k,j):
  //   for (i = 0..M)
  //     for (k = 0..K)
  //       float a_ik = a[i*K + k]
  //       for (j = 0..N)
  //         c[i*N + j] += a_ik * b[k*N + j]    // b row accessed sequentially!
  //
  // Phase 3 — Tiled (L2-aware blocking):
  //   Choose tile sizes TM, TN, TK such that:
  //     TM * TK + TK * TN + TM * TN <= L2_CACHE_SIZE / sizeof(float)
  //   For L2 = 256KB: TM=TN=TK=64 works well (48KB per tile set)
  //
  //   for (i0 = 0..M step TM)
  //     for (j0 = 0..N step TN)
  //       for (k0 = 0..K step TK)
  //         // Micro-kernel: multiply TM x TK block of A by TK x TN block of B
  //         // Prefetch next tile of B here: __builtin_prefetch(&b[(k0+TK)*N+j0])
  //
  // Phase 4 — SIMD: vectorize the inner j-loop with Highway
  //
  throw std::runtime_error(
      "SimdGemm not implemented yet — start with the naive triple loop!");
}

void SimdGemv(const float* a, const float* x, float* y,
              int64_t M, int64_t K) {
  // TODO: Implement GEMV
  //   for (i = 0..M)
  //     y[i] = dot_product(a + i*K, x, K)
  //
  throw std::runtime_error("SimdGemv not implemented yet");
}

float SimdDotProduct(const float* a, const float* b, int64_t n) {
  // TODO: Implement dot product
  //   sum = 0
  //   for (i = 0..n) sum += a[i] * b[i]
  //   return sum
  //
  // Then: vectorize with Highway hn::MulAdd + hn::ReduceSum
  //
  throw std::runtime_error("SimdDotProduct not implemented yet");
}

void SimdSoftmax(const float* input, float* output, int64_t n) {
  // TODO: Implement numerically stable softmax
  //   1. max_val = max(input[0..n])
  //   2. sum = 0; for each i: output[i] = exp(input[i] - max_val); sum += output[i]
  //   3. for each i: output[i] /= sum
  //
  throw std::runtime_error("SimdSoftmax not implemented yet");
}

void SimdRmsNorm(const float* input, const float* weight, float* output,
                 int64_t n, float eps) {
  // TODO: Implement RMS normalization
  //   1. ss = sum(input[i]^2) / n
  //   2. scale = 1.0 / sqrt(ss + eps)
  //   3. output[i] = input[i] * scale * weight[i]
  //
  throw std::runtime_error("SimdRmsNorm not implemented yet");
}

void SimdSilu(const float* input, float* output, int64_t n) {
  // TODO: Implement SiLU activation
  //   output[i] = input[i] / (1.0 + exp(-input[i]))
  //
  throw std::runtime_error("SimdSilu not implemented yet");
}

void SimdMul(const float* a, const float* b, float* c, int64_t n) {
  // TODO: Element-wise multiply
  throw std::runtime_error("SimdMul not implemented yet");
}

void SimdAdd(const float* a, const float* b, float* c, int64_t n) {
  // TODO: Element-wise add
  throw std::runtime_error("SimdAdd not implemented yet");
}

}  // namespace compute
}  // namespace ie
