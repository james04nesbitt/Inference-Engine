#include "engine/compute/simd_kernels.h"

#include <cmath>

// On Windows/MSVC, Highway's BCR BUILD sets HWY_SHARED_DEFINE which adds
// __declspec(dllimport) to function declarations. Since Bazel links statically,
// we must override this before including any Highway headers.
#ifdef HWY_SHARED_DEFINE
#undef HWY_SHARED_DEFINE
#endif
#ifndef HWY_STATIC_DEFINE
#define HWY_STATIC_DEFINE
#endif

// Highway headers — must be included in this specific order.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "engine/compute/simd_kernels.cc"
#include "hwy/foreach_target.h" // Must come before highway.h
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();

namespace ie {
namespace compute {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// --- Dot Product (Highway SIMD) ---
float SimdDotProductImpl(const float *a, const float *b, int64_t n) {
  const hn::ScalableTag<float> d;
  const int64_t lanes = static_cast<int64_t>(hn::Lanes(d));
  auto sum = hn::Zero(d);

  int64_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    auto va = hn::Load(d, a + i);
    auto vb = hn::Load(d, b + i);
    sum = hn::MulAdd(va, vb, sum);
  }
  float result = hn::ReduceSum(d, sum);

  // Scalar remainder.
  for (; i < n; ++i) {
    result += a[i] * b[i];
  }
  return result;
}

// --- GEMM: Cache-friendly i,k,j with SIMD inner loop ---
void SimdGemmImpl(const float *a, const float *b, float *c, int64_t M,
                  int64_t N, int64_t K) {
  const hn::ScalableTag<float> d;
  const int64_t lanes = static_cast<int64_t>(hn::Lanes(d));

  // Zero the output matrix.
  for (int64_t i = 0; i < M * N; ++i) {
    c[i] = 0.0f;
  }

  // Cache-friendly i,k,j order with SIMD inner loop.
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t k = 0; k < K; ++k) {
      const auto a_ik = hn::Set(d, a[i * K + k]);
      const float *b_row = b + k * N;
      float *c_row = c + i * N;

      int64_t j = 0;
      for (; j + lanes <= N; j += lanes) {
        auto cv = hn::Load(d, c_row + j);
        auto bv = hn::Load(d, b_row + j);
        cv = hn::MulAdd(a_ik, bv, cv);
        hn::Store(cv, d, c_row + j);
      }
      // Scalar remainder.
      float a_ik_s = a[i * K + k];
      for (; j < N; ++j) {
        c_row[j] += a_ik_s * b_row[j];
      }
    }
  }
}

// --- GEMV: Matrix-vector multiply (SIMD via dot product) ---
void SimdGemvImpl(const float *a, const float *x, float *y, int64_t M,
                  int64_t K) {
  for (int64_t i = 0; i < M; ++i) {
    y[i] = SimdDotProductImpl(a + i * K, x, K);
  }
}

// --- Softmax: Numerically stable (SIMD) ---
void SimdSoftmaxImpl(const float *input, float *output, int64_t n) {
  const hn::ScalableTag<float> d;
  const int64_t lanes = static_cast<int64_t>(hn::Lanes(d));

  // Step 1: Find max using SIMD.
  auto max_v = hn::Set(d, -INFINITY);
  int64_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    auto v = hn::Load(d, input + i);
    max_v = hn::Max(max_v, v);
  }
  float max_val = hn::ReduceMax(d, max_v);
  for (; i < n; ++i) {
    if (input[i] > max_val)
      max_val = input[i];
  }

  // Step 2: Exponentiate and sum (scalar — exp is transcendental).
  float sum = 0.0f;
  for (int64_t j = 0; j < n; ++j) {
    output[j] = std::exp(input[j] - max_val);
    sum += output[j];
  }

  // Step 3: Normalize with SIMD.
  const auto inv_sum = hn::Set(d, 1.0f / sum);
  i = 0;
  for (; i + lanes <= n; i += lanes) {
    auto v = hn::Load(d, output + i);
    v = hn::Mul(v, inv_sum);
    hn::Store(v, d, output + i);
  }
  float inv_sum_s = 1.0f / sum;
  for (; i < n; ++i) {
    output[i] *= inv_sum_s;
  }
}

// --- RMS Norm (SIMD) ---
void SimdRmsNormImpl(const float *input, const float *weight, float *output,
                     int64_t n, float eps) {
  const hn::ScalableTag<float> d;
  const int64_t lanes = static_cast<int64_t>(hn::Lanes(d));

  // Step 1: Compute sum of squares with SIMD.
  auto ss_v = hn::Zero(d);
  int64_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    auto v = hn::Load(d, input + i);
    ss_v = hn::MulAdd(v, v, ss_v);
  }
  float ss = hn::ReduceSum(d, ss_v);
  for (; i < n; ++i) {
    ss += input[i] * input[i];
  }
  ss /= static_cast<float>(n);

  // Step 2: Compute scale factor.
  float scale = 1.0f / std::sqrt(ss + eps);
  const auto scale_v = hn::Set(d, scale);

  // Step 3: Normalize and apply weight with SIMD.
  i = 0;
  for (; i + lanes <= n; i += lanes) {
    auto iv = hn::Load(d, input + i);
    auto wv = hn::Load(d, weight + i);
    auto result = hn::Mul(hn::Mul(iv, scale_v), wv);
    hn::Store(result, d, output + i);
  }
  for (; i < n; ++i) {
    output[i] = input[i] * scale * weight[i];
  }
}

// --- SiLU Activation (scalar — transcendental function) ---
void SimdSiluImpl(const float *input, float *output, int64_t n) {
  // SiLU = x / (1 + exp(-x)). The exp is transcendental so we stay scalar.
  // Highway does have ApproximateExp, but std::exp is more accurate.
  for (int64_t i = 0; i < n; ++i) {
    output[i] = input[i] / (1.0f + std::exp(-input[i]));
  }
}

// --- Element-wise Multiply (SIMD) ---
void SimdMulImpl(const float *a, const float *b, float *c, int64_t n) {
  const hn::ScalableTag<float> d;
  const int64_t lanes = static_cast<int64_t>(hn::Lanes(d));

  int64_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    auto va = hn::Load(d, a + i);
    auto vb = hn::Load(d, b + i);
    hn::Store(hn::Mul(va, vb), d, c + i);
  }
  for (; i < n; ++i) {
    c[i] = a[i] * b[i];
  }
}

// --- Element-wise Add (SIMD) ---
void SimdAddImpl(const float *a, const float *b, float *c, int64_t n) {
  const hn::ScalableTag<float> d;
  const int64_t lanes = static_cast<int64_t>(hn::Lanes(d));

  int64_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    auto va = hn::Load(d, a + i);
    auto vb = hn::Load(d, b + i);
    hn::Store(hn::Add(va, vb), d, c + i);
  }
  for (; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
}

} // namespace HWY_NAMESPACE
} // namespace compute
} // namespace ie

HWY_AFTER_NAMESPACE();

// ============================================================================
// Dynamic dispatch — Highway selects the best target at runtime.
// This block runs only once (not per-target like the code above).
// ============================================================================
#if HWY_ONCE

namespace ie {
namespace compute {

HWY_EXPORT(SimdDotProductImpl);
float SimdDotProduct(const float *a, const float *b, int64_t n) {
  return HWY_DYNAMIC_DISPATCH(SimdDotProductImpl)(a, b, n);
}

HWY_EXPORT(SimdGemmImpl);
void SimdGemm(const float *a, const float *b, float *c, int64_t M, int64_t N,
              int64_t K) {
  HWY_DYNAMIC_DISPATCH(SimdGemmImpl)(a, b, c, M, N, K);
}

HWY_EXPORT(SimdGemvImpl);
void SimdGemv(const float *a, const float *x, float *y, int64_t M, int64_t K) {
  HWY_DYNAMIC_DISPATCH(SimdGemvImpl)(a, x, y, M, K);
}

HWY_EXPORT(SimdSoftmaxImpl);
void SimdSoftmax(const float *input, float *output, int64_t n) {
  HWY_DYNAMIC_DISPATCH(SimdSoftmaxImpl)(input, output, n);
}

HWY_EXPORT(SimdRmsNormImpl);
void SimdRmsNorm(const float *input, const float *weight, float *output,
                 int64_t n, float eps) {
  HWY_DYNAMIC_DISPATCH(SimdRmsNormImpl)(input, weight, output, n, eps);
}

HWY_EXPORT(SimdSiluImpl);
void SimdSilu(const float *input, float *output, int64_t n) {
  HWY_DYNAMIC_DISPATCH(SimdSiluImpl)(input, output, n);
}

HWY_EXPORT(SimdMulImpl);
void SimdMul(const float *a, const float *b, float *c, int64_t n) {
  HWY_DYNAMIC_DISPATCH(SimdMulImpl)(a, b, c, n);
}

HWY_EXPORT(SimdAddImpl);
void SimdAdd(const float *a, const float *b, float *c, int64_t n) {
  HWY_DYNAMIC_DISPATCH(SimdAddImpl)(a, b, c, n);
}

} // namespace compute
} // namespace ie

#endif // HWY_ONCE
