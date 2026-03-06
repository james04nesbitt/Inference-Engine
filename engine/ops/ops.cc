#include "engine/ops/ops.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "engine/compute/simd_kernels.h"
#include "engine/compute/thread_pool.h"

namespace ie {
namespace ops {

// ============================================================================
// Global thread pool for parallel compute kernels.
// Initialized lazily on first use, persists for process lifetime.
// ============================================================================
static compute::ThreadPool &GetThreadPool() {
  static compute::ThreadPool pool(0); // 0 = hardware_concurrency
  return pool;
}

// Minimum M dimension to parallelize matmul across threads.
constexpr int64_t kParallelMatmulThreshold = 64;

// ============================================================================
// Element-wise Operations — SIMD accelerated
// ============================================================================

Tensor add(const Tensor &a, const Tensor &b) {
  if (!a.shape_equals(b)) {
    throw std::runtime_error("add: shapes are not equal");
  }
  DType out_dtype = a.dtype();
  Tensor a_f = a.to(DType::kFloat32).contiguous();
  Tensor b_f = b.to(DType::kFloat32).contiguous();
  Tensor c = Tensor::zeros(a_f.shape(), DType::kFloat32);

  compute::SimdAdd(a_f.data<float>(), b_f.data<float>(), c.data<float>(),
                   a_f.numel());
  return c.to(out_dtype);
}

Tensor mul(const Tensor &a, const Tensor &b) {
  if (!a.shape_equals(b)) {
    throw std::runtime_error("mul: shapes are not equal");
  }
  DType out_dtype = a.dtype();
  Tensor a_f = a.to(DType::kFloat32).contiguous();
  Tensor b_f = b.to(DType::kFloat32).contiguous();
  Tensor c = Tensor::zeros(a_f.shape(), DType::kFloat32);

  compute::SimdMul(a_f.data<float>(), b_f.data<float>(), c.data<float>(),
                   a_f.numel());
  return c.to(out_dtype);
}

// ============================================================================
// Matrix Multiply — SIMD + Multi-threaded
// ============================================================================
//
// For small M: call SimdGemm directly (single-threaded SIMD).
// For large M: partition rows across thread pool, each thread runs SimdGemm
//   on its slice of rows.
//
// The inner loop uses Highway-vectorized dot products with cache-friendly
// tiling and prefetching (as implemented in simd_kernels.cc).

Tensor matmul(const Tensor &a, const Tensor &b) {
  if (a.ndim() != 2 || b.ndim() != 2) {
    throw std::runtime_error("matmul: both inputs must be 2D");
  }
  if (a.size(1) != b.size(0)) {
    throw std::runtime_error("matmul: inner dimensions must match — "
                             "A is [M," +
                             std::to_string(a.size(1)) + "], B is [" +
                             std::to_string(b.size(0)) + ",N]");
  }

  const int64_t M = a.size(0);
  const int64_t K = a.size(1);
  const int64_t N = b.size(1);

  DType out_dtype = a.dtype();
  Tensor a_f = a.to(DType::kFloat32).contiguous();
  Tensor b_f = b.to(DType::kFloat32).contiguous();
  Tensor c = Tensor::zeros({M, N}, DType::kFloat32);

  const float *A = a_f.data<float>();
  const float *B = b_f.data<float>();
  float *C = c.data<float>();

  if (M >= kParallelMatmulThreshold) {
    // Multi-threaded: partition M rows across thread pool.
    auto &pool = GetThreadPool();
    int64_t n_threads = static_cast<int64_t>(pool.NumThreads());
    if (n_threads < 1)
      n_threads = 1;

    pool.ParallelFor(n_threads, [&](int64_t tid) {
      int64_t rows_per_thread = (M + n_threads - 1) / n_threads;
      int64_t start = tid * rows_per_thread;
      int64_t end = std::min(start + rows_per_thread, M);
      if (start >= M)
        return;

      int64_t chunk_m = end - start;
      // Each thread runs SIMD GEMM on its row slice.
      compute::SimdGemm(A + start * K, B, C + start * N, chunk_m, N, K);
    });
  } else {
    // Single-threaded SIMD GEMM for small matrices.
    compute::SimdGemm(A, B, C, M, N, K);
  }

  return c.to(out_dtype);
}

// ============================================================================
// RMS Normalization — SIMD accelerated
// ============================================================================

Tensor rms_norm(const Tensor &x, const Tensor &weight, float eps) {
  if (x.ndim() < 1) {
    throw std::runtime_error("rms_norm: input must have at least 1 dimension");
  }
  const int64_t last_dim = x.size(-1);
  if (!weight.shape_equals({last_dim})) {
    throw std::runtime_error("rms_norm: weight shape must match last dim of x");
  }

  DType out_dtype = x.dtype();
  Tensor x_f = x.to(DType::kFloat32).contiguous();
  Tensor w_f = weight.to(DType::kFloat32).contiguous();
  Tensor out = Tensor::zeros(x_f.shape(), DType::kFloat32);

  const float *x_data = x_f.data<float>();
  const float *w_data = w_f.data<float>();
  float *o_data = out.data<float>();

  const int64_t n_rows = x_f.numel() / last_dim;

  for (int64_t row = 0; row < n_rows; ++row) {
    compute::SimdRmsNorm(x_data + row * last_dim, w_data,
                         o_data + row * last_dim, last_dim, eps);
  }
  return out.to(out_dtype);
}

// ============================================================================
// SiLU Activation — SIMD accelerated
// ============================================================================

Tensor silu(const Tensor &x) {
  DType out_dtype = x.dtype();
  Tensor x_f = x.to(DType::kFloat32).contiguous();
  Tensor out = Tensor::zeros(x_f.shape(), DType::kFloat32);

  compute::SimdSilu(x_f.data<float>(), out.data<float>(), x_f.numel());
  return out.to(out_dtype);
}

// ============================================================================
// GeLU Activation — scalar (no SIMD kernel for this)
// ============================================================================

Tensor gelu(const Tensor &x) {
  DType out_dtype = x.dtype();
  Tensor x_f = x.to(DType::kFloat32).contiguous();
  Tensor out = Tensor::zeros(x_f.shape(), DType::kFloat32);

  const float *x_data = x_f.data<float>();
  float *o_data = out.data<float>();

  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  constexpr float kCoeff = 0.044715f;

  for (int64_t i = 0; i < x_f.numel(); ++i) {
    float val = x_data[i];
    float inner = kSqrt2OverPi * (val + kCoeff * val * val * val);
    o_data[i] = 0.5f * val * (1.0f + std::tanh(inner));
  }
  return out.to(out_dtype);
}

// ============================================================================
// Softmax — SIMD accelerated for last-dim contiguous rows
// ============================================================================

Tensor softmax(const Tensor &x, int64_t dim) {
  if (x.ndim() < 1) {
    throw std::runtime_error("softmax: input must have at least 1 dimension");
  }

  if (dim < 0) {
    dim += x.ndim();
  }
  if (dim < 0 || dim >= x.ndim()) {
    throw std::runtime_error("softmax: dim out of range");
  }

  DType out_dtype = x.dtype();
  Tensor x_f = x.to(DType::kFloat32).contiguous();
  Tensor out = Tensor::zeros(x_f.shape(), DType::kFloat32);

  const auto &shape = x_f.shape();
  const int64_t dim_size = shape[dim];

  // Fast path: last-dim softmax on contiguous data → use SIMD kernel.
  if (dim == x_f.ndim() - 1) {
    const float *x_data = x_f.data<float>();
    float *o_data = out.data<float>();
    int64_t n_rows = x_f.numel() / dim_size;

    for (int64_t row = 0; row < n_rows; ++row) {
      compute::SimdSoftmax(x_data + row * dim_size, o_data + row * dim_size,
                           dim_size);
    }
    return out.to(out_dtype);
  }

  // General path: arbitrary dim (rare in transformer inference).
  int64_t outer_size = 1;
  for (int64_t d = 0; d < dim; ++d) {
    outer_size *= shape[d];
  }
  int64_t inner_size = 1;
  for (int64_t d = dim + 1; d < x_f.ndim(); ++d) {
    inner_size *= shape[d];
  }

  const float *x_data = x_f.data<float>();
  float *o_data = out.data<float>();

  for (int64_t outer = 0; outer < outer_size; ++outer) {
    for (int64_t inner = 0; inner < inner_size; ++inner) {
      int64_t base = outer * dim_size * inner_size + inner;

      float max_val = x_data[base];
      for (int64_t d = 1; d < dim_size; ++d) {
        float v = x_data[base + d * inner_size];
        if (v > max_val)
          max_val = v;
      }

      float sum = 0.0f;
      for (int64_t d = 0; d < dim_size; ++d) {
        float e = std::exp(x_data[base + d * inner_size] - max_val);
        o_data[base + d * inner_size] = e;
        sum += e;
      }

      float inv_sum = 1.0f / sum;
      for (int64_t d = 0; d < dim_size; ++d) {
        o_data[base + d * inner_size] *= inv_sum;
      }
    }
  }
  return out.to(out_dtype);
}

// ============================================================================
// Rotary Positional Embedding (RoPE)
// ============================================================================

Tensor rope(const Tensor &x, const Tensor &positions, float freq_base) {
  if (x.ndim() != 4) {
    throw std::runtime_error(
        "rope: input must be 4D [batch, seq_len, n_heads, head_dim]");
  }
  if (positions.ndim() != 2) {
    throw std::runtime_error("rope: positions must be 2D [batch, seq_len]");
  }
  if (x.size(0) != positions.size(0) || x.size(1) != positions.size(1)) {
    throw std::runtime_error(
        "rope: batch and seq_len must match between x and positions");
  }
  if (x.size(3) % 2 != 0) {
    throw std::runtime_error("rope: head_dim must be even");
  }

  const int64_t batch = x.size(0);
  const int64_t seq_len = x.size(1);
  const int64_t n_heads = x.size(2);
  const int64_t head_dim = x.size(3);
  const int64_t half_dim = head_dim / 2;

  DType out_dtype = x.dtype();
  Tensor x_f = x.to(DType::kFloat32).contiguous();
  Tensor pos_f = positions.to(DType::kFloat32).contiguous();
  Tensor out = Tensor::zeros(x_f.shape(), DType::kFloat32);

  const float *x_data = x_f.data<float>();
  const float *pos_data = pos_f.data<float>();
  float *o_data = out.data<float>();

  std::vector<float> inv_freq(half_dim);
  for (int64_t i = 0; i < half_dim; ++i) {
    inv_freq[i] = 1.0f / std::pow(freq_base, static_cast<float>(2 * i) /
                                                 static_cast<float>(head_dim));
  }

  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq_len; ++s) {
      float pos = pos_data[b * seq_len + s];

      for (int64_t h = 0; h < n_heads; ++h) {
        int64_t offset = ((b * seq_len + s) * n_heads + h) * head_dim;

        for (int64_t i = 0; i < half_dim; ++i) {
          float theta = pos * inv_freq[i];
          float cos_theta = std::cos(theta);
          float sin_theta = std::sin(theta);

          float x0 = x_data[offset + 2 * i];
          float x1 = x_data[offset + 2 * i + 1];

          o_data[offset + 2 * i] = x0 * cos_theta - x1 * sin_theta;
          o_data[offset + 2 * i + 1] = x0 * sin_theta + x1 * cos_theta;
        }
      }
    }
  }
  return out.to(out_dtype);
}

// ============================================================================
// Embedding Table Lookup
// ============================================================================

Tensor embedding(const Tensor &table, const Tensor &indices) {
  if (table.ndim() != 2) {
    throw std::runtime_error("embedding: table must be 2D [vocab, embed_dim]");
  }
  if (indices.ndim() != 1) {
    throw std::runtime_error("embedding: indices must be 1D");
  }

  const int64_t vocab_size = table.size(0);
  const int64_t embed_dim = table.size(1);
  const int64_t n_tokens = indices.size(0);

  DType out_dtype = table.dtype();
  Tensor table_f = table.to(DType::kFloat32).contiguous();
  Tensor idx_f = indices.to(DType::kFloat32).contiguous();
  Tensor out = Tensor::zeros({n_tokens, embed_dim}, DType::kFloat32);

  const float *t_data = table_f.data<float>();
  const float *i_data = idx_f.data<float>();
  float *o_data = out.data<float>();

  for (int64_t t = 0; t < n_tokens; ++t) {
    int64_t idx = static_cast<int64_t>(i_data[t]);
    if (idx < 0 || idx >= vocab_size) {
      throw std::runtime_error("embedding: index " + std::to_string(idx) +
                               " out of range [0, " +
                               std::to_string(vocab_size) + ")");
    }

    const float *src = t_data + idx * embed_dim;
    float *dst = o_data + t * embed_dim;
    std::memcpy(dst, src, static_cast<size_t>(embed_dim) * sizeof(float));
  }
  return out.to(out_dtype);
}

} // namespace ops
} // namespace ie
