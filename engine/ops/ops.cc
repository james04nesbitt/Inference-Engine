#include "engine/ops/ops.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ie {
namespace ops {

// ============================================================================
// Element-wise Operations
// ============================================================================

Tensor add(const Tensor &a, const Tensor &b) {
  if (!a.shape_equals(b)) {
    throw std::runtime_error("add: shapes are not equal");
  }
  // Upcast to FP32 for computation, then convert back.
  DType out_dtype = a.dtype();
  Tensor a_f = a.to(DType::kFloat32).contiguous();
  Tensor b_f = b.to(DType::kFloat32).contiguous();
  Tensor c = Tensor::zeros(a_f.shape(), DType::kFloat32);

  const float *a_data = a_f.data<float>();
  const float *b_data = b_f.data<float>();
  float *c_data = c.data<float>();

  for (int64_t i = 0; i < a_f.numel(); ++i) {
    c_data[i] = a_data[i] + b_data[i];
  }
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

  const float *a_data = a_f.data<float>();
  const float *b_data = b_f.data<float>();
  float *c_data = c.data<float>();

  for (int64_t i = 0; i < a_f.numel(); ++i) {
    c_data[i] = a_data[i] * b_data[i];
  }
  return c.to(out_dtype);
}

// ============================================================================
// Matrix Multiply — Cache-friendly i,k,j loop order
// ============================================================================
//
// Standard matmul: C[M,N] = A[M,K] @ B[K,N]
//
// The naive i,j,k order accesses B with stride N (column-wise), which is
// cache-hostile. Reordering to i,k,j makes the inner loop stride-1 on both
// B and C, giving ~3-5x speedup by maximizing L1/L2 cache line utilization.
//
// Next optimization steps (not yet implemented):
//   - Tiling: break into L2-sized blocks (e.g. 64×64 for 256KB L2)
//   - SIMD: vectorize inner loop with Highway (hn::MulAdd)
//   - Threading: partition M dimension across thread pool

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

  // Always compute in FP32 for numerical accuracy, convert output back.
  DType out_dtype = a.dtype();
  Tensor a_f = a.to(DType::kFloat32).contiguous();
  Tensor b_f = b.to(DType::kFloat32).contiguous();
  Tensor c = Tensor::zeros({M, N}, DType::kFloat32);

  const float *A = a_f.data<float>();
  const float *B = b_f.data<float>();
  float *C = c.data<float>();

  // Cache-friendly i,k,j loop order:
  //   Inner loop touches C[i, 0..N-1] and B[k, 0..N-1] — both stride-1.
  //   A[i,k] is invariant in the inner loop → hoisted into a register.
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t k = 0; k < K; ++k) {
      const float a_ik = A[i * K + k];
      for (int64_t j = 0; j < N; ++j) {
        C[i * N + j] += a_ik * B[k * N + j];
      }
    }
  }
  return c.to(out_dtype);
}

// ============================================================================
// RMS Normalization
// ============================================================================
//
// Used by Gemma instead of LayerNorm. Simpler (no mean subtraction) and
// empirically just as effective.
//
// For each "row" (all dims except the last):
//   rms = sqrt(mean(x^2) + eps)
//   output = (x / rms) * weight
//
// weight has shape [last_dim] — one scale per feature.

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

  // Number of "rows" = total elements / last_dim
  const int64_t n_rows = x_f.numel() / last_dim;

  for (int64_t row = 0; row < n_rows; ++row) {
    const float *x_row = x_data + row * last_dim;
    float *o_row = o_data + row * last_dim;

    // 1. Compute mean of squares
    float sum_sq = 0.0f;
    for (int64_t j = 0; j < last_dim; ++j) {
      sum_sq += x_row[j] * x_row[j];
    }
    float rms = std::sqrt(sum_sq / static_cast<float>(last_dim) + eps);

    // 2. Normalize and scale by weight
    float inv_rms = 1.0f / rms;
    for (int64_t j = 0; j < last_dim; ++j) {
      o_row[j] = x_row[j] * inv_rms * w_data[j];
    }
  }
  return out.to(out_dtype);
}

// ============================================================================
// SiLU Activation (Sigmoid Linear Unit)
// ============================================================================
//
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// Used in Gemma's SwiGLU FFN: FFN(x) = silu(W_gate @ x) * (W_up @ x)

Tensor silu(const Tensor &x) {
  DType out_dtype = x.dtype();
  Tensor x_f = x.to(DType::kFloat32).contiguous();
  Tensor out = Tensor::zeros(x_f.shape(), DType::kFloat32);

  const float *x_data = x_f.data<float>();
  float *o_data = out.data<float>();

  for (int64_t i = 0; i < x_f.numel(); ++i) {
    float val = x_data[i];
    float sigmoid = 1.0f / (1.0f + std::exp(-val));
    o_data[i] = val * sigmoid;
  }
  return out.to(out_dtype);
}

// ============================================================================
// GeLU Activation (Gaussian Error Linear Unit)
// ============================================================================
//
// Approximation (tanh form, same as PyTorch):
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

Tensor gelu(const Tensor &x) {
  DType out_dtype = x.dtype();
  Tensor x_f = x.to(DType::kFloat32).contiguous();
  Tensor out = Tensor::zeros(x_f.shape(), DType::kFloat32);

  const float *x_data = x_f.data<float>();
  float *o_data = out.data<float>();

  constexpr float kSqrt2OverPi = 0.7978845608028654f; // sqrt(2/pi)
  constexpr float kCoeff = 0.044715f;

  for (int64_t i = 0; i < x_f.numel(); ++i) {
    float val = x_data[i];
    float inner = kSqrt2OverPi * (val + kCoeff * val * val * val);
    o_data[i] = 0.5f * val * (1.0f + std::tanh(inner));
  }
  return out.to(out_dtype);
}

// ============================================================================
// Softmax — Numerically Stable
// ============================================================================
//
// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
//
// Subtracting max prevents overflow in exp(). This is standard practice
// and mathematically equivalent to the naive formula.
//
// Operates along the specified dimension (default: last dim, i.e. -1).
// For transformer inference, this is always the sequence dimension in
// attention scores.

Tensor softmax(const Tensor &x, int64_t dim) {
  if (x.ndim() < 1) {
    throw std::runtime_error("softmax: input must have at least 1 dimension");
  }

  // Resolve negative dim
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

  // Compute the number of independent softmax operations.
  // For a tensor of shape [A, B, C] with dim=1, we do A*C softmax ops,
  // each over B elements.
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
      // Base offset into the contiguous buffer.
      // Elements along `dim` are spaced `inner_size` apart.
      int64_t base = outer * dim_size * inner_size + inner;

      // 1. Find max for numerical stability
      float max_val = x_data[base];
      for (int64_t d = 1; d < dim_size; ++d) {
        float v = x_data[base + d * inner_size];
        if (v > max_val)
          max_val = v;
      }

      // 2. Exponentiate and accumulate sum
      float sum = 0.0f;
      for (int64_t d = 0; d < dim_size; ++d) {
        float e = std::exp(x_data[base + d * inner_size] - max_val);
        o_data[base + d * inner_size] = e;
        sum += e;
      }

      // 3. Normalize
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
//
// Encodes position information by rotating pairs of dimensions in Q/K vectors.
//
// Input x:         [batch, seq_len, n_heads, head_dim]
// Input positions: [batch, seq_len]
//
// For dimension pair (2i, 2i+1) at position pos:
//   theta = pos * freq_base^(-2i / head_dim)
//   x'[..., 2i]   = x[..., 2i]   * cos(theta) - x[..., 2i+1] * sin(theta)
//   x'[..., 2i+1] = x[..., 2i]   * sin(theta) + x[..., 2i+1] * cos(theta)
//
// This is a 2D rotation matrix applied to each pair. The frequencies decrease
// geometrically, giving different dimensions sensitivity to different scales.

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

  // Precompute inverse frequencies: freq_base^(-2i/head_dim) for i in [0,
  // half_dim)
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
//
// table:   [vocab_size, embed_dim]
// indices: 1D tensor of token IDs
// output:  [num_tokens, embed_dim]
//
// Each output row is a copy of table[index]. This is the very first operation
// in the transformer forward pass: token IDs → dense vectors.

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

  // Upcast table to FP32 for computation. Indices are read as float and
  // cast to int64, so they work regardless of storage dtype.
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

    // Copy entire row: memcpy is faster than element-wise for contiguous data
    const float *src = t_data + idx * embed_dim;
    float *dst = o_data + t * embed_dim;
    std::memcpy(dst, src, static_cast<size_t>(embed_dim) * sizeof(float));
  }
  return out.to(out_dtype);
}

} // namespace ops
} // namespace ie
