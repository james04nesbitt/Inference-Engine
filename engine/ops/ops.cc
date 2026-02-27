#include "engine/ops/ops.h"

#include <cmath>
#include <stdexcept>

namespace ie {
namespace ops {

Tensor add(const Tensor &a, const Tensor &b) {
  if (!a.shape_equals(b)) {
    throw std::runtime_error("Shapes are not equal");
  }
  Tensor c = Tensor::zeros(a.shape(), a.dtype());
  a.contiguous();
  b.contiguous();

  const float *a_data = a.data<float>();
  const float *b_data = b.data<float>();
  float *c_data = c.data<float>();

  for (int64_t i = 0; i < a.numel(); ++i) {
    c_data[i] = a_data[i] + b_data[i];
  }
  return c;
}

Tensor mul(const Tensor &a, const Tensor &b) {
  if (!a.shape_equals(b)) {
    throw std::runtime_error("Shapes are not equal");
  }
  Tensor c = Tensor::zeros(a.shape(), a.dtype());
  a.contiguous();
  b.contiguous();

  const float *a_data = a.data<float>();
  const float *b_data = b.data<float>();
  float *c_data = c.data<float>();

  for (int64_t i = 0; i < a.numel(); ++i) {
    c_data[i] = a_data[i] * b_data[i];
  }
  return c;
}

Tensor matmul(const Tensor &a, const Tensor &b) {
  // TODO: Matrix multiply - START HERE, this is the most important kernel!
  //
  // Naive version (triple loop):
  //   for i in range(M):
  //     for j in range(N):
  //       for k in range(K):
  //         C[i][j] += A[i][k] * B[k][j]
  //
  // Then optimize with:
  //   1. Loop reordering (i,k,j is more cache-friendly than i,j,k)
  //   2. Tiling (break into cache-sized blocks)
  //   3. SIMD (process 4-8 floats at once with AVX)
  throw std::runtime_error("ops::matmul not implemented yet");
}

Tensor rms_norm(const Tensor &x, const Tensor &weight, float eps) {
  // TODO: RMS normalization
  // 1. Compute mean of squares along last dim
  // 2. Divide by sqrt(mean + eps)
  // 3. Multiply by weight
  throw std::runtime_error("ops::rms_norm not implemented yet");
}

Tensor silu(const Tensor &x) {
  // TODO: SiLU activation: x * sigmoid(x)
  // sigmoid(x) = 1 / (1 + exp(-x))
  throw std::runtime_error("ops::silu not implemented yet");
}

Tensor gelu(const Tensor &x) {
  // TODO: GeLU activation (tanh approximation)
  throw std::runtime_error("ops::gelu not implemented yet");
}

Tensor softmax(const Tensor &x, int64_t dim) {
  // TODO: Numerically stable softmax
  // 1. Subtract max for numerical stability
  // 2. Exponentiate
  // 3. Divide by sum
  throw std::runtime_error("ops::softmax not implemented yet");
}

Tensor rope(const Tensor &x, const Tensor &positions, float freq_base) {
  // TODO: Rotary Positional Embedding
  // For each position and head dimension pair:
  //   theta = position * freq_base^(-2i/d)
  //   Apply rotation: [cos(theta), -sin(theta); sin(theta), cos(theta)]
  throw std::runtime_error("ops::rope not implemented yet");
}

Tensor embedding(const Tensor &table, const Tensor &indices) {
  // TODO: Embedding table lookup
  // For each index, copy the corresponding row from the table
  throw std::runtime_error("ops::embedding not implemented yet");
}

} // namespace ops
} // namespace ie
