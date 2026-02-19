#include "engine/ops/ops.h"

#include <cmath>
#include <stdexcept>

namespace ie {
namespace ops {

Tensor add(const Tensor& a, const Tensor& b) {
  // TODO: Implement element-wise addition with broadcasting
  // Step 1: Validate shapes are broadcast-compatible
  // Step 2: Allocate output tensor
  // Step 3: Iterate and add (handle broadcasting)
  throw std::runtime_error("ops::add not implemented yet");
}

Tensor mul(const Tensor& a, const Tensor& b) {
  // TODO: Implement element-wise multiplication with broadcasting
  throw std::runtime_error("ops::mul not implemented yet");
}

Tensor matmul(const Tensor& a, const Tensor& b) {
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

Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps) {
  // TODO: RMS normalization
  // 1. Compute mean of squares along last dim
  // 2. Divide by sqrt(mean + eps)
  // 3. Multiply by weight
  throw std::runtime_error("ops::rms_norm not implemented yet");
}

Tensor silu(const Tensor& x) {
  // TODO: SiLU activation: x * sigmoid(x)
  // sigmoid(x) = 1 / (1 + exp(-x))
  throw std::runtime_error("ops::silu not implemented yet");
}

Tensor gelu(const Tensor& x) {
  // TODO: GeLU activation (tanh approximation)
  throw std::runtime_error("ops::gelu not implemented yet");
}

Tensor softmax(const Tensor& x, int64_t dim) {
  // TODO: Numerically stable softmax
  // 1. Subtract max for numerical stability
  // 2. Exponentiate
  // 3. Divide by sum
  throw std::runtime_error("ops::softmax not implemented yet");
}

Tensor rope(const Tensor& x, const Tensor& positions, float freq_base) {
  // TODO: Rotary Positional Embedding
  // For each position and head dimension pair:
  //   theta = position * freq_base^(-2i/d)
  //   Apply rotation: [cos(theta), -sin(theta); sin(theta), cos(theta)]
  throw std::runtime_error("ops::rope not implemented yet");
}

Tensor embedding(const Tensor& table, const Tensor& indices) {
  // TODO: Embedding table lookup
  // For each index, copy the corresponding row from the table
  throw std::runtime_error("ops::embedding not implemented yet");
}

}  // namespace ops
}  // namespace ie
