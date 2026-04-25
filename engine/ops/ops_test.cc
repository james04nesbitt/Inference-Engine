#include "engine/ops/ops.h"

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

namespace ie {
namespace ops {
namespace {

// ============================================================================
// add / mul — Element-wise operations
// ============================================================================

TEST(OpsTest, AddBasic) {
  Tensor a = Tensor::from_vector({1.0f, 2.0f, 3.0f});
  Tensor b = Tensor::from_vector({4.0f, 5.0f, 6.0f});
  Tensor c = add(a, b);
  EXPECT_TRUE(c.shape_equals({3}));
  EXPECT_FLOAT_EQ(c.at({0}), 5.0f);
  EXPECT_FLOAT_EQ(c.at({1}), 7.0f);
  EXPECT_FLOAT_EQ(c.at({2}), 9.0f);
}

TEST(OpsTest, Add2D) {
  Tensor a({2, 3});
  Tensor b({2, 3});
  a.fill(1.0f);
  b.fill(2.0f);
  Tensor c = add(a, b);
  EXPECT_TRUE(c.shape_equals({2, 3}));
  for (int64_t i = 0; i < c.numel(); ++i) {
    EXPECT_FLOAT_EQ(c.data<float>()[i], 3.0f);
  }
}

TEST(OpsTest, AddShapeMismatchThrows) {
  Tensor a({2, 3});
  Tensor b({3, 2});
  EXPECT_THROW(add(a, b), std::runtime_error);
}

TEST(OpsTest, MulBasic) {
  Tensor a = Tensor::from_vector({1.0f, 2.0f, 3.0f});
  Tensor b = Tensor::from_vector({4.0f, 5.0f, 6.0f});
  Tensor c = mul(a, b);
  EXPECT_FLOAT_EQ(c.at({0}), 4.0f);
  EXPECT_FLOAT_EQ(c.at({1}), 10.0f);
  EXPECT_FLOAT_EQ(c.at({2}), 18.0f);
}

TEST(OpsTest, MulShapeMismatchThrows) {
  Tensor a({4});
  Tensor b({5});
  EXPECT_THROW(mul(a, b), std::runtime_error);
}

// ============================================================================
// matmul
// ============================================================================

TEST(OpsTest, MatmulIdentity) {
  // Multiply by identity matrix: A @ I = A
  Tensor a({2, 3});
  a.set({0, 0}, 1.0f);
  a.set({0, 1}, 2.0f);
  a.set({0, 2}, 3.0f);
  a.set({1, 0}, 4.0f);
  a.set({1, 1}, 5.0f);
  a.set({1, 2}, 6.0f);

  // 3x3 identity
  Tensor eye({3, 3});
  eye.set({0, 0}, 1.0f);
  eye.set({1, 1}, 1.0f);
  eye.set({2, 2}, 1.0f);

  Tensor c = matmul(a, eye);
  EXPECT_TRUE(c.shape_equals({2, 3}));
  EXPECT_FLOAT_EQ(c.at({0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(c.at({0, 1}), 2.0f);
  EXPECT_FLOAT_EQ(c.at({0, 2}), 3.0f);
  EXPECT_FLOAT_EQ(c.at({1, 0}), 4.0f);
  EXPECT_FLOAT_EQ(c.at({1, 1}), 5.0f);
  EXPECT_FLOAT_EQ(c.at({1, 2}), 6.0f);
}

TEST(OpsTest, MatmulKnown2x2) {
  // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
  Tensor a({2, 2});
  a.set({0, 0}, 1.0f);
  a.set({0, 1}, 2.0f);
  a.set({1, 0}, 3.0f);
  a.set({1, 1}, 4.0f);

  Tensor b({2, 2});
  b.set({0, 0}, 5.0f);
  b.set({0, 1}, 6.0f);
  b.set({1, 0}, 7.0f);
  b.set({1, 1}, 8.0f);

  Tensor c = matmul(a, b);
  EXPECT_FLOAT_EQ(c.at({0, 0}), 19.0f);
  EXPECT_FLOAT_EQ(c.at({0, 1}), 22.0f);
  EXPECT_FLOAT_EQ(c.at({1, 0}), 43.0f);
  EXPECT_FLOAT_EQ(c.at({1, 1}), 50.0f);
}

TEST(OpsTest, MatmulNonSquare) {
  // [2, 3] @ [3, 4] = [2, 4]
  Tensor a({2, 3});
  Tensor b({3, 4});
  a.fill(1.0f);
  b.fill(1.0f);

  Tensor c = matmul(a, b);
  EXPECT_TRUE(c.shape_equals({2, 4}));
  // Each element should be 3.0 (dot product of ones vectors, length 3)
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(c.at({i, j}), 3.0f);
    }
  }
}

TEST(OpsTest, MatmulInnerDimMismatchThrows) {
  Tensor a({2, 3});
  Tensor b({4, 5});
  EXPECT_THROW(matmul(a, b), std::runtime_error);
}

TEST(OpsTest, MatmulNot2DThrows) {
  Tensor a({2, 3, 4});
  Tensor b({4, 5});
  EXPECT_THROW(matmul(a, b), std::runtime_error);
}

// ============================================================================
// rms_norm
// ============================================================================

TEST(OpsTest, RmsNormBasic) {
  // x = [1, 2, 3, 4], weight = [1, 1, 1, 1]
  // rms = sqrt(mean([1, 4, 9, 16]) + 1e-6) = sqrt(7.5 + 1e-6) ≈ 2.7386
  // output ≈ [0.3651, 0.7303, 1.0954, 1.4606]
  Tensor x = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f});
  Tensor w = Tensor::ones({4});
  Tensor out = rms_norm(x, w);

  // With (1+weight) formulation: weight=ones means effective weight = 2.0
  float rms = std::sqrt((1 + 4 + 9 + 16) / 4.0f + 1e-6f);
  EXPECT_NEAR(out.at({0}), 2.0f * 1.0f / rms, 1e-4f);
  EXPECT_NEAR(out.at({1}), 2.0f * 2.0f / rms, 1e-4f);
  EXPECT_NEAR(out.at({2}), 2.0f * 3.0f / rms, 1e-4f);
  EXPECT_NEAR(out.at({3}), 2.0f * 4.0f / rms, 1e-4f);
}

TEST(OpsTest, RmsNormWithWeight) {
  Tensor x = Tensor::from_vector({2.0f, 2.0f});
  Tensor w = Tensor::from_vector({3.0f, 0.5f});
  Tensor out = rms_norm(x, w);

  // rms = sqrt(mean([4, 4]) + 1e-6) = sqrt(4 + 1e-6) ≈ 2.0
  // normalized x ≈ [1, 1]
  // With (1+weight): effective weights are (1+3.0)=4.0, (1+0.5)=1.5
  // output ≈ [4.0, 1.5]
  EXPECT_NEAR(out.at({0}), 4.0f, 1e-4f);
  EXPECT_NEAR(out.at({1}), 1.5f, 1e-4f);
}

TEST(OpsTest, RmsNorm2D) {
  // 2D input: each row is normalized independently
  Tensor x({2, 3});
  x.set({0, 0}, 1.0f);
  x.set({0, 1}, 1.0f);
  x.set({0, 2}, 1.0f);
  x.set({1, 0}, 3.0f);
  x.set({1, 1}, 3.0f);
  x.set({1, 2}, 3.0f);
  Tensor w = Tensor::ones({3});

  Tensor out = rms_norm(x, w);
  EXPECT_TRUE(out.shape_equals({2, 3}));

  // Row 0: rms = sqrt(1 + 1e-6) ≈ 1.0, with (1+1)=2 weight, output ≈ [2, 2, 2]
  EXPECT_NEAR(out.at({0, 0}), 2.0f, 1e-4f);
  // Row 1: rms = sqrt(9 + 1e-6) ≈ 3.0, with (1+1)=2 weight, output ≈ [2, 2, 2]
  EXPECT_NEAR(out.at({1, 0}), 2.0f, 1e-4f);
}

TEST(OpsTest, RmsNormWeightMismatchThrows) {
  Tensor x({4});
  Tensor w({3}); // wrong size
  EXPECT_THROW(rms_norm(x, w), std::runtime_error);
}

// ============================================================================
// silu
// ============================================================================

TEST(OpsTest, SiluZero) {
  Tensor x = Tensor::from_vector({0.0f});
  Tensor out = silu(x);
  // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
  EXPECT_FLOAT_EQ(out.at({0}), 0.0f);
}

TEST(OpsTest, SiluKnownValues) {
  Tensor x = Tensor::from_vector({1.0f, -1.0f, 2.0f});
  Tensor out = silu(x);

  // silu(1) = 1 * sigmoid(1) = 1 / (1 + exp(-1)) ≈ 0.7311
  EXPECT_NEAR(out.at({0}), 0.7311f, 1e-3f);
  // silu(-1) = -1 * sigmoid(-1) = -1 / (1 + exp(1)) ≈ -0.2689
  EXPECT_NEAR(out.at({1}), -0.2689f, 1e-3f);
  // silu(2) ≈ 2 * 0.8808 = 1.7616
  EXPECT_NEAR(out.at({2}), 1.7616f, 1e-3f);
}

// ============================================================================
// gelu
// ============================================================================

TEST(OpsTest, GeluZero) {
  Tensor x = Tensor::from_vector({0.0f});
  Tensor out = gelu(x);
  EXPECT_NEAR(out.at({0}), 0.0f, 1e-5f);
}

TEST(OpsTest, GeluKnownValues) {
  Tensor x = Tensor::from_vector({1.0f, -1.0f});
  Tensor out = gelu(x);

  // gelu(1) ≈ 0.8412 (tanh approximation)
  EXPECT_NEAR(out.at({0}), 0.8412f, 1e-3f);
  // gelu(-1) ≈ -0.1588
  EXPECT_NEAR(out.at({1}), -0.1588f, 1e-3f);
}

// ============================================================================
// softmax
// ============================================================================

TEST(OpsTest, SoftmaxUniform) {
  // Uniform input → uniform output
  Tensor x = Tensor::full({4}, 1.0f);
  Tensor out = softmax(x);
  EXPECT_TRUE(out.shape_equals({4}));
  for (int64_t i = 0; i < 4; ++i) {
    EXPECT_NEAR(out.at({i}), 0.25f, 1e-5f);
  }
}

TEST(OpsTest, SoftmaxSumsToOne) {
  Tensor x = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f});
  Tensor out = softmax(x);
  float sum = 0.0f;
  for (int64_t i = 0; i < 4; ++i) {
    sum += out.at({i});
    EXPECT_GT(out.at({i}), 0.0f);
  }
  EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(OpsTest, SoftmaxMonotonic) {
  // Larger input → larger probability
  Tensor x = Tensor::from_vector({1.0f, 2.0f, 3.0f});
  Tensor out = softmax(x);
  EXPECT_LT(out.at({0}), out.at({1}));
  EXPECT_LT(out.at({1}), out.at({2}));
}

TEST(OpsTest, SoftmaxNumericalStability) {
  // Very large values — naive exp() would overflow
  Tensor x = Tensor::from_vector({1000.0f, 1001.0f, 1002.0f});
  Tensor out = softmax(x);
  float sum = 0.0f;
  for (int64_t i = 0; i < 3; ++i) {
    EXPECT_FALSE(std::isnan(out.at({i})));
    EXPECT_FALSE(std::isinf(out.at({i})));
    sum += out.at({i});
  }
  EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(OpsTest, Softmax2DDim1) {
  // 2D input, softmax along dim 1 (last dim, default)
  Tensor x({2, 3});
  x.set({0, 0}, 1.0f);
  x.set({0, 1}, 1.0f);
  x.set({0, 2}, 1.0f);
  x.set({1, 0}, 0.0f);
  x.set({1, 1}, 10.0f);
  x.set({1, 2}, 0.0f);

  Tensor out = softmax(x, 1);
  EXPECT_TRUE(out.shape_equals({2, 3}));

  // Row 0: uniform → 1/3 each
  EXPECT_NEAR(out.at({0, 0}), 1.0f / 3.0f, 1e-5f);
  EXPECT_NEAR(out.at({0, 1}), 1.0f / 3.0f, 1e-5f);

  // Row 1: dominated by index 1 (value 10)
  EXPECT_GT(out.at({1, 1}), 0.99f);

  // Each row sums to 1
  float sum0 = out.at({0, 0}) + out.at({0, 1}) + out.at({0, 2});
  float sum1 = out.at({1, 0}) + out.at({1, 1}) + out.at({1, 2});
  EXPECT_NEAR(sum0, 1.0f, 1e-5f);
  EXPECT_NEAR(sum1, 1.0f, 1e-5f);
}

TEST(OpsTest, Softmax2DDim0) {
  // Softmax along dim 0 (column-wise)
  Tensor x({3, 2});
  x.set({0, 0}, 1.0f);
  x.set({0, 1}, 1.0f);
  x.set({1, 0}, 1.0f);
  x.set({1, 1}, 1.0f);
  x.set({2, 0}, 1.0f);
  x.set({2, 1}, 1.0f);

  Tensor out = softmax(x, 0);
  // Uniform → each should be 1/3
  for (int64_t r = 0; r < 3; ++r) {
    for (int64_t c = 0; c < 2; ++c) {
      EXPECT_NEAR(out.at({r, c}), 1.0f / 3.0f, 1e-5f);
    }
  }
}

// ============================================================================
// rope
// ============================================================================

TEST(OpsTest, RopePreservesNorm) {
  // RoPE is a rotation — it should preserve the L2 norm of each head vector.
  const int64_t batch = 1, seq_len = 1, n_heads = 1, head_dim = 4;
  Tensor x({batch, seq_len, n_heads, head_dim});
  x.set({0, 0, 0, 0}, 1.0f);
  x.set({0, 0, 0, 1}, 2.0f);
  x.set({0, 0, 0, 2}, 3.0f);
  x.set({0, 0, 0, 3}, 4.0f);

  Tensor pos({batch, seq_len});
  pos.set({0, 0}, 1.0f);

  Tensor out = rope(x, pos);
  EXPECT_TRUE(out.shape_equals({batch, seq_len, n_heads, head_dim}));

  // Compute norms
  float norm_in = 0.0f, norm_out = 0.0f;
  for (int64_t i = 0; i < head_dim; ++i) {
    float xi = x.at({0, 0, 0, i});
    float oi = out.at({0, 0, 0, i});
    norm_in += xi * xi;
    norm_out += oi * oi;
  }
  EXPECT_NEAR(std::sqrt(norm_in), std::sqrt(norm_out), 1e-4f);
}

TEST(OpsTest, RopePositionZeroIsIdentity) {
  // At position 0, theta = 0 for all dims, so cos(0)=1, sin(0)=0.
  // Output should equal input.
  Tensor x({1, 1, 1, 4});
  x.set({0, 0, 0, 0}, 1.0f);
  x.set({0, 0, 0, 1}, 2.0f);
  x.set({0, 0, 0, 2}, 3.0f);
  x.set({0, 0, 0, 3}, 4.0f);

  Tensor pos = Tensor::zeros({1, 1});
  Tensor out = rope(x, pos);

  for (int64_t i = 0; i < 4; ++i) {
    EXPECT_NEAR(out.at({0, 0, 0, i}), x.at({0, 0, 0, i}), 1e-5f);
  }
}

TEST(OpsTest, RopeOddHeadDimThrows) {
  Tensor x({1, 1, 1, 3}); // odd head_dim
  Tensor pos({1, 1});
  EXPECT_THROW(rope(x, pos), std::runtime_error);
}

TEST(OpsTest, RopeBatchSeqMismatchThrows) {
  Tensor x({2, 3, 1, 4});
  Tensor pos({1, 3}); // batch mismatch
  EXPECT_THROW(rope(x, pos), std::runtime_error);
}

// ============================================================================
// embedding
// ============================================================================

TEST(OpsTest, EmbeddingBasic) {
  // 4 words, 3-dim embeddings
  Tensor table({4, 3});
  table.set({0, 0}, 0.1f);
  table.set({0, 1}, 0.2f);
  table.set({0, 2}, 0.3f);
  table.set({1, 0}, 1.1f);
  table.set({1, 1}, 1.2f);
  table.set({1, 2}, 1.3f);
  table.set({2, 0}, 2.1f);
  table.set({2, 1}, 2.2f);
  table.set({2, 2}, 2.3f);
  table.set({3, 0}, 3.1f);
  table.set({3, 1}, 3.2f);
  table.set({3, 2}, 3.3f);

  // Look up tokens 2, 0, 3
  Tensor indices = Tensor::from_vector({2.0f, 0.0f, 3.0f});
  Tensor out = embedding(table, indices);

  EXPECT_TRUE(out.shape_equals({3, 3}));
  // Row 0 = table[2]
  EXPECT_FLOAT_EQ(out.at({0, 0}), 2.1f);
  EXPECT_FLOAT_EQ(out.at({0, 1}), 2.2f);
  EXPECT_FLOAT_EQ(out.at({0, 2}), 2.3f);
  // Row 1 = table[0]
  EXPECT_FLOAT_EQ(out.at({1, 0}), 0.1f);
  // Row 2 = table[3]
  EXPECT_FLOAT_EQ(out.at({2, 0}), 3.1f);
}

TEST(OpsTest, EmbeddingOutOfBoundsThrows) {
  Tensor table({4, 3});
  Tensor indices = Tensor::from_vector({5.0f}); // out of range
  EXPECT_THROW(embedding(table, indices), std::runtime_error);
}

TEST(OpsTest, EmbeddingNegativeIndexThrows) {
  Tensor table({4, 3});
  Tensor indices = Tensor::from_vector({-1.0f});
  EXPECT_THROW(embedding(table, indices), std::runtime_error);
}

// ============================================================================
// Multi-dtype support — FP16 inputs/outputs
// ============================================================================

TEST(OpsTest, AddFloat16) {
  Tensor a({3}, DType::kFloat16);
  Tensor b({3}, DType::kFloat16);
  a.set({0}, 1.0f);
  a.set({1}, 2.0f);
  a.set({2}, 3.0f);
  b.set({0}, 4.0f);
  b.set({1}, 5.0f);
  b.set({2}, 6.0f);

  Tensor c = add(a, b);
  EXPECT_EQ(c.dtype(), DType::kFloat16); // output should preserve dtype
  EXPECT_NEAR(c.at({0}), 5.0f, 0.01f);
  EXPECT_NEAR(c.at({1}), 7.0f, 0.01f);
  EXPECT_NEAR(c.at({2}), 9.0f, 0.01f);
}

TEST(OpsTest, MulFloat16) {
  Tensor a({2}, DType::kFloat16);
  Tensor b({2}, DType::kFloat16);
  a.set({0}, 2.0f);
  a.set({1}, 3.0f);
  b.set({0}, 4.0f);
  b.set({1}, 5.0f);

  Tensor c = mul(a, b);
  EXPECT_EQ(c.dtype(), DType::kFloat16);
  EXPECT_NEAR(c.at({0}), 8.0f, 0.01f);
  EXPECT_NEAR(c.at({1}), 15.0f, 0.01f);
}

TEST(OpsTest, MatmulFloat16) {
  // FP16 matmul: compute in FP32, output as FP16
  Tensor a({2, 2}, DType::kFloat16);
  a.set({0, 0}, 1.0f);
  a.set({0, 1}, 2.0f);
  a.set({1, 0}, 3.0f);
  a.set({1, 1}, 4.0f);

  Tensor b({2, 2}, DType::kFloat16);
  b.set({0, 0}, 5.0f);
  b.set({0, 1}, 6.0f);
  b.set({1, 0}, 7.0f);
  b.set({1, 1}, 8.0f);

  Tensor c = matmul(a, b);
  EXPECT_EQ(c.dtype(), DType::kFloat16);
  EXPECT_NEAR(c.at({0, 0}), 19.0f, 0.1f);
  EXPECT_NEAR(c.at({0, 1}), 22.0f, 0.1f);
  EXPECT_NEAR(c.at({1, 0}), 43.0f, 0.1f);
  EXPECT_NEAR(c.at({1, 1}), 50.0f, 0.1f);
}

TEST(OpsTest, SiluFloat16) {
  Tensor x({2}, DType::kFloat16);
  x.set({0}, 1.0f);
  x.set({1}, 0.0f);

  Tensor out = silu(x);
  EXPECT_EQ(out.dtype(), DType::kFloat16);
  EXPECT_NEAR(out.at({0}), 0.7311f, 0.01f);
  EXPECT_NEAR(out.at({1}), 0.0f, 0.01f);
}

TEST(OpsTest, SoftmaxFloat16) {
  Tensor x({3}, DType::kFloat16);
  x.set({0}, 1.0f);
  x.set({1}, 1.0f);
  x.set({2}, 1.0f);

  Tensor out = softmax(x);
  EXPECT_EQ(out.dtype(), DType::kFloat16);
  for (int64_t i = 0; i < 3; ++i) {
    EXPECT_NEAR(out.at({i}), 1.0f / 3.0f, 0.01f);
  }
}

TEST(OpsTest, RmsNormFloat16) {
  Tensor x({2}, DType::kFloat16);
  x.set({0}, 2.0f);
  x.set({1}, 2.0f);
  Tensor w = Tensor::ones({2}, DType::kFloat16);

  Tensor out = rms_norm(x, w);
  EXPECT_EQ(out.dtype(), DType::kFloat16);
  // rms = sqrt(4 + 1e-6) ≈ 2.0, with (1+1)=2 weight, output ≈ [2, 2]
  EXPECT_NEAR(out.at({0}), 2.0f, 0.01f);
  EXPECT_NEAR(out.at({1}), 2.0f, 0.01f);
}

TEST(OpsTest, EmbeddingFloat16Table) {
  // FP16 embedding table — common in GGUF models
  Tensor table({3, 2}, DType::kFloat16);
  table.set({0, 0}, 1.0f);
  table.set({0, 1}, 2.0f);
  table.set({1, 0}, 3.0f);
  table.set({1, 1}, 4.0f);
  table.set({2, 0}, 5.0f);
  table.set({2, 1}, 6.0f);

  Tensor indices = Tensor::from_vector({1.0f, 2.0f});
  Tensor out = embedding(table, indices);

  EXPECT_EQ(out.dtype(), DType::kFloat16); // preserves table dtype
  EXPECT_TRUE(out.shape_equals({2, 2}));
  EXPECT_NEAR(out.at({0, 0}), 3.0f, 0.01f); // table[1]
  EXPECT_NEAR(out.at({0, 1}), 4.0f, 0.01f);
  EXPECT_NEAR(out.at({1, 0}), 5.0f, 0.01f); // table[2]
  EXPECT_NEAR(out.at({1, 1}), 6.0f, 0.01f);
}

} // namespace
} // namespace ops
} // namespace ie
