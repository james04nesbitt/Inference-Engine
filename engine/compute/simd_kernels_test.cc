#include "engine/compute/simd_kernels.h"

#include <cmath>
#include <vector>

#include "gtest/gtest.h"

namespace ie {
namespace compute {
namespace {

// ============================================================================
// SimdDotProduct Tests
// ============================================================================

TEST(SimdDotProduct, BasicDot) {
  std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> b = {5.0f, 6.0f, 7.0f, 8.0f};
  // 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32 = 70
  float result = SimdDotProduct(a.data(), b.data(), 4);
  EXPECT_NEAR(result, 70.0f, 1e-5f);
}

TEST(SimdDotProduct, Zeros) {
  std::vector<float> a(16, 0.0f);
  std::vector<float> b(16, 1.0f);
  EXPECT_NEAR(SimdDotProduct(a.data(), b.data(), 16), 0.0f, 1e-5f);
}

TEST(SimdDotProduct, LargeVector) {
  // Test with a vector larger than any SIMD register width.
  const int64_t n = 1024;
  std::vector<float> a(n, 1.0f);
  std::vector<float> b(n, 2.0f);
  float result = SimdDotProduct(a.data(), b.data(), n);
  EXPECT_NEAR(result, 2048.0f, 1e-3f);
}

TEST(SimdDotProduct, NonAlignedLength) {
  // Test with length that isn't a multiple of SIMD lanes.
  const int64_t n = 17;
  std::vector<float> a(n, 3.0f);
  std::vector<float> b(n, 4.0f);
  float result = SimdDotProduct(a.data(), b.data(), n);
  EXPECT_NEAR(result, 17.0f * 12.0f, 1e-4f);
}

// ============================================================================
// SimdGemm Tests
// ============================================================================

TEST(SimdGemm, SquareIdentity) {
  // A * I = A for 3x3 identity.
  std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> identity = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<float> c(9, 0.0f);

  SimdGemm(a.data(), identity.data(), c.data(), 3, 3, 3);

  for (int i = 0; i < 9; ++i) {
    EXPECT_NEAR(c[i], a[i], 1e-5f);
  }
}

TEST(SimdGemm, TwoByTwo) {
  // [1 2] @ [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
  // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
  std::vector<float> a = {1, 2, 3, 4};
  std::vector<float> b = {5, 6, 7, 8};
  std::vector<float> c(4, 0.0f);

  SimdGemm(a.data(), b.data(), c.data(), 2, 2, 2);

  EXPECT_NEAR(c[0], 19.0f, 1e-5f);
  EXPECT_NEAR(c[1], 22.0f, 1e-5f);
  EXPECT_NEAR(c[2], 43.0f, 1e-5f);
  EXPECT_NEAR(c[3], 50.0f, 1e-5f);
}

TEST(SimdGemm, Rectangular) {
  // [2x3] @ [3x2] = [2x2]
  std::vector<float> a = {1, 2, 3, 4, 5, 6};
  std::vector<float> b = {7, 8, 9, 10, 11, 12};
  std::vector<float> c(4, 0.0f);

  SimdGemm(a.data(), b.data(), c.data(), 2, 2, 3);

  // [1*7+2*9+3*11  1*8+2*10+3*12] = [58  64]
  // [4*7+5*9+6*11  4*8+5*10+6*12]   [139 154]
  EXPECT_NEAR(c[0], 58.0f, 1e-5f);
  EXPECT_NEAR(c[1], 64.0f, 1e-5f);
  EXPECT_NEAR(c[2], 139.0f, 1e-5f);
  EXPECT_NEAR(c[3], 154.0f, 1e-5f);
}

// ============================================================================
// SimdGemv Tests
// ============================================================================

TEST(SimdGemv, Basic) {
  // [1 2] @ [3] = [1*3+2*4] = [11]
  // [5 6]   [4]   [5*3+6*4]   [39]
  std::vector<float> a = {1, 2, 5, 6};
  std::vector<float> x = {3, 4};
  std::vector<float> y(2, 0.0f);

  SimdGemv(a.data(), x.data(), y.data(), 2, 2);

  EXPECT_NEAR(y[0], 11.0f, 1e-5f);
  EXPECT_NEAR(y[1], 39.0f, 1e-5f);
}

// ============================================================================
// SimdSoftmax Tests
// ============================================================================

TEST(SimdSoftmax, Uniform) {
  std::vector<float> input = {1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> output(4, 0.0f);

  SimdSoftmax(input.data(), output.data(), 4);

  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(output[i], 0.25f, 1e-5f);
  }
}

TEST(SimdSoftmax, SumToOne) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> output(5, 0.0f);

  SimdSoftmax(input.data(), output.data(), 5);

  float sum = 0.0f;
  for (int i = 0; i < 5; ++i)
    sum += output[i];
  EXPECT_NEAR(sum, 1.0f, 1e-5f);

  // Larger inputs should have larger outputs.
  for (int i = 0; i < 4; ++i) {
    EXPECT_LT(output[i], output[i + 1]);
  }
}

TEST(SimdSoftmax, NumericalStability) {
  // Very large values should not overflow.
  std::vector<float> input = {1000.0f, 1001.0f, 1002.0f};
  std::vector<float> output(3, 0.0f);

  SimdSoftmax(input.data(), output.data(), 3);

  float sum = 0.0f;
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(std::isfinite(output[i]));
    sum += output[i];
  }
  EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

// ============================================================================
// SimdRmsNorm Tests
// ============================================================================

TEST(SimdRmsNorm, OnesWeight) {
  std::vector<float> input = {3.0f, 4.0f};
  std::vector<float> weight = {1.0f, 1.0f};
  std::vector<float> output(2, 0.0f);

  SimdRmsNorm(input.data(), weight.data(), output.data(), 2, 1e-6f);

  // rms = sqrt((9 + 16) / 2) = sqrt(12.5) ≈ 3.5355
  // output[0] = 3.0 / 3.5355 ≈ 0.8485
  // output[1] = 4.0 / 3.5355 ≈ 1.1314
  float rms = std::sqrt(12.5f);
  EXPECT_NEAR(output[0], 3.0f / rms, 1e-4f);
  EXPECT_NEAR(output[1], 4.0f / rms, 1e-4f);
}

// ============================================================================
// SimdSilu Tests
// ============================================================================

TEST(SimdSilu, Known) {
  std::vector<float> input = {0.0f, 1.0f, -1.0f};
  std::vector<float> output(3, 0.0f);

  SimdSilu(input.data(), output.data(), 3);

  // silu(0) = 0 * sigmoid(0) = 0
  EXPECT_NEAR(output[0], 0.0f, 1e-5f);
  // silu(1) = 1 * sigmoid(1) = 1 / (1 + exp(-1)) ≈ 0.7311
  EXPECT_NEAR(output[1], 1.0f / (1.0f + std::exp(-1.0f)), 1e-4f);
  // silu(-1) = -1 * sigmoid(-1) = -1 / (1 + exp(1)) ≈ -0.2689
  EXPECT_NEAR(output[2], -1.0f / (1.0f + std::exp(1.0f)), 1e-4f);
}

// ============================================================================
// SimdMul / SimdAdd Tests
// ============================================================================

TEST(SimdMul, Basic) {
  std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> b = {2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> c(8, 0.0f);

  SimdMul(a.data(), b.data(), c.data(), 8);

  for (int i = 0; i < 8; ++i) {
    EXPECT_NEAR(c[i], a[i] * b[i], 1e-5f);
  }
}

TEST(SimdAdd, Basic) {
  std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> b = {8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<float> c(8, 0.0f);

  SimdAdd(a.data(), b.data(), c.data(), 8);

  for (int i = 0; i < 8; ++i) {
    EXPECT_NEAR(c[i], 9.0f, 1e-5f);
  }
}

} // namespace
} // namespace compute
} // namespace ie
