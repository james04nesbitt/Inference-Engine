#include "engine/quantize/quantize.h"

#include <cmath>
#include <vector>

#include "gtest/gtest.h"

namespace ie {
namespace {

// ============================================================================
// ComputeQuantParams Tests
// ============================================================================

TEST(ComputeQuantParams, SymmetricPerChannel) {
  // 2D tensor [2 channels, 3 elements each]
  Tensor t({2, 3});
  // Channel 0: [1.0, -2.0, 3.0]  -> absmax = 3.0, scale = 3.0/127
  // Channel 1: [4.0, -5.0, 6.0]  -> absmax = 6.0, scale = 6.0/127
  t.set({0, 0}, 1.0f);
  t.set({0, 1}, -2.0f);
  t.set({0, 2}, 3.0f);
  t.set({1, 0}, 4.0f);
  t.set({1, 1}, -5.0f);
  t.set({1, 2}, 6.0f);

  auto params = ComputeQuantParams(t, /*channel_dim=*/0, /*symmetric=*/true);

  EXPECT_EQ(params.num_channels, 2);
  EXPECT_TRUE(params.symmetric);
  EXPECT_NEAR(params.scales[0], 3.0f / 127.0f, 1e-6f);
  EXPECT_NEAR(params.scales[1], 6.0f / 127.0f, 1e-6f);
  EXPECT_EQ(params.zero_points[0], 0);
  EXPECT_EQ(params.zero_points[1], 0);
}

// ============================================================================
// QuantizeInt8 + DequantizeInt8 Round-Trip Tests
// ============================================================================

TEST(Quantize, RoundTripSmallValues) {
  // Values within a range that INT8 can represent well.
  Tensor t = Tensor::from_vector({1.0f, -1.0f, 0.5f, -0.5f, 0.0f});
  t = t.reshape({1, 5}); // [1 channel, 5 elements]

  auto params = ComputeQuantParams(t, 0, true);
  auto qt = QuantizeInt8(t, params, 0);

  // Compression ratio for small tensors won't be 4x due to metadata overhead.
  // Just verify it's > 1.0 (i.e., compressed).
  EXPECT_GT(qt.CompressionRatio(), 1.0f);

  Tensor recovered = DequantizeInt8(qt);

  // Values should be close to originals (within quantization error).
  EXPECT_EQ(recovered.size(0), 1);
  EXPECT_EQ(recovered.size(1), 5);
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_NEAR(recovered.at({0, i}), t.at({0, i}), 0.02f)
        << "Mismatch at index " << i;
  }
}

TEST(Quantize, RoundTripMultiChannel) {
  // [3 channels, 4 elements each]
  Tensor t({3, 4});
  float vals[] = {1, 2, 3, 4, 10, 20, 30, 40, 0.1f, 0.2f, 0.3f, 0.4f};
  for (int i = 0; i < 12; ++i) {
    t.set({i / 4, i % 4}, vals[i]);
  }

  auto params = ComputeQuantParams(t, 0, true);
  auto qt = QuantizeInt8(t, params, 0);
  Tensor recovered = DequantizeInt8(qt);

  // Each channel has different scale; check relative accuracy.
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < 4; ++i) {
      float orig = t.at({c, i});
      float recv = recovered.at({c, i});
      float tol = std::abs(orig) * 0.02f + 0.5f; // ~1% + quantization step
      EXPECT_NEAR(recv, orig, tol) << "c=" << c << " i=" << i;
    }
  }
}

// ============================================================================
// QuantizeKVCache Tests
// ============================================================================

TEST(QuantizeKVCache, BasicQuantization) {
  // Simulate a small KV cache: [4 tokens, 1 head, 8 dim]
  Tensor kv({4, 1, 8});
  for (int64_t s = 0; s < 4; ++s) {
    for (int64_t d = 0; d < 8; ++d) {
      kv.set({s, static_cast<int64_t>(0), d},
             static_cast<float>((s + 1) * (d + 1)) * 0.1f);
    }
  }

  auto qt = QuantizeKVCache(kv, 6.0f);

  // Should be INT8 data.
  EXPECT_EQ(qt.data.dtype(), DType::kInt8);
  // Compression ratio > 1.
  EXPECT_GT(qt.CompressionRatio(), 1.0f);
}

// ============================================================================
// QuantizedMatmul Tests
// ============================================================================

TEST(QuantizedMatmul, BasicGemm) {
  // A: [2, 3], B: [3, 2] quantized.
  Tensor A({2, 3});
  A.set({0, 0}, 1.0f);
  A.set({0, 1}, 2.0f);
  A.set({0, 2}, 3.0f);
  A.set({1, 0}, 4.0f);
  A.set({1, 1}, 5.0f);
  A.set({1, 2}, 6.0f);

  Tensor B({3, 2});
  B.set({0, 0}, 7.0f);
  B.set({0, 1}, 8.0f);
  B.set({1, 0}, 9.0f);
  B.set({1, 1}, 10.0f);
  B.set({2, 0}, 11.0f);
  B.set({2, 1}, 12.0f);

  // Quantize B.
  auto params = ComputeQuantParams(B, 0, true);
  auto qB = QuantizeInt8(B, params, 0);

  // Compute C = A @ B (quantized).
  Tensor C = QuantizedMatmul(A, qB);

  // Expected (exact): [1*7+2*9+3*11=58, 1*8+2*10+3*12=64]
  //                   [4*7+5*9+6*11=139, 4*8+5*10+6*12=154]
  EXPECT_EQ(C.size(0), 2);
  EXPECT_EQ(C.size(1), 2);
  // Allow some quantization error.
  EXPECT_NEAR(C.at({0, 0}), 58.0f, 2.0f);
  EXPECT_NEAR(C.at({0, 1}), 64.0f, 2.0f);
  EXPECT_NEAR(C.at({1, 0}), 139.0f, 3.0f);
  EXPECT_NEAR(C.at({1, 1}), 154.0f, 3.0f);
}

} // namespace
} // namespace ie
