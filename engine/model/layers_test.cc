#include "engine/model/layers.h"

#include <cmath>
#include <vector>

#include "engine/ops/ops.h"
#include "engine/tensor/tensor.h"
#include "gtest/gtest.h"

namespace ie {
namespace {

// ============================================================================
// RMSNorm Tests
// ============================================================================

TEST(RMSNorm, BasicNormalization) {
  // Weight of all ones — just normalizes by RMS.
  Tensor weight = Tensor::ones({4});
  RMSNorm norm(weight, 1e-6f);

  Tensor x = Tensor::from_vector({3.0f, 4.0f, 0.0f, 0.0f});
  x = x.reshape({1, 4}); // [1, 4]

  Tensor out = norm.forward(x);
  EXPECT_EQ(out.size(0), 1);
  EXPECT_EQ(out.size(1), 4);

  // rms = sqrt(mean(x^2)) = sqrt((9+16+0+0)/4) = sqrt(6.25) = 2.5
  float rms = std::sqrt(6.25f);
  EXPECT_NEAR(out.at({0, 0}), 3.0f / rms, 1e-4f);
  EXPECT_NEAR(out.at({0, 1}), 4.0f / rms, 1e-4f);
  EXPECT_NEAR(out.at({0, 2}), 0.0f, 1e-4f);
  EXPECT_NEAR(out.at({0, 3}), 0.0f, 1e-4f);
}

TEST(RMSNorm, WithWeightScaling) {
  Tensor weight = Tensor::from_vector({2.0f, 0.5f});
  RMSNorm norm(weight, 1e-6f);

  Tensor x = Tensor::from_vector({1.0f, 1.0f});
  x = x.reshape({1, 2});

  Tensor out = norm.forward(x);

  EXPECT_NEAR(out.at({0, 0}), 2.0f, 1e-4f);
  EXPECT_NEAR(out.at({0, 1}), 0.5f, 1e-4f);
}

// ============================================================================
// FeedForward Tests
// ============================================================================

TEST(FeedForward, SmallSwiGLU) {
  Tensor w_gate = Tensor::ones({3, 2});
  Tensor w_up = Tensor::ones({3, 2});
  Tensor w_down = Tensor::ones({2, 3});

  FeedForward ffn(w_gate, w_up, w_down);

  Tensor x = Tensor::ones({1, 2});

  Tensor out = ffn.forward(x);
  EXPECT_EQ(out.size(0), 1);
  EXPECT_EQ(out.size(1), 2);
  EXPECT_TRUE(std::isfinite(out.at({0, 0})));
  EXPECT_TRUE(std::isfinite(out.at({0, 1})));
}

// ============================================================================
// TransformerBlock Tests (with Paged KV Cache)
// ============================================================================

TEST(TransformerBlock, SmallBlock) {
  GemmaConfig config;
  config.embed_dim = 4;
  config.num_heads = 1;
  config.num_kv_heads = 1;
  config.head_dim = 4;
  config.hidden_dim = 8;
  config.num_layers = 1;
  config.max_seq_len = 64;
  config.rms_norm_eps = 1e-6f;
  config.rope_theta = 10000.0f;

  Tensor attn_norm_w = Tensor::ones({4});
  Tensor ffn_norm_w = Tensor::ones({4});
  Tensor wq = Tensor::full({4, 4}, 0.1f);
  Tensor wk = Tensor::full({4, 4}, 0.1f);
  Tensor wv = Tensor::full({4, 4}, 0.1f);
  Tensor wo = Tensor::full({4, 4}, 0.1f);
  Tensor w_gate = Tensor::full({8, 4}, 0.1f);
  Tensor w_up = Tensor::full({8, 4}, 0.1f);
  Tensor w_down = Tensor::full({4, 8}, 0.1f);

  RMSNorm attn_norm(attn_norm_w, config.rms_norm_eps);
  Attention attn(config, /*layer_idx=*/0, wq, wk, wv, wo);
  RMSNorm ffn_norm(ffn_norm_w, config.rms_norm_eps);
  FeedForward ffn(w_gate, w_up, w_down);

  TransformerBlock block(config, /*layer_idx=*/0, std::move(attn_norm),
                         std::move(attn), std::move(ffn_norm), std::move(ffn));

  // Create a KV cache manager for this test.
  KVCacheManager kv_cache(/*num_layers=*/1, /*num_kv_heads=*/1,
                          /*head_dim=*/4, /*max_blocks=*/20,
                          /*block_size=*/16);
  int64_t seq_id = kv_cache.AllocateSequence();

  // Input: [2, 4] (seq_len=2, embed_dim=4)
  Tensor x = Tensor::ones({2, 4});
  Tensor out = block.forward(x, 0, kv_cache, seq_id);

  EXPECT_EQ(out.size(0), 2);
  EXPECT_EQ(out.size(1), 4);
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 4; ++j) {
      EXPECT_TRUE(std::isfinite(out.at({i, j})))
          << "NaN/inf at (" << i << ", " << j << ")";
    }
  }

  kv_cache.FreeSequence(seq_id);
}

} // namespace
} // namespace ie
