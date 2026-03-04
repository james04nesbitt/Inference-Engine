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

  // rms = sqrt((1+1)/2) = 1.0
  // output = [1.0 * 2.0 / 1.0, 1.0 * 0.5 / 1.0] = [2.0, 0.5]
  EXPECT_NEAR(out.at({0, 0}), 2.0f, 1e-4f);
  EXPECT_NEAR(out.at({0, 1}), 0.5f, 1e-4f);
}

// ============================================================================
// FeedForward Tests
// ============================================================================

TEST(FeedForward, SmallSwiGLU) {
  // Create tiny weights: embed_dim=2, hidden_dim=3
  // Weight shape: [hidden_dim, embed_dim] = [3, 2]
  Tensor w_gate = Tensor::ones({3, 2});
  Tensor w_up = Tensor::ones({3, 2});
  // w_down: [embed_dim, hidden_dim] = [2, 3]
  Tensor w_down = Tensor::ones({2, 3});

  FeedForward ffn(w_gate, w_up, w_down);

  Tensor x = Tensor::ones({1, 2}); // [1, 2]

  Tensor out = ffn.forward(x);
  EXPECT_EQ(out.size(0), 1);
  EXPECT_EQ(out.size(1), 2);

  // x @ w_gate^T = [1,1] @ [[1,1,1],[1,1,1]]^T ... hmm let me just check shape.
  // gate = silu(x @ w_gate^T), gate is [1, 3]
  // up = x @ w_up^T, up is [1, 3]
  // out = (gate * up) @ w_down^T, out is [1, 2]
  // Just check it doesn't crash and output shape is right.
  EXPECT_TRUE(std::isfinite(out.at({0, 0})));
  EXPECT_TRUE(std::isfinite(out.at({0, 1})));
}

// ============================================================================
// TransformerBlock Tests
// ============================================================================

TEST(TransformerBlock, SmallBlock) {
  // Create a minimal transformer block: embed_dim=4, 1 head, head_dim=4,
  // hidden_dim=8.
  GemmaConfig config;
  config.embed_dim = 4;
  config.num_heads = 1;
  config.num_kv_heads = 1;
  config.head_dim = 4;
  config.hidden_dim = 8;
  config.rms_norm_eps = 1e-6f;
  config.rope_theta = 10000.0f;

  Tensor attn_norm_w = Tensor::ones({4});
  Tensor ffn_norm_w = Tensor::ones({4});
  Tensor wq = Tensor::full({4, 4}, 0.1f); // [q_dim, embed_dim]
  Tensor wk = Tensor::full({4, 4}, 0.1f); // [kv_dim, embed_dim]
  Tensor wv = Tensor::full({4, 4}, 0.1f);
  Tensor wo = Tensor::full({4, 4}, 0.1f);
  Tensor w_gate = Tensor::full({8, 4}, 0.1f);
  Tensor w_up = Tensor::full({8, 4}, 0.1f);
  Tensor w_down = Tensor::full({4, 8}, 0.1f);

  RMSNorm attn_norm(attn_norm_w, config.rms_norm_eps);
  Attention attn(config, wq, wk, wv, wo);
  RMSNorm ffn_norm(ffn_norm_w, config.rms_norm_eps);
  FeedForward ffn(w_gate, w_up, w_down);

  TransformerBlock block(config, std::move(attn_norm), std::move(attn),
                         std::move(ffn_norm), std::move(ffn));

  // Input: [2, 4] (seq_len=2, embed_dim=4)
  Tensor x = Tensor::ones({2, 4});
  Tensor out = block.forward(x, 0);

  EXPECT_EQ(out.size(0), 2);
  EXPECT_EQ(out.size(1), 4);
  // Check values are finite (no NaN/inf).
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 4; ++j) {
      EXPECT_TRUE(std::isfinite(out.at({i, j})))
          << "NaN/inf at (" << i << ", " << j << ")";
    }
  }
}

} // namespace
} // namespace ie
