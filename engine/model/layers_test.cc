#include "engine/model/layers.h"

#include <cmath>
#include <vector>

#include "engine/attention/kv_cache.h"
#include "engine/ops/ops.h"
#include "engine/tensor/tensor.h"
#include "gtest/gtest.h"

namespace ie {
namespace {

// ============================================================================
// RMSNorm Tests
// ============================================================================

TEST(RMSNorm, BasicNormalization) {
  Tensor weight = Tensor::ones({4});
  RMSNorm norm(weight, 1e-6f);

  Tensor x = Tensor::from_vector({3.0f, 4.0f, 0.0f, 0.0f});
  x = x.reshape({1, 4});

  Tensor out = norm.forward(x);
  EXPECT_EQ(out.size(0), 1);
  EXPECT_EQ(out.size(1), 4);

  // With (1+weight) formulation and weight=ones, weight factor is (1+1)=2.
  float rms = std::sqrt(6.25f);
  EXPECT_NEAR(out.at({0, 0}), 3.0f / rms * 2.0f, 1e-4f);
  EXPECT_NEAR(out.at({0, 1}), 4.0f / rms * 2.0f, 1e-4f);
  EXPECT_NEAR(out.at({0, 2}), 0.0f, 1e-4f);
  EXPECT_NEAR(out.at({0, 3}), 0.0f, 1e-4f);
}

TEST(RMSNorm, WithWeightScaling) {
  Tensor weight = Tensor::from_vector({2.0f, 0.5f});
  RMSNorm norm(weight, 1e-6f);

  Tensor x = Tensor::from_vector({1.0f, 1.0f});
  x = x.reshape({1, 2});

  Tensor out = norm.forward(x);

  // With (1+weight), effective weights are (1+2.0)=3.0 and (1+0.5)=1.5.
  // Input [1, 1] has RMS=1, so normalized = [1, 1], then * [3.0, 1.5].
  EXPECT_NEAR(out.at({0, 0}), 3.0f, 1e-4f);
  EXPECT_NEAR(out.at({0, 1}), 1.5f, 1e-4f);
}

// ============================================================================
// FeedForward Tests
// ============================================================================

TEST(FeedForward, SmallSwiGLU) {
  // Constructors now expect pre-transposed weights: [out, in] → [in, out]
  Tensor gate_t = Tensor::ones({3, 2}).transpose(0, 1).contiguous();
  Tensor up_t = Tensor::ones({3, 2}).transpose(0, 1).contiguous();
  Tensor down_t = Tensor::ones({2, 3}).transpose(0, 1).contiguous();

  FeedForward ffn(gate_t, up_t, down_t);

  Tensor x = Tensor::ones({1, 2});

  Tensor out = ffn.forward(x);
  EXPECT_EQ(out.size(0), 1);
  EXPECT_EQ(out.size(1), 2);
  EXPECT_TRUE(std::isfinite(out.at({0, 0})));
  EXPECT_TRUE(std::isfinite(out.at({0, 1})));
}

// ============================================================================
// TransformerBlock Tests (with externally-provided KV cache)
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
  config.rope_theta_local = 10000.0f;
  config.rope_theta_global = 1000000.0f;

  Tensor attn_norm_w = Tensor::ones({4});
  Tensor ffn_norm_w = Tensor::ones({4});
  // Pre-transpose weights: constructors now expect [in, out] layout.
  Tensor wq_t = Tensor::full({4, 4}, 0.1f).transpose(0, 1).contiguous();
  Tensor wk_t = Tensor::full({4, 4}, 0.1f).transpose(0, 1).contiguous();
  Tensor wv_t = Tensor::full({4, 4}, 0.1f).transpose(0, 1).contiguous();
  Tensor wo_t = Tensor::full({4, 4}, 0.1f).transpose(0, 1).contiguous();
  Tensor gate_t = Tensor::full({8, 4}, 0.1f).transpose(0, 1).contiguous();
  Tensor up_t = Tensor::full({8, 4}, 0.1f).transpose(0, 1).contiguous();
  Tensor down_t = Tensor::full({4, 8}, 0.1f).transpose(0, 1).contiguous();

  RMSNorm attn_norm(attn_norm_w, config.rms_norm_eps);
  Attention attn(config, /*layer_idx=*/0, wq_t, wk_t, wv_t, wo_t);
  RMSNorm post_attn_norm(Tensor::zeros({4}), config.rms_norm_eps);
  RMSNorm ffn_norm(ffn_norm_w, config.rms_norm_eps);
  FeedForward ffn(gate_t, up_t, down_t);
  RMSNorm post_ffn_norm(Tensor::zeros({4}), config.rms_norm_eps);

  TransformerBlock block(config, /*layer_idx=*/0, std::move(attn_norm),
                         std::move(attn), std::move(post_attn_norm),
                         std::move(ffn_norm), std::move(ffn),
                         std::move(post_ffn_norm));

  // Create external KV cache.
  KVCacheManager kv_cache(/*num_layers=*/1, /*num_kv_heads=*/1,
                          /*head_dim=*/4, /*max_blocks=*/20,
                          /*block_size=*/16);
  int64_t seq_id = kv_cache.AllocateSequence();

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
