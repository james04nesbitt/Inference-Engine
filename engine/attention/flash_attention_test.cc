#include "engine/attention/flash_attention.h"

#include <cmath>
#include <vector>

#include "engine/ops/ops.h"
#include "engine/tensor/tensor.h"
#include "gtest/gtest.h"

namespace ie {
namespace {

// Helper: compute standard attention for reference.
// query: [seq_q, num_heads, head_dim]
// key:   [seq_kv, num_heads, head_dim]  (already GQA-expanded)
// value: [seq_kv, num_heads, head_dim]
Tensor reference_attention(const Tensor &query, const Tensor &key,
                           const Tensor &value, float scale, bool causal) {
  int64_t seq_q = query.size(0);
  int64_t num_heads = query.size(1);
  int64_t head_dim = query.size(2);
  int64_t seq_kv = key.size(0);

  Tensor out({seq_q, num_heads, head_dim});

  for (int64_t h = 0; h < num_heads; ++h) {
    // Extract per-head slices.
    Tensor Q_h = query.select(1, h).contiguous(); // [seq_q, head_dim]
    Tensor K_h = key.select(1, h).contiguous();   // [seq_kv, head_dim]
    Tensor V_h = value.select(1, h).contiguous(); // [seq_kv, head_dim]

    // Scores = Q @ K^T * scale -> [seq_q, seq_kv]
    Tensor K_t = K_h.transpose(0, 1).contiguous();
    Tensor scores = ops::matmul(Q_h, K_t);
    float *s = scores.data<float>();
    for (int64_t i = 0; i < scores.numel(); ++i)
      s[i] *= scale;

    // Causal mask.
    if (causal) {
      for (int64_t qi = 0; qi < seq_q; ++qi) {
        for (int64_t ki = qi + 1; ki < seq_kv; ++ki) {
          scores.set({qi, ki}, -INFINITY);
        }
      }
    }

    // Softmax.
    Tensor weights = ops::softmax(scores);

    // Output = weights @ V -> [seq_q, head_dim]
    Tensor out_h = ops::matmul(weights, V_h);

    // Copy into output.
    const float *src = out_h.data<float>();
    for (int64_t qi = 0; qi < seq_q; ++qi) {
      for (int64_t d = 0; d < head_dim; ++d) {
        out.set({qi, h, d}, src[qi * head_dim + d]);
      }
    }
  }

  return out;
}

TEST(FlashAttention, MatchesStandard) {
  // Small test: seq_len=4, 2 heads, head_dim=4
  int64_t seq = 4, heads = 2, dim = 4;

  Tensor Q({seq, heads, dim});
  Tensor K({seq, heads, dim});
  Tensor V({seq, heads, dim});

  // Fill with deterministic values.
  float val = 0.1f;
  for (int64_t s = 0; s < seq; ++s) {
    for (int64_t h = 0; h < heads; ++h) {
      for (int64_t d = 0; d < dim; ++d) {
        Q.set({s, h, d}, val);
        val += 0.03f;
        K.set({s, h, d}, val);
        val += 0.02f;
        V.set({s, h, d}, val);
        val += 0.01f;
      }
    }
  }

  float scale = 1.0f / std::sqrt(static_cast<float>(dim));

  Tensor flash_out = flash_attention(Q, K, V, scale, /*causal=*/true);
  Tensor ref_out = reference_attention(Q, K, V, scale, /*causal=*/true);

  EXPECT_EQ(flash_out.size(0), seq);
  EXPECT_EQ(flash_out.size(1), heads);
  EXPECT_EQ(flash_out.size(2), dim);

  // Compare outputs — should be numerically very close.
  for (int64_t s = 0; s < seq; ++s) {
    for (int64_t h = 0; h < heads; ++h) {
      for (int64_t d = 0; d < dim; ++d) {
        EXPECT_NEAR(flash_out.at({s, h, d}), ref_out.at({s, h, d}), 1e-4f)
            << "Mismatch at [" << s << "," << h << "," << d << "]";
      }
    }
  }
}

TEST(FlashAttention, NoCausal) {
  int64_t seq = 3, heads = 1, dim = 4;
  Tensor Q({seq, heads, dim});
  Tensor K({seq, heads, dim});
  Tensor V({seq, heads, dim});

  float val = 0.5f;
  for (int64_t s = 0; s < seq; ++s) {
    for (int64_t d = 0; d < dim; ++d) {
      Q.set({s, static_cast<int64_t>(0), d}, val);
      val += 0.1f;
      K.set({s, static_cast<int64_t>(0), d}, val);
      val += 0.1f;
      V.set({s, static_cast<int64_t>(0), d}, val);
      val += 0.1f;
    }
  }

  float scale = 1.0f / std::sqrt(static_cast<float>(dim));
  Tensor flash_out = flash_attention(Q, K, V, scale, /*causal=*/false);
  Tensor ref_out = reference_attention(Q, K, V, scale, /*causal=*/false);

  for (int64_t s = 0; s < seq; ++s) {
    for (int64_t d = 0; d < dim; ++d) {
      EXPECT_NEAR(flash_out.at({s, static_cast<int64_t>(0), d}),
                  ref_out.at({s, static_cast<int64_t>(0), d}), 1e-4f);
    }
  }
}

TEST(FlashAttention, GQA) {
  // GQA: 4 query heads, 2 KV heads (groups=2)
  int64_t seq = 3, num_heads = 4, num_kv_heads = 2, dim = 4;

  Tensor Q({seq, num_heads, dim});
  Tensor K({seq, num_kv_heads, dim});
  Tensor V({seq, num_kv_heads, dim});

  // Fill with simple values.
  Q.fill(0.5f);
  K.fill(0.3f);
  V.fill(0.7f);

  float scale = 1.0f / std::sqrt(static_cast<float>(dim));
  Tensor out = flash_attention(Q, K, V, scale, true);

  EXPECT_EQ(out.size(0), seq);
  EXPECT_EQ(out.size(1), num_heads);
  EXPECT_EQ(out.size(2), dim);

  // All outputs should be finite.
  for (int64_t s = 0; s < seq; ++s) {
    for (int64_t h = 0; h < num_heads; ++h) {
      for (int64_t d = 0; d < dim; ++d) {
        EXPECT_TRUE(std::isfinite(out.at({s, h, d})));
      }
    }
  }
}

} // namespace
} // namespace ie
