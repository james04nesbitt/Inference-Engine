#include "engine/model/layers.h"

#include <cmath>
#include <iostream>

#include "engine/attention/flash_attention.h"
#include "engine/ops/ops.h"

namespace ie {

// ============================================================================
// RMSNorm::forward
// ============================================================================
Tensor RMSNorm::forward(const Tensor &x) const {
  return ops::rms_norm(x, weight_, eps_);
}

// ============================================================================
// FeedForward::forward — GeGLU (used in Gemma models)
// ============================================================================
Tensor FeedForward::forward(const Tensor &x) const {
  Tensor gate = ops::silu(ops::matmul(x, gate_t_));
  Tensor up = ops::matmul(x, up_t_);
  return ops::matmul(ops::mul(gate, up), down_t_);
}

// ============================================================================
// Naive Scaled Dot-Product Attention
// ============================================================================
// Replaces FlashAttention for correctness debugging.
// No tiling, no online softmax — just straightforward Q·K^T → mask → softmax → ·V.
// q_offset is the absolute position of the first query token (start_pos),
// needed so the causal mask works correctly during autoregressive decode.
static Tensor naive_sdpa(const Tensor &Q, const Tensor &K, const Tensor &V,
                         float scale, bool causal, int32_t q_offset) {
  const int64_t seq_q = Q.size(0);
  const int64_t num_heads = Q.size(1);
  const int64_t head_dim = Q.size(2);
  const int64_t seq_kv = K.size(0);

  Tensor Q_c = Q.contiguous();
  Tensor K_c = K.contiguous();
  Tensor V_c = V.contiguous();

  const float *q_data = Q_c.data<float>();
  const float *k_data = K_c.data<float>();
  const float *v_data = V_c.data<float>();

  Tensor output = Tensor::zeros({seq_q, num_heads, head_dim});
  float *o_data = output.data<float>();

  for (int64_t h = 0; h < num_heads; ++h) {
    for (int64_t qi = 0; qi < seq_q; ++qi) {
      const float *q_row =
          q_data + qi * num_heads * head_dim + h * head_dim;

      // Compute attention scores: Q[qi,h,:] dot K[ki,h,:] * scale
      std::vector<float> scores(seq_kv);
      for (int64_t ki = 0; ki < seq_kv; ++ki) {
        const float *k_row =
            k_data + ki * num_heads * head_dim + h * head_dim;
        float dot = 0.0f;
        for (int64_t d = 0; d < head_dim; ++d) {
          dot += q_row[d] * k_row[d];
        }
        scores[ki] = dot * scale;
      }

      // Causal mask using absolute positions.
      // During decode: q_offset = start_pos, qi = 0, so q_abs = start_pos.
      // The query can attend to KV positions 0..start_pos (all past tokens).
      if (causal) {
        int64_t q_abs = static_cast<int64_t>(q_offset) + qi;
        for (int64_t ki = 0; ki < seq_kv; ++ki) {
          if (ki > q_abs) {
            scores[ki] = -INFINITY;
          }
        }
      }

      // Softmax
      float max_val = scores[0];
      for (int64_t ki = 1; ki < seq_kv; ++ki) {
        if (scores[ki] > max_val) max_val = scores[ki];
      }
      float sum = 0.0f;
      for (int64_t ki = 0; ki < seq_kv; ++ki) {
        scores[ki] = std::exp(scores[ki] - max_val);
        sum += scores[ki];
      }
      if (sum > 0.0f) {
        for (int64_t ki = 0; ki < seq_kv; ++ki) {
          scores[ki] /= sum;
        }
      }

      // Weighted sum of values
      float *o_row = o_data + qi * num_heads * head_dim + h * head_dim;
      for (int64_t ki = 0; ki < seq_kv; ++ki) {
        const float *v_row =
            v_data + ki * num_heads * head_dim + h * head_dim;
        for (int64_t d = 0; d < head_dim; ++d) {
          o_row[d] += scores[ki] * v_row[d];
        }
      }
    }
  }

  return output;
}

// ============================================================================
// Attention::forward — with Paged KV Cache
// ============================================================================
Tensor Attention::forward(const Tensor &x, int32_t start_pos,
                          KVCacheManager &kv_cache, int64_t seq_id) const {
  int64_t seq_len = x.size(0);
  int32_t num_heads = config_.num_heads;
  int32_t num_kv_heads = config_.num_kv_heads;
  int32_t head_dim = config_.head_dim;
  int32_t num_groups = num_heads / num_kv_heads;

  // --- Step 1: Project to Q, K, V ---
  Tensor Q = ops::matmul(x, wq_t_);
  Tensor K_new = ops::matmul(x, wk_t_);
  Tensor V_new = ops::matmul(x, wv_t_);

  // --- Step 2: Reshape into heads ---
  Q = Q.reshape({seq_len, num_heads, head_dim});
  K_new = K_new.reshape({seq_len, num_kv_heads, head_dim});
  V_new = V_new.reshape({seq_len, num_kv_heads, head_dim});

  // --- Step 2.5: Apply QK Normalization (for Gemma 2/3) ---
  if (q_norm_) {
    Q = q_norm_->forward(Q);
  }
  if (k_norm_) {
    K_new = k_norm_->forward(K_new);
  }

  // --- Step 3: Apply RoPE ---
  Tensor positions({1, seq_len});
  for (int64_t i = 0; i < seq_len; ++i) {
    positions.set({static_cast<int64_t>(0), i},
                  static_cast<float>(start_pos + i));
  }
  float layer_theta = config_.rope_theta_for_layer(layer_idx_);
  Q = ops::rope(Q.unsqueeze(0), positions, layer_theta).squeeze(0);
  K_new =
      ops::rope(K_new.unsqueeze(0), positions, layer_theta).squeeze(0);



  // --- Step 4: Update paged KV cache ---
  for (int64_t t = 0; t < seq_len; ++t) {
    Tensor k_token = K_new.select(0, t).contiguous();
    Tensor v_token = V_new.select(0, t).contiguous();
    kv_cache.AppendToken(seq_id, layer_idx_, k_token, v_token);
  }

  Tensor K = kv_cache.GetKeys(seq_id, layer_idx_);
  Tensor V = kv_cache.GetValues(seq_id, layer_idx_);

  // --- Step 5: GQA expansion ---
  if (num_groups > 1) {
    K = K.repeat(1, num_groups);
    V = V.repeat(1, num_groups);
  }

  // --- Step 6: Naive SDPA (replaces FlashAttention for correctness debugging) ---
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  Tensor output = naive_sdpa(Q, K, V, scale, /*causal=*/true, start_pos);

  // --- Step 7: Concatenate heads and project ---
  Tensor concat =
      output.reshape({seq_len, static_cast<int64_t>(num_heads * head_dim)});
  return ops::matmul(concat, wo_t_);
}

// ============================================================================
// TransformerBlock::forward
// ============================================================================
Tensor TransformerBlock::forward(const Tensor &x, int32_t start_pos,
                                 KVCacheManager &kv_cache,
                                 int64_t seq_id) const {
  Tensor attn_out =
      attn_.forward(attn_norm_.forward(x), start_pos, kv_cache, seq_id);
  Tensor residual1 = ops::add(x, attn_out);

  Tensor ffn_out = ffn_.forward(ffn_norm_.forward(residual1));
  return ops::add(residual1, ffn_out);
}

// ============================================================================
// GemmaModel::forward — with externally-provided KV cache
// ============================================================================
Tensor GemmaModel::forward(const Tensor &tokens, int32_t start_pos,
                           KVCacheManager &kv_cache, int64_t seq_id) const {
  Tensor x = ops::embedding(token_embedding_, tokens);

  float scale = std::sqrt(static_cast<float>(config_.embed_dim));
  float *x_ptr = x.data<float>();
  for (int64_t i = 0; i < x.numel(); ++i) {
    x_ptr[i] *= scale;
  }

  // DEBUG: Check embedding output
  {
    const float *d = x.data<float>();
    float sum = 0, sum2 = 0;
    for (int64_t i = 0; i < x.numel(); ++i) { sum += d[i]; sum2 += d[i]*d[i]; }
    float mean = sum / x.numel();
    float rms = std::sqrt(sum2 / x.numel());
    std::cerr << "[DBG] After embed+scale: mean=" << mean << " rms=" << rms
              << " shape=[" << x.size(0) << "," << x.size(1) << "]" << std::endl;
  }

  int layer_idx = 0;
  for (auto &block : layers_) {
    x = block.forward(x, start_pos, kv_cache, seq_id);
    if (layer_idx == 0 || layer_idx == 25) {
      const float *d = x.data<float>();
      float sum = 0, sum2 = 0;
      for (int64_t i = 0; i < x.numel(); ++i) { sum += d[i]; sum2 += d[i]*d[i]; }
      float mean = sum / x.numel();
      float rms = std::sqrt(sum2 / x.numel());
      std::cerr << "[DBG] After layer " << layer_idx << ": mean=" << mean
                << " rms=" << rms << std::endl;
    }
    layer_idx++;
  }

  x = final_norm_.forward(x);

  int64_t seq_len = x.size(0);
  Tensor last_hidden = x.select(0, seq_len - 1).contiguous();
  last_hidden =
      last_hidden.reshape({1, static_cast<int64_t>(config_.embed_dim)});
  Tensor logits = ops::matmul(last_hidden, embed_t_);
  Tensor logits_flat = logits.reshape({static_cast<int64_t>(config_.vocab_size)});

  // DEBUG: Print logits stats and top-5 tokens
  {
    const float *ld = logits_flat.data<float>();
    int64_t n = logits_flat.numel();
    float lmin = ld[0], lmax = ld[0], lsum = 0;
    for (int64_t i = 0; i < n; ++i) {
      if (ld[i] < lmin) lmin = ld[i];
      if (ld[i] > lmax) lmax = ld[i];
      lsum += ld[i];
    }
    std::cerr << "[DBG] Logits: min=" << lmin << " max=" << lmax
              << " mean=" << lsum/n << std::endl;

    // Top-5
    std::vector<std::pair<float, int32_t>> top(5, {-1e30f, -1});
    for (int64_t i = 0; i < n; ++i) {
      if (ld[i] > top[4].first) {
        top[4] = {ld[i], static_cast<int32_t>(i)};
        for (int k = 3; k >= 0; --k) {
          if (top[k+1].first > top[k].first) std::swap(top[k], top[k+1]);
        }
      }
    }
    std::cerr << "[DBG] Top-5 logits:";
    for (auto &[val, idx] : top) {
      std::cerr << " [" << idx << "]=" << val;
    }
    std::cerr << std::endl;
  }

  return logits_flat;
}

} // namespace ie
