#include "engine/model/layers.h"

#include <cmath>

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
// FeedForward::forward — SwiGLU
// ============================================================================
Tensor FeedForward::forward(const Tensor &x) const {
  Tensor gate = ops::silu(ops::matmul(x, gate_t_));
  Tensor up = ops::matmul(x, up_t_);
  return ops::matmul(ops::mul(gate, up), down_t_);
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

  // --- Step 3: Apply RoPE ---
  Tensor positions({1, seq_len});
  for (int64_t i = 0; i < seq_len; ++i) {
    positions.set({static_cast<int64_t>(0), i},
                  static_cast<float>(start_pos + i));
  }
  Q = ops::rope(Q.unsqueeze(0), positions, config_.rope_theta).squeeze(0);
  K_new =
      ops::rope(K_new.unsqueeze(0), positions, config_.rope_theta).squeeze(0);

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

  // --- Step 6: FlashAttention ---
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  Tensor output = flash_attention(Q, K, V, scale, /*causal=*/true);

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

  for (auto &block : layers_) {
    x = block.forward(x, start_pos, kv_cache, seq_id);
  }

  x = final_norm_.forward(x);

  int64_t seq_len = x.size(0);
  Tensor last_hidden = x.select(0, seq_len - 1).contiguous();
  last_hidden =
      last_hidden.reshape({1, static_cast<int64_t>(config_.embed_dim)});
  Tensor logits = ops::matmul(last_hidden, embed_t_);
  return logits.reshape({static_cast<int64_t>(config_.vocab_size)});
}

} // namespace ie
