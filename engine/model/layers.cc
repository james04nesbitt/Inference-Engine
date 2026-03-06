#include "engine/model/layers.h"

#include <cmath>

#include "engine/attention/flash_attention.h"
#include "engine/ops/ops.h"

namespace ie {

// ============================================================================
// RMSNorm::forward
// ============================================================================
//   output = (x / sqrt(mean(x^2) + eps)) * weight
Tensor RMSNorm::forward(const Tensor &x) const {
  // x: [seq_len, dim]
  // Use the vectorized rms_norm op.
  return ops::rms_norm(x, weight_, eps_);
}

// ============================================================================
// FeedForward::forward — SwiGLU
// ============================================================================
//   gate = silu(x @ w_gate^T)
//   up   = x @ w_up^T
//   output = (gate * up) @ w_down^T
Tensor FeedForward::forward(const Tensor &x) const {
  // Transpose weights for matmul: [out, in] -> [in, out]
  Tensor gate_t = w_gate_.transpose(0, 1).contiguous();
  Tensor up_t = w_up_.transpose(0, 1).contiguous();
  Tensor down_t = w_down_.transpose(0, 1).contiguous();

  // gate = silu(x @ w_gate^T)
  Tensor gate = ops::silu(ops::matmul(x, gate_t));

  // up = x @ w_up^T
  Tensor up = ops::matmul(x, up_t);

  // output = (gate * up) @ w_down^T
  return ops::matmul(ops::mul(gate, up), down_t);
}

// ============================================================================
// Attention::forward — with Paged KV Cache
// ============================================================================
Tensor Attention::forward(const Tensor &x, int32_t start_pos,
                          KVCacheManager &kv_cache, int64_t seq_id) const {
  // x: [seq_len, embed_dim]
  int64_t seq_len = x.size(0);
  int32_t num_heads = config_.num_heads;
  int32_t num_kv_heads = config_.num_kv_heads;
  int32_t head_dim = config_.head_dim;
  int32_t num_groups = num_heads / num_kv_heads;

  // --- Step 1: Project to Q, K, V ---
  Tensor wq_t = wq_.transpose(0, 1).contiguous();
  Tensor wk_t = wk_.transpose(0, 1).contiguous();
  Tensor wv_t = wv_.transpose(0, 1).contiguous();
  Tensor wo_t = wo_.transpose(0, 1).contiguous();

  Tensor Q = ops::matmul(x, wq_t);
  Tensor K_new = ops::matmul(x, wk_t);
  Tensor V_new = ops::matmul(x, wv_t);

  // --- Step 2: Reshape into heads ---
  Q = Q.reshape({seq_len, num_heads, head_dim});
  K_new = K_new.reshape({seq_len, num_kv_heads, head_dim});
  V_new = V_new.reshape({seq_len, num_kv_heads, head_dim});

  // --- Step 3: Apply RoPE to Q and K_new ---
  Tensor positions({1, seq_len});
  for (int64_t i = 0; i < seq_len; ++i) {
    positions.set({static_cast<int64_t>(0), i},
                  static_cast<float>(start_pos + i));
  }
  Q = ops::rope(Q.unsqueeze(0), positions, config_.rope_theta).squeeze(0);
  K_new =
      ops::rope(K_new.unsqueeze(0), positions, config_.rope_theta).squeeze(0);

  // --- Step 4: Update paged KV cache ---
  // Append each new token's K/V data to the cache.
  for (int64_t t = 0; t < seq_len; ++t) {
    // Extract K and V for this token: [num_kv_heads, head_dim]
    Tensor k_token = K_new.select(0, t).contiguous();
    Tensor v_token = V_new.select(0, t).contiguous();
    kv_cache.AppendToken(seq_id, layer_idx_, k_token, v_token);
  }

  // Retrieve the full cached K/V history from the page table.
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
  return ops::matmul(concat, wo_t);
}

// ============================================================================
// TransformerBlock::forward
// ============================================================================
Tensor TransformerBlock::forward(const Tensor &x, int32_t start_pos,
                                 KVCacheManager &kv_cache,
                                 int64_t seq_id) const {
  // Pre-norm attention with residual connection.
  Tensor attn_out =
      attn_.forward(attn_norm_.forward(x), start_pos, kv_cache, seq_id);
  Tensor residual1 = ops::add(x, attn_out);

  // Pre-norm FFN with residual connection.
  Tensor ffn_out = ffn_.forward(ffn_norm_.forward(residual1));
  return ops::add(residual1, ffn_out);
}

// ============================================================================
// GemmaModel constructor — creates paged KV cache
// ============================================================================
GemmaModel::GemmaModel(GemmaConfig config, Tensor token_embedding,
                       std::vector<TransformerBlock> layers, RMSNorm final_norm)
    : config_(std::move(config)), token_embedding_(std::move(token_embedding)),
      layers_(std::move(layers)), final_norm_(std::move(final_norm)) {
  // Compute max blocks needed: max_seq_len / block_size, per layer.
  // Add extra blocks for headroom.
  int32_t blocks_per_seq =
      (config_.max_seq_len + kDefaultBlockSize - 1) / kDefaultBlockSize;
  int32_t max_blocks = blocks_per_seq * config_.num_layers + config_.num_layers;

  kv_cache_ = std::make_unique<KVCacheManager>(
      config_.num_layers, config_.num_kv_heads, config_.head_dim, max_blocks,
      kDefaultBlockSize);

  // Allocate an initial sequence.
  seq_id_ = kv_cache_->AllocateSequence();
}

// ============================================================================
// GemmaModel::forward
// ============================================================================
Tensor GemmaModel::forward(const Tensor &tokens, int32_t start_pos) const {
  // Step 1: Token embedding lookup.
  Tensor x = ops::embedding(token_embedding_, tokens);

  // Step 2: Gemma-specific embedding scaling.
  float scale = std::sqrt(static_cast<float>(config_.embed_dim));
  float *x_ptr = x.data<float>();
  for (int64_t i = 0; i < x.numel(); ++i) {
    x_ptr[i] *= scale;
  }

  // Step 3: Run through transformer blocks with paged KV cache.
  for (auto &block : layers_) {
    x = block.forward(x, start_pos, *kv_cache_, seq_id_);
  }

  // Step 4: Final normalization.
  x = final_norm_.forward(x);

  // Step 5: Compute logits via weight tying.
  int64_t seq_len = x.size(0);
  Tensor last_hidden = x.select(0, seq_len - 1).contiguous();
  last_hidden =
      last_hidden.reshape({1, static_cast<int64_t>(config_.embed_dim)});
  Tensor embed_t = token_embedding_.transpose(0, 1).contiguous();
  Tensor logits = ops::matmul(last_hidden, embed_t);
  return logits.reshape({static_cast<int64_t>(config_.vocab_size)});
}

// ============================================================================
// GemmaModel::ClearCache
// ============================================================================
void GemmaModel::ClearCache() const {
  if (kv_cache_ && seq_id_ >= 0) {
    kv_cache_->FreeSequence(seq_id_);
    seq_id_ = kv_cache_->AllocateSequence();
  }
}

} // namespace ie
