#pragma once

#include <memory>
#include <vector>

#include "engine/attention/kv_cache.h"
#include "engine/model/config.h"
#include "engine/tensor/tensor.h"

namespace ie {

// ============================================================================
// Transformer Layers
// ============================================================================

// --- RMS Layer Normalization ---
class RMSNorm {
public:
  RMSNorm(Tensor weight, float eps) : weight_(std::move(weight)), eps_(eps) {}

  Tensor forward(const Tensor &x) const;

private:
  Tensor weight_;
  float eps_;
};

// --- Multi-Head Attention with Grouped Query Attention (GQA) ---
class Attention {
public:
  Attention(const GemmaConfig &config, int32_t layer_idx, Tensor wq, Tensor wk,
            Tensor wv, Tensor wo)
      : config_(config), layer_idx_(layer_idx),
        wq_t_(wq.transpose(0, 1).contiguous()),
        wk_t_(wk.transpose(0, 1).contiguous()),
        wv_t_(wv.transpose(0, 1).contiguous()),
        wo_t_(wo.transpose(0, 1).contiguous()) {}

  Tensor forward(const Tensor &x, int32_t start_pos, KVCacheManager &kv_cache,
                 int64_t seq_id) const;

private:
  GemmaConfig config_;
  int32_t layer_idx_;
  Tensor wq_t_, wk_t_, wv_t_, wo_t_;
};

// --- Feed-Forward Network (SwiGLU variant) ---
class FeedForward {
public:
  FeedForward(Tensor w_gate, Tensor w_up, Tensor w_down)
      : gate_t_(w_gate.transpose(0, 1).contiguous()),
        up_t_(w_up.transpose(0, 1).contiguous()),
        down_t_(w_down.transpose(0, 1).contiguous()) {}

  Tensor forward(const Tensor &x) const;

private:
  Tensor gate_t_, up_t_, down_t_;
};

// --- Single Transformer Block ---
class TransformerBlock {
public:
  TransformerBlock(const GemmaConfig &config, int32_t layer_idx,
                   RMSNorm attn_norm, Attention attn, RMSNorm ffn_norm,
                   FeedForward ffn)
      : config_(config), layer_idx_(layer_idx),
        attn_norm_(std::move(attn_norm)), attn_(std::move(attn)),
        ffn_norm_(std::move(ffn_norm)), ffn_(std::move(ffn)) {}

  Tensor forward(const Tensor &x, int32_t start_pos, KVCacheManager &kv_cache,
                 int64_t seq_id) const;

private:
  GemmaConfig config_;
  int32_t layer_idx_;
  RMSNorm attn_norm_;
  Attention attn_;
  RMSNorm ffn_norm_;
  FeedForward ffn_;
};

// --- Full Gemma Model ---
// KV cache is externally owned — the caller (engine/scheduler) manages it.
class GemmaModel {
public:
  GemmaModel(GemmaConfig config, Tensor token_embedding,
             std::vector<TransformerBlock> layers, RMSNorm final_norm)
      : config_(std::move(config)),
        token_embedding_(std::move(token_embedding)),
        layers_(std::move(layers)), final_norm_(std::move(final_norm)),
        embed_t_(token_embedding_.transpose(0, 1).contiguous()) {}

  // Forward pass with externally-provided KV cache.
  Tensor forward(const Tensor &tokens, int32_t start_pos,
                 KVCacheManager &kv_cache, int64_t seq_id) const;

  const GemmaConfig &config() const { return config_; }

private:
  GemmaConfig config_;
  Tensor token_embedding_;
  std::vector<TransformerBlock> layers_;
  RMSNorm final_norm_;
  Tensor embed_t_;
};

} // namespace ie
