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
  // Constructor accepting pre-transposed weight matrices.
  // Weights should already be in [out_features, in_features] → [in, out]
  // transposed layout for efficient matmul.
  Attention(const GemmaConfig &config, int32_t layer_idx, Tensor wq_t,
            Tensor wk_t, Tensor wv_t, Tensor wo_t,
            Tensor q_norm_w = Tensor(), Tensor k_norm_w = Tensor())
      : config_(config), layer_idx_(layer_idx), wq_t_(std::move(wq_t)),
        wk_t_(std::move(wk_t)), wv_t_(std::move(wv_t)),
        wo_t_(std::move(wo_t)) {
    if (q_norm_w.ndim() > 0) {
      q_norm_ = std::make_unique<RMSNorm>(std::move(q_norm_w), config_.rms_norm_eps);
    }
    if (k_norm_w.ndim() > 0) {
      k_norm_ = std::make_unique<RMSNorm>(std::move(k_norm_w), config_.rms_norm_eps);
    }
  }

  Tensor forward(const Tensor &x, int32_t start_pos, KVCacheManager &kv_cache,
                 int64_t seq_id) const;

private:
  GemmaConfig config_;
  int32_t layer_idx_;
  Tensor wq_t_, wk_t_, wv_t_, wo_t_;
  std::unique_ptr<RMSNorm> q_norm_;
  std::unique_ptr<RMSNorm> k_norm_;
};

// --- Feed-Forward Network (SwiGLU variant) ---
class FeedForward {
public:
  // Constructor accepting pre-transposed weight matrices.
  FeedForward(Tensor gate_t, Tensor up_t, Tensor down_t)
      : gate_t_(std::move(gate_t)), up_t_(std::move(up_t)),
        down_t_(std::move(down_t)) {}

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
  // token_embedding: the raw embedding table [vocab_size, embed_dim]
  // embed_t: pre-transposed embedding table [embed_dim, vocab_size] for logit
  // projection
  GemmaModel(GemmaConfig config, Tensor token_embedding,
             std::vector<TransformerBlock> layers, RMSNorm final_norm,
             Tensor embed_t)
      : config_(std::move(config)),
        token_embedding_(std::move(token_embedding)),
        layers_(std::move(layers)), final_norm_(std::move(final_norm)),
        embed_t_(std::move(embed_t)) {}

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
