#pragma once

#include <memory>
#include <vector>

#include "engine/model/config.h"
#include "engine/tensor/tensor.h"

namespace ie {

// ============================================================================
// Transformer Layers — YOUR CORE ML LEARNING OBJECTIVE
//
// Each layer class stores references to its weight tensors (loaded from GGUF)
// and implements a forward() method that performs the computation.
//
// Implementation order suggestion:
//   1. RMSNorm (simplest, good warmup)
//   2. FeedForward (matmul + activation)
//   3. Attention (the main event — Q/K/V projections, scaled dot-product)
//   4. TransformerBlock (combines the above)
//   5. GemmaModel (embeddings + blocks + final norm)
// ============================================================================

// --- RMS Layer Normalization ---
class RMSNorm {
 public:
  RMSNorm(Tensor weight, float eps) : weight_(std::move(weight)), eps_(eps) {}

  // TODO: Implement
  //   output = (x / sqrt(mean(x^2) + eps)) * weight
  Tensor forward(const Tensor& x) const;

 private:
  Tensor weight_;
  float eps_;
};

// --- Multi-Head Attention with Grouped Query Attention (GQA) ---
class Attention {
 public:
  Attention(const GemmaConfig& config, Tensor wq, Tensor wk, Tensor wv,
            Tensor wo)
      : config_(config),
        wq_(std::move(wq)),
        wk_(std::move(wk)),
        wv_(std::move(wv)),
        wo_(std::move(wo)) {}

  // TODO: Implement attention forward pass
  //
  // Steps:
  //   1. Project input to Q, K, V using weight matrices
  //      Q = x @ wq  [batch, seq, num_heads * head_dim]
  //      K = x @ wk  [batch, seq, num_kv_heads * head_dim]
  //      V = x @ wv  [batch, seq, num_kv_heads * head_dim]
  //
  //   2. Reshape into heads
  //      Q = Q.view(batch, seq, num_heads, head_dim)
  //      K = K.view(batch, seq, num_kv_heads, head_dim)
  //
  //   3. Apply RoPE to Q and K
  //
  //   4. If using GQA (num_kv_heads < num_heads):
  //      Repeat K,V to match num_heads
  //
  //   5. Compute attention scores
  //      scores = (Q @ K^T) / sqrt(head_dim)
  //
  //   6. Apply causal mask (upper triangle = -inf)
  //
  //   7. Softmax over last dimension
  //
  //   8. Attention output = scores @ V
  //
  //   9. Concatenate heads and project
  //      output = concat @ wo
  //
  // Advanced: Add KV-cache support for efficient autoregressive generation
  Tensor forward(const Tensor& x, int32_t start_pos) const;

 private:
  GemmaConfig config_;
  Tensor wq_, wk_, wv_, wo_;
};

// --- Feed-Forward Network (SwiGLU variant used by Gemma) ---
class FeedForward {
 public:
  FeedForward(Tensor w_gate, Tensor w_up, Tensor w_down)
      : w_gate_(std::move(w_gate)),
        w_up_(std::move(w_up)),
        w_down_(std::move(w_down)) {}

  // TODO: Implement SwiGLU feed-forward
  //   gate = silu(x @ w_gate)
  //   up   = x @ w_up
  //   output = (gate * up) @ w_down
  Tensor forward(const Tensor& x) const;

 private:
  Tensor w_gate_, w_up_, w_down_;
};

// --- Single Transformer Block ---
class TransformerBlock {
 public:
  TransformerBlock(const GemmaConfig& config, RMSNorm attn_norm,
                   Attention attn, RMSNorm ffn_norm, FeedForward ffn)
      : config_(config),
        attn_norm_(std::move(attn_norm)),
        attn_(std::move(attn)),
        ffn_norm_(std::move(ffn_norm)),
        ffn_(std::move(ffn)) {}

  // TODO: Implement
  //   x = x + attn(attn_norm(x))     // Attention with residual
  //   x = x + ffn(ffn_norm(x))       // FFN with residual
  Tensor forward(const Tensor& x, int32_t start_pos) const;

 private:
  GemmaConfig config_;
  RMSNorm attn_norm_;
  Attention attn_;
  RMSNorm ffn_norm_;
  FeedForward ffn_;
};

// --- Full Gemma Model ---
class GemmaModel {
 public:
  GemmaModel(GemmaConfig config, Tensor token_embedding,
             std::vector<TransformerBlock> layers, RMSNorm final_norm)
      : config_(std::move(config)),
        token_embedding_(std::move(token_embedding)),
        layers_(std::move(layers)),
        final_norm_(std::move(final_norm)) {}

  // TODO: Implement the full forward pass
  //   1. Embed tokens: x = embedding_table[token_ids]
  //   2. Scale embeddings: x = x * sqrt(embed_dim)  (Gemma-specific)
  //   3. For each transformer block: x = block.forward(x, start_pos)
  //   4. Final norm: x = final_norm(x)
  //   5. Compute logits: logits = x @ embedding_table^T  (weight tying)
  //   6. Return logits for the last token position
  Tensor forward(const Tensor& tokens, int32_t start_pos) const;

  const GemmaConfig& config() const { return config_; }

 private:
  GemmaConfig config_;
  Tensor token_embedding_;
  std::vector<TransformerBlock> layers_;
  RMSNorm final_norm_;
};

}  // namespace ie
