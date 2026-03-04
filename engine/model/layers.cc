#include "engine/model/layers.h"

#include <cmath>

#include "engine/ops/ops.h"

namespace ie {

// ============================================================================
// RMSNorm::forward
// ============================================================================
// Delegates to the already-implemented ops::rms_norm kernel.
// output = (x / sqrt(mean(x^2) + eps)) * weight
Tensor RMSNorm::forward(const Tensor &x) const {
  return ops::rms_norm(x, weight_, eps_);
}

// ============================================================================
// FeedForward::forward — SwiGLU variant
// ============================================================================
//   gate = silu(x @ w_gate^T)
//   up   = x @ w_up^T
//   output = (gate * up) @ w_down^T
//
// Weight matrices have shape [hidden_dim, embed_dim] (GGUF stores them
// transposed), so we need x @ W^T = matmul(x, transpose(W)).
// The ops::matmul handles [seq, embed] @ [embed, hidden] = [seq, hidden],
// so we transpose the weight matrices before multiplication.
Tensor FeedForward::forward(const Tensor &x) const {
  // x: [seq_len, embed_dim]
  // w_gate_: [hidden_dim, embed_dim] -> transpose -> [embed_dim, hidden_dim]
  Tensor w_gate_t = w_gate_.transpose(0, 1).contiguous();
  Tensor w_up_t = w_up_.transpose(0, 1).contiguous();
  Tensor w_down_t = w_down_.transpose(0, 1).contiguous();

  // gate = silu(x @ w_gate^T)  ->  [seq_len, hidden_dim]
  Tensor gate = ops::silu(ops::matmul(x, w_gate_t));

  // up = x @ w_up^T  ->  [seq_len, hidden_dim]
  Tensor up = ops::matmul(x, w_up_t);

  // output = (gate * up) @ w_down^T  ->  [seq_len, embed_dim]
  // Note: w_down is [embed_dim, hidden_dim] so w_down^T is [hidden_dim,
  // embed_dim]
  return ops::matmul(ops::mul(gate, up), w_down_t);
}

// ============================================================================
// Attention::forward — Multi-Head Attention with GQA
// ============================================================================
// This implements the full attention forward pass without KV caching (prefill).
// start_pos is accepted for future KV cache support.
Tensor Attention::forward(const Tensor &x, int32_t start_pos) const {
  // x: [seq_len, embed_dim]
  int64_t seq_len = x.size(0);
  int32_t num_heads = config_.num_heads;
  int32_t num_kv_heads = config_.num_kv_heads;
  int32_t head_dim = config_.head_dim;
  int32_t num_groups = num_heads / num_kv_heads; // GQA group count

  // --- Step 1: Project to Q, K, V ---
  // Weights are [out_features, embed_dim], so transpose for x @ W^T
  Tensor wq_t = wq_.transpose(0, 1).contiguous();
  Tensor wk_t = wk_.transpose(0, 1).contiguous();
  Tensor wv_t = wv_.transpose(0, 1).contiguous();
  Tensor wo_t = wo_.transpose(0, 1).contiguous();

  Tensor Q = ops::matmul(x, wq_t); // [seq_len, num_heads * head_dim]
  Tensor K = ops::matmul(x, wk_t); // [seq_len, num_kv_heads * head_dim]
  Tensor V = ops::matmul(x, wv_t); // [seq_len, num_kv_heads * head_dim]

  // --- Step 2: Reshape into heads ---
  Q = Q.reshape({seq_len, num_heads, head_dim});    // [seq, nh, hd]
  K = K.reshape({seq_len, num_kv_heads, head_dim}); // [seq, nkv, hd]
  V = V.reshape({seq_len, num_kv_heads, head_dim}); // [seq, nkv, hd]

  // --- Step 3: Apply RoPE to Q and K ---
  // ops::rope expects 4D [batch, seq_len, n_heads, head_dim] and 2D [batch,
  // seq_len]. We add a batch dimension of 1, apply RoPE, then remove it.
  Tensor positions({1, seq_len});
  for (int64_t i = 0; i < seq_len; ++i) {
    positions.set({static_cast<int64_t>(0), i},
                  static_cast<float>(start_pos + i));
  }
  Q = ops::rope(Q.unsqueeze(0), positions, config_.rope_theta).squeeze(0);
  K = ops::rope(K.unsqueeze(0), positions, config_.rope_theta).squeeze(0);

  // --- Step 4: GQA expansion ---
  // If num_kv_heads < num_heads, repeat K and V to match.
  if (num_groups > 1) {
    K = K.repeat(1, num_groups); // [seq, nh, hd]
    V = V.repeat(1, num_groups); // [seq, nh, hd]
  }

  // --- Step 5-8: Per-head scaled dot-product attention ---
  // We compute attention head-by-head to avoid a 4D matmul.
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // Output accumulator: [seq_len, num_heads, head_dim]
  Tensor output({seq_len, static_cast<int64_t>(num_heads),
                 static_cast<int64_t>(head_dim)});

  for (int32_t h = 0; h < num_heads; ++h) {
    // Extract Q_h, K_h, V_h for this head: [seq_len, head_dim]
    Tensor Q_h = Q.select(1, h).contiguous(); // [seq, hd]
    Tensor K_h = K.select(1, h).contiguous(); // [seq, hd]
    Tensor V_h = V.select(1, h).contiguous(); // [seq, hd]

    // scores = Q_h @ K_h^T * scale  ->  [seq, seq]
    Tensor K_h_t = K_h.transpose(0, 1).contiguous();
    Tensor scores = ops::matmul(Q_h, K_h_t); // [seq, seq]

    // Scale the scores.
    float *scores_ptr = scores.data<float>();
    for (int64_t i = 0; i < scores.numel(); ++i) {
      scores_ptr[i] *= scale;
    }

    // Apply causal mask: set upper triangle to -inf.
    for (int64_t qi = 0; qi < seq_len; ++qi) {
      for (int64_t ki = qi + 1; ki < seq_len; ++ki) {
        scores.set({qi, ki}, -INFINITY);
      }
    }

    // Softmax over last dimension.
    Tensor attn_weights = ops::softmax(scores); // [seq, seq]

    // Attention output = attn_weights @ V_h  ->  [seq, hd]
    Tensor attn_out = ops::matmul(attn_weights, V_h); // [seq, hd]

    // Copy into output[:, h, :]
    const float *src = attn_out.data<float>();
    for (int64_t s = 0; s < seq_len; ++s) {
      for (int64_t d = 0; d < head_dim; ++d) {
        output.set({s, static_cast<int64_t>(h), d}, src[s * head_dim + d]);
      }
    }
  }

  // --- Step 9: Concatenate heads and project ---
  // Reshape: [seq_len, num_heads, head_dim] -> [seq_len, num_heads * head_dim]
  Tensor concat =
      output.reshape({seq_len, static_cast<int64_t>(num_heads * head_dim)});

  // Final projection: concat @ wo^T  ->  [seq_len, embed_dim]
  return ops::matmul(concat, wo_t);
}

// ============================================================================
// TransformerBlock::forward
// ============================================================================
//   x = x + attn(attn_norm(x), start_pos)   // Attention with residual
//   x = x + ffn(ffn_norm(x))                // FFN with residual
Tensor TransformerBlock::forward(const Tensor &x, int32_t start_pos) const {
  // Pre-norm attention with residual connection.
  Tensor attn_out = attn_.forward(attn_norm_.forward(x), start_pos);
  Tensor residual1 = ops::add(x, attn_out);

  // Pre-norm FFN with residual connection.
  Tensor ffn_out = ffn_.forward(ffn_norm_.forward(residual1));
  return ops::add(residual1, ffn_out);
}

// ============================================================================
// GemmaModel::forward
// ============================================================================
//   1. Embed tokens
//   2. Scale by sqrt(embed_dim)  (Gemma-specific)
//   3. Run through all transformer blocks
//   4. Final RMS norm
//   5. Compute logits via weight tying (embed^T)
//   6. Return logits for the last position
Tensor GemmaModel::forward(const Tensor &tokens, int32_t start_pos) const {
  // Step 1: Token embedding lookup.
  // tokens is a 1D tensor of int token IDs.
  Tensor x = ops::embedding(token_embedding_, tokens); // [seq_len, embed_dim]

  // Step 2: Gemma-specific embedding scaling.
  float scale = std::sqrt(static_cast<float>(config_.embed_dim));
  float *x_ptr = x.data<float>();
  for (int64_t i = 0; i < x.numel(); ++i) {
    x_ptr[i] *= scale;
  }

  // Step 3: Run through transformer blocks.
  for (const auto &block : layers_) {
    x = block.forward(x, start_pos);
  }

  // Step 4: Final normalization.
  x = final_norm_.forward(x);

  // Step 5: Compute logits via weight tying.
  // logits = x @ embedding_table^T
  // embedding_table: [vocab_size, embed_dim]
  // x (last position): [embed_dim]
  // logits: [vocab_size]
  int64_t seq_len = x.size(0);
  Tensor last_hidden = x.select(0, seq_len - 1).contiguous(); // [embed_dim]

  // Reshape to [1, embed_dim] for matmul.
  last_hidden =
      last_hidden.reshape({1, static_cast<int64_t>(config_.embed_dim)});

  // embedding^T: [embed_dim, vocab_size]
  Tensor embed_t = token_embedding_.transpose(0, 1).contiguous();

  // logits = [1, vocab_size]
  Tensor logits = ops::matmul(last_hidden, embed_t);

  // Squeeze to [vocab_size]
  return logits.reshape({static_cast<int64_t>(config_.vocab_size)});
}

} // namespace ie
