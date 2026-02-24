#pragma once

#include "engine/tensor/tensor.h"

namespace ie {

// ============================================================================
// FlashAttention — Memory-Efficient Fused Attention
//
// Standard attention computes the full N×N attention matrix, requiring
// O(N²) memory. FlashAttention avoids materializing this matrix by:
//   1. Tiling the Q, K, V matrices into blocks
//   2. Computing attention one tile at a time in SRAM (registers/L1)
//   3. Using an online softmax algorithm to accumulate results
//
// The key trick is the online softmax: you can compute softmax incrementally
// without ever having the full row. You maintain a running max and running sum,
// and rescale previous results when a new max is found.
//
// Online softmax algorithm:
//   m_0 = -inf, l_0 = 0, o_0 = 0
//   For each tile j of K:
//     s_j = Q_tile @ K_tile_j^T / sqrt(d)           // tile scores
//     m_new = max(m_old, max(s_j))                   // running max
//     l_new = l_old * exp(m_old - m_new) +           // rescaled old sum
//             sum(exp(s_j - m_new))                   // new tile sum
//     o_new = o_old * (l_old * exp(m_old - m_new) / l_new) +   // rescale
//             exp(s_j - m_new) @ V_tile_j / l_new               // new contrib
//     m_old = m_new, l_old = l_new
//
// Papers to read:
//   - "FlashAttention: Fast and Memory-Efficient Exact Attention with
//      IO-Awareness" (Dao et al., 2022): https://arxiv.org/abs/2205.14135
//   - "FlashAttention-2" (Dao, 2023): https://arxiv.org/abs/2307.08691
// ============================================================================

// Block size for tiling — chosen to fit in L1/register file.
// Typical values: 64-128 for GPU, 32-64 for CPU.
constexpr int32_t kFlashBlockSize = 32;

// Compute attention output using the FlashAttention algorithm.
//
// Parameters:
//   query:  [seq_len_q, num_heads, head_dim]
//   key:    [seq_len_kv, num_kv_heads, head_dim]
//   value:  [seq_len_kv, num_kv_heads, head_dim]
//   scale:  typically 1/sqrt(head_dim)
//   causal: if true, apply causal mask (upper triangle = -inf)
//
// Returns:
//   output: [seq_len_q, num_heads, head_dim]
//
// TODO: Implement the tiled, fused attention algorithm
//
Tensor flash_attention(const Tensor& query, const Tensor& key,
                       const Tensor& value, float scale, bool causal = true);

// Fused attention + RoPE variant
// Applies rotary embeddings inline during tiled attention computation,
// avoiding a separate RoPE pass over Q and K.
//
// TODO: Implement (advanced optimization)
//
Tensor flash_attention_rope(const Tensor& query, const Tensor& key,
                            const Tensor& value, const Tensor& positions,
                            float scale, float rope_theta = 10000.0f,
                            bool causal = true);

}  // namespace ie
