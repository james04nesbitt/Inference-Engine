#include "engine/attention/flash_attention.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ie {

// ============================================================================
// FlashAttention — Tiled attention with online softmax
// ============================================================================
//
// Instead of materializing the full [seq_q, seq_kv] attention matrix, we
// process it in tiles using an online softmax algorithm that maintains:
//   m = running max of scores (for numerical stability)
//   l = running sum of exp(scores - m) (for normalization)
//   o = running weighted sum of values (the output)
//
// When a new tile produces a larger max, we rescale the running sums.
//
Tensor flash_attention(const Tensor &query, const Tensor &key,
                       const Tensor &value, float scale, bool causal) {
  // Expected shapes:
  //   query:  [seq_len_q, num_heads, head_dim]
  //   key:    [seq_len_kv, num_kv_heads, head_dim]
  //   value:  [seq_len_kv, num_kv_heads, head_dim]
  if (query.ndim() != 3 || key.ndim() != 3 || value.ndim() != 3) {
    throw std::runtime_error(
        "flash_attention: inputs must be 3D [seq, heads, dim]");
  }

  const int64_t seq_q = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t head_dim = query.size(2);
  const int64_t seq_kv = key.size(0);
  const int64_t num_kv_heads = key.size(1);

  if (head_dim != key.size(2) || head_dim != value.size(2)) {
    throw std::runtime_error("flash_attention: head_dim mismatch");
  }

  const int64_t gqa_groups = num_heads / num_kv_heads;
  const int32_t Bq = kFlashBlockSize;
  const int32_t Bkv = kFlashBlockSize;

  // Get contiguous data pointers.
  Tensor q_c = query.contiguous();
  Tensor k_c = key.contiguous();
  Tensor v_c = value.contiguous();
  const float *Q = q_c.data<float>();
  const float *K = k_c.data<float>();
  const float *V = v_c.data<float>();

  // Allocate output: [seq_q, num_heads, head_dim]
  Tensor output({seq_q, num_heads, head_dim});
  float *O = output.data<float>();
  std::memset(O, 0, seq_q * num_heads * head_dim * sizeof(float));

  // For each query head:
  for (int64_t h = 0; h < num_heads; ++h) {
    int64_t kv_h = h / gqa_groups; // GQA: map query head to KV head

    // For each query tile:
    for (int64_t q_start = 0; q_start < seq_q; q_start += Bq) {
      int64_t q_end = std::min(q_start + Bq, seq_q);
      int64_t tile_q = q_end - q_start;

      // Online softmax state for each query in the tile.
      std::vector<float> m(tile_q, -INFINITY); // running max
      std::vector<float> l(tile_q, 0.0f);      // running exp sum
      // Running output: [tile_q, head_dim]
      std::vector<float> o(tile_q * head_dim, 0.0f);

      // For each KV tile:
      for (int64_t kv_start = 0; kv_start < seq_kv; kv_start += Bkv) {
        int64_t kv_end = std::min(kv_start + Bkv, seq_kv);
        int64_t tile_kv = kv_end - kv_start;

        // Causal optimization: skip if all positions are masked.
        if (causal && kv_start >= q_end)
          break;

        // Compute tile scores: S[qi][ki] = Q[q_start+qi, h, :] dot
        // K[kv_start+ki, kv_h, :] * scale S shape: [tile_q, tile_kv]
        std::vector<float> S(tile_q * tile_kv);

        for (int64_t qi = 0; qi < tile_q; ++qi) {
          const float *q_row =
              Q + (q_start + qi) * num_heads * head_dim + h * head_dim;
          for (int64_t ki = 0; ki < tile_kv; ++ki) {
            const float *k_row =
                K + (kv_start + ki) * num_kv_heads * head_dim + kv_h * head_dim;
            float dot = 0.0f;
            for (int64_t d = 0; d < head_dim; ++d) {
              dot += q_row[d] * k_row[d];
            }
            S[qi * tile_kv + ki] = dot * scale;
          }
        }

        // Apply causal mask within this tile.
        if (causal) {
          for (int64_t qi = 0; qi < tile_q; ++qi) {
            for (int64_t ki = 0; ki < tile_kv; ++ki) {
              if ((q_start + qi) < (kv_start + ki)) {
                S[qi * tile_kv + ki] = -INFINITY;
              }
            }
          }
        }

        // Online softmax update for each query in the tile.
        for (int64_t qi = 0; qi < tile_q; ++qi) {
          // Find row max for this query.
          float row_max = -INFINITY;
          for (int64_t ki = 0; ki < tile_kv; ++ki) {
            float s = S[qi * tile_kv + ki];
            if (s > row_max)
              row_max = s;
          }

          float m_new = std::max(m[qi], row_max);

          // Compute P = exp(S - m_new) and row sum.
          float p_sum = 0.0f;
          std::vector<float> P(tile_kv);
          for (int64_t ki = 0; ki < tile_kv; ++ki) {
            P[ki] = std::exp(S[qi * tile_kv + ki] - m_new);
            p_sum += P[ki];
          }

          // Rescale running sum and output.
          float rescale = std::exp(m[qi] - m_new);
          float l_new = l[qi] * rescale + p_sum;

          // Update running output:
          // o_new = o_old * (l_old * rescale / l_new) + P @ V_tile / l_new
          float old_weight = (l[qi] * rescale) / l_new;
          float new_weight = 1.0f / l_new;

          float *o_row = o.data() + qi * head_dim;
          for (int64_t d = 0; d < head_dim; ++d) {
            // Rescale old output.
            o_row[d] *= old_weight;

            // Add new contribution: P @ V_tile for this dimension.
            float pv = 0.0f;
            for (int64_t ki = 0; ki < tile_kv; ++ki) {
              const float *v_row = V +
                                   (kv_start + ki) * num_kv_heads * head_dim +
                                   kv_h * head_dim;
              pv += P[ki] * v_row[d];
            }
            o_row[d] += pv * new_weight;
          }

          m[qi] = m_new;
          l[qi] = l_new;
        }
      }

      // Write output tile back.
      for (int64_t qi = 0; qi < tile_q; ++qi) {
        float *dst = O + (q_start + qi) * num_heads * head_dim + h * head_dim;
        const float *src = o.data() + qi * head_dim;
        std::memcpy(dst, src, head_dim * sizeof(float));
      }
    }
  }

  return output;
}

// ============================================================================
// FlashAttention + RoPE fused variant
// ============================================================================
// Applies RoPE inline during tiled computation, avoiding a separate pass.
Tensor flash_attention_rope(const Tensor &query, const Tensor &key,
                            const Tensor &value, const Tensor &positions,
                            float scale, float rope_theta, bool causal) {
  if (query.ndim() != 3 || key.ndim() != 3 || value.ndim() != 3) {
    throw std::runtime_error(
        "flash_attention_rope: inputs must be 3D [seq, heads, dim]");
  }

  const int64_t seq_q = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t head_dim = query.size(2);
  const int64_t seq_kv = key.size(0);
  const int64_t num_kv_heads = key.size(1);
  const int64_t half_dim = head_dim / 2;
  const int64_t gqa_groups = num_heads / num_kv_heads;

  Tensor q_c = query.contiguous();
  Tensor k_c = key.contiguous();
  Tensor v_c = value.contiguous();
  Tensor p_c = positions.contiguous();
  const float *Q_src = q_c.data<float>();
  const float *K_src = k_c.data<float>();
  const float *V = v_c.data<float>();
  const float *pos_data = p_c.data<float>();

  // Precompute inverse frequencies.
  std::vector<float> inv_freq(half_dim);
  for (int64_t i = 0; i < half_dim; ++i) {
    inv_freq[i] = 1.0f / std::pow(rope_theta, static_cast<float>(2 * i) /
                                                  static_cast<float>(head_dim));
  }

  // Helper: apply RoPE to a single [head_dim] vector in-place.
  auto apply_rope = [&](float *vec, float pos) {
    for (int64_t i = 0; i < half_dim; ++i) {
      float theta = pos * inv_freq[i];
      float cos_t = std::cos(theta);
      float sin_t = std::sin(theta);
      float x0 = vec[2 * i];
      float x1 = vec[2 * i + 1];
      vec[2 * i] = x0 * cos_t - x1 * sin_t;
      vec[2 * i + 1] = x0 * sin_t + x1 * cos_t;
    }
  };

  // Make copies of Q and K, then apply RoPE in-place.
  std::vector<float> Q_rope(seq_q * num_heads * head_dim);
  std::vector<float> K_rope(seq_kv * num_kv_heads * head_dim);

  std::memcpy(Q_rope.data(), Q_src, Q_rope.size() * sizeof(float));
  std::memcpy(K_rope.data(), K_src, K_rope.size() * sizeof(float));

  for (int64_t s = 0; s < seq_q; ++s) {
    float pos = pos_data[s];
    for (int64_t h = 0; h < num_heads; ++h) {
      apply_rope(Q_rope.data() + s * num_heads * head_dim + h * head_dim, pos);
    }
  }
  for (int64_t s = 0; s < seq_kv; ++s) {
    float pos = pos_data[s < positions.numel() ? s : positions.numel() - 1];
    for (int64_t h = 0; h < num_kv_heads; ++h) {
      apply_rope(K_rope.data() + s * num_kv_heads * head_dim + h * head_dim,
                 pos);
    }
  }

  // Now run the standard FlashAttention algorithm on the RoPE'd data.
  // We create temporary tensors wrapping the RoPE'd data.
  auto q_buf = std::shared_ptr<uint8_t[]>(
      reinterpret_cast<uint8_t *>(Q_rope.data()), [](uint8_t *) {});
  auto k_buf = std::shared_ptr<uint8_t[]>(
      reinterpret_cast<uint8_t *>(K_rope.data()), [](uint8_t *) {});

  Tensor q_roped =
      Tensor::from_buffer(q_buf, {seq_q, num_heads, head_dim}, DType::kFloat32);
  Tensor k_roped = Tensor::from_buffer(k_buf, {seq_kv, num_kv_heads, head_dim},
                                       DType::kFloat32);

  return flash_attention(q_roped, k_roped, value, scale, causal);
}

} // namespace ie
