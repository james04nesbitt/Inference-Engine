#pragma once

#include <cstdint>

namespace ie {

// ============================================================================
// GemmaConfig — Model hyperparameters for Gemma 3.
//
// These values are read from the GGUF metadata at load time.
// The defaults below match Gemma 3 1B Instruct (F16).
// ============================================================================
struct GemmaConfig {
  // Architecture
  int32_t num_layers = 26;         // Number of transformer blocks
  int32_t embed_dim = 1152;        // Embedding/hidden dimension
  int32_t num_heads = 4;           // Number of attention heads
  int32_t num_kv_heads = 1;        // Number of key/value heads (GQA)
  int32_t head_dim = 256;          // Dimension per head
  int32_t hidden_dim = 6912;       // Feed-forward intermediate dimension
  int32_t vocab_size = 262144;     // Vocabulary size

  // Context
  int32_t max_seq_len = 32768;     // Maximum sequence length

  // Normalization
  float rms_norm_eps = 1e-6f;      // RMS norm epsilon

  // RoPE
  float rope_theta = 10000.0f;     // RoPE base frequency

  // Derived properties
  int32_t kv_dim() const { return num_kv_heads * head_dim; }
  int32_t q_dim() const { return num_heads * head_dim; }

  // Populate from GGUF metadata keys.
  // TODO: Implement this to read from GGUFFile metadata
  // Keys to look for:
  //   gemma.block_count -> num_layers
  //   gemma.embedding_length -> embed_dim
  //   gemma.attention.head_count -> num_heads
  //   gemma.attention.head_count_kv -> num_kv_heads
  //   gemma.attention.key_length -> head_dim
  //   gemma.feed_forward_length -> hidden_dim
};

}  // namespace ie
