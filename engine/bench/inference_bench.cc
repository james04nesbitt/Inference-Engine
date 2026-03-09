#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "engine/attention/kv_cache.h"
#include "engine/model/config.h"
#include "engine/model/layers.h"
#include "engine/ops/ops.h"
#include "engine/tensor/tensor.h"

// ============================================================================
// End-to-End Inference Benchmark Suite
//
// Measures tokens/sec, prefill throughput, time-to-first-token (TTFT),
// and per-token decode latency using a synthetic model (random weights).
//
// This avoids requiring the actual GGUF model file for benchmarking.
// The compute is identical to real inference — only the weight values differ.
//
// Run with: bazel run --config=release //engine/bench:inference_bench
// ============================================================================

static std::mt19937 g_rng(42);

static ie::Tensor RandomTensor(std::vector<int64_t> shape) {
  ie::Tensor t(shape);
  float *data = t.data<float>();
  std::normal_distribution<float> dist(0.0f, 0.02f);
  for (int64_t i = 0; i < t.numel(); ++i) {
    data[i] = dist(g_rng);
  }
  return t;
}

// Build a full GemmaModel with random weights at real Gemma-3 1B dimensions.
static ie::GemmaModel BuildSyntheticModel(const ie::GemmaConfig &config) {
  // Token embedding: [vocab_size, embed_dim]
  // Use a small subset for benchmarking to avoid 1.2GB allocation.
  int32_t bench_vocab = 1024;
  ie::Tensor token_embedding = RandomTensor({bench_vocab, config.embed_dim});

  // Build transformer blocks.
  std::vector<ie::TransformerBlock> layers;
  for (int32_t i = 0; i < config.num_layers; ++i) {
    // Attention weights.
    ie::Tensor wq =
        RandomTensor({config.num_heads * config.head_dim, config.embed_dim});
    ie::Tensor wk =
        RandomTensor({config.num_kv_heads * config.head_dim, config.embed_dim});
    ie::Tensor wv =
        RandomTensor({config.num_kv_heads * config.head_dim, config.embed_dim});
    ie::Tensor wo =
        RandomTensor({config.embed_dim, config.num_heads * config.head_dim});

    ie::Attention attn(config, i, std::move(wq), std::move(wk), std::move(wv),
                       std::move(wo));

    // FFN weights (SwiGLU).
    ie::Tensor w_gate = RandomTensor({config.hidden_dim, config.embed_dim});
    ie::Tensor w_up = RandomTensor({config.hidden_dim, config.embed_dim});
    ie::Tensor w_down = RandomTensor({config.embed_dim, config.hidden_dim});
    ie::FeedForward ffn(std::move(w_gate), std::move(w_up), std::move(w_down));

    // Layer norms.
    ie::Tensor attn_norm_w = ie::Tensor::ones({config.embed_dim});
    ie::Tensor ffn_norm_w = ie::Tensor::ones({config.embed_dim});
    ie::RMSNorm attn_norm(std::move(attn_norm_w), config.rms_norm_eps);
    ie::RMSNorm ffn_norm(std::move(ffn_norm_w), config.rms_norm_eps);

    layers.emplace_back(config, i, std::move(attn_norm), std::move(attn),
                        std::move(ffn_norm), std::move(ffn));
  }

  // Final norm.
  ie::Tensor final_norm_w = ie::Tensor::ones({config.embed_dim});
  ie::RMSNorm final_norm(std::move(final_norm_w), config.rms_norm_eps);

  return ie::GemmaModel(config, std::move(token_embedding), std::move(layers),
                        std::move(final_norm));
}

// Simulate greedy sampling: argmax over logits.
static int32_t SampleGreedy(const ie::Tensor &logits) {
  const float *data = logits.data<float>();
  int64_t n = logits.numel();
  int32_t best = 0;
  float best_val = data[0];
  for (int64_t i = 1; i < n; ++i) {
    if (data[i] > best_val) {
      best_val = data[i];
      best = static_cast<int32_t>(i);
    }
  }
  return best;
}

int main() {
  std::cout << "============================================================"
            << std::endl;
  std::cout << "  Inference Engine — End-to-End Inference Benchmark"
            << std::endl;
  std::cout << "  (Synthetic weights — measures compute throughput only)"
            << std::endl;
  std::cout << "============================================================"
            << std::endl;

  ie::GemmaConfig config;
  // Use default Gemma-3 1B config, but reduce vocab for benchmark
  // (avoids 1.2GB embedding allocation; compute shapes stay the same).
  config.vocab_size = 1024;

  std::cout << std::endl;
  std::cout << "  Model config:" << std::endl;
  std::cout << "    Layers:     " << config.num_layers << std::endl;
  std::cout << "    Embed dim:  " << config.embed_dim << std::endl;
  std::cout << "    Heads:      " << config.num_heads
            << " (KV: " << config.num_kv_heads << ")" << std::endl;
  std::cout << "    Head dim:   " << config.head_dim << std::endl;
  std::cout << "    FFN dim:    " << config.hidden_dim << std::endl;

  std::cout << std::endl;
  std::cout << "  Building synthetic model (random weights)..." << std::flush;
  auto model_start = std::chrono::high_resolution_clock::now();
  auto model = BuildSyntheticModel(config);
  auto model_end = std::chrono::high_resolution_clock::now();
  double model_ms =
      std::chrono::duration<double, std::milli>(model_end - model_start)
          .count();
  std::cout << " done (" << std::fixed << std::setprecision(0) << model_ms
            << "ms)" << std::endl;

  // KV cache.
  int32_t max_blocks = 2048;
  ie::KVCacheManager kv_cache(config.num_layers, config.num_kv_heads,
                              config.head_dim, max_blocks);

  // ---- Prefill Benchmark ----
  {
    std::cout << std::endl;
    std::cout << "--- Prefill Throughput (Time-to-First-Token) ---"
              << std::endl;
    std::cout << std::setw(12) << "Prompt Len" << "  " << std::setw(14)
              << "TTFT (ms)" << "  " << std::setw(16) << "Prefill Tok/sec"
              << std::endl;
    std::cout << std::string(48, '-') << std::endl;

    int prompt_lens[] = {4, 8, 16};
    for (auto prompt_len : prompt_lens) {
      // Create token tensor (use small token IDs within our bench vocab).
      ie::Tensor tokens({prompt_len}, ie::DType::kFloat32);
      float *tok_data = tokens.data<float>();
      for (int i = 0; i < prompt_len; ++i) {
        tok_data[i] = static_cast<float>(i % 1024);
      }

      int64_t seq_id = kv_cache.AllocateSequence();

      auto start = std::chrono::high_resolution_clock::now();
      ie::Tensor logits = model.forward(tokens, 0, kv_cache, seq_id);
      auto end = std::chrono::high_resolution_clock::now();

      double ttft_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      double prefill_tok_sec = (prompt_len / ttft_ms) * 1000.0;

      std::cout << std::setw(12) << prompt_len << "  " << std::fixed
                << std::setprecision(1) << std::setw(12) << ttft_ms << "ms"
                << "  " << std::setprecision(1) << std::setw(14)
                << prefill_tok_sec << std::endl;

      kv_cache.FreeSequence(seq_id);
    }
  }

  // ---- Decode (Token Generation) Benchmark ----
  {
    std::cout << std::endl;
    std::cout << "--- Decode Throughput (Tokens/sec) ---" << std::endl;
    std::cout << std::setw(14) << "Gen Tokens" << "  " << std::setw(14)
              << "Total (ms)" << "  " << std::setw(16) << "Per-Token (ms)"
              << "  " << std::setw(14) << "Tokens/sec" << std::endl;
    std::cout << std::string(64, '-') << std::endl;

    int gen_counts[] = {4, 8, 16};
    int prompt_len = 4;

    for (auto gen_count : gen_counts) {
      int64_t seq_id = kv_cache.AllocateSequence();

      // Prefill.
      ie::Tensor prompt_tokens({prompt_len}, ie::DType::kFloat32);
      float *tok_data = prompt_tokens.data<float>();
      for (int i = 0; i < prompt_len; ++i) {
        tok_data[i] = static_cast<float>(i % 1024);
      }
      ie::Tensor logits = model.forward(prompt_tokens, 0, kv_cache, seq_id);
      int32_t next_token = SampleGreedy(logits);

      // Decode loop: measure only the decode phase.
      auto start = std::chrono::high_resolution_clock::now();
      for (int t = 0; t < gen_count; ++t) {
        ie::Tensor tok({1}, ie::DType::kFloat32);
        tok.data<float>()[0] = static_cast<float>(next_token % 1024);
        logits = model.forward(tok, prompt_len + t, kv_cache, seq_id);
        next_token = SampleGreedy(logits);
      }
      auto end = std::chrono::high_resolution_clock::now();

      double total_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      double per_token_ms = total_ms / gen_count;
      double tokens_per_sec = (gen_count / total_ms) * 1000.0;

      std::cout << std::setw(14) << gen_count << "  " << std::fixed
                << std::setprecision(1) << std::setw(12) << total_ms << "ms"
                << "  " << std::setprecision(2) << std::setw(14) << per_token_ms
                << "ms" << "  " << std::setprecision(2) << std::setw(12)
                << tokens_per_sec << std::endl;

      kv_cache.FreeSequence(seq_id);
    }
  }

  // ---- Per-Layer Latency Breakdown ----
  {
    std::cout << std::endl;
    std::cout << "--- Per-Layer Latency Breakdown (Single Token) ---"
              << std::endl;

    int64_t seq_id = kv_cache.AllocateSequence();

    // Prefill with a short prompt first.
    ie::Tensor prompt({4}, ie::DType::kFloat32);
    for (int i = 0; i < 4; ++i)
      prompt.data<float>()[i] = static_cast<float>(i);
    model.forward(prompt, 0, kv_cache, seq_id);

    // Measure a single decode step.
    ie::Tensor tok({1}, ie::DType::kFloat32);
    tok.data<float>()[0] = 42.0f;

    auto start = std::chrono::high_resolution_clock::now();
    ie::Tensor logits = model.forward(tok, 4, kv_cache, seq_id);
    auto end = std::chrono::high_resolution_clock::now();

    double single_step_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    double per_layer_ms = single_step_ms / config.num_layers;

    std::cout << "  Full forward pass: " << std::fixed << std::setprecision(1)
              << single_step_ms << " ms" << std::endl;
    std::cout << "  Per layer (avg):   " << std::setprecision(2) << per_layer_ms
              << " ms" << std::endl;
    std::cout << "  Layers:            " << config.num_layers << std::endl;
    std::cout << "  Decode tok/sec:    " << std::setprecision(2)
              << (1000.0 / single_step_ms) << std::endl;

    kv_cache.FreeSequence(seq_id);
  }

  // ---- Memory Usage Summary ----
  {
    std::cout << std::endl;
    std::cout << "--- Estimated Memory Usage ---" << std::endl;

    // Model weights (approximate).
    double embed_mb =
        (1024.0 * config.embed_dim * sizeof(float)) / (1024.0 * 1024.0);
    double attn_mb_per_layer =
        (config.embed_dim * config.num_heads * config.head_dim * 2 +
         config.embed_dim * config.num_kv_heads * config.head_dim * 2) *
        sizeof(float) / (1024.0 * 1024.0);
    double ffn_mb_per_layer = (config.embed_dim * config.hidden_dim * 3) *
                              sizeof(float) / (1024.0 * 1024.0);
    double model_mb =
        embed_mb + (attn_mb_per_layer + ffn_mb_per_layer) * config.num_layers;

    // KV cache (per token, all layers).
    double kv_per_token_kb = 2.0 * config.num_kv_heads * config.head_dim *
                             sizeof(float) * config.num_layers / 1024.0;

    std::cout << "  Model weights (bench):   " << std::fixed
              << std::setprecision(1) << model_mb << " MB" << std::endl;
    std::cout << "  KV cache per token:      " << std::setprecision(1)
              << kv_per_token_kb << " KB" << std::endl;
    std::cout << "  KV cache at 1K tokens:   " << std::setprecision(1)
              << (kv_per_token_kb * 1024 / 1024.0) << " MB" << std::endl;
    std::cout << "  KV cache at 4K tokens:   " << std::setprecision(1)
              << (kv_per_token_kb * 4096 / 1024.0) << " MB" << std::endl;
  }

  std::cout << std::endl;
  std::cout << "============================================================"
            << std::endl;
  std::cout << "  Benchmark complete." << std::endl;
  std::cout << "============================================================"
            << std::endl;

  return 0;
}
