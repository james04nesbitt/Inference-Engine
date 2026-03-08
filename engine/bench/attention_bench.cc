#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "engine/attention/flash_attention.h"
#include "engine/tensor/tensor.h"

// ============================================================================
// FlashAttention Benchmark Suite
//
// Measures attention latency, throughput, and memory characteristics.
// Tests across varying sequence lengths with Gemma-3 dimensions.
//
// Run with: bazel run --config=release //engine/bench:attention_bench
// ============================================================================

// Naive standard attention for comparison:
// output = softmax(Q @ K^T / scale) @ V — materializes full NxN matrix.
static ie::Tensor NaiveAttention(const ie::Tensor &query, const ie::Tensor &key,
                                 const ie::Tensor &value, float scale,
                                 bool causal) {
  const int64_t seq_q = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t head_dim = query.size(2);
  const int64_t seq_kv = key.size(0);
  const int64_t num_kv_heads = key.size(1);
  const int64_t gqa_groups = num_heads / num_kv_heads;

  ie::Tensor q_c = query.contiguous();
  ie::Tensor k_c = key.contiguous();
  ie::Tensor v_c = value.contiguous();
  const float *Q = q_c.data<float>();
  const float *K = k_c.data<float>();
  const float *V = v_c.data<float>();

  ie::Tensor output({seq_q, num_heads, head_dim});
  float *O = output.data<float>();
  std::memset(O, 0, seq_q * num_heads * head_dim * sizeof(float));

  for (int64_t h = 0; h < num_heads; ++h) {
    int64_t kv_h = h / gqa_groups;

    // Allocate full attention matrix: O(N^2) memory.
    std::vector<float> scores(seq_q * seq_kv);

    // Compute Q @ K^T.
    for (int64_t qi = 0; qi < seq_q; ++qi) {
      const float *q_row = Q + qi * num_heads * head_dim + h * head_dim;
      for (int64_t ki = 0; ki < seq_kv; ++ki) {
        const float *k_row = K + ki * num_kv_heads * head_dim + kv_h * head_dim;
        float dot = 0.0f;
        for (int64_t d = 0; d < head_dim; ++d) {
          dot += q_row[d] * k_row[d];
        }
        scores[qi * seq_kv + ki] = dot * scale;
      }
    }

    // Apply causal mask.
    if (causal) {
      for (int64_t qi = 0; qi < seq_q; ++qi) {
        for (int64_t ki = qi + 1; ki < seq_kv; ++ki) {
          scores[qi * seq_kv + ki] = -INFINITY;
        }
      }
    }

    // Softmax per row.
    for (int64_t qi = 0; qi < seq_q; ++qi) {
      float max_val = -INFINITY;
      for (int64_t ki = 0; ki < seq_kv; ++ki) {
        max_val = std::max(max_val, scores[qi * seq_kv + ki]);
      }
      float sum = 0.0f;
      for (int64_t ki = 0; ki < seq_kv; ++ki) {
        scores[qi * seq_kv + ki] = std::exp(scores[qi * seq_kv + ki] - max_val);
        sum += scores[qi * seq_kv + ki];
      }
      for (int64_t ki = 0; ki < seq_kv; ++ki) {
        scores[qi * seq_kv + ki] /= sum;
      }
    }

    // scores @ V.
    for (int64_t qi = 0; qi < seq_q; ++qi) {
      float *o_row = O + qi * num_heads * head_dim + h * head_dim;
      for (int64_t ki = 0; ki < seq_kv; ++ki) {
        float w = scores[qi * seq_kv + ki];
        const float *v_row = V + ki * num_kv_heads * head_dim + kv_h * head_dim;
        for (int64_t d = 0; d < head_dim; ++d) {
          o_row[d] += w * v_row[d];
        }
      }
    }
  }

  return output;
}

// Generate random tensors for attention benchmarking.
static ie::Tensor RandomTensor(std::vector<int64_t> shape, std::mt19937 &rng) {
  ie::Tensor t(shape);
  float *data = t.data<float>();
  std::normal_distribution<float> dist(0.0f, 0.1f);
  for (int64_t i = 0; i < t.numel(); ++i) {
    data[i] = dist(rng);
  }
  return t;
}

struct AttentionResult {
  double avg_ms;
  double peak_memory_mb;
};

static AttentionResult BenchFlash(int64_t seq_len, int64_t num_heads,
                                  int64_t num_kv_heads, int64_t head_dim,
                                  bool causal, int warmup, int iters,
                                  std::mt19937 &rng) {
  ie::Tensor Q = RandomTensor({seq_len, num_heads, head_dim}, rng);
  ie::Tensor K = RandomTensor({seq_len, num_kv_heads, head_dim}, rng);
  ie::Tensor V = RandomTensor({seq_len, num_kv_heads, head_dim}, rng);
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  for (int i = 0; i < warmup; ++i) {
    ie::flash_attention(Q, K, V, scale, causal);
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    ie::flash_attention(Q, K, V, scale, causal);
  }
  auto end = std::chrono::high_resolution_clock::now();

  double secs = std::chrono::duration<double>(end - start).count();
  double avg_ms = (secs / iters) * 1000.0;

  // FlashAttention peak memory: O(B_q * B_kv) per tile + output
  // = block_size^2 * sizeof(float) per tile + seq*heads*dim output
  double flash_mem =
      (ie::kFlashBlockSize * ie::kFlashBlockSize * sizeof(float)) +
      seq_len * num_heads * head_dim * sizeof(float);

  return {avg_ms, flash_mem / (1024.0 * 1024.0)};
}

static AttentionResult BenchNaive(int64_t seq_len, int64_t num_heads,
                                  int64_t num_kv_heads, int64_t head_dim,
                                  bool causal, int warmup, int iters,
                                  std::mt19937 &rng) {
  ie::Tensor Q = RandomTensor({seq_len, num_heads, head_dim}, rng);
  ie::Tensor K = RandomTensor({seq_len, num_kv_heads, head_dim}, rng);
  ie::Tensor V = RandomTensor({seq_len, num_kv_heads, head_dim}, rng);
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  for (int i = 0; i < warmup; ++i) {
    NaiveAttention(Q, K, V, scale, causal);
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    NaiveAttention(Q, K, V, scale, causal);
  }
  auto end = std::chrono::high_resolution_clock::now();

  double secs = std::chrono::duration<double>(end - start).count();
  double avg_ms = (secs / iters) * 1000.0;

  // Naive attention peak memory: full N*N score matrix + output
  double naive_mem = seq_len * seq_len * sizeof(float) * num_heads +
                     seq_len * num_heads * head_dim * sizeof(float);

  return {avg_ms, naive_mem / (1024.0 * 1024.0)};
}

int main() {
  std::cout << "============================================================"
            << std::endl;
  std::cout << "  Inference Engine — FlashAttention Benchmark Suite"
            << std::endl;
  std::cout << "============================================================"
            << std::endl;

  std::mt19937 rng(42);

  // Gemma-3 1B dimensions.
  const int64_t num_heads = 4;
  const int64_t num_kv_heads = 1;
  const int64_t head_dim = 256;

  // ---- FlashAttention vs Naive at varying sequence lengths ----
  {
    std::cout << std::endl;
    std::cout << "--- FlashAttention vs Naive (Causal, GQA 4:1) ---"
              << std::endl;
    std::cout << std::setw(10) << "SeqLen" << "  " << std::setw(12)
              << "Naive (ms)" << "  " << std::setw(12) << "Flash (ms)" << "  "
              << std::setw(10) << "Speedup" << "  " << std::setw(14)
              << "Naive Mem(MB)" << "  " << std::setw(14) << "Flash Mem(MB)"
              << "  " << std::setw(12) << "Mem Ratio" << std::endl;
    std::cout << std::string(96, '-') << std::endl;

    int64_t seq_lens[] = {32, 64, 128, 256, 512};
    for (auto seq_len : seq_lens) {
      int iters = (seq_len <= 128) ? 20 : (seq_len <= 256 ? 10 : 5);

      auto naive = BenchNaive(seq_len, num_heads, num_kv_heads, head_dim, true,
                              2, iters, rng);
      auto flash = BenchFlash(seq_len, num_heads, num_kv_heads, head_dim, true,
                              2, iters, rng);

      double speedup = naive.avg_ms / flash.avg_ms;
      double mem_ratio = naive.peak_memory_mb / flash.peak_memory_mb;

      std::cout << std::setw(10) << seq_len << "  " << std::fixed
                << std::setprecision(2) << std::setw(10) << naive.avg_ms << "ms"
                << "  " << std::setw(10) << flash.avg_ms << "ms"
                << "  " << std::setw(8) << speedup << "x" << "  "
                << std::setprecision(3) << std::setw(12) << naive.peak_memory_mb
                << "MB" << "  " << std::setw(12) << flash.peak_memory_mb << "MB"
                << "  " << std::setprecision(1) << std::setw(10) << mem_ratio
                << "x" << std::endl;
    }
  }

  // ---- FlashAttention: causal vs non-causal ----
  {
    std::cout << std::endl;
    std::cout << "--- FlashAttention: Causal vs Non-Causal ---" << std::endl;
    std::cout << std::setw(10) << "SeqLen" << "  " << std::setw(14)
              << "Causal (ms)" << "  " << std::setw(14) << "Non-Causal (ms)"
              << "  " << std::setw(10) << "Ratio" << std::endl;
    std::cout << std::string(54, '-') << std::endl;

    int64_t seq_lens[] = {64, 128, 256, 512};
    for (auto seq_len : seq_lens) {
      int iters = (seq_len <= 128) ? 20 : 10;

      auto causal = BenchFlash(seq_len, num_heads, num_kv_heads, head_dim, true,
                               2, iters, rng);
      auto non_causal = BenchFlash(seq_len, num_heads, num_kv_heads, head_dim,
                                   false, 2, iters, rng);

      std::cout << std::setw(10) << seq_len << "  " << std::fixed
                << std::setprecision(2) << std::setw(12) << causal.avg_ms
                << "ms" << "  " << std::setw(12) << non_causal.avg_ms << "ms"
                << "  " << std::setw(8) << (non_causal.avg_ms / causal.avg_ms)
                << "x" << std::endl;
    }
  }

  // ---- FlashAttention: throughput (sequences/sec) ----
  {
    std::cout << std::endl;
    std::cout << "--- FlashAttention: Throughput ---" << std::endl;
    std::cout << std::setw(10) << "SeqLen" << "  " << std::setw(12)
              << "Latency (ms)" << "  " << std::setw(12) << "Seq/sec"
              << std::endl;
    std::cout << std::string(40, '-') << std::endl;

    int64_t seq_lens[] = {32, 64, 128, 256, 512, 1024};
    for (auto seq_len : seq_lens) {
      int iters = (seq_len <= 128) ? 30 : (seq_len <= 512 ? 10 : 3);

      auto result = BenchFlash(seq_len, num_heads, num_kv_heads, head_dim, true,
                               2, iters, rng);
      double seq_per_sec = 1000.0 / result.avg_ms;

      std::cout << std::setw(10) << seq_len << "  " << std::fixed
                << std::setprecision(2) << std::setw(10) << result.avg_ms
                << "ms" << "  " << std::setprecision(1) << std::setw(10)
                << seq_per_sec << std::endl;
    }
  }

  // ---- Correctness validation (small check) ----
  {
    std::cout << std::endl;
    std::cout << "--- Correctness Check (Flash vs Naive) ---" << std::endl;

    int64_t seq_len = 32;
    ie::Tensor Q = RandomTensor({seq_len, num_heads, head_dim}, rng);
    ie::Tensor K = RandomTensor({seq_len, num_kv_heads, head_dim}, rng);
    ie::Tensor V = RandomTensor({seq_len, num_kv_heads, head_dim}, rng);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto naive_out = NaiveAttention(Q, K, V, scale, true);
    auto flash_out = ie::flash_attention(Q, K, V, scale, true);

    const float *naive_data = naive_out.data<float>();
    const float *flash_data = flash_out.data<float>();
    int64_t total = naive_out.numel();

    double max_err = 0.0;
    double sum_err = 0.0;
    for (int64_t i = 0; i < total; ++i) {
      double err = std::abs(naive_data[i] - flash_data[i]);
      max_err = std::max(max_err, err);
      sum_err += err;
    }
    double mean_err = sum_err / total;

    std::cout << "  Max error:  " << std::scientific << std::setprecision(4)
              << max_err << std::endl;
    std::cout << "  Mean error: " << mean_err << std::endl;
    std::cout << "  Status:     " << (max_err < 1e-4 ? "PASS" : "WARN (>1e-4)")
              << std::endl;
  }

  std::cout << std::endl;
  std::cout << "============================================================"
            << std::endl;
  std::cout << "  Benchmark complete." << std::endl;
  std::cout << "============================================================"
            << std::endl;

  return 0;
}
