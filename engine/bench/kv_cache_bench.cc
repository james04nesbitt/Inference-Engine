#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "engine/attention/kv_cache.h"
#include "engine/tensor/tensor.h"

// ============================================================================
// KV Cache Benchmark Suite
//
// Measures allocation, append, lookup, and multi-sequence throughput
// for the PagedAttention-style KV cache manager.
//
// Run with: bazel run --config=release //engine/bench:kv_cache_bench
// ============================================================================

// Generate a random [1, num_kv_heads, head_dim] tensor (single token KV).
static ie::Tensor RandomKV(int64_t num_kv_heads, int64_t head_dim,
                           std::mt19937 &rng) {
  ie::Tensor t({1, num_kv_heads, head_dim});
  float *data = t.data<float>();
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (int64_t i = 0; i < t.numel(); ++i) {
    data[i] = dist(rng);
  }
  return t;
}

int main() {
  std::cout << "============================================================"
            << std::endl;
  std::cout << "  Inference Engine — KV Cache Benchmark Suite" << std::endl;
  std::cout << "============================================================"
            << std::endl;

  std::mt19937 rng(42);

  // Gemma-3 1B dimensions.
  const int32_t num_layers = 26;
  const int32_t num_kv_heads = 1;
  const int32_t head_dim = 256;
  const int32_t block_size = 16;
  const int32_t max_blocks = 65536;

  // ---- Allocation/Free Throughput ----
  {
    std::cout << std::endl;
    std::cout << "--- Sequence Allocation/Free Throughput ---" << std::endl;

    const int num_allocs = 100;
    ie::KVCacheManager cache(num_layers, num_kv_heads, head_dim, max_blocks,
                             block_size);

    // Measure allocation.
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int64_t> seq_ids;
    for (int i = 0; i < num_allocs; ++i) {
      seq_ids.push_back(cache.AllocateSequence());
    }
    auto end = std::chrono::high_resolution_clock::now();
    double alloc_us =
        std::chrono::duration<double, std::micro>(end - start).count() /
        num_allocs;

    // Measure deallocation.
    start = std::chrono::high_resolution_clock::now();
    for (auto id : seq_ids) {
      cache.FreeSequence(id);
    }
    end = std::chrono::high_resolution_clock::now();
    double free_us =
        std::chrono::duration<double, std::micro>(end - start).count() /
        num_allocs;

    std::cout << "  Allocate:  " << std::fixed << std::setprecision(2)
              << alloc_us << " us/seq  (" << (1e6 / alloc_us) << " allocs/sec)"
              << std::endl;
    std::cout << "  Free:      " << free_us << " us/seq  (" << (1e6 / free_us)
              << " frees/sec)" << std::endl;
  }

  // ---- Single-Sequence Append Throughput ----
  {
    std::cout << std::endl;
    std::cout << "--- Single-Sequence Append Throughput ---" << std::endl;
    std::cout << std::setw(12) << "Tokens" << "  " << std::setw(12)
              << "Total (ms)" << "  " << std::setw(14) << "Per-Token (us)"
              << "  " << std::setw(14) << "Tokens/sec" << std::endl;
    std::cout << std::string(58, '-') << std::endl;

    int token_counts[] = {32, 64, 128, 256, 512};
    for (auto num_tokens : token_counts) {
      ie::KVCacheManager cache(num_layers, num_kv_heads, head_dim, max_blocks,
                               block_size);
      int64_t seq_id = cache.AllocateSequence();

      // Pre-generate KV data.
      std::vector<ie::Tensor> keys, values;
      for (int t = 0; t < num_tokens; ++t) {
        keys.push_back(RandomKV(num_kv_heads, head_dim, rng));
        values.push_back(RandomKV(num_kv_heads, head_dim, rng));
      }

      auto start = std::chrono::high_resolution_clock::now();
      for (int t = 0; t < num_tokens; ++t) {
        for (int32_t layer = 0; layer < num_layers; ++layer) {
          cache.AppendToken(seq_id, layer, keys[t], values[t]);
        }
      }
      auto end = std::chrono::high_resolution_clock::now();

      double total_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      double per_token_us = (total_ms / num_tokens) * 1000.0;
      double tokens_per_sec = (num_tokens / total_ms) * 1000.0;

      std::cout << std::setw(12) << num_tokens << "  " << std::fixed
                << std::setprecision(2) << std::setw(10) << total_ms << "ms"
                << "  " << std::setprecision(1) << std::setw(12) << per_token_us
                << "us" << "  " << std::setprecision(0) << std::setw(12)
                << tokens_per_sec << std::endl;

      cache.FreeSequence(seq_id);
    }
  }

  // ---- Lookup Latency ----
  {
    std::cout << std::endl;
    std::cout << "--- KV Lookup Latency ---" << std::endl;
    std::cout << std::setw(12) << "CacheLen" << "  " << std::setw(14)
              << "GetKeys (us)" << "  " << std::setw(14) << "GetValues (us)"
              << std::endl;
    std::cout << std::string(46, '-') << std::endl;

    int seq_lens[] = {32, 64, 128, 256, 512};
    for (auto seq_len : seq_lens) {
      ie::KVCacheManager cache(num_layers, num_kv_heads, head_dim, max_blocks,
                               block_size);
      int64_t seq_id = cache.AllocateSequence();

      // Fill the cache.
      for (int t = 0; t < seq_len; ++t) {
        ie::Tensor k = RandomKV(num_kv_heads, head_dim, rng);
        ie::Tensor v = RandomKV(num_kv_heads, head_dim, rng);
        for (int32_t layer = 0; layer < num_layers; ++layer) {
          cache.AppendToken(seq_id, layer, k, v);
        }
      }

      // Measure lookup (average across layers).
      const int lookup_iters = 100;
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < lookup_iters; ++i) {
        for (int32_t layer = 0; layer < num_layers; ++layer) {
          auto keys = cache.GetKeys(seq_id, layer);
          (void)keys;
        }
      }
      auto end = std::chrono::high_resolution_clock::now();
      double keys_us =
          std::chrono::duration<double, std::micro>(end - start).count() /
          (lookup_iters * num_layers);

      start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < lookup_iters; ++i) {
        for (int32_t layer = 0; layer < num_layers; ++layer) {
          auto values = cache.GetValues(seq_id, layer);
          (void)values;
        }
      }
      end = std::chrono::high_resolution_clock::now();
      double vals_us =
          std::chrono::duration<double, std::micro>(end - start).count() /
          (lookup_iters * num_layers);

      std::cout << std::setw(12) << seq_len << "  " << std::fixed
                << std::setprecision(2) << std::setw(12) << keys_us << "us"
                << "  " << std::setw(12) << vals_us << "us" << std::endl;

      cache.FreeSequence(seq_id);
    }
  }

  // ---- Multi-Sequence Concurrent Throughput ----
  {
    std::cout << std::endl;
    std::cout << "--- Multi-Sequence Concurrent Throughput ---" << std::endl;
    std::cout << std::setw(12) << "Sequences" << "  " << std::setw(14)
              << "Tok/Seq" << "  " << std::setw(14) << "Total (ms)" << "  "
              << std::setw(14) << "Tokens/sec" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    int seq_counts[] = {1, 2, 4, 8, 16};
    int tokens_per_seq = 64;

    for (auto num_seqs : seq_counts) {
      ie::KVCacheManager cache(num_layers, num_kv_heads, head_dim, max_blocks,
                               block_size);

      // Allocate all sequences.
      std::vector<int64_t> seq_ids;
      for (int s = 0; s < num_seqs; ++s) {
        seq_ids.push_back(cache.AllocateSequence());
      }

      // Pre-generate KV data.
      std::vector<ie::Tensor> keys, values;
      for (int t = 0; t < tokens_per_seq; ++t) {
        keys.push_back(RandomKV(num_kv_heads, head_dim, rng));
        values.push_back(RandomKV(num_kv_heads, head_dim, rng));
      }

      // Simulate round-robin append (interleaved across sequences).
      auto start = std::chrono::high_resolution_clock::now();
      for (int t = 0; t < tokens_per_seq; ++t) {
        for (int s = 0; s < num_seqs; ++s) {
          for (int32_t layer = 0; layer < num_layers; ++layer) {
            cache.AppendToken(seq_ids[s], layer, keys[t], values[t]);
          }
        }
      }
      auto end = std::chrono::high_resolution_clock::now();

      double total_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      int total_tokens = num_seqs * tokens_per_seq;
      double tokens_per_sec = (total_tokens / total_ms) * 1000.0;

      std::cout << std::setw(12) << num_seqs << "  " << std::setw(14)
                << tokens_per_seq << "  " << std::fixed << std::setprecision(2)
                << std::setw(12) << total_ms << "ms" << "  "
                << std::setprecision(0) << std::setw(12) << tokens_per_sec
                << std::endl;

      for (auto id : seq_ids) {
        cache.FreeSequence(id);
      }
    }
  }

  // ---- Block Utilization ----
  {
    std::cout << std::endl;
    std::cout << "--- Block Utilization ---" << std::endl;
    std::cout << std::setw(12) << "SeqLen" << "  " << std::setw(14)
              << "Blocks Used" << "  " << std::setw(14) << "Free Blocks"
              << "  " << std::setw(14) << "Utilization" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    int seq_lens[] = {16, 32, 64, 128, 256};
    for (auto seq_len : seq_lens) {
      ie::KVCacheManager cache(num_layers, num_kv_heads, head_dim, max_blocks,
                               block_size);
      int64_t seq_id = cache.AllocateSequence();

      int32_t initial_free = cache.NumFreeBlocks();

      for (int t = 0; t < seq_len; ++t) {
        ie::Tensor k = RandomKV(num_kv_heads, head_dim, rng);
        ie::Tensor v = RandomKV(num_kv_heads, head_dim, rng);
        for (int32_t layer = 0; layer < num_layers; ++layer) {
          cache.AppendToken(seq_id, layer, k, v);
        }
      }

      int32_t blocks_used = initial_free - cache.NumFreeBlocks();
      double utilization = static_cast<double>(blocks_used) / max_blocks * 100;

      std::cout << std::setw(12) << seq_len << "  " << std::setw(14)
                << blocks_used << "  " << std::setw(14) << cache.NumFreeBlocks()
                << "  " << std::fixed << std::setprecision(1) << std::setw(12)
                << utilization << "%" << std::endl;

      cache.FreeSequence(seq_id);
    }
  }

  std::cout << std::endl;
  std::cout << "============================================================"
            << std::endl;
  std::cout << "  Benchmark complete." << std::endl;
  std::cout << "============================================================"
            << std::endl;

  return 0;
}
