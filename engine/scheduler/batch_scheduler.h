#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <string>
#include <vector>

#include "engine/attention/kv_cache.h"
#include "engine/model/layers.h"
#include "engine/tokenizer/tokenizer.h"

namespace ie {

// ============================================================================
// Continuous Batching Scheduler
//
// Manages multiple concurrent generation requests, scheduling them
// round-robin through the model. Each request gets its own KV cache
// sequence, and the scheduler processes one decode token per active
// sequence per step.
//
// Based on concepts from:
//   - "Efficient Memory Management for Large Language Model Serving with
//     PagedAttention" (Kwon et al., 2023)
//   - Orca continuous batching (Yu et al., 2022)
// ============================================================================

// A generation request submitted to the scheduler.
struct Request {
  std::string prompt;
  int32_t max_tokens = 128;

  // Streaming callback: called with each generated token string.
  std::function<void(const std::string &)> on_token;

  // Completed callback: called with the full generated text when done.
  std::function<void(const std::string &)> on_complete;
};

// Internal state for an active (in-flight) sequence.
struct ActiveSequence {
  int64_t seq_id = -1;         // KV cache sequence handle
  std::vector<int32_t> tokens; // All tokens (prompt + generated)
  int32_t prompt_len = 0;      // Number of prompt tokens
  int32_t generated = 0;       // Number of tokens generated so far
  int32_t max_tokens = 128;    // Max tokens to generate
  bool finished = false;       // Whether generation is complete

  // Callbacks from the request.
  std::function<void(const std::string &)> on_token;
  std::function<void(const std::string &)> on_complete;
};

// The batch scheduler manages request lifecycle and decode scheduling.
class BatchScheduler {
public:
  // Create a scheduler with shared model, tokenizer, and KV cache.
  BatchScheduler(GemmaModel &model, Tokenizer &tokenizer,
                 KVCacheManager &kv_cache);

  // Submit a new generation request.
  void AddRequest(Request request);

  // Process one decode step for all active sequences.
  // Returns the number of sequences that are still active.
  int32_t Step();

  // Run until all pending and active requests are complete.
  void Run();

  // Accessors.
  int32_t NumActive() const { return static_cast<int32_t>(active_.size()); }
  int32_t NumPending() const { return static_cast<int32_t>(pending_.size()); }
  int32_t NumCompleted() const { return completed_count_; }

private:
  GemmaModel &model_;
  Tokenizer &tokenizer_;
  KVCacheManager &kv_cache_;

  std::deque<Request> pending_;
  std::vector<ActiveSequence> active_;
  int32_t completed_count_ = 0;

  // Maximum number of concurrent active sequences.
  int32_t max_batch_size_ = 8;

  // Prefill a pending request and move it to active.
  void PrefillNext();

  // Greedy sampling (scheduler uses greedy by default).
  int32_t SampleGreedy(const Tensor &logits) const;
};

} // namespace ie
