#include "engine/scheduler/batch_scheduler.h"

#include <iostream>
#include <stdexcept>

namespace ie {

BatchScheduler::BatchScheduler(GemmaModel &model, Tokenizer &tokenizer,
                               KVCacheManager &kv_cache)
    : model_(model), tokenizer_(tokenizer), kv_cache_(kv_cache) {}

void BatchScheduler::AddRequest(Request request) {
  pending_.push_back(std::move(request));
}

void BatchScheduler::PrefillNext() {
  if (pending_.empty())
    return;

  Request req = std::move(pending_.front());
  pending_.pop_front();

  // Tokenize and prepend BOS.
  std::vector<int32_t> tokens = tokenizer_.Encode(req.prompt);
  tokens.insert(tokens.begin(), tokenizer_.BosId());

  // Allocate a KV cache sequence.
  int64_t seq_id = kv_cache_.AllocateSequence();

  // Create float token vector for the model.
  std::vector<float> token_floats(tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    token_floats[i] = static_cast<float>(tokens[i]);
  }
  Tensor input_tensor = Tensor::from_vector(token_floats);

  // Run prefill: process the entire prompt.
  Tensor logits =
      model_.forward(input_tensor, /*start_pos=*/0, kv_cache_, seq_id);

  // Sample the first generated token.
  int32_t next_token = SampleGreedy(logits);
  tokens.push_back(next_token);

  // Stream the first token.
  if (req.on_token) {
    std::string decoded = tokenizer_.Decode({next_token});
    req.on_token(decoded);
  }

  // Create an active sequence.
  ActiveSequence seq;
  seq.seq_id = seq_id;
  seq.tokens = std::move(tokens);
  seq.prompt_len = static_cast<int32_t>(token_floats.size());
  seq.generated = 1;
  seq.max_tokens = req.max_tokens;
  seq.finished = (next_token == tokenizer_.EosId());
  seq.on_token = std::move(req.on_token);
  seq.on_complete = std::move(req.on_complete);

  active_.push_back(std::move(seq));
}

int32_t BatchScheduler::Step() {
  // Prefill pending requests up to max_batch_size.
  while (!pending_.empty() &&
         static_cast<int32_t>(active_.size()) < max_batch_size_) {
    PrefillNext();
  }

  // Process one decode step for each active sequence.
  for (auto &seq : active_) {
    if (seq.finished)
      continue;

    int32_t last_token = seq.tokens.back();

    // Create single-token input.
    Tensor single_token = Tensor::from_vector({static_cast<float>(last_token)});

    // start_pos = total tokens processed minus the current one.
    int32_t start_pos = static_cast<int32_t>(seq.tokens.size()) - 1;

    Tensor logits =
        model_.forward(single_token, start_pos, kv_cache_, seq.seq_id);

    int32_t next_token = SampleGreedy(logits);
    seq.tokens.push_back(next_token);
    seq.generated++;

    // Stream callback.
    if (seq.on_token) {
      std::string decoded = tokenizer_.Decode({next_token});
      seq.on_token(decoded);
    }

    // Check termination conditions.
    if (next_token == tokenizer_.EosId() || seq.generated >= seq.max_tokens) {
      seq.finished = true;
    }
  }

  // Remove finished sequences, call completion callbacks, free KV cache.
  auto it = active_.begin();
  while (it != active_.end()) {
    if (it->finished) {
      // Decode full output (skip BOS).
      std::vector<int32_t> output_tokens(it->tokens.begin() + 1,
                                         it->tokens.end());
      std::string full_text = tokenizer_.Decode(output_tokens);

      // Call completion callback.
      if (it->on_complete) {
        it->on_complete(full_text);
      }

      // Free KV cache blocks.
      kv_cache_.FreeSequence(it->seq_id);
      completed_count_++;

      it = active_.erase(it);
    } else {
      ++it;
    }
  }

  return static_cast<int32_t>(active_.size());
}

void BatchScheduler::Run() {
  while (!pending_.empty() || !active_.empty()) {
    Step();
  }
}

int32_t BatchScheduler::SampleGreedy(const Tensor &logits) const {
  const float *data = logits.data<float>();
  int64_t n = logits.numel();
  int32_t best_idx = 0;
  float best_val = data[0];
  for (int64_t i = 1; i < n; ++i) {
    if (data[i] > best_val) {
      best_val = data[i];
      best_idx = static_cast<int32_t>(i);
    }
  }
  return best_idx;
}

} // namespace ie
