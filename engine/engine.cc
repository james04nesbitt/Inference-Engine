#include "engine/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

#include "engine/model/model_builder.h"
#include "engine/ops/ops.h"
#include "engine/scheduler/batch_scheduler.h"

namespace ie {

bool InferenceEngine::LoadModel(const std::string &model_path) {
  std::cout << "Loading model from: " << model_path << std::endl;

  if (!gguf_.Open(model_path)) {
    std::cerr << "Failed to open GGUF file" << std::endl;
    return false;
  }

  gguf_.PrintSummary();

  // Build tokenizer from GGUF metadata.
  tokenizer_ = ie::BuildTokenizer(gguf_);
  if (!tokenizer_) {
    std::cerr << "Warning: Failed to build tokenizer" << std::endl;
  }

  // Build model, config, and KV cache from GGUF tensors.
  try {
    ModelBundle bundle = ie::BuildGemmaModel(gguf_);
    config_ = std::move(bundle.config);
    model_ = std::move(bundle.model);
    kv_cache_ = std::move(bundle.kv_cache);
  } catch (const std::exception &e) {
    std::cerr << "Warning: Failed to build model: " << e.what() << std::endl;
  }

  std::cout << "\nModel loaded successfully!" << std::endl;
  return true;
}

// ============================================================================
// Generate — One-shot generation
// ============================================================================
std::string InferenceEngine::Generate(const std::string &prompt,
                                      int32_t max_tokens) {
  SamplingConfig config;
  config.strategy = SamplingStrategy::kGreedy;
  return GenerateStreaming(prompt, max_tokens, config, nullptr);
}

// ============================================================================
// GenerateStreaming — Token-by-token generation with callback
// ============================================================================
std::string InferenceEngine::GenerateStreaming(
    const std::string &prompt, int32_t max_tokens,
    const SamplingConfig &sampling_config,
    std::function<void(const std::string &)> on_token) {
  if (!model_ || !tokenizer_ || !kv_cache_) {
    throw std::runtime_error(
        "Model or tokenizer not loaded. Call LoadModel() first.");
  }

  // Allocate a fresh sequence in the KV cache.
  if (seq_id_ >= 0) {
    kv_cache_->FreeSequence(seq_id_);
  }
  seq_id_ = kv_cache_->AllocateSequence();

  // Step 1: Build token sequence with proper special token injection.
  // Instead of encoding the chat template string through BPE (which shreds
  // control tokens like <start_of_turn> into sub-pieces), we inject the
  // special token IDs directly and only BPE-encode the user text.
  int32_t start_of_turn_id = tokenizer_->TokenToId("<start_of_turn>");
  int32_t end_of_turn_id = tokenizer_->TokenToId("<end_of_turn>");
  if (start_of_turn_id < 0 || end_of_turn_id < 0) {
    throw std::runtime_error(
        "Could not find <start_of_turn>/<end_of_turn> in vocab. "
        "start_of_turn=" + std::to_string(start_of_turn_id) +
        " end_of_turn=" + std::to_string(end_of_turn_id));
  }

  std::vector<int32_t> tokens;
  tokens.push_back(tokenizer_->BosId());             // BOS
  tokens.push_back(start_of_turn_id);                // <start_of_turn>
  {
    auto user_tokens = tokenizer_->Encode("user\n" + prompt);
    tokens.insert(tokens.end(), user_tokens.begin(), user_tokens.end());
  }
  tokens.push_back(end_of_turn_id);                  // <end_of_turn>
  {
    auto newline_tokens = tokenizer_->Encode("\n");
    tokens.insert(tokens.end(), newline_tokens.begin(), newline_tokens.end());
  }
  tokens.push_back(start_of_turn_id);                // <start_of_turn>
  {
    auto model_tokens = tokenizer_->Encode("model\n");
    tokens.insert(tokens.end(), model_tokens.begin(), model_tokens.end());
  }
  int32_t prompt_len = static_cast<int32_t>(tokens.size());

  std::cerr << "Input tokens: [";
  for (int32_t t : tokens)
    std::cerr << t << ", ";
  std::cerr << "]\n";

  // Step 2: Prefill.
  std::vector<float> token_floats(tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    token_floats[i] = static_cast<float>(tokens[i]);
  }
  Tensor input_tensor = Tensor::from_vector(token_floats);

  auto t_start = std::chrono::high_resolution_clock::now();

  Tensor logits =
      model_->forward(input_tensor, /*start_pos=*/0, *kv_cache_, seq_id_);

  auto t_prefill = std::chrono::high_resolution_clock::now();
  double prefill_ms =
      std::chrono::duration<double, std::milli>(t_prefill - t_start).count();

  int32_t next_token = Sample(logits, sampling_config);
  tokens.push_back(next_token);
  std::cerr << "[GEN_0] " << next_token << std::endl;

  if (on_token) {
    std::string decoded = tokenizer_->Decode({next_token});
    on_token(decoded);
  }

  // Step 4: Autoregressive decode.
  int32_t generated_count = 1;
  for (int32_t i = 1; i < max_tokens; ++i) {
    if (next_token == tokenizer_->EosId()) {
      break;
    }

    Tensor single_token = Tensor::from_vector({static_cast<float>(next_token)});
    int32_t start_pos = prompt_len + i - 1;
    logits = model_->forward(single_token, start_pos, *kv_cache_, seq_id_);

    // DEBUG: Verify KV cache is growing and RoPE position is advancing.
    std::cerr << "[STEP " << i << "] kv_len="
              << kv_cache_->SequenceLength(seq_id_)
              << " rope_pos=" << start_pos << std::endl;

    next_token = Sample(logits, sampling_config);
    tokens.push_back(next_token);
    generated_count++;
    std::cerr << "[GEN_" << generated_count << "] " << next_token << std::endl;

    if (on_token) {
      std::string decoded = tokenizer_->Decode({next_token});
      on_token(decoded);
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double total_ms =
      std::chrono::duration<double, std::milli>(t_end - t_start).count();
  double decode_ms =
      std::chrono::duration<double, std::milli>(t_end - t_prefill).count();

  double tokens_per_sec =
      (decode_ms > 0) ? (generated_count * 1000.0 / decode_ms) : 0;
  std::cerr << "\n\n--- Stats ---"
            << "\n  Prompt tokens:  " << prompt_len
            << "\n  Generated:      " << generated_count
            << "\n  Prefill:        " << static_cast<int>(prefill_ms) << " ms"
            << "\n  Decode:         " << static_cast<int>(decode_ms) << " ms"
            << "\n  Total:          " << static_cast<int>(total_ms) << " ms"
            << "\n  Tokens/sec:     " << std::fixed << std::setprecision(2)
            << tokens_per_sec << std::endl;

  std::vector<int32_t> output_tokens(tokens.begin() + 1, tokens.end());
  return tokenizer_->Decode(output_tokens);
}

// ============================================================================
// GenerateBatch — Multiple prompts via BatchScheduler
// ============================================================================
void InferenceEngine::GenerateBatch(
    const std::vector<std::string> &prompts, int32_t max_tokens,
    const SamplingConfig &config,
    std::function<void(int32_t idx, const std::string &token)> on_token,
    std::function<void(int32_t idx, const std::string &result)> on_complete) {
  if (!model_ || !tokenizer_ || !kv_cache_) {
    throw std::runtime_error(
        "Model or tokenizer not loaded. Call LoadModel() first.");
  }

  BatchScheduler scheduler(*model_, *tokenizer_, *kv_cache_);

  for (int32_t i = 0; i < static_cast<int32_t>(prompts.size()); ++i) {
    Request req;
    req.prompt = prompts[i];
    req.max_tokens = max_tokens;
    if (on_token) {
      req.on_token = [on_token, i](const std::string &tok) {
        on_token(i, tok);
      };
    }
    if (on_complete) {
      req.on_complete = [on_complete, i](const std::string &result) {
        on_complete(i, result);
      };
    }
    scheduler.AddRequest(std::move(req));
  }

  scheduler.Run();
}

void InferenceEngine::ClearCache() {
  if (kv_cache_ && seq_id_ >= 0) {
    kv_cache_->FreeSequence(seq_id_);
    seq_id_ = -1;
  }
}

// ============================================================================
// Sample — Dispatch to selected sampling strategy
// ============================================================================
int32_t InferenceEngine::Sample(const Tensor &logits,
                                const SamplingConfig &config) const {
  switch (config.strategy) {
  case SamplingStrategy::kGreedy:
    return SampleGreedy(logits);
  case SamplingStrategy::kTopK:
    return SampleTopK(logits, config.top_k, config.temperature);
  case SamplingStrategy::kTopP:
    return SampleTopP(logits, config.top_p, config.temperature);
  default:
    return SampleGreedy(logits);
  }
}

// BuildModel() and BuildTokenizer() have been moved to
// engine/model/model_builder.{h,cc} as standalone functions.

// ============================================================================
// Sampling Strategies
// ============================================================================

int32_t InferenceEngine::SampleGreedy(const Tensor &logits) const {
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

int32_t InferenceEngine::SampleTopK(const Tensor &logits, int32_t k,
                                    float temperature) const {
  const float *data = logits.data<float>();
  int64_t n = logits.numel();

  std::vector<std::pair<float, int32_t>> indexed(n);
  for (int64_t i = 0; i < n; ++i) {
    indexed[i] = {data[i] / temperature, static_cast<int32_t>(i)};
  }

  k = std::min(k, static_cast<int32_t>(n));
  std::partial_sort(
      indexed.begin(), indexed.begin() + k, indexed.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });

  float max_val = indexed[0].first;
  std::vector<float> probs(k);
  float sum = 0.0f;
  for (int32_t i = 0; i < k; ++i) {
    probs[i] = std::exp(indexed[i].first - max_val);
    sum += probs[i];
  }
  for (int32_t i = 0; i < k; ++i) {
    probs[i] /= sum;
  }

  static std::mt19937 rng(42);
  std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
  return indexed[dist(rng)].second;
}

int32_t InferenceEngine::SampleTopP(const Tensor &logits, float p,
                                    float temperature) const {
  const float *data = logits.data<float>();
  int64_t n = logits.numel();

  std::vector<std::pair<float, int32_t>> indexed(n);
  for (int64_t i = 0; i < n; ++i) {
    indexed[i] = {data[i] / temperature, static_cast<int32_t>(i)};
  }

  std::sort(indexed.begin(), indexed.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  float max_val = indexed[0].first;
  std::vector<float> probs(n);
  float sum = 0.0f;
  for (int64_t i = 0; i < n; ++i) {
    probs[i] = std::exp(indexed[i].first - max_val);
    sum += probs[i];
  }
  for (int64_t i = 0; i < n; ++i) {
    probs[i] /= sum;
  }

  float cumsum = 0.0f;
  int32_t cutoff = static_cast<int32_t>(n);
  for (int64_t i = 0; i < n; ++i) {
    cumsum += probs[i];
    if (cumsum >= p) {
      cutoff = static_cast<int32_t>(i + 1);
      break;
    }
  }

  std::vector<float> kept_probs(probs.begin(), probs.begin() + cutoff);
  float kept_sum = 0.0f;
  for (float prob : kept_probs)
    kept_sum += prob;
  for (float &prob : kept_probs)
    prob /= kept_sum;

  static std::mt19937 rng(42);
  std::discrete_distribution<int32_t> dist(kept_probs.begin(),
                                           kept_probs.end());
  return indexed[dist(rng)].second;
}

} // namespace ie
