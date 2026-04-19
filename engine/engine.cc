#include "engine/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

#include "engine/ops/ops.h"
#include "engine/scheduler/batch_scheduler.h"
#include "engine/tokenizer/bpe_tokenizer.h"

namespace ie {

bool InferenceEngine::LoadModel(const std::string &model_path) {
  std::cout << "Loading model from: " << model_path << std::endl;

  if (!gguf_.Open(model_path)) {
    std::cerr << "Failed to open GGUF file" << std::endl;
    return false;
  }

  gguf_.PrintSummary();

  if (!BuildTokenizer()) {
    std::cerr << "Warning: Failed to build tokenizer" << std::endl;
  }

  if (!BuildModel()) {
    std::cerr << "Warning: Failed to build model" << std::endl;
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

  // Step 1: Tokenize and prepend BOS.
  std::vector<int32_t> tokens = tokenizer_->Encode(prompt);
  tokens.insert(tokens.begin(), tokenizer_->BosId());
  int32_t prompt_len = static_cast<int32_t>(tokens.size());

  std::cerr << "Input tokens: [";
  for (int32_t t : tokens) std::cerr << t << ", ";
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

bool InferenceEngine::BuildModel() {
  // Read model architecture name to determine key prefix.
  // Gemma 3 GGUFs use "gemma3." prefix; some older ones use "gemma.".
  std::string arch = gguf_.GetString("general.architecture", "gemma3");

  auto getInt = [&](const std::string &suffix, int64_t def) -> int64_t {
    int64_t val = gguf_.GetInt(arch + "." + suffix, -1);
    if (val != -1) return val;
    return gguf_.GetInt("gemma." + suffix, def);
  };
  auto getFloat = [&](const std::string &suffix, float def) -> float {
    float val = gguf_.GetFloat(arch + "." + suffix, -1.0f);
    if (val != -1.0f) return val;
    return gguf_.GetFloat("gemma." + suffix, def);
  };

  config_.num_layers =
      static_cast<int32_t>(getInt("block_count", 26));
  config_.embed_dim =
      static_cast<int32_t>(getInt("embedding_length", 1152));
  config_.num_heads =
      static_cast<int32_t>(getInt("attention.head_count", 4));
  config_.num_kv_heads =
      static_cast<int32_t>(getInt("attention.head_count_kv", 1));
  config_.head_dim =
      static_cast<int32_t>(getInt("attention.key_length", 256));
  config_.hidden_dim =
      static_cast<int32_t>(getInt("feed_forward_length", 6912));
  config_.vocab_size =
      static_cast<int32_t>(getInt("vocab_size", 262144));
  config_.rms_norm_eps =
      getFloat("attention.layer_norm_rms_epsilon", 1e-6f);
  config_.rope_theta_global =
      gguf_.GetFloat(arch + ".rope.freq_base", gguf_.GetFloat("gemma.rope.freq_base", 1e6f));
  config_.rope_theta_local = 10000.0f;
  config_.sliding_window =
      static_cast<int32_t>(getInt("attention.sliding_window", 512));

  std::cout << "\nModel config:"
            << "\n  layers:       " << config_.num_layers
            << "\n  embed_dim:    " << config_.embed_dim
            << "\n  num_heads:    " << config_.num_heads
            << "\n  num_kv_heads: " << config_.num_kv_heads
            << "\n  head_dim:     " << config_.head_dim
            << "\n  hidden_dim:   " << config_.hidden_dim
            << "\n  vocab_size:   " << config_.vocab_size << std::endl;

  Tensor token_embedding = gguf_.LoadTensor("token_embd.weight");
  // Pre-transpose embedding for logit projection: [vocab, embed] → [embed, vocab]
  Tensor embed_t = gguf_.LoadTensorTransposed("token_embd.weight");

  std::vector<TransformerBlock> layers;
  layers.reserve(config_.num_layers);

  for (int32_t i = 0; i < config_.num_layers; ++i) {
    std::string prefix = "blk." + std::to_string(i) + ".";

    // Load weight matrices pre-transposed (fused load + F16→F32 + transpose).
    // This avoids 8 intermediate allocations per layer.
    Tensor wq_t = gguf_.LoadTensorTransposed(prefix + "attn_q.weight");
    if (i == 0) {
      Tensor check = wq_t;
      std::cerr << "L0 wq_t.weight [0..4]: " << check.data<float>()[0] << ", " 
                << check.data<float>()[1] << ", " << check.data<float>()[2] << ", "
                << check.data<float>()[3] << ", " << check.data<float>()[4] << std::endl;
    }
    Tensor wk_t = gguf_.LoadTensorTransposed(prefix + "attn_k.weight");
    Tensor wv_t = gguf_.LoadTensorTransposed(prefix + "attn_v.weight");
    Tensor wo_t = gguf_.LoadTensorTransposed(prefix + "attn_output.weight");

    Tensor q_norm_w;
    if (gguf_.GetTensorInfo(prefix + "attn_q_norm.weight")) {
      q_norm_w = gguf_.LoadTensor(prefix + "attn_q_norm.weight");
      if (i == 0) {
        Tensor check = q_norm_w.to(DType::kFloat32);
        std::cerr << "L0 q_norm.weight [0..4]: " << check.data<float>()[0] << ", " 
                  << check.data<float>()[1] << ", " << check.data<float>()[2] << ", "
                  << check.data<float>()[3] << ", " << check.data<float>()[4] << std::endl;
      }
    }
    Tensor k_norm_w;
    if (gguf_.GetTensorInfo(prefix + "attn_k_norm.weight")) {
      k_norm_w = gguf_.LoadTensor(prefix + "attn_k_norm.weight");
    }

    Tensor gate_t = gguf_.LoadTensorTransposed(prefix + "ffn_gate.weight");
    Tensor up_t = gguf_.LoadTensorTransposed(prefix + "ffn_up.weight");
    Tensor down_t = gguf_.LoadTensorTransposed(prefix + "ffn_down.weight");

    Tensor attn_norm_w = gguf_.LoadTensor(prefix + "attn_norm.weight");
    Tensor ffn_norm_w = gguf_.LoadTensor(prefix + "ffn_norm.weight");

    RMSNorm attn_norm(std::move(attn_norm_w), config_.rms_norm_eps);
    Attention attn(config_, i, std::move(wq_t), std::move(wk_t),
                   std::move(wv_t), std::move(wo_t),
                   std::move(q_norm_w), std::move(k_norm_w));
    RMSNorm ffn_norm(std::move(ffn_norm_w), config_.rms_norm_eps);
    FeedForward ffn(std::move(gate_t), std::move(up_t), std::move(down_t));

    layers.emplace_back(config_, i, std::move(attn_norm), std::move(attn),
                        std::move(ffn_norm), std::move(ffn));

    std::cout << "  Loaded block " << i << std::endl;
  }

  Tensor final_norm_w = gguf_.LoadTensor("output_norm.weight");
  RMSNorm final_norm(std::move(final_norm_w), config_.rms_norm_eps);

  model_ = std::make_unique<GemmaModel>(config_, std::move(token_embedding),
                                        std::move(layers),
                                        std::move(final_norm),
                                        std::move(embed_t));

  // Create the paged KV cache.
  int32_t blocks_per_seq =
      (config_.max_seq_len + kDefaultBlockSize - 1) / kDefaultBlockSize;
  // Allocate enough blocks for a few concurrent sequences.
  int32_t max_blocks =
      blocks_per_seq * config_.num_layers * 4 + config_.num_layers;

  kv_cache_ = std::make_unique<KVCacheManager>(
      config_.num_layers, config_.num_kv_heads, config_.head_dim, max_blocks,
      kDefaultBlockSize);

  std::cout << "Model built successfully (" << config_.num_layers << " layers, "
            << max_blocks << " KV cache blocks)" << std::endl;
  return true;
}

bool InferenceEngine::BuildTokenizer() {
  auto *tokens_val = gguf_.GetMetadata("tokenizer.ggml.tokens");
  if (!tokens_val) {
    std::cerr << "BuildTokenizer: missing tokenizer.ggml.tokens" << std::endl;
    return false;
  }
  auto *vocab = std::get_if<std::vector<std::string>>(tokens_val);
  if (!vocab) {
    std::cerr << "BuildTokenizer: tokenizer.ggml.tokens is not a string array"
              << std::endl;
    return false;
  }

  auto *scores_val = gguf_.GetMetadata("tokenizer.ggml.scores");
  if (!scores_val) {
    std::cerr << "BuildTokenizer: missing tokenizer.ggml.scores" << std::endl;
    return false;
  }
  auto *scores = std::get_if<std::vector<float>>(scores_val);
  if (!scores) {
    std::cerr << "BuildTokenizer: tokenizer.ggml.scores is not a float array"
              << std::endl;
    return false;
  }

  int32_t bos_id =
      static_cast<int32_t>(gguf_.GetInt("tokenizer.ggml.bos_token_id", 2));
  int32_t eos_id =
      static_cast<int32_t>(gguf_.GetInt("tokenizer.ggml.eos_token_id", 1));
  int32_t pad_id =
      static_cast<int32_t>(gguf_.GetInt("tokenizer.ggml.padding_token_id", 0));

  tokenizer_ =
      std::make_unique<BPETokenizer>(*vocab, *scores, bos_id, eos_id, pad_id);

  std::cout << "Tokenizer built: " << tokenizer_->VocabSize() << " tokens"
            << ", BOS=" << bos_id << ", EOS=" << eos_id << ", PAD=" << pad_id
            << std::endl;
  return true;
}

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
