#include "engine/engine.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

#include "engine/ops/ops.h"
#include "engine/tokenizer/bpe_tokenizer.h"

namespace ie {

bool InferenceEngine::LoadModel(const std::string &model_path) {
  std::cout << "Loading model from: " << model_path << std::endl;

  // Step 1: Parse the GGUF file.
  if (!gguf_.Open(model_path)) {
    std::cerr << "Failed to open GGUF file" << std::endl;
    return false;
  }

  gguf_.PrintSummary();

  // Step 2: Build tokenizer from GGUF metadata.
  if (!BuildTokenizer()) {
    std::cerr << "Warning: Failed to build tokenizer" << std::endl;
  }

  // Step 3: Build the model from weights.
  if (!BuildModel()) {
    std::cerr << "Warning: Failed to build model" << std::endl;
  }

  std::cout << "\nModel loaded successfully!" << std::endl;
  return true;
}

std::string InferenceEngine::Generate(const std::string &prompt,
                                      int32_t max_tokens) {
  if (!model_ || !tokenizer_) {
    throw std::runtime_error(
        "Model or tokenizer not loaded. Call LoadModel() first.");
  }

  // Step 1: Tokenize the prompt and prepend BOS.
  std::vector<int32_t> tokens = tokenizer_->Encode(prompt);
  tokens.insert(tokens.begin(), tokenizer_->BosId());
  int32_t prompt_len = static_cast<int32_t>(tokens.size());

  std::cout << "Prompt tokens: " << prompt_len << std::endl;

  // Step 2: Prefill — run the full prompt through the model.
  // This populates the KV cache for all prompt positions.
  std::vector<float> token_floats(tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    token_floats[i] = static_cast<float>(tokens[i]);
  }
  Tensor input_tensor = Tensor::from_vector(token_floats);

  Tensor logits = model_->forward(input_tensor, /*start_pos=*/0);

  // Step 3: Sample the first generated token.
  int32_t next_token = SampleGreedy(logits);
  tokens.push_back(next_token);

  std::cout << "Generating..." << std::flush;

  // Step 4: Autoregressive decode loop with KV cache.
  // Each step only passes the single new token; the KV cache holds the history.
  for (int32_t i = 1; i < max_tokens; ++i) {
    if (next_token == tokenizer_->EosId()) {
      break;
    }

    // Create input with just the new token.
    Tensor single_token = Tensor::from_vector({static_cast<float>(next_token)});

    // start_pos = total tokens generated so far (prompt + generated).
    int32_t start_pos = prompt_len + i - 1;
    logits = model_->forward(single_token, start_pos);

    next_token = SampleGreedy(logits);
    tokens.push_back(next_token);

    std::cout << "." << std::flush;
  }

  std::cout << " done! (" << tokens.size() << " tokens)" << std::endl;

  // Step 5: Decode tokens back to text (skip BOS).
  std::vector<int32_t> output_tokens(tokens.begin() + 1, tokens.end());
  return tokenizer_->Decode(output_tokens);
}

bool InferenceEngine::BuildModel() {
  // Read model config from GGUF metadata.
  config_.num_layers =
      static_cast<int32_t>(gguf_.GetInt("gemma.block_count", 26));
  config_.embed_dim =
      static_cast<int32_t>(gguf_.GetInt("gemma.embedding_length", 1152));
  config_.num_heads =
      static_cast<int32_t>(gguf_.GetInt("gemma.attention.head_count", 4));
  config_.num_kv_heads =
      static_cast<int32_t>(gguf_.GetInt("gemma.attention.head_count_kv", 1));
  config_.head_dim =
      static_cast<int32_t>(gguf_.GetInt("gemma.attention.key_length", 256));
  config_.hidden_dim =
      static_cast<int32_t>(gguf_.GetInt("gemma.feed_forward_length", 6912));
  config_.vocab_size =
      static_cast<int32_t>(gguf_.GetInt("gemma.vocab_size", 262144));
  config_.rms_norm_eps =
      gguf_.GetFloat("gemma.attention.layer_norm_rms_epsilon", 1e-6f);

  std::cout << "\nModel config:"
            << "\n  layers:       " << config_.num_layers
            << "\n  embed_dim:    " << config_.embed_dim
            << "\n  num_heads:    " << config_.num_heads
            << "\n  num_kv_heads: " << config_.num_kv_heads
            << "\n  head_dim:     " << config_.head_dim
            << "\n  hidden_dim:   " << config_.hidden_dim
            << "\n  vocab_size:   " << config_.vocab_size << std::endl;

  // Load the token embedding table.
  Tensor token_embedding = gguf_.LoadTensor("token_embd.weight");

  // Build each transformer block.
  std::vector<TransformerBlock> layers;
  layers.reserve(config_.num_layers);

  for (int32_t i = 0; i < config_.num_layers; ++i) {
    std::string prefix = "blk." + std::to_string(i) + ".";

    // Attention weights.
    Tensor wq = gguf_.LoadTensor(prefix + "attn_q.weight");
    Tensor wk = gguf_.LoadTensor(prefix + "attn_k.weight");
    Tensor wv = gguf_.LoadTensor(prefix + "attn_v.weight");
    Tensor wo = gguf_.LoadTensor(prefix + "attn_output.weight");

    // FFN weights.
    Tensor w_gate = gguf_.LoadTensor(prefix + "ffn_gate.weight");
    Tensor w_up = gguf_.LoadTensor(prefix + "ffn_up.weight");
    Tensor w_down = gguf_.LoadTensor(prefix + "ffn_down.weight");

    // Normalization weights.
    Tensor attn_norm_w = gguf_.LoadTensor(prefix + "attn_norm.weight");
    Tensor ffn_norm_w = gguf_.LoadTensor(prefix + "ffn_norm.weight");

    // Construct the block.
    RMSNorm attn_norm(std::move(attn_norm_w), config_.rms_norm_eps);
    Attention attn(config_, std::move(wq), std::move(wk), std::move(wv),
                   std::move(wo));
    RMSNorm ffn_norm(std::move(ffn_norm_w), config_.rms_norm_eps);
    FeedForward ffn(std::move(w_gate), std::move(w_up), std::move(w_down));

    layers.emplace_back(config_, std::move(attn_norm), std::move(attn),
                        std::move(ffn_norm), std::move(ffn));

    std::cout << "  Loaded block " << i << std::endl;
  }

  // Load final norm weight.
  Tensor final_norm_w = gguf_.LoadTensor("output_norm.weight");
  RMSNorm final_norm(std::move(final_norm_w), config_.rms_norm_eps);

  // Assemble the model.
  model_ =
      std::make_unique<GemmaModel>(config_, std::move(token_embedding),
                                   std::move(layers), std::move(final_norm));

  std::cout << "Model built successfully (" << config_.num_layers << " layers)"
            << std::endl;
  return true;
}

bool InferenceEngine::BuildTokenizer() {
  // Extract vocabulary tokens from GGUF metadata.
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

  // Extract scores.
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

  // Extract special token IDs (with reasonable defaults).
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
  // Argmax: find the token with the highest logit value.
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

  // Create index-value pairs and partial sort to get top-k.
  std::vector<std::pair<float, int32_t>> indexed(n);
  for (int64_t i = 0; i < n; ++i) {
    indexed[i] = {data[i] / temperature, static_cast<int32_t>(i)};
  }

  k = std::min(k, static_cast<int32_t>(n));
  std::partial_sort(
      indexed.begin(), indexed.begin() + k, indexed.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });

  // Softmax over top-k logits.
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

  // Sample from the distribution.
  static std::mt19937 rng(42);
  std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
  return indexed[dist(rng)].second;
}

int32_t InferenceEngine::SampleTopP(const Tensor &logits, float p,
                                    float temperature) const {
  const float *data = logits.data<float>();
  int64_t n = logits.numel();

  // Temperature-scale and create index-value pairs.
  std::vector<std::pair<float, int32_t>> indexed(n);
  for (int64_t i = 0; i < n; ++i) {
    indexed[i] = {data[i] / temperature, static_cast<int32_t>(i)};
  }

  // Sort by logit descending.
  std::sort(indexed.begin(), indexed.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  // Softmax over all logits.
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

  // Nucleus: keep tokens until cumulative probability >= p.
  float cumsum = 0.0f;
  int32_t cutoff = static_cast<int32_t>(n);
  for (int64_t i = 0; i < n; ++i) {
    cumsum += probs[i];
    if (cumsum >= p) {
      cutoff = static_cast<int32_t>(i + 1);
      break;
    }
  }

  // Renormalize the kept probabilities.
  std::vector<float> kept_probs(probs.begin(), probs.begin() + cutoff);
  float kept_sum = 0.0f;
  for (float prob : kept_probs)
    kept_sum += prob;
  for (float &prob : kept_probs)
    prob /= kept_sum;

  // Sample from the distribution.
  static std::mt19937 rng(42);
  std::discrete_distribution<int32_t> dist(kept_probs.begin(),
                                           kept_probs.end());
  return indexed[dist(rng)].second;
}

} // namespace ie
