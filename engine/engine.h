#pragma once

#include <functional>
#include <memory>
#include <string>

#include "engine/attention/kv_cache.h"
#include "engine/gguf/gguf_loader.h"
#include "engine/model/config.h"
#include "engine/model/layers.h"
#include "engine/model/model_builder.h"
#include "engine/tokenizer/tokenizer.h"

namespace ie {

// ============================================================================
// Sampling Configuration
// ============================================================================
enum class SamplingStrategy { kGreedy, kTopK, kTopP };

struct SamplingConfig {
  SamplingStrategy strategy = SamplingStrategy::kGreedy;
  float temperature = 1.0f;
  int32_t top_k = 40;
  float top_p = 0.9f;
};

// ============================================================================
// InferenceEngine — Top-level class that ties everything together.
// ============================================================================
class InferenceEngine {
public:
  InferenceEngine() = default;

  bool LoadModel(const std::string &model_path);

  // Single-request generation.
  std::string Generate(const std::string &prompt, int32_t max_tokens = 128);

  // Streaming generation with per-token callback.
  std::string GenerateStreaming(
      const std::string &prompt, int32_t max_tokens,
      const SamplingConfig &config,
      std::function<void(const std::string &)> on_token = nullptr);

  // Batch generation: submit multiple prompts, get results via callbacks.
  void GenerateBatch(
      const std::vector<std::string> &prompts, int32_t max_tokens,
      const SamplingConfig &config,
      std::function<void(int32_t idx, const std::string &token)> on_token,
      std::function<void(int32_t idx, const std::string &result)> on_complete);

  // Clear KV caches.
  void ClearCache();

  // Accessors for scheduler integration.
  GemmaModel *model() { return model_.get(); }
  Tokenizer *tokenizer() { return tokenizer_.get(); }
  KVCacheManager *kv_cache() { return kv_cache_.get(); }

private:
  GGUFFile gguf_;
  GemmaConfig config_;
  std::unique_ptr<GemmaModel> model_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<KVCacheManager> kv_cache_;

  // Active sequence for single-request mode.
  int64_t seq_id_ = -1;

  // --- Sampling ---
  int32_t Sample(const Tensor &logits, const SamplingConfig &config) const;
  int32_t SampleGreedy(const Tensor &logits) const;
  int32_t SampleTopK(const Tensor &logits, int32_t k, float temperature) const;
  int32_t SampleTopP(const Tensor &logits, float p, float temperature) const;
};

} // namespace ie
