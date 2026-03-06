#pragma once

#include <functional>
#include <memory>
#include <string>

#include "engine/gguf/gguf_loader.h"
#include "engine/model/config.h"
#include "engine/model/layers.h"
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
//
// Usage:
//   InferenceEngine engine;
//   engine.LoadModel("model/gemma-3-1b-it-f16.gguf");
//
//   // One-shot:
//   std::string output = engine.Generate("Hello!", 128);
//
//   // Streaming:
//   engine.GenerateStreaming("Hello!", 128, config,
//       [](const std::string& token) { std::cout << token << std::flush; });
// ============================================================================
class InferenceEngine {
public:
  InferenceEngine() = default;

  // Load a GGUF model from disk.
  bool LoadModel(const std::string &model_path);

  // Generate text from a prompt (returns full output string).
  std::string Generate(const std::string &prompt, int32_t max_tokens = 128);

  // Generate text with streaming token-by-token output.
  // The callback is called with each decoded token as it's generated.
  // Returns the full generated string.
  std::string GenerateStreaming(
      const std::string &prompt, int32_t max_tokens,
      const SamplingConfig &config,
      std::function<void(const std::string &)> on_token = nullptr);

  // Clear KV caches (call between separate prompts in interactive mode).
  void ClearCache();

private:
  GGUFFile gguf_;
  GemmaConfig config_;
  std::unique_ptr<GemmaModel> model_;
  std::unique_ptr<Tokenizer> tokenizer_;

  bool BuildModel();
  bool BuildTokenizer();

  // --- Sampling ---
  int32_t Sample(const Tensor &logits, const SamplingConfig &config) const;
  int32_t SampleGreedy(const Tensor &logits) const;
  int32_t SampleTopK(const Tensor &logits, int32_t k, float temperature) const;
  int32_t SampleTopP(const Tensor &logits, float p, float temperature) const;
};

} // namespace ie
