#pragma once

#include <memory>
#include <string>

#include "engine/gguf/gguf_loader.h"
#include "engine/model/config.h"
#include "engine/model/layers.h"
#include "engine/tokenizer/tokenizer.h"

namespace ie {

// ============================================================================
// InferenceEngine — Top-level class that ties everything together.
//
// Usage:
//   InferenceEngine engine;
//   engine.LoadModel("model/gemma-3-1b-it-f16.gguf");
//   std::string output = engine.Generate("Hello, world!", 128);
// ============================================================================
class InferenceEngine {
 public:
  InferenceEngine() = default;

  // Load a GGUF model from disk.
  bool LoadModel(const std::string& model_path);

  // Generate text from a prompt.
  // TODO: Implement the generation loop:
  //   1. Tokenize the prompt
  //   2. Forward pass through the model → logits
  //   3. Sample next token from logits (greedy, top-k, top-p, temperature)
  //   4. Append to sequence, repeat until max_tokens or EOS
  //   5. Decode tokens back to text
  std::string Generate(const std::string& prompt, int32_t max_tokens = 128);

 private:
  GGUFFile gguf_;
  GemmaConfig config_;
  std::unique_ptr<GemmaModel> model_;
  std::unique_ptr<Tokenizer> tokenizer_;

  // Build the model from loaded GGUF weights.
  // TODO: Extract weights by name and construct layer objects
  bool BuildModel();

  // Build the tokenizer from GGUF metadata.
  // TODO: Read vocab, scores, and special token IDs from metadata
  bool BuildTokenizer();

  // --- Sampling Strategies ---
  // TODO: Implement these
  int32_t SampleGreedy(const Tensor& logits) const;
  int32_t SampleTopK(const Tensor& logits, int32_t k,
                     float temperature) const;
  int32_t SampleTopP(const Tensor& logits, float p, float temperature) const;
};

}  // namespace ie
