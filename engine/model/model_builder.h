#pragma once

#include <memory>
#include <string>

#include "engine/attention/kv_cache.h"
#include "engine/gguf/gguf_loader.h"
#include "engine/model/config.h"
#include "engine/model/layers.h"
#include "engine/tokenizer/tokenizer.h"

namespace ie {

// ============================================================================
// Model Builder — Standalone functions that construct model components
// from a parsed GGUF file. Extracted from InferenceEngine to keep the
// engine focused on orchestration and generation.
// ============================================================================

// Result of BuildGemmaModel: the model, its config, and a KV cache manager.
struct ModelBundle {
  GemmaConfig config;
  std::unique_ptr<GemmaModel> model;
  std::unique_ptr<KVCacheManager> kv_cache;
};

// Build the GemmaModel, GemmaConfig, and KVCacheManager from a GGUF file.
// Loads all weight tensors and constructs the transformer layers.
ModelBundle BuildGemmaModel(GGUFFile &gguf);

// Build a tokenizer from GGUF metadata (vocab, scores, special token IDs).
std::unique_ptr<Tokenizer> BuildTokenizer(GGUFFile &gguf);

// ============================================================================
// Chat Template — Wraps a user message in the Gemma 3 instruction format.
//
// Format:
//   <start_of_turn>user
//   {message}<end_of_turn>
//   <start_of_turn>model
//
// The BOS token is handled separately by the generation code (prepended
// before tokenization), so it is NOT included here.
// ============================================================================
std::string ApplyGemma3ChatTemplate(const std::string &user_message);

}  // namespace ie
