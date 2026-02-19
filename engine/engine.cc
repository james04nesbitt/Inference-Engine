#include "engine/engine.h"

#include <iostream>
#include <stdexcept>

namespace ie {

bool InferenceEngine::LoadModel(const std::string& model_path) {
  std::cout << "Loading model from: " << model_path << std::endl;

  // Step 1: Parse the GGUF file
  if (!gguf_.Open(model_path)) {
    std::cerr << "Failed to open GGUF file" << std::endl;
    return false;
  }

  gguf_.PrintSummary();

  // Step 2: Build tokenizer from GGUF metadata
  if (!BuildTokenizer()) {
    std::cerr << "Warning: Failed to build tokenizer" << std::endl;
    // Don't fail — we can still inspect the model
  }

  // Step 3: Build the model from weights
  if (!BuildModel()) {
    std::cerr << "Warning: Failed to build model" << std::endl;
    // Don't fail — we can still inspect metadata
  }

  std::cout << "\nModel loaded successfully!" << std::endl;
  return true;
}

std::string InferenceEngine::Generate(const std::string& prompt,
                                      int32_t max_tokens) {
  // TODO: Implement the autoregressive generation loop
  //
  // Pseudocode:
  //   tokens = tokenizer_->Encode(prompt)
  //   tokens.insert(tokens.begin(), tokenizer_->BosId())
  //
  //   for (int i = 0; i < max_tokens; ++i) {
  //     Tensor input_tokens = ...  // Create tensor from token IDs
  //     Tensor logits = model_->forward(input_tokens, start_pos)
  //     int32_t next_token = SampleGreedy(logits)
  //     if (next_token == tokenizer_->EosId()) break
  //     tokens.push_back(next_token)
  //   }
  //
  //   return tokenizer_->Decode(tokens)

  throw std::runtime_error(
      "Generate() not implemented yet — implement the forward pass first!");
}

bool InferenceEngine::BuildModel() {
  // TODO: Implement model construction
  //
  // Steps:
  //   1. Read config from GGUF metadata
  //      config_.num_layers = gguf_.GetInt("gemma.block_count", 26);
  //      config_.embed_dim = gguf_.GetInt("gemma.embedding_length", 1152);
  //      ... etc
  //
  //   2. For each layer, load weight tensors:
  //      Tensor wq = gguf_.LoadTensor("blk.0.attn_q.weight");
  //      Tensor wk = gguf_.LoadTensor("blk.0.attn_k.weight");
  //      ... etc
  //
  //   3. Construct layer objects and assemble the model

  std::cerr << "BuildModel: NOT YET IMPLEMENTED" << std::endl;
  return false;
}

bool InferenceEngine::BuildTokenizer() {
  // TODO: Implement tokenizer construction from GGUF metadata
  //
  // The GGUF file stores tokenizer data under these keys:
  //   tokenizer.ggml.model       -> "llama" (SentencePiece BPE)
  //   tokenizer.ggml.tokens      -> array of token strings
  //   tokenizer.ggml.scores      -> array of float scores
  //   tokenizer.ggml.bos_token_id -> int
  //   tokenizer.ggml.eos_token_id -> int
  //   tokenizer.ggml.padding_token_id -> int
  //
  // Once you've parsed these from metadata (in ReadMetadata):
  //   auto tokens = std::get<std::vector<std::string>>(
  //       *gguf_.GetMetadata("tokenizer.ggml.tokens"));
  //   auto scores = std::get<std::vector<float>>(
  //       *gguf_.GetMetadata("tokenizer.ggml.scores"));
  //   tokenizer_ = std::make_unique<BPETokenizer>(tokens, scores, bos, eos, pad);

  std::cerr << "BuildTokenizer: NOT YET IMPLEMENTED" << std::endl;
  return false;
}

int32_t InferenceEngine::SampleGreedy(const Tensor& logits) const {
  // TODO: Return the index of the maximum logit value
  // This is argmax — the simplest sampling strategy.
  throw std::runtime_error("SampleGreedy not implemented yet");
}

int32_t InferenceEngine::SampleTopK(const Tensor& logits, int32_t k,
                                    float temperature) const {
  // TODO: Top-K sampling
  //   1. Divide logits by temperature
  //   2. Keep only the top-k values
  //   3. Apply softmax
  //   4. Sample from the resulting distribution
  throw std::runtime_error("SampleTopK not implemented yet");
}

int32_t InferenceEngine::SampleTopP(const Tensor& logits, float p,
                                    float temperature) const {
  // TODO: Nucleus (top-p) sampling
  //   1. Divide logits by temperature
  //   2. Sort probabilities descending
  //   3. Keep tokens until cumulative probability >= p
  //   4. Sample from those tokens
  throw std::runtime_error("SampleTopP not implemented yet");
}

}  // namespace ie
