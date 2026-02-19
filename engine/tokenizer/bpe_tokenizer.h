#pragma once

#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "engine/tokenizer/tokenizer.h"

namespace ie {

// ============================================================================
// BPETokenizer — Byte Pair Encoding tokenizer.
//
// This tokenizer reads vocabulary and merge rules from the GGUF metadata.
// BPE works by:
//   1. Start with each character as its own token
//   2. Repeatedly merge the most frequent adjacent pair
//   3. Until no more merges apply
//
// The vocabulary and merge rules are learned during model training and
// stored in the GGUF file under tokenizer.ggml.* keys.
// ============================================================================
class BPETokenizer : public Tokenizer {
 public:
  // Build a tokenizer from vocab and merge tables.
  //   vocab: list of token strings, indexed by token ID
  //   scores: score for each token (higher = higher priority merge)
  //   merges: BPE merge rules as "tokenA tokenB" strings
  BPETokenizer(std::vector<std::string> vocab, std::vector<float> scores,
               int32_t bos_id, int32_t eos_id, int32_t pad_id);

  std::vector<int32_t> Encode(const std::string& text) const override;
  std::string Decode(const std::vector<int32_t>& tokens) const override;
  int32_t VocabSize() const override {
    return static_cast<int32_t>(vocab_.size());
  }
  int32_t BosId() const override { return bos_id_; }
  int32_t EosId() const override { return eos_id_; }
  int32_t PadId() const override { return pad_id_; }

 private:
  std::vector<std::string> vocab_;        // id -> token string
  std::vector<float> scores_;             // id -> merge score
  std::unordered_map<std::string, int32_t> token_to_id_;  // token -> id

  int32_t bos_id_;
  int32_t eos_id_;
  int32_t pad_id_;

  // Find the token ID for a string, or -1 if not in vocab.
  int32_t LookupToken(const std::string& token) const;
};

}  // namespace ie
