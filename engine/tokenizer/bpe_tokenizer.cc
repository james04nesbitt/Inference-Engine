#include "engine/tokenizer/bpe_tokenizer.h"

#include <iostream>
#include <limits>
#include <sstream>

namespace ie {

BPETokenizer::BPETokenizer(std::vector<std::string> vocab,
                           std::vector<float> scores, int32_t bos_id,
                           int32_t eos_id, int32_t pad_id)
    : vocab_(std::move(vocab)),
      scores_(std::move(scores)),
      bos_id_(bos_id),
      eos_id_(eos_id),
      pad_id_(pad_id) {
  // Build reverse lookup map
  for (int32_t i = 0; i < static_cast<int32_t>(vocab_.size()); ++i) {
    token_to_id_[vocab_[i]] = i;
  }
  std::cout << "BPETokenizer initialized with " << vocab_.size() << " tokens"
            << std::endl;
}

std::vector<int32_t> BPETokenizer::Encode(const std::string& text) const {
  // ============================================================================
  // BPE Encoding Algorithm — Implement this!
  //
  // The algorithm:
  //   1. Convert input text to a list of individual UTF-8 byte tokens
  //      (or characters, depending on the tokenizer variant).
  //   2. Look up each byte/char in the vocab to get initial token IDs.
  //   3. Repeatedly find the adjacent pair with the highest merge score:
  //      a. For each adjacent pair (tokens[i], tokens[i+1]):
  //         - Concatenate them: merged = vocab_[tokens[i]] + vocab_[tokens[i+1]]
  //         - Check if merged exists in token_to_id_
  //         - If yes, record its score from scores_
  //      b. Find the pair with the highest score
  //      c. Replace all occurrences of that pair with the merged token
  //   4. Repeat step 3 until no more merges are possible.
  //   5. Return the final list of token IDs.
  //
  // Example: "hello" ->
  //   ['h', 'e', 'l', 'l', 'o']
  //   -> ['h', 'e', 'll', 'o']     (merge 'l'+'l' -> 'll')
  //   -> ['he', 'll', 'o']         (merge 'h'+'e' -> 'he')
  //   -> ['hell', 'o']             (merge 'he'+'ll' -> 'hell')
  //   -> ['hello']                 (merge 'hell'+'o' -> 'hello')
  //
  // Edge cases to handle:
  //   - Unknown bytes (not in vocab)
  //   - UTF-8 multi-byte characters
  //   - SentencePiece-style space handling (▁ prefix)
  // ============================================================================

  // Placeholder: split into individual characters and look up each one
  std::vector<int32_t> tokens;
  for (size_t i = 0; i < text.size(); ++i) {
    std::string ch(1, text[i]);
    int32_t id = LookupToken(ch);
    if (id >= 0) {
      tokens.push_back(id);
    } else {
      // Unknown byte — you'll need a fallback strategy
      std::cerr << "Warning: unknown byte in input: 0x" << std::hex
                << static_cast<int>(static_cast<uint8_t>(text[i])) << std::dec
                << std::endl;
    }
  }

  // TODO: Implement the BPE merge loop here!
  // The tokens vector currently contains per-character IDs.
  // You need to iteratively merge adjacent pairs.

  return tokens;
}

std::string BPETokenizer::Decode(const std::vector<int32_t>& tokens) const {
  std::string result;
  for (int32_t id : tokens) {
    if (id >= 0 && id < static_cast<int32_t>(vocab_.size())) {
      result += vocab_[id];
    }
  }
  // TODO: Handle SentencePiece-style space tokens (replace ▁ with space)
  return result;
}

int32_t BPETokenizer::LookupToken(const std::string& token) const {
  auto it = token_to_id_.find(token);
  return (it != token_to_id_.end()) ? it->second : -1;
}

}  // namespace ie
