#include "engine/tokenizer/bpe_tokenizer.h"

#include <cstdio>
#include <limits>

namespace ie {

// The SentencePiece space marker (U+2581, "▁").
// SentencePiece encodes leading spaces as this character in the vocabulary.
static const std::string kSpaceSymbol = "\xe2\x96\x81"; // UTF-8 for ▁

BPETokenizer::BPETokenizer(std::vector<std::string> vocab,
                           std::vector<float> scores, int32_t bos_id,
                           int32_t eos_id, int32_t pad_id)
    : vocab_(std::move(vocab)), scores_(std::move(scores)), bos_id_(bos_id),
      eos_id_(eos_id), pad_id_(pad_id) {
  // Build reverse lookup map
  for (int32_t i = 0; i < static_cast<int32_t>(vocab_.size()); ++i) {
    token_to_id_[vocab_[i]] = i;
  }
}

// ============================================================================
// Encode — SentencePiece-style BPE encoding.
//
// 1. Prepend the ▁ space symbol (SentencePiece convention).
// 2. Split into individual UTF-8 characters and look up each in the vocab.
// 3. Iteratively merge the adjacent pair with the highest score until
//    no more merges are possible.
// ============================================================================
std::vector<int32_t> BPETokenizer::Encode(const std::string &text) const {
  if (text.empty())
    return {};

  // Step 1: Prepend the SentencePiece space marker.
  std::string processed = kSpaceSymbol + text;

  // Step 2: Split into UTF-8 characters and look up initial token IDs.
  // We need to keep both the token string and its ID for merging.
  std::vector<int32_t> tokens;

  size_t i = 0;
  while (i < processed.size()) {
    // Determine the length of the current UTF-8 character.
    uint8_t byte = static_cast<uint8_t>(processed[i]);
    size_t char_len = 1;
    if ((byte & 0x80) == 0) {
      char_len = 1; // ASCII
    } else if ((byte & 0xE0) == 0xC0) {
      char_len = 2; // 2-byte UTF-8
    } else if ((byte & 0xF0) == 0xE0) {
      char_len = 3; // 3-byte UTF-8 (includes ▁)
    } else if ((byte & 0xF8) == 0xF0) {
      char_len = 4; // 4-byte UTF-8
    }

    // Don't read past the end of the string.
    if (i + char_len > processed.size()) {
      char_len = 1;
    }

    std::string ch = processed.substr(i, char_len);
    int32_t id = LookupToken(ch);
    if (id >= 0) {
      tokens.push_back(id);
    } else {
      // Fallback: try each byte individually (byte-level BPE).
      for (size_t b = 0; b < char_len; ++b) {
        std::string single_byte(1, processed[i + b]);
        int32_t byte_id = LookupToken(single_byte);
        if (byte_id >= 0) {
          tokens.push_back(byte_id);
        }
        // If even the byte isn't in vocab, silently skip it.
      }
    }
    i += char_len;
  }

  // Step 3: BPE merge loop.
  // Repeatedly find the adjacent pair that, when merged, has the highest
  // score in scores_[]. Replace it, and repeat until no merges remain.
  while (tokens.size() >= 2) {
    // Find the best merge: the pair whose merged token has the highest score.
    float best_score = -std::numeric_limits<float>::infinity();
    int32_t best_id = -1;
    size_t best_idx = 0;

    for (size_t j = 0; j < tokens.size() - 1; ++j) {
      std::string merged = vocab_[tokens[j]] + vocab_[tokens[j + 1]];
      int32_t merged_id = LookupToken(merged);
      if (merged_id >= 0 && scores_[merged_id] > best_score) {
        best_score = scores_[merged_id];
        best_id = merged_id;
        best_idx = j;
      }
    }

    // No valid merge found — we're done.
    if (best_id < 0)
      break;

    // Apply the merge: replace tokens[best_idx] and tokens[best_idx+1]
    // with the merged token.
    tokens[best_idx] = best_id;
    tokens.erase(tokens.begin() + static_cast<ptrdiff_t>(best_idx) + 1);
  }

  return tokens;
}

// ============================================================================
// Decode — Convert token IDs back to text.
//
// Concatenate all token strings, then replace ▁ with spaces and trim
// the leading space that was added during encoding.
// ============================================================================
std::string BPETokenizer::Decode(const std::vector<int32_t> &tokens) const {
  std::string result;
  for (int32_t id : tokens) {
    if (id >= 0 && id < static_cast<int32_t>(vocab_.size())) {
      const std::string &text = vocab_[id];
      // Detect byte-fallback tokens: <0xXX> (6 chars: '<','0','x',H,H,'>')
      if (text.size() == 6 && text[0] == '<' && text[1] == '0' &&
          text[2] == 'x' && text[5] == '>') {
        unsigned int byte_val = 0;
        if (std::sscanf(text.c_str(), "<0x%02X>", &byte_val) == 1) {
          result += static_cast<char>(byte_val);
        } else {
          result += text;
        }
      } else {
        result += text;
      }
    }
  }

  // Replace all SentencePiece space markers (▁) with regular spaces.
  size_t pos = 0;
  while ((pos = result.find(kSpaceSymbol, pos)) != std::string::npos) {
    result.replace(pos, kSpaceSymbol.size(), " ");
    pos += 1; // Move past the replacement.
  }

  // Trim the leading space that was prepended during Encode().
  if (!result.empty() && result[0] == ' ') {
    result.erase(0, 1);
  }

  return result;
}

int32_t BPETokenizer::LookupToken(const std::string &token) const {
  auto it = token_to_id_.find(token);
  return (it != token_to_id_.end()) ? it->second : -1;
}

} // namespace ie
