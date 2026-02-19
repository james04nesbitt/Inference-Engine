#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ie {

// ============================================================================
// Tokenizer — Abstract interface for text tokenization.
//
// Tokenizers convert between human-readable text and integer token IDs
// that the model understands. Different models use different tokenization
// schemes (BPE, SentencePiece, WordPiece, etc.).
// ============================================================================
class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  // Encode text into token IDs.
  virtual std::vector<int32_t> Encode(const std::string& text) const = 0;

  // Decode token IDs back into text.
  virtual std::string Decode(const std::vector<int32_t>& tokens) const = 0;

  // Get the vocabulary size.
  virtual int32_t VocabSize() const = 0;

  // Special token IDs
  virtual int32_t BosId() const = 0;  // Beginning of sequence
  virtual int32_t EosId() const = 0;  // End of sequence
  virtual int32_t PadId() const = 0;  // Padding
};

}  // namespace ie
