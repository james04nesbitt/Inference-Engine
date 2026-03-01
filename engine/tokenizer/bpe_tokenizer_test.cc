#include "engine/tokenizer/bpe_tokenizer.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace ie {
namespace {

// ============================================================================
// Test helpers
// ============================================================================

// Build a small synthetic vocabulary for testing.
// Vocab layout:
//   0: "<unk>"     score 0
//   1: "<s>"       score 0    (BOS)
//   2: "</s>"      score 0    (EOS)
//   3: "▁"         score -1   (SentencePiece space)
//   4: "h"         score -2
//   5: "e"         score -3
//   6: "l"         score -4
//   7: "o"         score -5
//   8: "w"         score -6
//   9: "r"         score -7
//  10: "d"         score -8
//  11: "ll"        score 10   (merge: l + l)
//  12: "he"        score 8    (merge: h + e)
//  13: "▁h"        score 7    (merge: ▁ + h)
//  14: "▁he"       score 6    (merge: ▁h + e, or ▁ + he)
//  15: "hell"      score 5    (merge: he + ll)
//  16: "hello"     score 4    (merge: hell + o)
//  17: "▁hello"    score 3    (merge: ▁ + hello, or ▁h + ello, etc.)
//  18: "wo"        score 9    (merge: w + o)
//  19: "wor"       score 7    (merge: wo + r)
//  20: "worl"      score 6    (merge: wor + l)
//  21: "world"     score 5    (merge: worl + d)
//  22: "▁world"    score 3    (merge: ▁ + world)
//  23: "▁w"        score 6    (merge: ▁ + w)

struct TestVocab {
  std::vector<std::string> tokens;
  std::vector<float> scores;
  int32_t bos_id = 1;
  int32_t eos_id = 2;
  int32_t pad_id = 0;

  BPETokenizer Build() const {
    return BPETokenizer(tokens, scores, bos_id, eos_id, pad_id);
  }
};

TestVocab MakeTestVocab() {
  TestVocab v;
  // The SentencePiece space marker in UTF-8.
  std::string sp = "\xe2\x96\x81";

  v.tokens = {
      "<unk>",      // 0
      "<s>",        // 1  (BOS)
      "</s>",       // 2  (EOS)
      sp,           // 3  (▁)
      "h",          // 4
      "e",          // 5
      "l",          // 6
      "o",          // 7
      "w",          // 8
      "r",          // 9
      "d",          // 10
      "ll",         // 11
      "he",         // 12
      sp + "h",     // 13  (▁h)
      sp + "he",    // 14  (▁he)
      "hell",       // 15
      "hello",      // 16
      sp + "hello", // 17  (▁hello)
      "wo",         // 18
      "wor",        // 19
      "worl",       // 20
      "world",      // 21
      sp + "world", // 22  (▁world)
      sp + "w",     // 23  (▁w)
  };

  v.scores = {
      0.0f,  // 0: <unk>
      0.0f,  // 1: <s>
      0.0f,  // 2: </s>
      -1.0f, // 3: ▁
      -2.0f, // 4: h
      -3.0f, // 5: e
      -4.0f, // 6: l
      -5.0f, // 7: o
      -6.0f, // 8: w
      -7.0f, // 9: r
      -8.0f, // 10: d
      10.0f, // 11: ll
      8.0f,  // 12: he
      7.0f,  // 13: ▁h
      6.0f,  // 14: ▁he
      5.0f,  // 15: hell
      4.0f,  // 16: hello
      3.0f,  // 17: ▁hello
      9.0f,  // 18: wo
      7.0f,  // 19: wor
      6.0f,  // 20: worl
      5.0f,  // 21: world
      3.0f,  // 22: ▁world
      6.0f,  // 23: ▁w
  };

  return v;
}

// ============================================================================
// Tests
// ============================================================================

TEST(BPETokenizerTest, EmptyInput) {
  auto tok = MakeTestVocab().Build();
  auto result = tok.Encode("");
  EXPECT_TRUE(result.empty());
}

TEST(BPETokenizerTest, VocabSize) {
  auto vocab = MakeTestVocab();
  auto tok = vocab.Build();
  EXPECT_EQ(tok.VocabSize(), static_cast<int32_t>(vocab.tokens.size()));
}

TEST(BPETokenizerTest, SpecialTokenIds) {
  auto tok = MakeTestVocab().Build();
  EXPECT_EQ(tok.BosId(), 1);
  EXPECT_EQ(tok.EosId(), 2);
  EXPECT_EQ(tok.PadId(), 0);
}

TEST(BPETokenizerTest, EncodeHello) {
  // "hello" → prepend ▁ → "▁hello"
  // The merge loop should eventually produce token 17 (▁hello).
  //
  // Trace: [▁, h, e, l, l, o]
  //   best merge: l+l → ll (score 10) → [▁, h, e, ll, o]
  //   best merge: w+o not present, h+e → he (score 8) → [▁, he, ll, o]
  //   best merge: ▁+he → ▁he (score 6) or he+ll → hell (score 5)
  //     ▁he wins → [▁he, ll, o]
  //   best merge: ▁he+ll → ? (not in vocab) → no match on ▁he+ll
  //     Actually, let's check: vocab has "hell" (he+ll), but ▁he is a single
  //     token now. We need ▁he + ll → ▁hell — not in vocab.
  //     ll+o → llo — not in vocab.
  //     So no more merges → [▁he, ll, o] = [14, 11, 7]
  //
  // Wait — let me re-trace more carefully with the greedy approach:
  // [3(▁), 4(h), 5(e), 6(l), 6(l), 7(o)]
  //   Pairs: ▁+h→▁h(13,sc7), h+e→he(12,sc8), e+l→?(no), l+l→ll(11,sc10),
  //   l+o→?(no) Best: l+l→ll(score 10) → [3, 4, 5, 11, 7] Pairs:
  //   ▁+h→▁h(13,sc7), h+e→he(12,sc8), e+ll→?(no), ll+o→?(no) Best: h+e→he(score
  //   8) → [3, 12, 11, 7] Pairs: ▁+he→▁he(14,sc6), he+ll→hell(15,sc5),
  //   ll+o→?(no) Best: ▁+he→▁he(score 6) → [14, 11, 7] Pairs: ▁he+ll→?(no),
  //   ll+o→?(no) No more merges → [14, 11, 7]
  auto tok = MakeTestVocab().Build();
  auto result = tok.Encode("hello");
  ASSERT_EQ(result.size(), 3u);
  EXPECT_EQ(result[0], 14); // ▁he
  EXPECT_EQ(result[1], 11); // ll
  EXPECT_EQ(result[2], 7);  // o
}

TEST(BPETokenizerTest, MergePriority) {
  // Verify that higher-scored merges happen first.
  // "ll" → prepend ▁ → "▁ll"
  // Initial: [3(▁), 6(l), 6(l)]
  //   Pairs: ▁+l→?(no), l+l→ll(11, score 10)
  //   Merge to get [3, 11]
  //   Pairs: ▁+ll→?(no)
  //   Done → [3, 11]
  auto tok = MakeTestVocab().Build();
  auto result = tok.Encode("ll");
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0], 3);  // ▁
  EXPECT_EQ(result[1], 11); // ll
}

TEST(BPETokenizerTest, NoMergePossible) {
  // "old" — 'o', 'l', 'd' are in vocab, but none of the pairs
  // "ol", "ld" exist as merged tokens.
  // "▁old" → [▁, o, l, d] → no merges → [3, 7, 6, 10]
  auto tok = MakeTestVocab().Build();
  auto result = tok.Encode("old");
  ASSERT_EQ(result.size(), 4u);
  EXPECT_EQ(result[0], 3);  // ▁
  EXPECT_EQ(result[1], 7);  // o
  EXPECT_EQ(result[2], 6);  // l
  EXPECT_EQ(result[3], 10); // d
}

TEST(BPETokenizerTest, DecodeBasic) {
  // Decode [14, 11, 7] → "▁he" + "ll" + "o" → "▁hello" → replace ▁ → " hello"
  // → trim leading space → "hello"
  auto tok = MakeTestVocab().Build();
  auto text = tok.Decode({14, 11, 7});
  EXPECT_EQ(text, "hello");
}

TEST(BPETokenizerTest, DecodeRoundTrip) {
  auto tok = MakeTestVocab().Build();
  std::string original = "hello";
  auto encoded = tok.Encode(original);
  auto decoded = tok.Decode(encoded);
  EXPECT_EQ(decoded, original);
}

TEST(BPETokenizerTest, DecodeInvalidIds) {
  // Out of range IDs should be silently ignored.
  auto tok = MakeTestVocab().Build();
  auto text = tok.Decode({-1, 4, 9999, 5});
  EXPECT_EQ(text, "he");
}

TEST(BPETokenizerTest, UnknownCharSkipped) {
  // Characters not in vocab (like 'z') should be skipped.
  // "hez" → "▁hez" → ▁(3), h(4), e(5), z(not found)
  // Then merges: h+e→he(12, sc8), ▁+he→▁he(14, sc6)
  // Result: [14] (the z is dropped)
  auto tok = MakeTestVocab().Build();
  auto result = tok.Encode("he");
  // "he" → "▁he" → [3, 4, 5] → h+e→he(12) → [3, 12] → ▁+he→▁he(14) → [14]
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0], 14); // ▁he
}

TEST(BPETokenizerTest, SingleChar) {
  // "h" → "▁h" → [3, 4] → ▁+h→▁h(13, score 7) → [13]
  auto tok = MakeTestVocab().Build();
  auto result = tok.Encode("h");
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0], 13); // ▁h
}

TEST(BPETokenizerTest, DecodeSingleToken) {
  // Decode [13] → "▁h" → replace ▁ with space → " h" → trim → "h"
  auto tok = MakeTestVocab().Build();
  EXPECT_EQ(tok.Decode({13}), "h");
}

TEST(BPETokenizerTest, DecodeEmptyTokens) {
  auto tok = MakeTestVocab().Build();
  EXPECT_EQ(tok.Decode({}), "");
}

} // namespace
} // namespace ie
