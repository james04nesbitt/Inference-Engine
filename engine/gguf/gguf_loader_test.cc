#include "engine/gguf/gguf_loader.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace ie {
namespace {

// ============================================================================
// Helper: write raw bytes to an output stream
// ============================================================================

template <typename T> void WriteRaw(std::ofstream &out, T val) {
  out.write(reinterpret_cast<const char *>(&val), sizeof(val));
}

void WriteGGUFString(std::ofstream &out, const std::string &s) {
  uint64_t len = s.size();
  WriteRaw(out, len);
  out.write(s.data(), static_cast<std::streamsize>(len));
}

// Write a GGUF header with given tensor/metadata counts.
void WriteHeader(std::ofstream &out, uint64_t tensor_count,
                 uint64_t metadata_count) {
  WriteRaw<uint32_t>(out, kGGUFMagic);
  WriteRaw<uint32_t>(out, kGGUFVersion3);
  WriteRaw<uint64_t>(out, tensor_count);
  WriteRaw<uint64_t>(out, metadata_count);
}

// ============================================================================
// A RAII temp file helper
// ============================================================================

class TempFile {
public:
  TempFile() {
    path_ = std::filesystem::temp_directory_path() /
            ("gguf_test_" + std::to_string(counter_++) + ".gguf");
  }
  ~TempFile() { std::filesystem::remove(path_); }

  std::string path() const { return path_.string(); }

private:
  std::filesystem::path path_;
  static int counter_;
};
int TempFile::counter_ = 0;

// ============================================================================
// Synthetic GGUF Tests — validate the parser on small hand-crafted files
// ============================================================================

TEST(GGUFLoaderTest, InvalidMagic) {
  TempFile tmp;
  {
    std::ofstream out(tmp.path(), std::ios::binary);
    WriteRaw<uint32_t>(out, 0xDEADBEEF); // bad magic
    WriteRaw<uint32_t>(out, 3);
    WriteRaw<uint64_t>(out, 0);
    WriteRaw<uint64_t>(out, 0);
  }
  GGUFFile gguf;
  EXPECT_FALSE(gguf.Open(tmp.path()));
}

TEST(GGUFLoaderTest, HeaderOnly) {
  TempFile tmp;
  {
    std::ofstream out(tmp.path(), std::ios::binary);
    WriteHeader(out, /*tensor_count=*/0, /*metadata_count=*/0);
  }
  GGUFFile gguf;
  ASSERT_TRUE(gguf.Open(tmp.path()));
  EXPECT_EQ(gguf.header().version, kGGUFVersion3);
  EXPECT_EQ(gguf.header().tensor_count, 0u);
  EXPECT_EQ(gguf.header().metadata_kv_count, 0u);
}

TEST(GGUFLoaderTest, ReadMetadataStringAndUint32) {
  TempFile tmp;
  {
    std::ofstream out(tmp.path(), std::ios::binary);
    WriteHeader(out, /*tensor_count=*/0, /*metadata_count=*/2);

    // Entry 1: string metadata
    WriteGGUFString(out, "general.name");
    WriteRaw<uint32_t>(out, static_cast<uint32_t>(GGUFValueType::kString));
    WriteGGUFString(out, "test-model");

    // Entry 2: uint32 metadata
    WriteGGUFString(out, "general.context_length");
    WriteRaw<uint32_t>(out, static_cast<uint32_t>(GGUFValueType::kUint32));
    WriteRaw<uint32_t>(out, 8192);
  }

  GGUFFile gguf;
  ASSERT_TRUE(gguf.Open(tmp.path()));
  EXPECT_EQ(gguf.GetString("general.name"), "test-model");
  EXPECT_EQ(gguf.GetInt("general.context_length"), 8192);
  // Non-existent key returns default
  EXPECT_EQ(gguf.GetString("nonexistent", "default"), "default");
  EXPECT_EQ(gguf.GetInt("nonexistent", -1), -1);
}

TEST(GGUFLoaderTest, ReadMetadataFloat) {
  TempFile tmp;
  {
    std::ofstream out(tmp.path(), std::ios::binary);
    WriteHeader(out, 0, 1);

    WriteGGUFString(out, "my.temperature");
    WriteRaw<uint32_t>(out, static_cast<uint32_t>(GGUFValueType::kFloat32));
    WriteRaw<float>(out, 0.7f);
  }

  GGUFFile gguf;
  ASSERT_TRUE(gguf.Open(tmp.path()));
  EXPECT_FLOAT_EQ(gguf.GetFloat("my.temperature"), 0.7f);
}

TEST(GGUFLoaderTest, ReadMetadataBool) {
  TempFile tmp;
  {
    std::ofstream out(tmp.path(), std::ios::binary);
    WriteHeader(out, 0, 1);

    WriteGGUFString(out, "my.flag");
    WriteRaw<uint32_t>(out, static_cast<uint32_t>(GGUFValueType::kBool));
    WriteRaw<uint8_t>(out, 1);
  }

  GGUFFile gguf;
  ASSERT_TRUE(gguf.Open(tmp.path()));
  auto *val = gguf.GetMetadata("my.flag");
  ASSERT_NE(val, nullptr);
  ASSERT_TRUE(std::holds_alternative<bool>(*val));
  EXPECT_TRUE(std::get<bool>(*val));
}

TEST(GGUFLoaderTest, ReadMetadataArrayOfStrings) {
  TempFile tmp;
  {
    std::ofstream out(tmp.path(), std::ios::binary);
    WriteHeader(out, 0, 1);

    WriteGGUFString(out, "tokenizer.tokens");
    WriteRaw<uint32_t>(out, static_cast<uint32_t>(GGUFValueType::kArray));
    WriteRaw<uint32_t>(out, static_cast<uint32_t>(GGUFValueType::kString));
    WriteRaw<uint64_t>(out, 3); // count = 3
    WriteGGUFString(out, "hello");
    WriteGGUFString(out, "world");
    WriteGGUFString(out, "!");
  }

  GGUFFile gguf;
  ASSERT_TRUE(gguf.Open(tmp.path()));
  auto *val = gguf.GetMetadata("tokenizer.tokens");
  ASSERT_NE(val, nullptr);
  ASSERT_TRUE(std::holds_alternative<std::vector<std::string>>(*val));
  auto &tokens = std::get<std::vector<std::string>>(*val);
  ASSERT_EQ(tokens.size(), 3u);
  EXPECT_EQ(tokens[0], "hello");
  EXPECT_EQ(tokens[1], "world");
  EXPECT_EQ(tokens[2], "!");
}

TEST(GGUFLoaderTest, LoadTensorF32) {
  // Build a synthetic GGUF with one 1D F32 tensor of 4 elements
  TempFile tmp;
  const int64_t numel = 4;
  const float values[4] = {1.0f, 2.5f, -3.0f, 42.0f};

  {
    std::ofstream out(tmp.path(), std::ios::binary);
    WriteHeader(out, 1, 0);

    WriteGGUFString(out, "test_tensor");
    WriteRaw<uint32_t>(out, 1); // n_dims
    WriteRaw<uint64_t>(out, static_cast<uint64_t>(numel));
    WriteRaw<uint32_t>(out, static_cast<uint32_t>(GGMLType::kF32));
    WriteRaw<uint64_t>(out, 0); // offset = 0

    // Pad to 32-byte alignment (GGUF default)
    auto pos = out.tellp();
    uint64_t aligned = (static_cast<uint64_t>(pos) + 31) & ~uint64_t(31);
    while (static_cast<uint64_t>(out.tellp()) < aligned) {
      WriteRaw<uint8_t>(out, 0);
    }

    // Write the tensor data
    out.write(reinterpret_cast<const char *>(values), sizeof(values));
  }

  GGUFFile gguf;
  ASSERT_TRUE(gguf.Open(tmp.path()));

  Tensor t = gguf.LoadTensor("test_tensor");
  EXPECT_TRUE(t.shape_equals({numel}));
  EXPECT_EQ(t.dtype(), DType::kFloat32);
  EXPECT_EQ(t.numel(), numel);
  for (int64_t i = 0; i < numel; ++i) {
    EXPECT_FLOAT_EQ(t.at({i}), values[i]) << "Mismatch at index " << i;
  }
}

TEST(GGUFLoaderTest, LoadTensorBF16) {
  // Build a synthetic GGUF with one 1D BF16 tensor of 4 elements
  TempFile tmp;
  const int64_t numel = 4;
  const float src_values[4] = {1.0f, -2.0f, 0.5f, 100.0f};

  // Convert to BF16: just truncate lower 16 bits of FP32
  uint16_t bf16_values[4];
  for (int i = 0; i < 4; ++i) {
    uint32_t bits;
    std::memcpy(&bits, &src_values[i], sizeof(bits));
    bf16_values[i] = static_cast<uint16_t>(bits >> 16);
  }

  {
    std::ofstream out(tmp.path(), std::ios::binary);
    WriteHeader(out, 1, 0);

    WriteGGUFString(out, "bf16_tensor");
    WriteRaw<uint32_t>(out, 1);
    WriteRaw<uint64_t>(out, static_cast<uint64_t>(numel));
    WriteRaw<uint32_t>(out, static_cast<uint32_t>(GGMLType::kBF16));
    WriteRaw<uint64_t>(out, 0);

    // Pad
    auto pos = out.tellp();
    uint64_t aligned = (static_cast<uint64_t>(pos) + 31) & ~uint64_t(31);
    while (static_cast<uint64_t>(out.tellp()) < aligned) {
      WriteRaw<uint8_t>(out, 0);
    }

    out.write(reinterpret_cast<const char *>(bf16_values), sizeof(bf16_values));
  }

  GGUFFile gguf;
  ASSERT_TRUE(gguf.Open(tmp.path()));

  Tensor t = gguf.LoadTensor("bf16_tensor");
  EXPECT_TRUE(t.shape_equals({numel}));
  EXPECT_EQ(t.dtype(), DType::kBFloat16);
  // BF16 values match source exactly for these simple values
  EXPECT_FLOAT_EQ(t.at({0}), 1.0f);
  EXPECT_FLOAT_EQ(t.at({1}), -2.0f);
  EXPECT_FLOAT_EQ(t.at({2}), 0.5f);
  EXPECT_FLOAT_EQ(t.at({3}), 100.0f);
}

TEST(GGUFLoaderTest, LoadTensorNotFound) {
  TempFile tmp;
  {
    std::ofstream out(tmp.path(), std::ios::binary);
    WriteHeader(out, 0, 0);
  }
  GGUFFile gguf;
  ASSERT_TRUE(gguf.Open(tmp.path()));
  EXPECT_THROW(gguf.LoadTensor("nonexistent"), std::runtime_error);
}

TEST(GGUFLoaderTest, BF16TensorConvertToF32) {
  // Verify that BF16 tensors can be converted to F32 via Tensor::to()
  TempFile tmp;
  const int64_t numel = 4;
  const float src_values[4] = {1.0f, -2.0f, 0.5f, 100.0f};

  uint16_t bf16_values[4];
  for (int i = 0; i < 4; ++i) {
    uint32_t bits;
    std::memcpy(&bits, &src_values[i], sizeof(bits));
    bf16_values[i] = static_cast<uint16_t>(bits >> 16);
  }

  {
    std::ofstream out(tmp.path(), std::ios::binary);
    WriteHeader(out, 1, 0);

    WriteGGUFString(out, "bf16_tensor");
    WriteRaw<uint32_t>(out, 1);
    WriteRaw<uint64_t>(out, static_cast<uint64_t>(numel));
    WriteRaw<uint32_t>(out, static_cast<uint32_t>(GGMLType::kBF16));
    WriteRaw<uint64_t>(out, 0);

    auto pos = out.tellp();
    uint64_t aligned = (static_cast<uint64_t>(pos) + 31) & ~uint64_t(31);
    while (static_cast<uint64_t>(out.tellp()) < aligned) {
      WriteRaw<uint8_t>(out, 0);
    }

    out.write(reinterpret_cast<const char *>(bf16_values), sizeof(bf16_values));
  }

  GGUFFile gguf;
  ASSERT_TRUE(gguf.Open(tmp.path()));

  Tensor t = gguf.LoadTensor("bf16_tensor");
  EXPECT_EQ(t.dtype(), DType::kBFloat16);

  // Convert to F32 — this is what the ops pipeline does
  Tensor t_f32 = t.to(DType::kFloat32);
  EXPECT_EQ(t_f32.dtype(), DType::kFloat32);
  EXPECT_FLOAT_EQ(t_f32.at({0}), 1.0f);
  EXPECT_FLOAT_EQ(t_f32.at({1}), -2.0f);
  EXPECT_FLOAT_EQ(t_f32.at({2}), 0.5f);
  EXPECT_FLOAT_EQ(t_f32.at({3}), 100.0f);
}

TEST(GGUFLoaderTest, DebugModeDoesNotCrash) {
  TempFile tmp;
  {
    std::ofstream out(tmp.path(), std::ios::binary);
    WriteHeader(out, 0, 1);

    WriteGGUFString(out, "general.name");
    WriteRaw<uint32_t>(out, static_cast<uint32_t>(GGUFValueType::kString));
    WriteGGUFString(out, "test-model");
  }

  GGUFFile gguf;
  // Open with debug=true — should print debug info without crashing
  ASSERT_TRUE(gguf.Open(tmp.path(), /*debug=*/true));
  EXPECT_EQ(gguf.GetString("general.name"), "test-model");
}

// ============================================================================
// GGMLType helper tests
// ============================================================================

TEST(GGMLTypeTest, TypeToDType) {
  EXPECT_EQ(GGMLTypeToDType(GGMLType::kF32), DType::kFloat32);
  EXPECT_EQ(GGMLTypeToDType(GGMLType::kBF16), DType::kBFloat16);
  // Unsupported types throw.
  EXPECT_THROW(GGMLTypeToDType(static_cast<GGMLType>(1)), std::runtime_error);
}

TEST(GGMLTypeTest, TypeSize) {
  EXPECT_EQ(GGMLTypeSize(GGMLType::kF32), 4u);
  EXPECT_EQ(GGMLTypeSize(GGMLType::kBF16), 2u);
}

TEST(GGMLTypeTest, TypeName) {
  EXPECT_EQ(GGMLTypeName(GGMLType::kF32), "GGML_TYPE_F32");
  EXPECT_EQ(GGMLTypeName(GGMLType::kBF16), "BF16");
}

// ============================================================================
// Real Model Tests — load the actual gemma-3-1b-it-BF16.gguf
//
// These tests validate against the Python GGUF parser reference output
// in engine/gguf/parsedbf16.txt.
// ============================================================================

// Path to the real BF16 model. Adjust if your model is elsewhere.
static const char *kModelPath =
    "C:/Users/james/Coding/Projects/Inference-Engine/bazel-inference-engine/"
    "model/gemma-3-1b-it-BF16.gguf";

class RealModelTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!std::filesystem::exists(kModelPath)) {
      GTEST_SKIP() << "Model file not found: " << kModelPath;
    }
    ASSERT_TRUE(gguf_.Open(kModelPath, /*debug=*/false));
  }

  GGUFFile gguf_;
};

TEST_F(RealModelTest, HeaderIsCorrect) {
  EXPECT_EQ(gguf_.header().magic, kGGUFMagic);
  EXPECT_EQ(gguf_.header().version, 3u);
  // Gemma 3 1B BF16 has 340 tensors
  EXPECT_EQ(gguf_.header().tensor_count, 340u);
}

TEST_F(RealModelTest, ArchitectureMetadata) {
  EXPECT_EQ(gguf_.GetString("general.architecture"), "gemma3");
  EXPECT_EQ(gguf_.GetString("general.type"), "model");
  EXPECT_EQ(gguf_.GetString("general.name"), "Gemma-3-1B-It");
}

TEST_F(RealModelTest, ModelDimensionMetadata) {
  EXPECT_EQ(gguf_.GetInt("gemma3.embedding_length"), 1152);
  EXPECT_EQ(gguf_.GetInt("gemma3.block_count"), 26);
  EXPECT_EQ(gguf_.GetInt("gemma3.feed_forward_length"), 6912);
  EXPECT_EQ(gguf_.GetInt("gemma3.attention.head_count"), 4);
  EXPECT_EQ(gguf_.GetInt("gemma3.attention.head_count_kv"), 1);
  EXPECT_EQ(gguf_.GetInt("gemma3.attention.key_length"), 256);
  EXPECT_EQ(gguf_.GetInt("gemma3.attention.value_length"), 256);
  EXPECT_EQ(gguf_.GetInt("gemma3.attention.sliding_window"), 512);
  EXPECT_FLOAT_EQ(gguf_.GetFloat("gemma3.rope.freq_base"), 1000000.0f);
}

TEST_F(RealModelTest, TokenizerMetadata) {
  EXPECT_EQ(gguf_.GetString("tokenizer.ggml.model"), "llama");
  EXPECT_EQ(gguf_.GetInt("tokenizer.ggml.bos_token_id"), 2);
  EXPECT_EQ(gguf_.GetInt("tokenizer.ggml.eos_token_id"), 106);
  EXPECT_EQ(gguf_.GetInt("tokenizer.ggml.unknown_token_id"), 3);
  EXPECT_EQ(gguf_.GetInt("tokenizer.ggml.padding_token_id"), 0);

  // Verify tokens array exists and has expected size
  auto *tokens_val = gguf_.GetMetadata("tokenizer.ggml.tokens");
  ASSERT_NE(tokens_val, nullptr);
  auto *tokens = std::get_if<std::vector<std::string>>(tokens_val);
  ASSERT_NE(tokens, nullptr);
  // 262144 tokens = 50 shown + 262094 more from parsedbf16.txt
  EXPECT_EQ(tokens->size(), 262144u);
  // Check first few tokens match reference
  EXPECT_EQ((*tokens)[0], "<pad>");
  EXPECT_EQ((*tokens)[1], "<eos>");
  EXPECT_EQ((*tokens)[2], "<bos>");
  EXPECT_EQ((*tokens)[3], "<unk>");
}

TEST_F(RealModelTest, EmbeddingTensorInfo) {
  // From parsedbf16.txt:
  //   Name: token_embd.weight, Shape: (1152, 262144), Type: BF16, Offset: 0
  auto *info = gguf_.GetTensorInfo("token_embd.weight");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->type, GGMLType::kBF16);
  EXPECT_EQ(info->offset, 0u);
  ASSERT_EQ(info->dimensions.size(), 2u);
  EXPECT_EQ(info->dimensions[0], 1152u);
  EXPECT_EQ(info->dimensions[1], 262144u);
}

TEST_F(RealModelTest, NormTensorIsF32) {
  // From parsedbf16.txt:
  //   Name: blk.0.attn_norm.weight, Shape: (1152,), Type: GGML_TYPE_F32
  auto *info = gguf_.GetTensorInfo("blk.0.attn_norm.weight");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->type, GGMLType::kF32);
  ASSERT_EQ(info->dimensions.size(), 1u);
  EXPECT_EQ(info->dimensions[0], 1152u);
}

TEST_F(RealModelTest, LoadEmbeddingTensorBF16) {
  // Load the embedding table — should be BF16, zero-copy
  Tensor emb = gguf_.LoadTensor("token_embd.weight");

  // Shape should be reversed from GGUF column-major: [262144, 1152]
  EXPECT_TRUE(emb.shape_equals({262144, 1152}));
  EXPECT_EQ(emb.dtype(), DType::kBFloat16);
  EXPECT_TRUE(emb.is_contiguous());

  // Verify we can read values
  float val = emb.at({0, 0});
  // Just check it's a valid float (not NaN/Inf)
  EXPECT_TRUE(std::isfinite(val));
}

TEST_F(RealModelTest, LoadNormTensorF32) {
  Tensor norm = gguf_.LoadTensor("blk.0.attn_norm.weight");
  EXPECT_TRUE(norm.shape_equals({1152}));
  EXPECT_EQ(norm.dtype(), DType::kFloat32);
  EXPECT_TRUE(norm.is_contiguous());

  // Check the values are reasonable
  float val = norm.at({0});
  EXPECT_TRUE(std::isfinite(val));
}

TEST_F(RealModelTest, LoadWeightTensorBF16) {
  // blk.0.ffn_gate.weight: (1152, 6912), BF16
  Tensor gate = gguf_.LoadTensor("blk.0.ffn_gate.weight");
  EXPECT_TRUE(gate.shape_equals({6912, 1152})); // reversed from GGUF
  EXPECT_EQ(gate.dtype(), DType::kBFloat16);
  EXPECT_TRUE(gate.is_contiguous());
}

TEST_F(RealModelTest, TensorCount) {
  auto names = gguf_.TensorNames();
  EXPECT_EQ(names.size(), 340u);
}

TEST_F(RealModelTest, BF16TensorConvertToF32Pipeline) {
  // Simulate the ops pipeline: load BF16 → convert to F32 → contiguous
  Tensor norm_weight = gguf_.LoadTensor("blk.0.attn_norm.weight");
  Tensor norm_f32 = norm_weight.to(DType::kFloat32).contiguous();
  EXPECT_EQ(norm_f32.dtype(), DType::kFloat32);
  EXPECT_TRUE(norm_f32.is_contiguous());

  // Load a BF16 weight and convert
  Tensor gate = gguf_.LoadTensor("blk.0.attn_k.weight");
  Tensor gate_f32 = gate.to(DType::kFloat32).contiguous();
  EXPECT_EQ(gate_f32.dtype(), DType::kFloat32);
  EXPECT_TRUE(gate_f32.is_contiguous());
}

TEST_F(RealModelTest, DebugModePrintsAllInfo) {
  // Just verify debug mode doesn't crash on the real model
  gguf_.PrintDebugInfo();
}

TEST_F(RealModelTest, OutputNormTensor) {
  // output_norm.weight: (1152,), F32 — the final layer norm
  auto *info = gguf_.GetTensorInfo("output_norm.weight");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->type, GGMLType::kF32);

  Tensor t = gguf_.LoadTensor("output_norm.weight");
  EXPECT_TRUE(t.shape_equals({1152}));
  EXPECT_EQ(t.dtype(), DType::kFloat32);
}

TEST_F(RealModelTest, AllTensorsLoadable) {
  // Verify every tensor in the model can be loaded without errors
  auto names = gguf_.TensorNames();
  for (const auto &name : names) {
    ASSERT_NO_THROW(gguf_.LoadTensor(name)) << "Failed to load: " << name;
  }
}

} // namespace
} // namespace ie
