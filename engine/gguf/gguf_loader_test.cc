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
// Tests
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

TEST(GGUFLoaderTest, ReadTensorInfos) {
  TempFile tmp;
  {
    std::ofstream out(tmp.path(), std::ios::binary);
    WriteHeader(out, /*tensor_count=*/2, /*metadata_count=*/0);

    // Tensor 1: "weight" [4, 3] F32 at offset 0
    WriteGGUFString(out, "weight");
    WriteRaw<uint32_t>(out, 2); // n_dims
    WriteRaw<uint64_t>(out, 4); // dim 0
    WriteRaw<uint64_t>(out, 3); // dim 1
    WriteRaw<uint32_t>(out, static_cast<uint32_t>(GGMLType::kF32));
    WriteRaw<uint64_t>(out, 0); // offset

    // Tensor 2: "bias" [3] F16 at offset 48
    WriteGGUFString(out, "bias");
    WriteRaw<uint32_t>(out, 1); // n_dims
    WriteRaw<uint64_t>(out, 3); // dim 0
    WriteRaw<uint32_t>(out, static_cast<uint32_t>(GGMLType::kF16));
    WriteRaw<uint64_t>(out, 48); // offset
  }

  GGUFFile gguf;
  ASSERT_TRUE(gguf.Open(tmp.path()));

  auto names = gguf.TensorNames();
  EXPECT_EQ(names.size(), 2u);

  auto *w = gguf.GetTensorInfo("weight");
  ASSERT_NE(w, nullptr);
  EXPECT_EQ(w->dimensions.size(), 2u);
  EXPECT_EQ(w->dimensions[0], 4u);
  EXPECT_EQ(w->dimensions[1], 3u);
  EXPECT_EQ(w->type, GGMLType::kF32);
  EXPECT_EQ(w->offset, 0u);

  auto *b = gguf.GetTensorInfo("bias");
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(b->dimensions.size(), 1u);
  EXPECT_EQ(b->dimensions[0], 3u);
  EXPECT_EQ(b->type, GGMLType::kF16);
  EXPECT_EQ(b->offset, 48u);
}

TEST(GGUFLoaderTest, LoadTensorF32) {
  // Build a synthetic GGUF with one 1D F32 tensor of 4 elements
  TempFile tmp;
  const int64_t numel = 4;
  const float values[4] = {1.0f, 2.5f, -3.0f, 42.0f};

  {
    std::ofstream out(tmp.path(), std::ios::binary);
    // Header: 1 tensor, 0 metadata
    WriteHeader(out, 1, 0);

    // Tensor info: "test_tensor" [4] F32, offset=0
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

TEST(GGUFLoaderTest, LoadTensorF32_2D) {
  // 2D tensor [2, 3]
  TempFile tmp;
  const float values[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  {
    std::ofstream out(tmp.path(), std::ios::binary);
    WriteHeader(out, 1, 0);

    WriteGGUFString(out, "matrix");
    WriteRaw<uint32_t>(out, 2); // 2 dims
    WriteRaw<uint64_t>(out, 2); // dim 0
    WriteRaw<uint64_t>(out, 3); // dim 1
    WriteRaw<uint32_t>(out, static_cast<uint32_t>(GGMLType::kF32));
    WriteRaw<uint64_t>(out, 0);

    // Pad to alignment
    auto pos = out.tellp();
    uint64_t aligned = (static_cast<uint64_t>(pos) + 31) & ~uint64_t(31);
    while (static_cast<uint64_t>(out.tellp()) < aligned) {
      WriteRaw<uint8_t>(out, 0);
    }

    out.write(reinterpret_cast<const char *>(values), sizeof(values));
  }

  GGUFFile gguf;
  ASSERT_TRUE(gguf.Open(tmp.path()));

  Tensor t = gguf.LoadTensor("matrix");
  EXPECT_TRUE(t.shape_equals({2, 3}));
  EXPECT_EQ(t.dtype(), DType::kFloat32);
  EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(t.at({0, 2}), 3.0f);
  EXPECT_FLOAT_EQ(t.at({1, 0}), 4.0f);
  EXPECT_FLOAT_EQ(t.at({1, 2}), 6.0f);
}

// Helper: convert float to FP16 (matches tensor.cc's FloatToHalf)
static uint16_t FloatToHalf(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  uint32_t sign = (bits >> 16) & 0x8000;
  int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
  uint32_t mantissa = bits & 0x007FFFFF;
  if (exponent <= 0) {
    return static_cast<uint16_t>(sign);
  } else if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00);
  }
  return static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
}

TEST(GGUFLoaderTest, LoadTensorF16) {
  TempFile tmp;
  const int64_t numel = 4;
  const float src_values[4] = {1.0f, -2.0f, 0.5f, 100.0f};
  uint16_t half_values[4];
  for (int i = 0; i < 4; ++i) {
    half_values[i] = FloatToHalf(src_values[i]);
  }

  {
    std::ofstream out(tmp.path(), std::ios::binary);
    WriteHeader(out, 1, 0);

    WriteGGUFString(out, "half_tensor");
    WriteRaw<uint32_t>(out, 1);
    WriteRaw<uint64_t>(out, static_cast<uint64_t>(numel));
    WriteRaw<uint32_t>(out, static_cast<uint32_t>(GGMLType::kF16));
    WriteRaw<uint64_t>(out, 0);

    // Pad
    auto pos = out.tellp();
    uint64_t aligned = (static_cast<uint64_t>(pos) + 31) & ~uint64_t(31);
    while (static_cast<uint64_t>(out.tellp()) < aligned) {
      WriteRaw<uint8_t>(out, 0);
    }

    out.write(reinterpret_cast<const char *>(half_values), sizeof(half_values));
  }

  GGUFFile gguf;
  ASSERT_TRUE(gguf.Open(tmp.path()));

  Tensor t = gguf.LoadTensor("half_tensor");
  EXPECT_TRUE(t.shape_equals({numel}));
  EXPECT_EQ(t.dtype(), DType::kFloat16);
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

TEST(GGUFLoaderTest, MultipleTensors) {
  // Two tensors in one file — verify both load correctly
  TempFile tmp;
  const float w_data[6] = {1, 2, 3, 4, 5, 6};
  const float b_data[3] = {0.1f, 0.2f, 0.3f};

  {
    std::ofstream out(tmp.path(), std::ios::binary);
    WriteHeader(out, 2, 0);

    // Tensor "weights" [2, 3] F32, offset = 0
    WriteGGUFString(out, "weights");
    WriteRaw<uint32_t>(out, 2);
    WriteRaw<uint64_t>(out, 2);
    WriteRaw<uint64_t>(out, 3);
    WriteRaw<uint32_t>(out, static_cast<uint32_t>(GGMLType::kF32));
    WriteRaw<uint64_t>(out, 0);

    // Tensor "bias" [3] F32, offset = 24 (6 floats * 4 bytes)
    WriteGGUFString(out, "bias");
    WriteRaw<uint32_t>(out, 1);
    WriteRaw<uint64_t>(out, 3);
    WriteRaw<uint32_t>(out, static_cast<uint32_t>(GGMLType::kF32));
    WriteRaw<uint64_t>(out, 24);

    // Pad to alignment
    auto pos = out.tellp();
    uint64_t aligned = (static_cast<uint64_t>(pos) + 31) & ~uint64_t(31);
    while (static_cast<uint64_t>(out.tellp()) < aligned) {
      WriteRaw<uint8_t>(out, 0);
    }

    // Write tensor data: weights then bias
    out.write(reinterpret_cast<const char *>(w_data), sizeof(w_data));
    out.write(reinterpret_cast<const char *>(b_data), sizeof(b_data));
  }

  GGUFFile gguf;
  ASSERT_TRUE(gguf.Open(tmp.path()));

  Tensor w = gguf.LoadTensor("weights");
  EXPECT_TRUE(w.shape_equals({2, 3}));
  EXPECT_FLOAT_EQ(w.at({0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(w.at({1, 2}), 6.0f);

  Tensor b = gguf.LoadTensor("bias");
  EXPECT_TRUE(b.shape_equals({3}));
  EXPECT_FLOAT_EQ(b.at({0}), 0.1f);
  EXPECT_FLOAT_EQ(b.at({2}), 0.3f);
}

TEST(GGUFLoaderTest, MetadataAndTensorsTogether) {
  // Full integration: metadata + tensors in one file
  TempFile tmp;
  const float data[4] = {10, 20, 30, 40};

  {
    std::ofstream out(tmp.path(), std::ios::binary);
    WriteHeader(out, 1, 2);

    // Metadata 1: string
    WriteGGUFString(out, "general.architecture");
    WriteRaw<uint32_t>(out, static_cast<uint32_t>(GGUFValueType::kString));
    WriteGGUFString(out, "gemma");

    // Metadata 2: uint32
    WriteGGUFString(out, "gemma.embedding_length");
    WriteRaw<uint32_t>(out, static_cast<uint32_t>(GGUFValueType::kUint32));
    WriteRaw<uint32_t>(out, 1152);

    // Tensor: "embed" [4] F32
    WriteGGUFString(out, "embed");
    WriteRaw<uint32_t>(out, 1);
    WriteRaw<uint64_t>(out, 4);
    WriteRaw<uint32_t>(out, static_cast<uint32_t>(GGMLType::kF32));
    WriteRaw<uint64_t>(out, 0);

    // Pad
    auto pos = out.tellp();
    uint64_t aligned = (static_cast<uint64_t>(pos) + 31) & ~uint64_t(31);
    while (static_cast<uint64_t>(out.tellp()) < aligned) {
      WriteRaw<uint8_t>(out, 0);
    }

    out.write(reinterpret_cast<const char *>(data), sizeof(data));
  }

  GGUFFile gguf;
  ASSERT_TRUE(gguf.Open(tmp.path()));

  // Verify metadata
  EXPECT_EQ(gguf.GetString("general.architecture"), "gemma");
  EXPECT_EQ(gguf.GetInt("gemma.embedding_length"), 1152);

  // Verify tensor
  Tensor t = gguf.LoadTensor("embed");
  EXPECT_TRUE(t.shape_equals({4}));
  EXPECT_FLOAT_EQ(t.at({0}), 10.0f);
  EXPECT_FLOAT_EQ(t.at({3}), 40.0f);
}

// ============================================================================
// GGMLType helper tests
// ============================================================================

TEST(GGMLTypeTest, TypeToDType) {
  EXPECT_EQ(GGMLTypeToDType(GGMLType::kF32), DType::kFloat32);
  EXPECT_EQ(GGMLTypeToDType(GGMLType::kF16), DType::kFloat16);
  EXPECT_THROW(GGMLTypeToDType(GGMLType::kQ4_0), std::runtime_error);
  EXPECT_THROW(GGMLTypeToDType(GGMLType::kQ8_0), std::runtime_error);
}

TEST(GGMLTypeTest, TypeSize) {
  EXPECT_EQ(GGMLTypeSize(GGMLType::kF32), 4u);
  EXPECT_EQ(GGMLTypeSize(GGMLType::kF16), 2u);
  EXPECT_THROW(GGMLTypeSize(GGMLType::kQ4_0), std::runtime_error);
}

} // namespace
} // namespace ie
