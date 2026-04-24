#pragma once

#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include "engine/tensor/tensor.h"

namespace ie {

// GGUF format constants
constexpr uint32_t kGGUFMagic = 0x46554747; // "GGUF" in little-endian
constexpr uint32_t kGGUFVersion3 = 3;

// ============================================================================
// GGUF metadata value types (from the spec).
// See: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
// ============================================================================
enum class GGUFValueType : uint32_t {
  kUint8 = 0,
  kInt8 = 1,
  kUint16 = 2,
  kInt16 = 3,
  kUint32 = 4,
  kInt32 = 5,
  kFloat32 = 6,
  kBool = 7,
  kString = 8,
  kArray = 9,
  kUint64 = 10,
  kInt64 = 11,
  kFloat64 = 12,
};

// ============================================================================
// GGUF tensor types — only BF16 and F32 are supported.
// This engine exclusively loads BF16-quantized GGUF files.
// ============================================================================
enum class GGMLType : uint32_t {
  kF32 = 0,
  kBF16 = 30, // BFloat16
};

// Convert GGMLType to our Tensor DType.
DType GGMLTypeToDType(GGMLType type);

// Returns the number of bytes per element for a given GGMLType.
size_t GGMLTypeSize(GGMLType type);

// Returns a human-readable name for a GGMLType.
std::string GGMLTypeName(GGMLType type);

// ============================================================================
// Parsed structures from a GGUF file
// ============================================================================

struct GGUFHeader {
  uint32_t magic;
  uint32_t version;
  uint64_t tensor_count;
  uint64_t metadata_kv_count;
};

// A single metadata value. Uses std::variant for type safety.
using GGUFMetadataValue =
    std::variant<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, float,
                 bool, std::string, uint64_t, int64_t, double,
                 std::vector<int32_t>,    // Array of ints (e.g., token types)
                 std::vector<float>,      // Array of floats (e.g., scores)
                 std::vector<std::string> // Array of strings (e.g., tokens)
                 >;

struct GGUFTensorInfo {
  std::string name;
  std::vector<uint64_t> dimensions;
  GGMLType type;
  uint64_t offset; // Offset from start of tensor data section
};

// ============================================================================
// GGUFFile — Reads and provides access to a BF16 GGUF model file.
//
// Uses Boost.Interprocess memory-mapped I/O for zero-copy tensor loading.
// BF16 tensors are returned as views into the mmap'd region — no data is
// copied. The mmap region owns the memory; tensors hold a shared_ptr to
// the GGUFFile to prevent it from being destroyed while tensors are live.
//
// Only BF16 and F32 tensor types are supported.
// ============================================================================
class GGUFFile : public std::enable_shared_from_this<GGUFFile> {
public:
  GGUFFile() = default;
  ~GGUFFile() = default;

  // Non-copyable (owns mmap resources).
  GGUFFile(const GGUFFile &) = delete;
  GGUFFile &operator=(const GGUFFile &) = delete;

  // Opens and parses the GGUF file header, metadata, and tensor info.
  // Memory-maps the entire file for zero-copy tensor loading.
  // If debug=true, prints all metadata and tensor info to stdout.
  // Returns false on failure.
  bool Open(const std::string &path, bool debug = false);

  // --- Accessors ---
  const GGUFHeader &header() const { return header_; }

  // Get a metadata value by key. Returns nullptr if not found.
  const GGUFMetadataValue *GetMetadata(const std::string &key) const;

  // Get string metadata (convenience).
  std::string GetString(const std::string &key,
                        const std::string &default_val = "") const;

  // Get integer metadata (convenience).
  int64_t GetInt(const std::string &key, int64_t default_val = 0) const;

  // Get float metadata (convenience).
  float GetFloat(const std::string &key, float default_val = 0.0f) const;

  // Get info about a specific tensor by name.
  const GGUFTensorInfo *GetTensorInfo(const std::string &name) const;

  // Load a tensor as a zero-copy view into the mmap'd file.
  // BF16 tensors → DType::kBFloat16, F32 tensors → DType::kFloat32.
  // The returned Tensor shares ownership of the mmap region.
  Tensor LoadTensor(const std::string &name) const;

  // List all tensor names.
  std::vector<std::string> TensorNames() const;

  // Print a summary of the file contents (first 10 tensors).
  void PrintSummary() const;

  // Print full debug info: all metadata + all tensors with shapes/types/offsets.
  // Output format matches the Python GGUF parser reference.
  void PrintDebugInfo() const;

private:
  GGUFHeader header_{};
  std::map<std::string, GGUFMetadataValue> metadata_;
  std::map<std::string, GGUFTensorInfo> tensors_;
  std::string file_path_;
  uint64_t tensor_data_offset_ = 0; // Byte offset where tensor data starts
  uint64_t file_size_ = 0;

  // Boost.Interprocess memory-mapped file.
  // Replaces ~80 lines of Win32/POSIX mmap code with 2 objects.
  std::unique_ptr<boost::interprocess::file_mapping> file_mapping_;
  std::unique_ptr<boost::interprocess::mapped_region> mapped_region_;

  // Raw pointer into the mapped region (convenience).
  const uint8_t *mapped_data_ = nullptr;

  // --- Internal parsing helpers ---
  bool ReadHeader(std::ifstream &file);
  bool ReadMetadata(std::ifstream &file);
  bool ReadTensorInfos(std::ifstream &file);

  // Low-level read helpers
  std::string ReadString(std::ifstream &file);
  GGUFMetadataValue ReadValue(std::ifstream &file, GGUFValueType type);
};

} // namespace ie
