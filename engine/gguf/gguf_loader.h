#pragma once

#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <variant>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

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
// GGUF tensor types — maps to quantization formats.
// Start with F32 and F16, add quantized types as you learn them.
// ============================================================================
enum class GGMLType : uint32_t {
  kF32 = 0,
  kF16 = 1,
  kQ4_0 = 2,
  kQ4_1 = 3,
  kQ5_0 = 6,
  kQ5_1 = 7,
  kQ8_0 = 8,
  kQ8_1 = 9,
  kQ2_K = 10,
  kQ3_K = 11,
  kQ4_K = 12,
  kQ5_K = 13,
  kQ6_K = 14,
  kQ8_K = 15,
  kBF16 = 30, // BFloat16
  // TODO: Add more as needed
};

// Convert GGMLType to our Tensor DType.
// Throws for quantized types that are not yet supported.
DType GGMLTypeToDType(GGMLType type);

// Returns the number of bytes per element for a given GGMLType.
// For block-quantized types, returns the output element size (FP32 = 4).
size_t GGMLTypeSize(GGMLType type);


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
// GGUFFile — Reads and provides access to a GGUF model file.
//
// Uses memory-mapped I/O for zero-copy tensor loading. The file is mapped
// once in Open() and all tensor reads go through the mapped memory region,
// eliminating per-tensor file open/seek/read overhead.
// ============================================================================
class GGUFFile {
public:
  GGUFFile() = default;
  ~GGUFFile();

  // Non-copyable due to mmap resources.
  GGUFFile(const GGUFFile &) = delete;
  GGUFFile &operator=(const GGUFFile &) = delete;

  // Opens and parses the GGUF file header and metadata.
  // Memory-maps the entire file for fast tensor loading.
  // Returns false on failure.
  bool Open(const std::string &path);

  // Closes the memory-mapped file and releases resources.
  void Close();

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

  // Load a tensor's data into a Tensor object.
  // Reads directly from memory-mapped region — no file I/O per call.
  Tensor LoadTensor(const std::string &name) const;

  // Load a 2D tensor and transpose it in a single fused pass.
  // Avoids the intermediate allocation from separate load + transpose +
  // contiguous. For F16 tensors, also fuses the F16→F32 conversion.
  //
  // Input shape in GGUF: [rows, cols]
  // Output Tensor shape:  [cols, rows] (transposed, contiguous)
  Tensor LoadTensorTransposed(const std::string &name) const;

  // List all tensor names.
  std::vector<std::string> TensorNames() const;

  // Print a summary of the file contents.
  void PrintSummary() const;

private:
  GGUFHeader header_{};
  std::map<std::string, GGUFMetadataValue> metadata_;
  std::map<std::string, GGUFTensorInfo> tensors_;
  std::string file_path_;
  uint64_t tensor_data_offset_ = 0; // Byte offset where tensor data starts
  uint64_t file_size_ = 0;

  // Memory-mapped file state
  const uint8_t *mapped_data_ = nullptr;
#ifdef _WIN32
  HANDLE file_handle_ = INVALID_HANDLE_VALUE;
  HANDLE mapping_handle_ = nullptr;
#else
  int fd_ = -1;
#endif

  // --- Internal parsing helpers ---
  bool ReadHeader(std::ifstream &file);
  bool ReadMetadata(std::ifstream &file);
  bool ReadTensorInfos(std::ifstream &file);

  // Memory-map the file. Called from Open() after parsing headers.
  bool MapFile();

  // Low-level read helpers
  std::string ReadString(std::ifstream &file);
  GGUFMetadataValue ReadValue(std::ifstream &file, GGUFValueType type);
};

} // namespace ie
