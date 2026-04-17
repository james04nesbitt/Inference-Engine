#include "engine/gguf/gguf_loader.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace ie {

// ============================================================================
// GGMLType helpers
// ============================================================================

DType GGMLTypeToDType(GGMLType type) {
  switch (type) {
  case GGMLType::kF32:
    return DType::kFloat32;
  case GGMLType::kF16:
    return DType::kFloat16;
  case GGMLType::kQ8_0:
  case GGMLType::kQ4_0:
    // Quantized types are dequantized to FP32 during loading.
    return DType::kFloat32;
  default:
    throw std::runtime_error("GGMLTypeToDType: unsupported/quantized type " +
                             std::to_string(static_cast<uint32_t>(type)));
  }
}

size_t GGMLTypeSize(GGMLType type) {
  switch (type) {
  case GGMLType::kF32:
    return 4;
  case GGMLType::kF16:
    return 2;
  case GGMLType::kQ8_0:
  case GGMLType::kQ4_0:
    // Block-quantized: use GGMLBlockSize() for raw byte computation.
    // Return the output element size (FP32) since we dequantize on load.
    return 4;
  default:
    throw std::runtime_error("GGMLTypeSize: unsupported/quantized type " +
                             std::to_string(static_cast<uint32_t>(type)));
  }
}

// Number of elements per quantization block.
int64_t GGMLBlockElements(GGMLType type) {
  switch (type) {
  case GGMLType::kQ8_0:
    return 32; // 32 INT8 values per block
  case GGMLType::kQ4_0:
    return 32; // 32 INT4 values per block (packed into 16 bytes)
  default:
    return 1; // Non-block types
  }
}

// Raw byte size per quantization block.
size_t GGMLBlockBytes(GGMLType type) {
  switch (type) {
  case GGMLType::kQ8_0:
    return 2 + 32; // FP16 scale (2 bytes) + 32 INT8 values
  case GGMLType::kQ4_0:
    return 2 + 16; // FP16 scale (2 bytes) + 16 bytes (32 nibbles)
  default:
    return 0;
  }
}

// ============================================================================
// FP16 → FP32 conversion helper
// ============================================================================
static float Fp16ToFp32(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;

  if (exp == 0) {
    if (mant == 0) {
      // ±zero
      uint32_t bits = sign << 31;
      float f;
      std::memcpy(&f, &bits, sizeof(f));
      return f;
    }
    // Subnormal: convert to normalized FP32
    float val = std::ldexp(static_cast<float>(mant), -24);
    return sign ? -val : val;
  }
  if (exp == 31) {
    // Inf/NaN
    uint32_t bits = (sign << 31) | 0x7F800000 | (mant << 13);
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
  }
  // Normal number
  uint32_t bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// ============================================================================
// Q8_0 dequantization
// ============================================================================
// Block format: [FP16 scale (2 bytes)] [32 x INT8 values (32 bytes)] = 34 bytes
// Dequantized: value[i] = scale * quant[i]
static void DequantizeQ8_0(const uint8_t *src, float *dst, int64_t numel) {
  constexpr int64_t BLOCK_SIZE = 32;
  constexpr size_t BLOCK_BYTES = 2 + 32; // FP16 + 32 INT8s
  int64_t n_blocks = numel / BLOCK_SIZE;

  for (int64_t b = 0; b < n_blocks; ++b) {
    const uint8_t *block = src + b * BLOCK_BYTES;

    // Read FP16 scale
    uint16_t scale_fp16;
    std::memcpy(&scale_fp16, block, sizeof(scale_fp16));
    float scale = Fp16ToFp32(scale_fp16);

    // Read and dequantize 32 INT8 values
    const int8_t *quants = reinterpret_cast<const int8_t *>(block + 2);
    for (int64_t i = 0; i < BLOCK_SIZE; ++i) {
      dst[b * BLOCK_SIZE + i] = scale * static_cast<float>(quants[i]);
    }
  }
}

// ============================================================================
// Q4_0 dequantization
// ============================================================================
// Block format: [FP16 scale (2 bytes)] [16 bytes = 32 nibbles] = 18 bytes
// Each byte holds two 4-bit values (low nibble first).
// Values are unsigned [0, 15] with zero_point = 8: dequant = scale * (val - 8)
static void DequantizeQ4_0(const uint8_t *src, float *dst, int64_t numel) {
  constexpr int64_t BLOCK_SIZE = 32;
  constexpr size_t BLOCK_BYTES = 2 + 16; // FP16 + 16 nibble bytes
  int64_t n_blocks = numel / BLOCK_SIZE;

  for (int64_t b = 0; b < n_blocks; ++b) {
    const uint8_t *block = src + b * BLOCK_BYTES;

    // Read FP16 scale
    uint16_t scale_fp16;
    std::memcpy(&scale_fp16, block, sizeof(scale_fp16));
    float scale = Fp16ToFp32(scale_fp16);

    // Read 16 bytes = 32 nibbles
    const uint8_t *nibbles = block + 2;
    for (int64_t i = 0; i < 16; ++i) {
      uint8_t byte = nibbles[i];
      int8_t lo = static_cast<int8_t>(byte & 0x0F) - 8;
      int8_t hi = static_cast<int8_t>(byte >> 4) - 8;
      dst[b * BLOCK_SIZE + i * 2] = scale * static_cast<float>(lo);
      dst[b * BLOCK_SIZE + i * 2 + 1] = scale * static_cast<float>(hi);
    }
  }
}

// ============================================================================
// GGUFFile destructor
// ============================================================================

GGUFFile::~GGUFFile() { Close(); }

void GGUFFile::Close() {
  if (mapped_data_) {
#ifdef _WIN32
    UnmapViewOfFile(mapped_data_);
    mapped_data_ = nullptr;
    if (mapping_handle_) {
      CloseHandle(mapping_handle_);
      mapping_handle_ = nullptr;
    }
    if (file_handle_ != INVALID_HANDLE_VALUE) {
      CloseHandle(file_handle_);
      file_handle_ = INVALID_HANDLE_VALUE;
    }
#else
    munmap(const_cast<uint8_t *>(mapped_data_),
           static_cast<size_t>(file_size_));
    mapped_data_ = nullptr;
    if (fd_ >= 0) {
      close(fd_);
      fd_ = -1;
    }
#endif
  }
  file_size_ = 0;
}

// ============================================================================
// MapFile — memory-map the entire GGUF file
// ============================================================================

bool GGUFFile::MapFile() {
#ifdef _WIN32
  file_handle_ = CreateFileA(file_path_.c_str(), GENERIC_READ, FILE_SHARE_READ,
                             nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                             nullptr);
  if (file_handle_ == INVALID_HANDLE_VALUE) {
    std::cerr << "MapFile: failed to open file: " << file_path_ << std::endl;
    return false;
  }

  LARGE_INTEGER size;
  if (!GetFileSizeEx(file_handle_, &size)) {
    std::cerr << "MapFile: failed to get file size" << std::endl;
    CloseHandle(file_handle_);
    file_handle_ = INVALID_HANDLE_VALUE;
    return false;
  }
  file_size_ = static_cast<uint64_t>(size.QuadPart);

  mapping_handle_ =
      CreateFileMappingA(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!mapping_handle_) {
    std::cerr << "MapFile: failed to create file mapping" << std::endl;
    CloseHandle(file_handle_);
    file_handle_ = INVALID_HANDLE_VALUE;
    return false;
  }

  mapped_data_ = static_cast<const uint8_t *>(
      MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0));
  if (!mapped_data_) {
    std::cerr << "MapFile: failed to map view of file" << std::endl;
    CloseHandle(mapping_handle_);
    mapping_handle_ = nullptr;
    CloseHandle(file_handle_);
    file_handle_ = INVALID_HANDLE_VALUE;
    return false;
  }
#else
  fd_ = open(file_path_.c_str(), O_RDONLY);
  if (fd_ < 0) {
    std::cerr << "MapFile: failed to open file: " << file_path_ << std::endl;
    return false;
  }

  struct stat st;
  if (fstat(fd_, &st) != 0) {
    std::cerr << "MapFile: failed to stat file" << std::endl;
    close(fd_);
    fd_ = -1;
    return false;
  }
  file_size_ = static_cast<uint64_t>(st.st_size);

  void *ptr =
      mmap(nullptr, static_cast<size_t>(file_size_), PROT_READ, MAP_PRIVATE,
           fd_, 0);
  if (ptr == MAP_FAILED) {
    std::cerr << "MapFile: mmap failed" << std::endl;
    close(fd_);
    fd_ = -1;
    return false;
  }
  mapped_data_ = static_cast<const uint8_t *>(ptr);
#endif

  return true;
}

// ============================================================================
// Open — top-level entry point
// ============================================================================

bool GGUFFile::Open(const std::string &path) {
  file_path_ = path;
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << path << std::endl;
    return false;
  }

  // Phase 1: Read and validate header
  if (!ReadHeader(file))
    return false;

  // Phase 2: Read metadata key-value pairs
  if (!ReadMetadata(file))
    return false;

  // Phase 3: Read tensor info entries
  if (!ReadTensorInfos(file))
    return false;

  // Record where tensor data starts (for LoadTensor)
  tensor_data_offset_ = static_cast<uint64_t>(file.tellg());

  // Align to 32 bytes (GGUF spec)
  uint64_t alignment = 32;
  if (auto *align_val = GetMetadata("general.alignment")) {
    if (auto *v = std::get_if<uint32_t>(align_val)) {
      alignment = *v;
    }
  }
  tensor_data_offset_ =
      (tensor_data_offset_ + alignment - 1) & ~(alignment - 1);

  // Close the ifstream — we'll use mmap from here on.
  file.close();

  // Phase 4: Memory-map the file for fast tensor loading.
  if (!MapFile()) {
    std::cerr << "Warning: memory-map failed, falling back to ifstream"
              << std::endl;
    // mapped_data_ remains nullptr; LoadTensor will fall back to ifstream.
  }

  return true;
}

// ============================================================================
// ReadHeader
// ============================================================================

bool GGUFFile::ReadHeader(std::ifstream &file) {
  // Read the 4-byte magic number
  file.read(reinterpret_cast<char *>(&header_.magic), sizeof(header_.magic));
  if (header_.magic != kGGUFMagic) {
    std::cerr << "Invalid GGUF magic number: 0x" << std::hex << header_.magic
              << std::dec << " (expected 0x" << std::hex << kGGUFMagic << ")"
              << std::dec << std::endl;
    return false;
  }

  // Read version, tensor count, metadata count
  file.read(reinterpret_cast<char *>(&header_.version),
            sizeof(header_.version));
  file.read(reinterpret_cast<char *>(&header_.tensor_count),
            sizeof(header_.tensor_count));
  file.read(reinterpret_cast<char *>(&header_.metadata_kv_count),
            sizeof(header_.metadata_kv_count));

  std::cout << "GGUF v" << header_.version << " | " << header_.tensor_count
            << " tensors | " << header_.metadata_kv_count << " metadata entries"
            << std::endl;

  return true;
}

// ============================================================================
// ReadMetadata — parse all metadata key-value pairs
// ============================================================================

bool GGUFFile::ReadMetadata(std::ifstream &file) {
  for (uint64_t i = 0; i < header_.metadata_kv_count; ++i) {
    // 1. Read key string
    std::string key = ReadString(file);

    // 2. Read value type
    uint32_t type_raw = 0;
    file.read(reinterpret_cast<char *>(&type_raw), sizeof(type_raw));
    auto type = static_cast<GGUFValueType>(type_raw);

    // 3. Read value based on type
    if (type == GGUFValueType::kArray) {
      // Arrays: read element type + count, then each element
      uint32_t elem_type_raw = 0;
      file.read(reinterpret_cast<char *>(&elem_type_raw),
                sizeof(elem_type_raw));
      auto elem_type = static_cast<GGUFValueType>(elem_type_raw);

      uint64_t count = 0;
      file.read(reinterpret_cast<char *>(&count), sizeof(count));

      switch (elem_type) {
      case GGUFValueType::kInt32: {
        std::vector<int32_t> arr(count);
        for (uint64_t j = 0; j < count; ++j) {
          file.read(reinterpret_cast<char *>(&arr[j]), sizeof(int32_t));
        }
        metadata_[key] = std::move(arr);
        break;
      }
      case GGUFValueType::kFloat32: {
        std::vector<float> arr(count);
        for (uint64_t j = 0; j < count; ++j) {
          file.read(reinterpret_cast<char *>(&arr[j]), sizeof(float));
        }
        metadata_[key] = std::move(arr);
        break;
      }
      case GGUFValueType::kString: {
        std::vector<std::string> arr;
        arr.reserve(count);
        for (uint64_t j = 0; j < count; ++j) {
          arr.push_back(ReadString(file));
        }
        metadata_[key] = std::move(arr);
        break;
      }
      default: {
        // For unsupported array element types, skip the raw bytes.
        // Each element's size depends on type; for now skip by reading them
        // individually with ReadValue (which may throw for truly unsupported
        // types, but handles most scalar types).
        for (uint64_t j = 0; j < count; ++j) {
          ReadValue(file, elem_type); // discard result
        }
        // Store nothing for this key — we don't have a variant type for it
        break;
      }
      }
    } else {
      // Scalar types
      metadata_[key] = ReadValue(file, type);
    }

    if (file.fail()) {
      std::cerr << "ReadMetadata: read error at entry " << i << " (key=\""
                << key << "\")" << std::endl;
      return false;
    }
  }
  return true;
}

// ============================================================================
// ReadTensorInfos — parse tensor info entries
// ============================================================================

bool GGUFFile::ReadTensorInfos(std::ifstream &file) {
  for (uint64_t i = 0; i < header_.tensor_count; ++i) {
    GGUFTensorInfo info;

    // 1. Read tensor name
    info.name = ReadString(file);

    // 2. Read number of dimensions
    uint32_t n_dims = 0;
    file.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));

    // 3. Read each dimension
    info.dimensions.resize(n_dims);
    for (uint32_t d = 0; d < n_dims; ++d) {
      file.read(reinterpret_cast<char *>(&info.dimensions[d]),
                sizeof(uint64_t));
    }

    // 4. Read tensor type
    uint32_t type_raw = 0;
    file.read(reinterpret_cast<char *>(&type_raw), sizeof(type_raw));
    info.type = static_cast<GGMLType>(type_raw);

    // 5. Read data offset (relative to start of tensor data section)
    file.read(reinterpret_cast<char *>(&info.offset), sizeof(info.offset));

    if (file.fail()) {
      std::cerr << "ReadTensorInfos: read error at entry " << i << std::endl;
      return false;
    }

    // 6. Store in map
    tensors_[info.name] = std::move(info);
  }
  return true;
}

// ============================================================================
// String reading helper
// ============================================================================

std::string GGUFFile::ReadString(std::ifstream &file) {
  // GGUF strings are length-prefixed: uint64_t length, then `length` bytes
  uint64_t length = 0;
  file.read(reinterpret_cast<char *>(&length), sizeof(length));
  std::string result(length, '\0');
  file.read(result.data(), static_cast<std::streamsize>(length));
  return result;
}

GGUFMetadataValue GGUFFile::ReadValue(std::ifstream &file, GGUFValueType type) {
  switch (type) {
  case GGUFValueType::kUint32: {
    uint32_t val;
    file.read(reinterpret_cast<char *>(&val), sizeof(val));
    return val;
  }
  case GGUFValueType::kInt32: {
    int32_t val;
    file.read(reinterpret_cast<char *>(&val), sizeof(val));
    return val;
  }
  case GGUFValueType::kFloat32: {
    float val;
    file.read(reinterpret_cast<char *>(&val), sizeof(val));
    return val;
  }
  case GGUFValueType::kBool: {
    uint8_t val;
    file.read(reinterpret_cast<char *>(&val), sizeof(val));
    return static_cast<bool>(val);
  }
  case GGUFValueType::kString:
    return ReadString(file);
  case GGUFValueType::kUint64: {
    uint64_t val;
    file.read(reinterpret_cast<char *>(&val), sizeof(val));
    return val;
  }
  case GGUFValueType::kInt64: {
    int64_t val;
    file.read(reinterpret_cast<char *>(&val), sizeof(val));
    return val;
  }
  case GGUFValueType::kUint8: {
    uint8_t val;
    file.read(reinterpret_cast<char *>(&val), sizeof(val));
    return val;
  }
  case GGUFValueType::kInt8: {
    int8_t val;
    file.read(reinterpret_cast<char *>(&val), sizeof(val));
    return val;
  }
  case GGUFValueType::kUint16: {
    uint16_t val;
    file.read(reinterpret_cast<char *>(&val), sizeof(val));
    return val;
  }
  case GGUFValueType::kInt16: {
    int16_t val;
    file.read(reinterpret_cast<char *>(&val), sizeof(val));
    return val;
  }
  case GGUFValueType::kFloat64: {
    double val;
    file.read(reinterpret_cast<char *>(&val), sizeof(val));
    return val;
  }
  default:
    throw std::runtime_error("ReadValue: unsupported type " +
                             std::to_string(static_cast<uint32_t>(type)));
  }
}

// ============================================================================
// Accessors
// ============================================================================

const GGUFMetadataValue *GGUFFile::GetMetadata(const std::string &key) const {
  auto it = metadata_.find(key);
  return (it != metadata_.end()) ? &it->second : nullptr;
}

std::string GGUFFile::GetString(const std::string &key,
                                const std::string &default_val) const {
  if (auto *val = GetMetadata(key)) {
    if (auto *s = std::get_if<std::string>(val))
      return *s;
  }
  return default_val;
}

int64_t GGUFFile::GetInt(const std::string &key, int64_t default_val) const {
  if (auto *val = GetMetadata(key)) {
    if (auto *v = std::get_if<int32_t>(val))
      return *v;
    if (auto *v = std::get_if<uint32_t>(val))
      return *v;
    if (auto *v = std::get_if<int64_t>(val))
      return *v;
    if (auto *v = std::get_if<uint64_t>(val))
      return static_cast<int64_t>(*v);
  }
  return default_val;
}

float GGUFFile::GetFloat(const std::string &key, float default_val) const {
  if (auto *val = GetMetadata(key)) {
    if (auto *v = std::get_if<float>(val))
      return *v;
  }
  return default_val;
}

const GGUFTensorInfo *GGUFFile::GetTensorInfo(const std::string &name) const {
  auto it = tensors_.find(name);
  return (it != tensors_.end()) ? &it->second : nullptr;
}

// ============================================================================
// LoadTensor — read tensor data from memory-mapped region
// ============================================================================

Tensor GGUFFile::LoadTensor(const std::string &name) const {
  const GGUFTensorInfo *info = GetTensorInfo(name);
  if (!info) {
    throw std::runtime_error("LoadTensor: tensor not found: " + name);
  }

  // Compute logical shape and element count.
  int64_t numel = 1;
  std::vector<int64_t> shape;
  shape.reserve(info->dimensions.size());
  for (uint64_t d : info->dimensions) {
    shape.push_back(static_cast<int64_t>(d));
    numel *= static_cast<int64_t>(d);
  }

  uint64_t data_pos = tensor_data_offset_ + info->offset;

  // --- Memory-mapped path (fast) ---
  if (mapped_data_) {
    const uint8_t *src = mapped_data_ + data_pos;

    // Block-quantized types: dequantize from mmap'd memory.
    if (info->type == GGMLType::kQ8_0 || info->type == GGMLType::kQ4_0) {
      size_t out_bytes = static_cast<size_t>(numel) * sizeof(float);
      auto buf = std::shared_ptr<uint8_t[]>(new uint8_t[out_bytes]);
      float *out = reinterpret_cast<float *>(buf.get());

      if (info->type == GGMLType::kQ8_0) {
        DequantizeQ8_0(src, out, numel);
      } else {
        DequantizeQ4_0(src, out, numel);
      }

      return Tensor::from_buffer(std::move(buf), std::move(shape),
                                 DType::kFloat32);
    }

    // Non-quantized types: memcpy from mmap'd memory.
    DType dtype = GGMLTypeToDType(info->type);
    size_t elem_size =
        (info->type == GGMLType::kF16) ? 2 : GGMLTypeSize(info->type);
    size_t nbytes = static_cast<size_t>(numel) * elem_size;

    auto buf = std::shared_ptr<uint8_t[]>(new uint8_t[nbytes]);
    std::memcpy(buf.get(), src, nbytes);

    return Tensor::from_buffer(std::move(buf), std::move(shape), dtype);
  }

  // --- Fallback: ifstream path (if mmap failed) ---
  std::ifstream file(file_path_, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("LoadTensor: cannot open file: " + file_path_);
  }
  file.seekg(static_cast<std::streamoff>(data_pos));
  if (file.fail()) {
    throw std::runtime_error("LoadTensor: seek failed for tensor: " + name);
  }

  // Block-quantized types require dequantization on load.
  if (info->type == GGMLType::kQ8_0 || info->type == GGMLType::kQ4_0) {
    int64_t block_elems = GGMLBlockElements(info->type);
    size_t block_bytes = GGMLBlockBytes(info->type);
    int64_t n_blocks = numel / block_elems;
    size_t raw_bytes = static_cast<size_t>(n_blocks) * block_bytes;

    std::vector<uint8_t> raw(raw_bytes);
    file.read(reinterpret_cast<char *>(raw.data()),
              static_cast<std::streamsize>(raw_bytes));
    if (file.fail()) {
      throw std::runtime_error("LoadTensor: read failed for tensor: " + name);
    }

    size_t out_bytes = static_cast<size_t>(numel) * sizeof(float);
    auto buf = std::shared_ptr<uint8_t[]>(new uint8_t[out_bytes]);
    float *out = reinterpret_cast<float *>(buf.get());

    if (info->type == GGMLType::kQ8_0) {
      DequantizeQ8_0(raw.data(), out, numel);
    } else {
      DequantizeQ4_0(raw.data(), out, numel);
    }

    return Tensor::from_buffer(std::move(buf), std::move(shape),
                               DType::kFloat32);
  }

  // Non-quantized types: direct load.
  DType dtype = GGMLTypeToDType(info->type);
  size_t elem_size = GGMLTypeSize(info->type);
  size_t nbytes = static_cast<size_t>(numel) * elem_size;

  auto buf = std::shared_ptr<uint8_t[]>(new uint8_t[nbytes]);
  file.read(reinterpret_cast<char *>(buf.get()),
            static_cast<std::streamsize>(nbytes));
  if (file.fail()) {
    throw std::runtime_error("LoadTensor: read failed for tensor: " + name);
  }

  return Tensor::from_buffer(std::move(buf), std::move(shape), dtype);
}

// ============================================================================
// LoadTensorTransposed — fused load + transpose for 2D weight matrices
//
// Instead of: LoadTensor → transpose(0,1) → contiguous()  (2 allocations)
// This does:  read from mmap → write directly in transposed layout (1 alloc)
//
// For F16 tensors, also fuses the F16→F32 conversion, so the result is
// always an FP32 contiguous tensor in transposed layout.
// ============================================================================

Tensor GGUFFile::LoadTensorTransposed(const std::string &name) const {
  const GGUFTensorInfo *info = GetTensorInfo(name);
  if (!info) {
    throw std::runtime_error("LoadTensorTransposed: tensor not found: " + name);
  }

  if (info->dimensions.size() != 2) {
    throw std::runtime_error(
        "LoadTensorTransposed: expected 2D tensor, got " +
        std::to_string(info->dimensions.size()) + "D for: " + name);
  }

  int64_t rows = static_cast<int64_t>(info->dimensions[0]);
  int64_t cols = static_cast<int64_t>(info->dimensions[1]);
  int64_t numel = rows * cols;

  // Output shape is transposed: [cols, rows]
  std::vector<int64_t> out_shape = {cols, rows};
  size_t out_bytes = static_cast<size_t>(numel) * sizeof(float);
  auto buf = std::shared_ptr<uint8_t[]>(new uint8_t[out_bytes]);
  float *out = reinterpret_cast<float *>(buf.get());

  uint64_t data_pos = tensor_data_offset_ + info->offset;

  if (mapped_data_) {
    const uint8_t *src = mapped_data_ + data_pos;

    if (info->type == GGMLType::kF16) {
      // Fused F16→F32 conversion + transpose.
      // Source layout: row-major [rows, cols] as FP16
      // Output layout: row-major [cols, rows] as FP32
      const uint16_t *fp16_src = reinterpret_cast<const uint16_t *>(src);
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          out[c * rows + r] = Fp16ToFp32(fp16_src[r * cols + c]);
        }
      }
    } else if (info->type == GGMLType::kF32) {
      // Transpose F32 data.
      const float *f32_src = reinterpret_cast<const float *>(src);
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          out[c * rows + r] = f32_src[r * cols + c];
        }
      }
    } else if (info->type == GGMLType::kQ8_0 ||
               info->type == GGMLType::kQ4_0) {
      // Dequantize into a temporary buffer, then transpose.
      // (Quantized data is block-structured so we can't easily fuse.)
      std::vector<float> tmp(numel);
      if (info->type == GGMLType::kQ8_0) {
        DequantizeQ8_0(src, tmp.data(), numel);
      } else {
        DequantizeQ4_0(src, tmp.data(), numel);
      }
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          out[c * rows + r] = tmp[r * cols + c];
        }
      }
    } else {
      throw std::runtime_error(
          "LoadTensorTransposed: unsupported type for: " + name);
    }
  } else {
    // Fallback: load normally then transpose.
    Tensor t = LoadTensor(name);
    Tensor transposed = t.transpose(0, 1).contiguous();
    return transposed;
  }

  return Tensor::from_buffer(std::move(buf), std::move(out_shape),
                             DType::kFloat32);
}

// ============================================================================
// Utilities
// ============================================================================

std::vector<std::string> GGUFFile::TensorNames() const {
  std::vector<std::string> names;
  names.reserve(tensors_.size());
  for (const auto &[name, _] : tensors_) {
    names.push_back(name);
  }
  return names;
}

void GGUFFile::PrintSummary() const {
  std::cout << "\n=== GGUF File Summary ===" << std::endl;
  std::cout << "Path: " << file_path_ << std::endl;
  std::cout << "Version: " << header_.version << std::endl;
  std::cout << "Tensors: " << header_.tensor_count << std::endl;
  std::cout << "Metadata entries: " << header_.metadata_kv_count << std::endl;

  std::cout << "\n--- Metadata ---" << std::endl;
  for (const auto &[key, val] : metadata_) {
    std::cout << "  " << key << " = ";
    std::visit(
        [](const auto &v) {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, std::string>) {
            std::cout << "\"" << v << "\"";
          } else if constexpr (std::is_same_v<T, bool>) {
            std::cout << (v ? "true" : "false");
          } else if constexpr (std::is_same_v<T, std::vector<int32_t>> ||
                               std::is_same_v<T, std::vector<float>> ||
                               std::is_same_v<T, std::vector<std::string>>) {
            std::cout << "[array, size=" << v.size() << "]";
          } else {
            std::cout << v;
          }
        },
        val);
    std::cout << std::endl;
  }

  if (!tensors_.empty()) {
    std::cout << "\n--- Tensors (first 10) ---" << std::endl;
    int count = 0;
    for (const auto &[name, info] : tensors_) {
      if (count++ >= 10) {
        std::cout << "  ... and " << (tensors_.size() - 10) << " more"
                  << std::endl;
        break;
      }
      std::cout << "  " << name << " [";
      for (size_t i = 0; i < info.dimensions.size(); ++i) {
        if (i > 0)
          std::cout << ", ";
        std::cout << info.dimensions[i];
      }
      std::cout << "] type=" << static_cast<uint32_t>(info.type) << std::endl;
    }
  }
}

} // namespace ie
