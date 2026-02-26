#include "engine/gguf/gguf_loader.h"

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
  default:
    throw std::runtime_error("GGMLTypeSize: unsupported/quantized type " +
                             std::to_string(static_cast<uint32_t>(type)));
  }
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
// LoadTensor — read tensor data from file into a Tensor object
// ============================================================================

Tensor GGUFFile::LoadTensor(const std::string &name) const {
  // 1. Look up tensor info
  const GGUFTensorInfo *info = GetTensorInfo(name);
  if (!info) {
    throw std::runtime_error("LoadTensor: tensor not found: " + name);
  }

  // 2. Convert GGMLType to our DType (throws for unsupported quantized types)
  DType dtype = GGMLTypeToDType(info->type);
  size_t elem_size = GGMLTypeSize(info->type);

  // 3. Compute total number of elements and bytes
  int64_t numel = 1;
  std::vector<int64_t> shape;
  shape.reserve(info->dimensions.size());
  for (uint64_t d : info->dimensions) {
    shape.push_back(static_cast<int64_t>(d));
    numel *= static_cast<int64_t>(d);
  }
  size_t nbytes = static_cast<size_t>(numel) * elem_size;

  // 4. Open the file and seek to the tensor's data
  std::ifstream file(file_path_, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("LoadTensor: cannot open file: " + file_path_);
  }
  uint64_t seek_pos = tensor_data_offset_ + info->offset;
  file.seekg(static_cast<std::streamoff>(seek_pos));
  if (file.fail()) {
    throw std::runtime_error("LoadTensor: seek failed for tensor: " + name);
  }

  // 5. Allocate buffer and read raw bytes
  auto buf = std::shared_ptr<uint8_t[]>(new uint8_t[nbytes]);
  file.read(reinterpret_cast<char *>(buf.get()),
            static_cast<std::streamsize>(nbytes));
  if (file.fail()) {
    throw std::runtime_error("LoadTensor: read failed for tensor: " + name);
  }

  // 6. Wrap in a Tensor — zero-copy, from_buffer shares the buffer
  return Tensor::from_buffer(std::move(buf), std::move(shape), dtype);
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
