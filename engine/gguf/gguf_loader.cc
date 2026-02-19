#include "engine/gguf/gguf_loader.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace ie {

bool GGUFFile::Open(const std::string& path) {
  file_path_ = path;
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << path << std::endl;
    return false;
  }

  // Phase 1: Read and validate header
  if (!ReadHeader(file)) return false;

  // Phase 2: Read metadata key-value pairs
  if (!ReadMetadata(file)) return false;

  // Phase 3: Read tensor info entries
  if (!ReadTensorInfos(file)) return false;

  // Record where tensor data starts (for LoadTensor)
  tensor_data_offset_ = static_cast<uint64_t>(file.tellg());

  // Align to 32 bytes (GGUF spec)
  uint64_t alignment = 32;
  if (auto* align_val = GetMetadata("general.alignment")) {
    if (auto* v = std::get_if<uint32_t>(align_val)) {
      alignment = *v;
    }
  }
  tensor_data_offset_ =
      (tensor_data_offset_ + alignment - 1) & ~(alignment - 1);

  return true;
}

bool GGUFFile::ReadHeader(std::ifstream& file) {
  // Read the 4-byte magic number
  file.read(reinterpret_cast<char*>(&header_.magic), sizeof(header_.magic));
  if (header_.magic != kGGUFMagic) {
    std::cerr << "Invalid GGUF magic number: 0x" << std::hex << header_.magic
              << std::dec << " (expected 0x" << std::hex << kGGUFMagic << ")"
              << std::dec << std::endl;
    return false;
  }

  // Read version, tensor count, metadata count
  file.read(reinterpret_cast<char*>(&header_.version), sizeof(header_.version));
  file.read(reinterpret_cast<char*>(&header_.tensor_count),
            sizeof(header_.tensor_count));
  file.read(reinterpret_cast<char*>(&header_.metadata_kv_count),
            sizeof(header_.metadata_kv_count));

  std::cout << "GGUF v" << header_.version << " | " << header_.tensor_count
            << " tensors | " << header_.metadata_kv_count << " metadata entries"
            << std::endl;

  return true;
}

bool GGUFFile::ReadMetadata(std::ifstream& file) {
  // TODO: Parse each metadata key-value pair
  //
  // For each of metadata_kv_count entries:
  //   1. Read key string (length-prefixed: uint64_t length, then chars)
  //   2. Read value type (uint32_t -> GGUFValueType enum)
  //   3. Read value based on type
  //   4. Store in metadata_ map
  //
  // The tricky part is handling arrays and nested types.
  // Start with simple types (uint32, string, float) and add more as needed.
  //
  // Hint: Use ReadString() and ReadValue() helpers below.

  std::cerr << "ReadMetadata: NOT YET IMPLEMENTED — skipping "
            << header_.metadata_kv_count << " entries" << std::endl;
  return true;  // Return true so you can test header reading first
}

bool GGUFFile::ReadTensorInfos(std::ifstream& file) {
  // TODO: Parse tensor info entries
  //
  // For each of tensor_count entries:
  //   1. Read tensor name (string)
  //   2. Read n_dimensions (uint32_t)
  //   3. Read each dimension (uint64_t × n_dimensions)
  //   4. Read tensor type (uint32_t -> GGMLType enum)
  //   5. Read tensor data offset (uint64_t, relative to tensor data start)
  //   6. Store in tensors_ map

  std::cerr << "ReadTensorInfos: NOT YET IMPLEMENTED — skipping "
            << header_.tensor_count << " entries" << std::endl;
  return true;
}

// --- String reading helper ---

std::string GGUFFile::ReadString(std::ifstream& file) {
  // GGUF strings are length-prefixed: uint64_t length, then `length` bytes
  uint64_t length = 0;
  file.read(reinterpret_cast<char*>(&length), sizeof(length));
  std::string result(length, '\0');
  file.read(result.data(), static_cast<std::streamsize>(length));
  return result;
}

GGUFMetadataValue GGUFFile::ReadValue(std::ifstream& file,
                                      GGUFValueType type) {
  // TODO: Read a value based on its type
  // Switch on type and read the appropriate number of bytes
  switch (type) {
    case GGUFValueType::kUint32: {
      uint32_t val;
      file.read(reinterpret_cast<char*>(&val), sizeof(val));
      return val;
    }
    case GGUFValueType::kInt32: {
      int32_t val;
      file.read(reinterpret_cast<char*>(&val), sizeof(val));
      return val;
    }
    case GGUFValueType::kFloat32: {
      float val;
      file.read(reinterpret_cast<char*>(&val), sizeof(val));
      return val;
    }
    case GGUFValueType::kBool: {
      uint8_t val;
      file.read(reinterpret_cast<char*>(&val), sizeof(val));
      return static_cast<bool>(val);
    }
    case GGUFValueType::kString:
      return ReadString(file);
    case GGUFValueType::kUint64: {
      uint64_t val;
      file.read(reinterpret_cast<char*>(&val), sizeof(val));
      return val;
    }
    case GGUFValueType::kInt64: {
      int64_t val;
      file.read(reinterpret_cast<char*>(&val), sizeof(val));
      return val;
    }
    default:
      throw std::runtime_error(
          "ReadValue: unsupported type " +
          std::to_string(static_cast<uint32_t>(type)));
  }
}

// --- Accessors ---

const GGUFMetadataValue* GGUFFile::GetMetadata(const std::string& key) const {
  auto it = metadata_.find(key);
  return (it != metadata_.end()) ? &it->second : nullptr;
}

std::string GGUFFile::GetString(const std::string& key,
                                const std::string& default_val) const {
  if (auto* val = GetMetadata(key)) {
    if (auto* s = std::get_if<std::string>(val)) return *s;
  }
  return default_val;
}

int64_t GGUFFile::GetInt(const std::string& key, int64_t default_val) const {
  if (auto* val = GetMetadata(key)) {
    if (auto* v = std::get_if<int32_t>(val)) return *v;
    if (auto* v = std::get_if<uint32_t>(val)) return *v;
    if (auto* v = std::get_if<int64_t>(val)) return *v;
    if (auto* v = std::get_if<uint64_t>(val))
      return static_cast<int64_t>(*v);
  }
  return default_val;
}

float GGUFFile::GetFloat(const std::string& key, float default_val) const {
  if (auto* val = GetMetadata(key)) {
    if (auto* v = std::get_if<float>(val)) return *v;
  }
  return default_val;
}

const GGUFTensorInfo* GGUFFile::GetTensorInfo(const std::string& name) const {
  auto it = tensors_.find(name);
  return (it != tensors_.end()) ? &it->second : nullptr;
}

Tensor GGUFFile::LoadTensor(const std::string& name) const {
  // TODO: Load tensor data from the file
  //
  // Steps:
  //   1. Look up tensor info by name
  //   2. Open file, seek to tensor_data_offset_ + tensor.offset
  //   3. Read the raw bytes
  //   4. If type is F32 or F16, wrap in a Tensor
  //   5. If type is quantized, you'll need dequantization (later!)
  throw std::runtime_error("LoadTensor not implemented yet");
}

std::vector<std::string> GGUFFile::TensorNames() const {
  std::vector<std::string> names;
  names.reserve(tensors_.size());
  for (const auto& [name, _] : tensors_) {
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
  for (const auto& [key, val] : metadata_) {
    std::cout << "  " << key << " = ";
    std::visit(
        [](const auto& v) {
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
    for (const auto& [name, info] : tensors_) {
      if (count++ >= 10) {
        std::cout << "  ... and " << (tensors_.size() - 10) << " more"
                  << std::endl;
        break;
      }
      std::cout << "  " << name << " [";
      for (size_t i = 0; i < info.dimensions.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << info.dimensions[i];
      }
      std::cout << "] type=" << static_cast<uint32_t>(info.type) << std::endl;
    }
  }
}

}  // namespace ie
