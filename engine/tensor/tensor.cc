#include "engine/tensor/tensor.h"

#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace ie {

// ============================================================================
// DType helpers — these are done for you since they're just boilerplate.
// ============================================================================

size_t DTypeSize(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return 4;
    case DType::kFloat16:
      return 2;
    case DType::kInt8:
      return 1;
    default:
      throw std::runtime_error("Unknown DType");
  }
}

std::string DTypeName(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return "float32";
    case DType::kFloat16:
      return "float16";
    case DType::kInt8:
      return "int8";
    default:
      return "unknown";
  }
}


Tensor::Tensor() : shape_({}), dtype_(DType::kFloat32) {}

Tensor::Tensor(std::vector<int64_t> shape, DType dtype)
    : shape_(shape), dtype_(dtype) {
  // TODO: Allocate memory for the tensor data!
  //
  // Steps:
  //   1. Compute total bytes: numel() * DTypeSize(dtype)
  //   2. Allocate that many bytes (however you chose to store data_)
  //   3. Zero-fill the memory
  //
  // If using std::vector<uint8_t>:
  //   data_.resize(nbytes(), 0);
  //
  // If using std::unique_ptr:
  //   data_ = std::make_unique<uint8_t[]>(nbytes());
  //   std::memset(data_.get(), 0, nbytes());
}

int64_t Tensor::numel() const {
  if (shape_.empty()) return 1;  // A scalar tensor has 1 element
  // Multiply all dimensions together: {2, 3, 4} -> 2 * 3 * 4 = 24
  int64_t result = 1;
  for (int64_t dim : shape_) {
    result *= dim;
  }
  return result;
}

size_t Tensor::nbytes() const {
  return static_cast<size_t>(numel()) * DTypeSize(dtype_);
}

void* Tensor::data_ptr() {
  // TODO: Return a pointer to your data storage.
  // If using std::vector<uint8_t>:  return data_.data();
  // If using unique_ptr:            return data_.get();
  return nullptr;
}

const void* Tensor::data_ptr() const {
  // Same as above but const.
  return nullptr;
}

float Tensor::at(std::vector<int64_t> indices) const {
  // TODO: Implement element access!
  //
  // For a tensor with shape {2, 3}, element at indices {1, 2}:
  //   flat_index = 1 * 3 + 2 = 5
  //
  // General formula for N dimensions:
  //   flat_index = 0
  //   for i in range(ndim):
  //     flat_index = flat_index * shape[i] + indices[i]
  //
  // Then read the float at that position in your data array:
  //   return reinterpret_cast<const float*>(data_ptr())[flat_index];
  //
  // BONUS: Add bounds checking! Throw if indices[i] >= shape[i].
  throw std::runtime_error("at() not implemented yet — this is your first TODO!");
}

void Tensor::set(std::vector<int64_t> indices, float value) {
  // TODO: Same logic as at(), but write instead of read.
  throw std::runtime_error("set() not implemented yet");
}

std::string Tensor::to_string() const {
  std::ostringstream ss;
  ss << "Tensor(shape=[";
  for (size_t i = 0; i < shape_.size(); ++i) {
    if (i > 0) ss << ", ";
    ss << shape_[i];
  }
  ss << "], dtype=" << DTypeName(dtype_) << ", numel=" << numel() << ")";
  return ss.str();
}

}  // namespace ie
