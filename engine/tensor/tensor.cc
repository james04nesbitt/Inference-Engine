#include "engine/tensor/tensor.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace ie {

size_t DTypeSize(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return 4;
    case DType::kFloat16:
      return 2;
    case DType::kBFloat16:
      return 2;
    case DType::kInt32:
      return 4;
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
    case DType::kBFloat16:
      return "bfloat16";
    case DType::kInt32:
      return "int32";
    case DType::kInt8:
      return "int8";
    default:
      return "unknown";
  }
}

// --- Construction ---

Tensor::Tensor() : shape_({}), strides_({}), dtype_(DType::kFloat32) {}

Tensor::Tensor(std::vector<int64_t> shape, DType dtype)
    : shape_(std::move(shape)),
      dtype_(dtype),
      strides_(ComputeStrides(shape_, dtype_)) {
  size_t total_bytes = nbytes();
  data_ = std::shared_ptr<uint8_t[]>(new uint8_t[total_bytes]());
}

Tensor::Tensor(std::vector<int64_t> shape, DType dtype,
               std::shared_ptr<uint8_t[]> data)
    : shape_(std::move(shape)),
      dtype_(dtype),
      strides_(ComputeStrides(shape_, dtype_)),
      data_(std::move(data)) {}

// --- Factory Methods ---

Tensor Tensor::Zeros(std::vector<int64_t> shape, DType dtype) {
  return Tensor(std::move(shape), dtype);  // Already zero-initialized
}

Tensor Tensor::Ones(std::vector<int64_t> shape, DType dtype) {
  return Full(std::move(shape), 1.0f, dtype);
}

Tensor Tensor::Full(std::vector<int64_t> shape, float value, DType dtype) {
  Tensor t(shape, dtype);
  if (dtype == DType::kFloat32) {
    float* ptr = t.data<float>();
    int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i) {
      ptr[i] = value;
    }
  } else {
    // TODO: Handle other dtypes
    throw std::runtime_error("Full() only supports float32 for now");
  }
  return t;
}

Tensor Tensor::FromBuffer(void* data, std::vector<int64_t> shape,
                          DType dtype) {
  // TODO: Implement non-owning view from external buffer
  // Hint: You'll need a custom deleter that does nothing
  throw std::runtime_error("FromBuffer not implemented yet");
}

// --- Properties ---

int64_t Tensor::numel() const {
  if (shape_.empty()) return 1;  // Scalar
  return std::accumulate(shape_.begin(), shape_.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

size_t Tensor::nbytes() const {
  return static_cast<size_t>(numel()) * DTypeSize(dtype_);
}

bool Tensor::is_contiguous() const {
  // TODO: Check if strides match the default contiguous layout
  // Hint: Compare strides_ with ComputeStrides(shape_, dtype_)
  return true;  // Placeholder
}

// --- Element Access ---

float Tensor::at(std::vector<int64_t> indices) const {
  // TODO: Implement with proper bounds checking and stride-based indexing
  // This is your first real exercise! Think about:
  //   offset = sum(indices[i] * strides_[i]) for each dimension
  throw std::runtime_error("at() not implemented yet");
}

void Tensor::set(std::vector<int64_t> indices, float value) {
  // TODO: Mirror the at() implementation
  throw std::runtime_error("set() not implemented yet");
}

// --- Shape Manipulation ---

Tensor Tensor::reshape(std::vector<int64_t> new_shape) const {
  // TODO: Implement reshape
  // Key insight: reshape is only valid if the total number of elements
  // doesn't change. If the tensor is contiguous, you can just change
  // the shape and strides. If not, you need to copy first.
  throw std::runtime_error("reshape() not implemented yet");
}

Tensor Tensor::view(std::vector<int64_t> new_shape) const {
  // TODO: Like reshape but requires contiguity (no copy allowed)
  throw std::runtime_error("view() not implemented yet");
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
  // TODO: Swap dimensions and strides (no data copy needed!)
  // This is the beauty of stride-based tensors.
  throw std::runtime_error("transpose() not implemented yet");
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end) const {
  // TODO: Return a view into a sub-range of one dimension
  // Hint: Adjust the data pointer and shape, keep same strides
  throw std::runtime_error("slice() not implemented yet");
}

Tensor Tensor::contiguous() const {
  if (is_contiguous()) return *this;  // Already contiguous, share data
  // TODO: Copy data into a new contiguous buffer
  throw std::runtime_error("contiguous() not implemented yet");
}

// --- Debugging ---

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

void Tensor::print() const { std::cout << to_string() << std::endl; }

// --- Private Helpers ---

std::vector<int64_t> Tensor::ComputeStrides(const std::vector<int64_t>& shape,
                                            DType dtype) {
  if (shape.empty()) return {};
  std::vector<int64_t> strides(shape.size());
  int64_t stride = static_cast<int64_t>(DTypeSize(dtype));
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

}  // namespace ie
