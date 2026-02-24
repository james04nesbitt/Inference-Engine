#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ie {


enum class DType {
  kFloat32,
  kFloat16,
  kInt8,
};


size_t DTypeSize(DType dtype);

std::string DTypeName(DType dtype);


class Tensor {
 public:

  Tensor();

 
  Tensor(std::vector<int64_t> shape, DType dtype = DType::kFloat32);

  // --- Properties ---
  const std::vector<int64_t>& shape() const { return shape_; }
  DType dtype() const { return dtype_; }

  // Number of dimensions. A vector is 1D, a matrix is 2D, etc.
  int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }

  // Total number of elements. For shape {2, 3, 4} this is 2*3*4 = 24.
  int64_t numel() const;

  // Total size in bytes. numel() * DTypeSize(dtype).
  size_t nbytes() const;

  // Raw untyped pointer — use data<T>() instead when you know the type.
  void* data_ptr();
  const void* data_ptr() const;

  // Typed pointer for direct access in kernels (SIMD, matmul, etc.).
  // Usage: float* p = tensor.data<float>();
  template <typename T>
  T* data() { return reinterpret_cast<T*>(data_.get()); }
  template <typename T>
  const T* data() const { return reinterpret_cast<const T*>(data_.get()); }

  // Element access — always works in float regardless of storage dtype.
  // Converts to/from the underlying dtype automatically.
  // These are for debugging/testing, NOT for hot paths.
  float at(std::vector<int64_t> indices) const;
  void set(std::vector<int64_t> indices, float value);

  // Compute flat index from multi-dimensional indices.
  int64_t flat_index(const std::vector<int64_t>& indices) const;

  std::string to_string() const;

 private:
  std::vector<int64_t> shape_;
  DType dtype_;

  std::shared_ptr<uint8_t[]> data_;     
};

}  // namespace ie
