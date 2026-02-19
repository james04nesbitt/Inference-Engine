#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "absl/types/span.h"

namespace ie {

// Supported data types for tensor elements.
// You'll extend this as you add quantization support.
enum class DType {
  kFloat32,
  kFloat16,
  kBFloat16,
  kInt32,
  kInt8,
  // TODO: Add quantized types (Q4_0, Q4_K_M, Q8_0, etc.)
};

// Returns the size in bytes of a single element of the given dtype.
size_t DTypeSize(DType dtype);

// Returns a human-readable name for the dtype.
std::string DTypeName(DType dtype);

// ============================================================================
// Tensor - The core multi-dimensional array class.
//
// YOUR MAIN LEARNING OBJECTIVE: Implement this class fully.
//
// Design notes:
//   - Tensors own their data via shared_ptr (enables cheap views/slices).
//   - Shape is stored as a vector of int64_t dimensions.
//   - Strides enable non-contiguous views (slicing, transposing).
//
// Concepts to explore as you implement:
//   - Memory layout (row-major vs column-major)
//   - Broadcasting rules (NumPy-style)
//   - Views vs copies (when does data get shared?)
//   - Move semantics and RAII
// ============================================================================
class Tensor {
 public:
  // --- Construction ---

  // Creates an empty (scalar) tensor.
  Tensor();

  // Creates a tensor with the given shape, filled with zeros.
  Tensor(std::vector<int64_t> shape, DType dtype = DType::kFloat32);

  // Creates a tensor from an existing data buffer (takes ownership).
  // The buffer must contain exactly `product(shape) * DTypeSize(dtype)` bytes.
  Tensor(std::vector<int64_t> shape, DType dtype,
         std::shared_ptr<uint8_t[]> data);

  // --- Factory Methods ---
  static Tensor Zeros(std::vector<int64_t> shape,
                      DType dtype = DType::kFloat32);
  static Tensor Ones(std::vector<int64_t> shape,
                     DType dtype = DType::kFloat32);
  static Tensor Full(std::vector<int64_t> shape, float value,
                     DType dtype = DType::kFloat32);

  // Creates a tensor from a raw buffer without taking ownership.
  // WARNING: The caller must ensure the buffer outlives the tensor.
  static Tensor FromBuffer(void* data, std::vector<int64_t> shape,
                           DType dtype = DType::kFloat32);

  // --- Properties ---
  const std::vector<int64_t>& shape() const { return shape_; }
  const std::vector<int64_t>& strides() const { return strides_; }
  DType dtype() const { return dtype_; }
  int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
  int64_t numel() const;  // Total number of elements
  size_t nbytes() const;  // Total size in bytes
  bool is_contiguous() const;

  // --- Data Access ---
  void* data_ptr() { return data_.get(); }
  const void* data_ptr() const { return data_.get(); }

  // Typed data access. Throws if dtype doesn't match.
  template <typename T>
  T* data() {
    // TODO: Add dtype check
    return reinterpret_cast<T*>(data_.get());
  }

  template <typename T>
  const T* data() const {
    // TODO: Add dtype check
    return reinterpret_cast<const T*>(data_.get());
  }

  // Element access (for debugging, not performance-critical).
  // TODO: Implement multi-dimensional indexing
  float at(std::vector<int64_t> indices) const;
  void set(std::vector<int64_t> indices, float value);

  // --- Shape Manipulation ---
  // TODO: Implement these operations
  Tensor reshape(std::vector<int64_t> new_shape) const;
  Tensor view(std::vector<int64_t> new_shape) const;  // Must be contiguous
  Tensor transpose(int64_t dim0, int64_t dim1) const;
  Tensor slice(int64_t dim, int64_t start, int64_t end) const;
  Tensor contiguous() const;  // Returns a contiguous copy if needed

  // --- Debugging ---
  std::string to_string() const;
  void print() const;

 private:
  std::vector<int64_t> shape_;
  DType dtype_;  // Must be declared before strides_ (init order!)
  std::vector<int64_t> strides_;
  std::shared_ptr<uint8_t[]> data_;

  // Computes default strides for a contiguous tensor with the given shape.
  static std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape,
                                             DType dtype);
};

}  // namespace ie
