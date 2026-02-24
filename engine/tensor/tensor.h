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

  // Creates a tensor with the given shape, filled with zeros.
  //
  // Example: Tensor t({2, 3});  // Creates a 2x3 matrix of zeros
  //
  // What you need to do:
  //   1. Store the shape
  //   2. Store the dtype
  //   3. Calculate how many total elements: 2 * 3 = 6
  //   4. Allocate memory: 6 elements * 4 bytes each = 24 bytes
  //   5. Zero out the memory
  //
  // QUESTION FOR YOU: How should you allocate the memory?
  //   Option A: new float[6]                    — raw pointer, you manage delete[]
  //   Option B: std::vector<uint8_t>(24)        — vector manages memory for you
  //   Option C: std::unique_ptr<uint8_t[]>(...)  — smart pointer, auto-deletes
  //   Option D: std::shared_ptr<uint8_t[]>(...)  — reference-counted smart pointer
  //
  //   Hint: Option D is what PyTorch uses. Why? Because when you "slice" or
  //   "reshape" a tensor, you want multiple Tensor objects to share the SAME
  //   underlying data. shared_ptr lets you do that safely.
  //
  //   But start with whatever makes sense to you! You can refactor later.
  //
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

  void* data_ptr();
  const void* data_ptr() const;

  // --- Element Access ---
  // Given indices like {1, 2} for a 2D tensor, return the value at that position.
  //
  // THE KEY INSIGHT: Your data is stored as a flat 1D array.
  // For a {2, 3} tensor, the layout in memory is:
  //
  //   Logical view:     Flat memory:
  //   [0,0] [0,1] [0,2]    [0] [1] [2] [3] [4] [5]
  //   [1,0] [1,1] [1,2]
  //
  // So element [1, 2] is at flat index: 1 * 3 + 2 = 5
  // General formula: flat_index = indices[0] * dim[1] + indices[1]
  //
  // For 3D {2, 3, 4}: flat_index = i * (3*4) + j * 4 + k
  //
  float at(std::vector<int64_t> indices) const;
  void set(std::vector<int64_t> indices, float value);

  std::string to_string() const;

 private:
  std::vector<int64_t> shape_;
  DType dtype_;

  std::shared_ptr<uint8_t[]> data_;     
};

}  // namespace ie
