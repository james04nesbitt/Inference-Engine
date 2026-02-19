#pragma once

#include <cstddef>
#include <cstdint>
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

// ============================================================================
// Tensor — A multi-dimensional array of numbers.
//
// This is THE fundamental data structure in ML. Everything — weights,
// activations, embeddings — is a tensor.
//
// YOUR JOB: Build this class from scratch. Start simple and grow it.
//
// A tensor needs to know three things:
//   1. SHAPE:  How many dimensions and how big each one is.
//              Example: {2, 3} means a 2x3 matrix (2 rows, 3 columns).
//
//   2. DATA:   The actual numbers, stored as a flat array in memory.
//              A {2, 3} tensor has 6 numbers stored contiguously.
//
//   3. DTYPE:  What kind of number each element is (float32, int8, etc.)
//
// Key questions to answer as you build:
//   - Who OWNS the data? (What happens when a Tensor goes out of scope?)
//   - How do you go from multi-dimensional indices like [row][col] to a
//     single flat index into the data array?
//   - What happens when you "reshape" a tensor? Does the data move?
//
// ============================================================================
class Tensor {
 public:
  // --- Constructors (START HERE) ---

  // Default constructor: Creates an empty tensor.
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

  // --- Data Access ---
  // Returns a raw pointer to the underlying data.
  // You decide the return type based on how you store the data.
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
  // TODO: Implement this!
  float at(std::vector<int64_t> indices) const;
  void set(std::vector<int64_t> indices, float value);

  // --- Debugging ---
  std::string to_string() const;

 private:
  std::vector<int64_t> shape_;
  DType dtype_;

  // ⬇️ YOU DECIDE: How to store the actual data.
  //
  // Some options:
  //   std::vector<uint8_t> data_;           // Simple, vector manages memory
  //   std::unique_ptr<uint8_t[]> data_;     // Smart pointer, you own it
  //   std::shared_ptr<uint8_t[]> data_;     // Shared ownership (PyTorch style)
  //
  // Start with std::vector<uint8_t> if you're not sure — it's the simplest.
  // You can always refactor to shared_ptr later when you need tensor views.
};

}  // namespace ie
