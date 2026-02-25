#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/types/optional.h"

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

  // Allocates 64-byte aligned memory for Highway SIMD compatibility.
  explicit Tensor(std::vector<int64_t> shape, DType dtype = DType::kFloat32);

  // --- Static Factories ---

  // Wrap an existing buffer in a Tensor. Takes ownership via shared_ptr.
  // Used by GGUF loader: read bytes from file → wrap as Tensor without copying.
  //
  // Example:
  //   auto buf = std::shared_ptr<uint8_t[]>(new uint8_t[nbytes]);
  //   file.read(buf.get(), nbytes);
  //   Tensor weights = Tensor::from_buffer(buf, {out, in}, DType::kFloat16);
  //
  static Tensor from_buffer(std::shared_ptr<uint8_t[]> data,
                            std::vector<int64_t> shape, DType dtype);

  // Create a 1D Tensor from a vector of floats.
  // Convenient for tests and small tensors (position indices, scales, etc.).
  static Tensor from_vector(const std::vector<float> &values);

  // Create a Tensor filled with a specific value.
  // Example: auto m = Tensor::full({seq_len}, -INFINITY);  // FlashAttention
  static Tensor full(std::vector<int64_t> shape, float value,
                     DType dtype = DType::kFloat32);

  // Convenience: zero-filled and ones-filled tensors.
  static Tensor zeros(std::vector<int64_t> shape,
                      DType dtype = DType::kFloat32);
  static Tensor ones(std::vector<int64_t> shape, DType dtype = DType::kFloat32);

  // Concatenate a list of tensors along a dimension.
  // All tensors must have the same shape except in the cat dimension.
  // Needed by KV cache (appending new K/V) and attention (concat heads).
  static Tensor cat(const std::vector<Tensor> &tensors, int64_t dim);

  // --- Properties ---
  const std::vector<int64_t> &shape() const { return shape_; }
  DType dtype() const { return dtype_; }

  // Returns the size of a specific dimension. PyTorch-style accessor.
  // Example: int64_t seq_len = x.size(1);
  int64_t size(int64_t dim) const;

  // Number of dimensions. A vector is 1D, a matrix is 2D, etc.
  int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }

  // Total number of elements. For shape {2, 3, 4} this is 2*3*4 = 24.
  int64_t numel() const;

  // Total size in bytes. numel() * DTypeSize(dtype).
  size_t nbytes() const;

  // Returns true if the tensor is C-contiguous (standard row-major layout).
  bool is_contiguous() const;

  // Raw untyped pointer, adjusted for offset_ so sliced views work.
  void *data_ptr();
  const void *data_ptr() const;

  // Typed pointer for direct access in kernels (SIMD, matmul, etc.).
  // Accounts for offset_ so sliced views point to the correct element.
  template <typename T> T *data() {
    return reinterpret_cast<T *>(data_.get()) + offset_;
  }
  template <typename T> const T *data() const {
    return reinterpret_cast<const T *>(data_.get()) + offset_;
  }

  // Element access — always works in float regardless of storage dtype.
  // Converts to/from the underlying dtype automatically.
  // These are for debugging/testing, NOT for hot paths.
  float at(std::vector<int64_t> indices) const;
  void set(std::vector<int64_t> indices, float value);

  // Compute flat index from multi-dimensional indices.
  int64_t flat_index(const std::vector<int64_t> &indices) const;

  // --- View and Reshape ---
  absl::optional<Tensor> view(std::vector<int64_t> new_shape) const;
  Tensor reshape(std::vector<int64_t> new_shape) const;
  Tensor transpose(int64_t dim0, int64_t dim1) const;
  Tensor permute(std::vector<int64_t> dims) const;

  // --- Slicing ---
  // Returns a zero-copy sub-tensor view along a given dimension.
  // Example: For a [4, 8, 16] tensor, slice(0, 1, 3) returns a [2, 8, 16] view.
  Tensor slice(int64_t dim, int64_t start, int64_t end) const;

  // --- Utilities ---
  // Fill entire tensor with a scalar value (respects dtype).
  void fill(float val);

  // Returns a contiguous (C-order) copy of this tensor.
  // If already contiguous, returns a shallow view sharing the same data.
  // Needed after transpose/permute before passing to SIMD kernels that
  // assume contiguous memory.
  Tensor contiguous() const;

  // --- Quantization Metadata ---
  // Whether this tensor has quantization parameters attached.
  bool is_quantized() const;

  // Attach per-channel quantization parameters.
  // scales: one scale factor per output channel (size = shape[0])
  // zero_points: one zero-point per output channel (size = shape[0])
  // Formula: float_val = (int8_val - zero_point) * scale
  void set_quantization_params(std::vector<float> scales,
                               std::vector<int32_t> zero_points);

  const std::vector<float> &scales() const;
  const std::vector<int32_t> &zero_points() const;

  // --- DType Conversion ---
  // Returns a new Tensor with the specified dtype, converting each element.
  // If already the target dtype, returns a view (no copy).
  Tensor to(DType target_dtype) const;

  // --- Deep Copy ---
  // Returns a deep copy (new allocation, data copied).
  // Always returns a contiguous tensor regardless of source layout.
  Tensor clone() const;

  // --- Shape Manipulation ---

  // Select a single index along a dimension, returning an (N-1)D view.
  // Example: For embedding table [vocab, embed_dim]:
  //   Tensor embed = table.select(0, token_id);  // returns [embed_dim]
  Tensor select(int64_t dim, int64_t index) const;

  // Insert a size-1 dimension at the given position.
  // Example: [3, 4] → unsqueeze(0) → [1, 3, 4]
  // Needed for broadcasting in attention masks.
  Tensor unsqueeze(int64_t dim) const;

  // Remove a size-1 dimension. Throws if shape[dim] != 1.
  // Example: [1, 3, 4] → squeeze(0) → [3, 4]
  Tensor squeeze(int64_t dim) const;

  // Repeat tensor along a dimension. Returns a new tensor.
  // Example: For GQA with num_heads=4, num_kv_heads=1:
  //   K_expanded = K.repeat(head_dim_axis, 4);  // repeat K 4x
  Tensor repeat(int64_t dim, int64_t times) const;

  // --- Utilities ---

  // Check if this tensor has the given shape.
  bool shape_equals(const std::vector<int64_t> &other_shape) const;
  bool shape_equals(const Tensor &other) const;

  std::string to_string() const;

private:
  // Private constructor for creating views that share data.
  Tensor(std::vector<int64_t> shape, std::vector<int64_t> strides, DType dtype,
         std::shared_ptr<uint8_t[]> data, int64_t offset = 0);

  std::vector<int64_t> shape_;
  std::vector<int64_t> strides_;
  DType dtype_;

  std::shared_ptr<uint8_t[]> data_;

  // Element offset into the shared data buffer.
  // Zero for tensors that own their data from the start.
  // Non-zero for sliced views — slice() sets this so data<T>()
  // points to the correct starting element without copying.
  int64_t offset_ = 0;

  // --- Quantization parameters (empty if not quantized) ---
  // Per-channel scale factors: float_val = (int8_val - zero_point) * scale
  std::vector<float> scales_;
  std::vector<int32_t> zero_points_;
};

} // namespace ie
