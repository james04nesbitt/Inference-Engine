#include "engine/tensor/tensor.h"

#include "absl/types/optional.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "hwy/highway.h"

namespace ie {
namespace internal {

static std::vector<int64_t> ComputeStrides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size());
  if (shape.empty())
    return strides;
  strides.back() = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

bool IncrementIndex(std::vector<int64_t> &indices,
                    const std::vector<int64_t> &shape) {
  for (int64_t i = static_cast<int64_t>(indices.size()) - 1; i >= 0; --i) {
    ++indices[i];
    if (indices[i] < shape[i]) {
      return true;
    }
    indices[i] = 0;
  }
  return false;
}

} // namespace internal

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

Tensor::Tensor() : shape_({}), strides_({}), dtype_(DType::kFloat32) {}

Tensor::Tensor(std::vector<int64_t> shape, DType dtype)
    : shape_(shape), strides_(internal::ComputeStrides(shape)), dtype_(dtype) {
  constexpr size_t kAlignment = 64;
  size_t alloc_size = nbytes();
  if (alloc_size == 0)
    alloc_size = kAlignment;

  // macOS std::aligned_alloc strictly requires size to be a multiple of
  // alignment
  alloc_size = (alloc_size + kAlignment - 1) & ~(kAlignment - 1);

#ifdef _WIN32
  void *raw = _aligned_malloc(alloc_size, kAlignment);
#else
  void *raw = std::aligned_alloc(kAlignment, alloc_size);
#endif
  if (!raw)
    throw std::runtime_error("Failed to allocate aligned memory");
  std::memset(raw, 0, alloc_size);
#ifdef _WIN32
  data_ = std::shared_ptr<uint8_t[]>(static_cast<uint8_t *>(raw),
                                     [](uint8_t *p) { _aligned_free(p); });
#else
  data_ = std::shared_ptr<uint8_t[]>(static_cast<uint8_t *>(raw),
                                     [](uint8_t *p) { std::free(p); });
#endif
}

Tensor::Tensor(std::vector<int64_t> shape, std::vector<int64_t> strides,
               DType dtype, std::shared_ptr<uint8_t[]> data, int64_t offset)
    : shape_(std::move(shape)), strides_(std::move(strides)), dtype_(dtype),
      data_(std::move(data)), offset_(offset) {}

int64_t Tensor::numel() const {
  if (shape_.empty())
    return 1;
  int64_t result = 1;
  for (int64_t dim : shape_) {
    result *= dim;
  }
  return result;
}

size_t Tensor::nbytes() const {
  return static_cast<size_t>(numel()) * DTypeSize(dtype_);
}

void *Tensor::data_ptr() {
  return static_cast<void *>(data_.get() + offset_ * DTypeSize(dtype_));
}

const void *Tensor::data_ptr() const {
  return static_cast<const void *>(data_.get() + offset_ * DTypeSize(dtype_));
}

static float HalfToFloat(uint16_t h) {
  uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exponent = (h >> 10) & 0x1F;
  uint32_t mantissa = h & 0x03FF;

  if (exponent == 0) {
    if (mantissa == 0) {
      uint32_t bits = sign;
      float result;
      std::memcpy(&result, &bits, sizeof(result));
      return result;
    }
    exponent = 1;
    while ((mantissa & 0x0400) == 0) {
      mantissa <<= 1;
      exponent--;
    }
    mantissa &= 0x03FF;
    exponent = exponent + (127 - 15);
  } else if (exponent == 31) {
    exponent = 255;
  } else {
    exponent = exponent + (127 - 15);
  }

  uint32_t bits = sign | (exponent << 23) | (mantissa << 13);
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

static uint16_t FloatToHalf(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));

  uint32_t sign = (bits >> 16) & 0x8000;
  int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
  uint32_t mantissa = bits & 0x007FFFFF;

  if (exponent <= 0) {
    return static_cast<uint16_t>(sign);
  } else if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00);
  }
  return static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
}

int64_t Tensor::flat_index(const std::vector<int64_t> &indices) const {
  int64_t idx = 0;
  for (int64_t i = 0; i < ndim(); ++i) {
    idx += indices[i] * strides_[i];
  }
  return idx;
}

float Tensor::at(const std::vector<int64_t> &indices) const {
  int64_t idx = flat_index(indices);
  switch (dtype_) {
  case DType::kFloat32:
    return data<float>()[idx];
  case DType::kFloat16:
    return HalfToFloat(data<uint16_t>()[idx]);
  case DType::kInt8:
    return static_cast<float>(data<int8_t>()[idx]);
  }
  throw std::runtime_error("Unknown dtype");
}

void Tensor::set(const std::vector<int64_t> &indices, float value) {
  int64_t idx = flat_index(indices);
  switch (dtype_) {
  case DType::kFloat32:
    data<float>()[idx] = value;
    return;
  case DType::kFloat16:
    data<uint16_t>()[idx] = FloatToHalf(value);
    return;
  case DType::kInt8:
    data<int8_t>()[idx] = static_cast<int8_t>(value);
    return;
  }
  throw std::runtime_error("Unknown dtype");
}

absl::optional<Tensor> Tensor::view(std::vector<int64_t> new_shape) const {
  if (!is_contiguous()) {
    return absl::nullopt;
  }
  int64_t new_numel = 1;
  for (int64_t d : new_shape)
    new_numel *= d;
  if (new_numel != numel()) {
    return absl::nullopt;
  }
  auto strides = internal::ComputeStrides(new_shape);
  return Tensor(std::move(new_shape), std::move(strides), dtype_, data_,
                offset_);
}

Tensor Tensor::reshape(std::vector<int64_t> new_shape) const {
  auto maybe_view = view(new_shape);
  if (maybe_view.has_value()) {
    return *std::move(maybe_view);
  }
  // Non-contiguous: must copy element-by-element
  Tensor result(new_shape, dtype_);
  std::vector<int64_t> indices(ndim(), 0);
  int64_t dst_flat = 0;
  do {
    float val = this->at(indices);
    switch (dtype_) {
    case DType::kFloat32:
      result.data<float>()[dst_flat] = val;
      break;
    case DType::kFloat16:
      result.data<uint16_t>()[dst_flat] = FloatToHalf(val);
      break;
    case DType::kInt8:
      result.data<int8_t>()[dst_flat] = static_cast<int8_t>(val);
      break;
    }
    ++dst_flat;
  } while (internal::IncrementIndex(indices, shape_));

  return result;
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
  auto new_shape = shape_;
  auto new_strides = strides_;
  std::swap(new_shape[dim0], new_shape[dim1]);
  std::swap(new_strides[dim0], new_strides[dim1]);
  return Tensor(std::move(new_shape), std::move(new_strides), dtype_, data_,
                offset_);
}

Tensor Tensor::permute(std::vector<int64_t> dims) const {
  std::vector<int64_t> new_shape(ndim());
  std::vector<int64_t> new_strides(ndim());
  for (int64_t i = 0; i < ndim(); ++i) {
    new_shape[i] = shape_[dims[i]];
    new_strides[i] = strides_[dims[i]];
  }
  return Tensor(std::move(new_shape), std::move(new_strides), dtype_, data_,
                offset_);
}

std::string Tensor::to_string() const {
  std::ostringstream ss;
  ss << "Tensor(shape=[";
  for (size_t i = 0; i < shape_.size(); ++i) {
    if (i > 0)
      ss << ", ";
    ss << shape_[i];
  }
  ss << "], dtype=" << DTypeName(dtype_) << ", numel=" << numel() << ")";
  return ss.str();
}

bool Tensor::is_contiguous() const {
  return strides_ == internal::ComputeStrides(shape_);
}

// ============================================================================
// Slicing
// ============================================================================

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end) const {
  if (dim < 0 || dim >= ndim() || start < 0 || end < start ||
      end > shape_[dim]) {
    throw std::runtime_error("Invalid slice parameters");
  }
  std::vector<int64_t> new_shape = shape_;
  new_shape[dim] = end - start;
  int64_t new_offset = offset_ + start * strides_[dim];
  return Tensor(new_shape, strides_, dtype_, data_, new_offset);
}

// ============================================================================
// Fill
// ============================================================================

void Tensor::fill(float val) {
  // Contiguous fast path: write directly to flat buffer
  if (is_contiguous()) {
    for (int64_t i = 0; i < numel(); ++i) {
      switch (dtype_) {
      case DType::kFloat32:
        data<float>()[i] = val;
        break;
      case DType::kFloat16:
        data<uint16_t>()[i] = FloatToHalf(val);
        break;
      case DType::kInt8:
        data<int8_t>()[i] = static_cast<int8_t>(val);
        break;
      }
    }
    return;
  }
  // Non-contiguous: must iterate with strides
  std::vector<int64_t> indices(ndim(), 0);
  do {
    set(indices, val);
  } while (internal::IncrementIndex(indices, shape_));
}

// ============================================================================
// Contiguous
// ============================================================================

Tensor Tensor::contiguous() const {
  if (is_contiguous() && offset_ == 0) {
    return *this;
  }
  Tensor result(shape_, dtype_);
  std::vector<int64_t> indices(ndim(), 0);
  int64_t dst_flat = 0;
  do {
    float val = this->at(indices);
    switch (dtype_) {
    case DType::kFloat32:
      result.data<float>()[dst_flat] = val;
      break;
    case DType::kFloat16:
      result.data<uint16_t>()[dst_flat] = FloatToHalf(val);
      break;
    case DType::kInt8:
      result.data<int8_t>()[dst_flat] = static_cast<int8_t>(val);
      break;
    }
    ++dst_flat;
  } while (internal::IncrementIndex(indices, shape_));
  return result;
}

// ============================================================================
// Quantization
// ============================================================================

bool Tensor::is_quantized() const { return !scales_.empty(); }

void Tensor::set_quantization_params(std::vector<float> scales,
                                     std::vector<int32_t> zero_points) {
  if (scales.size() != zero_points.size()) {
    throw std::runtime_error("scales and zero_points must have the same size");
  }
  if (!shape_.empty() && static_cast<int64_t>(scales.size()) != shape_[0]) {
    throw std::runtime_error(
        "scales size must match shape[0] (number of output channels)");
  }
  scales_ = std::move(scales);
  zero_points_ = std::move(zero_points);
}

const std::vector<float> &Tensor::scales() const { return scales_; }

const std::vector<int32_t> &Tensor::zero_points() const { return zero_points_; }

// ============================================================================
// Static Factories
// ============================================================================

Tensor Tensor::from_buffer(std::shared_ptr<uint8_t[]> data,
                           std::vector<int64_t> shape, DType dtype) {
  auto strides = internal::ComputeStrides(shape);
  return Tensor(std::move(shape), std::move(strides), dtype, std::move(data));
}

Tensor Tensor::from_vector(const std::vector<float> &values) {
  Tensor t({static_cast<int64_t>(values.size())}, DType::kFloat32);
  std::memcpy(t.data<float>(), values.data(), values.size() * sizeof(float));
  return t;
}

Tensor Tensor::full(std::vector<int64_t> shape, float value, DType dtype) {
  Tensor t(std::move(shape), dtype);
  t.fill(value);
  return t;
}

Tensor Tensor::zeros(std::vector<int64_t> shape, DType dtype) {
  // Constructor already zero-initializes via memset
  return Tensor(std::move(shape), dtype);
}

Tensor Tensor::ones(std::vector<int64_t> shape, DType dtype) {
  return full(std::move(shape), 1.0f, dtype);
}

Tensor Tensor::cat(const std::vector<Tensor> &tensors, int64_t dim) {
  if (tensors.empty()) {
    throw std::runtime_error("cat() requires at least one tensor");
  }
  const Tensor &first = tensors[0];
  DType dtype = first.dtype();
  int64_t ndims = first.ndim();

  // Validate all tensors have matching ndim, dtype, and shapes (except dim)
  for (size_t t = 1; t < tensors.size(); ++t) {
    if (tensors[t].ndim() != ndims) {
      throw std::runtime_error("cat(): all tensors must have same ndim");
    }
    if (tensors[t].dtype() != dtype) {
      throw std::runtime_error("cat(): all tensors must have same dtype");
    }
    for (int64_t d = 0; d < ndims; ++d) {
      if (d != dim && tensors[t].shape()[d] != first.shape()[d]) {
        throw std::runtime_error(
            "cat(): shapes must match except in cat dimension");
      }
    }
  }

  // Compute output shape
  std::vector<int64_t> out_shape = first.shape();
  out_shape[dim] = 0;
  for (const auto &t : tensors) {
    out_shape[dim] += t.shape()[dim];
  }

  Tensor result(out_shape, dtype);

  // Copy each tensor into the result at the correct offset along dim
  int64_t offset_along_dim = 0;
  for (const auto &src : tensors) {
    int64_t src_dim_size = src.shape()[dim];
    // Iterate over all elements in src, compute where they go in result
    std::vector<int64_t> src_indices(ndims, 0);
    do {
      std::vector<int64_t> dst_indices = src_indices;
      dst_indices[dim] += offset_along_dim;
      result.set(dst_indices, src.at(src_indices));
    } while (internal::IncrementIndex(src_indices, src.shape()));
    offset_along_dim += src_dim_size;
  }

  return result;
}

// ============================================================================
// Accessor
// ============================================================================

int64_t Tensor::size(int64_t dim) const {
  if (dim < 0)
    dim += ndim();
  if (dim < 0 || dim >= ndim()) {
    throw std::runtime_error("size(): dimension out of range");
  }
  return shape_[dim];
}

// ============================================================================
// DType Conversion
// ============================================================================

Tensor Tensor::to(DType target_dtype) const {
  if (dtype_ == target_dtype) {
    if (is_contiguous() && offset_ == 0) {
      // Return a view sharing the same data
      return Tensor(shape_, strides_, dtype_, data_, 0);
    }
    // Same dtype but non-contiguous — make a contiguous copy
    return contiguous();
  }

  Tensor result(shape_, target_dtype);
  std::vector<int64_t> indices(ndim(), 0);
  do {
    // at() reads as float regardless of source dtype
    // set() writes as float, converting to target dtype
    result.set(indices, this->at(indices));
  } while (internal::IncrementIndex(indices, shape_));

  return result;
}

// ============================================================================
// Deep Copy
// ============================================================================

Tensor Tensor::clone() const {
  Tensor result(shape_, dtype_);
  if (is_contiguous() && offset_ == 0) {
    std::memcpy(result.data_ptr(), data_ptr(), nbytes());
  } else {
    std::vector<int64_t> indices(ndim(), 0);
    do {
      result.set(indices, this->at(indices));
    } while (internal::IncrementIndex(indices, shape_));
  }
  return result;
}

// ============================================================================
// Shape Manipulation
// ============================================================================

Tensor Tensor::select(int64_t dim, int64_t index) const {
  if (dim < 0)
    dim += ndim();
  if (dim < 0 || dim >= ndim()) {
    throw std::runtime_error("select(): dimension out of range");
  }
  if (index < 0 || index >= shape_[dim]) {
    throw std::runtime_error("select(): index out of range");
  }

  int64_t new_offset = offset_ + index * strides_[dim];

  // Remove the selected dimension
  std::vector<int64_t> new_shape;
  std::vector<int64_t> new_strides;
  new_shape.reserve(ndim() - 1);
  new_strides.reserve(ndim() - 1);
  for (int64_t i = 0; i < ndim(); ++i) {
    if (i != dim) {
      new_shape.push_back(shape_[i]);
      new_strides.push_back(strides_[i]);
    }
  }

  return Tensor(std::move(new_shape), std::move(new_strides), dtype_, data_,
                new_offset);
}

Tensor Tensor::unsqueeze(int64_t dim) const {
  if (dim < 0)
    dim += ndim() + 1;
  if (dim < 0 || dim > ndim()) {
    throw std::runtime_error("unsqueeze(): dimension out of range");
  }

  std::vector<int64_t> new_shape = shape_;
  std::vector<int64_t> new_strides = strides_;

  // Compute the stride for the new dimension
  int64_t new_stride;
  if (dim < ndim()) {
    new_stride = shape_[dim] * strides_[dim];
  } else {
    // Inserting at the end
    new_stride = 1;
  }

  new_shape.insert(new_shape.begin() + dim, 1);
  new_strides.insert(new_strides.begin() + dim, new_stride);

  return Tensor(std::move(new_shape), std::move(new_strides), dtype_, data_,
                offset_);
}

Tensor Tensor::squeeze(int64_t dim) const {
  if (dim < 0)
    dim += ndim();
  if (dim < 0 || dim >= ndim()) {
    throw std::runtime_error("squeeze(): dimension out of range");
  }
  if (shape_[dim] != 1) {
    throw std::runtime_error(
        "squeeze(): can only squeeze dimensions of size 1");
  }

  std::vector<int64_t> new_shape;
  std::vector<int64_t> new_strides;
  new_shape.reserve(ndim() - 1);
  new_strides.reserve(ndim() - 1);
  for (int64_t i = 0; i < ndim(); ++i) {
    if (i != dim) {
      new_shape.push_back(shape_[i]);
      new_strides.push_back(strides_[i]);
    }
  }

  return Tensor(std::move(new_shape), std::move(new_strides), dtype_, data_,
                offset_);
}

Tensor Tensor::repeat(int64_t dim, int64_t times) const {
  if (dim < 0)
    dim += ndim();
  if (dim < 0 || dim >= ndim()) {
    throw std::runtime_error("repeat(): dimension out of range");
  }
  if (times < 1) {
    throw std::runtime_error("repeat(): times must be >= 1");
  }

  std::vector<int64_t> out_shape = shape_;
  out_shape[dim] *= times;
  Tensor result(out_shape, dtype_);

  int64_t src_dim_size = shape_[dim];

  // Copy data for each repetition
  std::vector<int64_t> src_indices(ndim(), 0);
  do {
    float val = this->at(src_indices);
    // Write this value into each repetition
    for (int64_t t = 0; t < times; ++t) {
      std::vector<int64_t> dst_indices = src_indices;
      dst_indices[dim] += t * src_dim_size;
      result.set(dst_indices, val);
    }
  } while (internal::IncrementIndex(src_indices, shape_));

  return result;
}

// ============================================================================
// Utilities
// ============================================================================

bool Tensor::shape_equals(const std::vector<int64_t> &other_shape) const {
  return shape_ == other_shape;
}

bool Tensor::shape_equals(const Tensor &other) const {
  return shape_equals(other.shape());
}

} // namespace ie
