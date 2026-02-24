#include "engine/tensor/tensor.h"

#include "absl/types/optional.h"
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace ie {
namespace internal {

// Forward declarations for internal helpers
bool IncrementIndex(std::vector<int64_t> &indices,
                    const std::vector<int64_t> &shape);
static std::vector<int64_t> ComputeStrides(const std::vector<int64_t> &shape);

// Compute row-major (C-contiguous) strides from shape.
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

Tensor::Tensor() : shape_({}), strides_({}), dtype_(DType::kFloat32) {}

Tensor::Tensor(std::vector<int64_t> shape, DType dtype)
    : shape_(shape), strides_(internal::ComputeStrides(shape)), dtype_(dtype) {
  data_ = std::shared_ptr<uint8_t[]>(new uint8_t[nbytes()]);
  std::memset(data_.get(), 0, nbytes());
}

Tensor::Tensor(std::vector<int64_t> shape, std::vector<int64_t> strides,
               DType dtype, std::shared_ptr<uint8_t[]> data)
    : shape_(std::move(shape)), strides_(std::move(strides)), dtype_(dtype),
      data_(std::move(data)) {}

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

void *Tensor::data_ptr() { return data_.get(); }

const void *Tensor::data_ptr() const { return data_.get(); }

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

float Tensor::at(std::vector<int64_t> indices) const {
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

void Tensor::set(std::vector<int64_t> indices, float value) {
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
  return Tensor(std::move(new_shape), internal::ComputeStrides(new_shape),
                dtype_, data_);
}

Tensor Tensor::reshape(std::vector<int64_t> new_shape) const {
  auto maybe_view = view(new_shape);
  if (maybe_view.has_value()) {
    return *std::move(maybe_view);
  }
  Tensor result(new_shape, dtype_);
  std::vector<int64_t> indices(ndim(), 0);
  int64_t dst_flat = 0;
  do {
    float val = this->at(indices);
    result.data<float>()[dst_flat] = val;
    ++dst_flat;
  } while (internal::IncrementIndex(indices, shape_));

  return result;
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
  auto new_shape = shape_;
  auto new_strides = strides_;
  std::swap(new_shape[dim0], new_shape[dim1]);
  std::swap(new_strides[dim0], new_strides[dim1]);
  return Tensor(std::move(new_shape), std::move(new_strides), dtype_, data_);
}

Tensor Tensor::permute(std::vector<int64_t> dims) const {
  std::vector<int64_t> new_shape(ndim());
  std::vector<int64_t> new_strides(ndim());
  for (int64_t i = 0; i < ndim(); ++i) {
    new_shape[i] = shape_[dims[i]];
    new_strides[i] = strides_[dims[i]];
  }
  return Tensor(std::move(new_shape), std::move(new_strides), dtype_, data_);
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

} // namespace ie
