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
    data_ = std::shared_ptr<uint8_t[]>(new uint8_t[nbytes()]);
    std::memset(data_.get(), 0, nbytes());
}

//return the number of elements in the tensor
int64_t Tensor::numel() const {
  if (shape_.empty()) return 1;  
  int64_t result = 1;
  for (int64_t dim : shape_) {
    result *= dim;
  }
  return result;
}

//return the number of bytes in the tensor
size_t Tensor::nbytes() const {
  return static_cast<size_t>(numel()) * DTypeSize(dtype_);
}

void* Tensor::data_ptr() {
  return data_.get();
}

const void* Tensor::data_ptr() const {
  return data_.get();
}

// --- FP16 conversion helpers ---
// IEEE 754 half-precision: 1 sign, 5 exponent, 10 mantissa bits.
// These convert between uint16_t (storage) and float (compute).

static float HalfToFloat(uint16_t h) {
  uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exponent = (h >> 10) & 0x1F;
  uint32_t mantissa = h & 0x03FF;

  if (exponent == 0) {
    // Subnormal or zero
    if (mantissa == 0) {
      uint32_t bits = sign;
      float result;
      std::memcpy(&result, &bits, sizeof(result));
      return result;
    }
    // Subnormal: normalize it
    exponent = 1;
    while ((mantissa & 0x0400) == 0) {
      mantissa <<= 1;
      exponent--;
    }
    mantissa &= 0x03FF;
    exponent = exponent + (127 - 15);
  } else if (exponent == 31) {
    exponent = 255;  // Inf or NaN
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
    return static_cast<uint16_t>(sign);  // Flush to zero
  } else if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00);  // Inf
  }
  return static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
}

int64_t Tensor::flat_index(const std::vector<int64_t>& indices) const {
  int64_t idx = 0;
  for (int64_t i = 0; i < ndim(); ++i) {
    idx = idx * shape_[i] + indices[i];
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
