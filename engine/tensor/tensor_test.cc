#include "engine/tensor/tensor.h"

#include <cmath>
#include <cstring>
#include <limits>

#include <gtest/gtest.h>

namespace ie {
namespace {

// ============================================================================
// Construction & Properties
// ============================================================================

TEST(TensorTest, DefaultConstruction) {
  Tensor t;
  EXPECT_EQ(t.ndim(), 0);
  EXPECT_EQ(t.numel(), 1);  // Scalar = 1 element
  EXPECT_EQ(t.dtype(), DType::kFloat32);
}

TEST(TensorTest, ShapedConstruction) {
  Tensor t({3, 4, 5});
  EXPECT_EQ(t.ndim(), 3);
  EXPECT_EQ(t.shape()[0], 3);
  EXPECT_EQ(t.shape()[1], 4);
  EXPECT_EQ(t.shape()[2], 5);
  EXPECT_EQ(t.numel(), 60);
  EXPECT_EQ(t.nbytes(), 60 * 4);
  EXPECT_EQ(t.dtype(), DType::kFloat32);
}

TEST(TensorTest, OneDimensional) {
  Tensor t({10});
  EXPECT_EQ(t.ndim(), 1);
  EXPECT_EQ(t.numel(), 10);
  EXPECT_EQ(t.nbytes(), 40);
}

TEST(TensorTest, SingleElement) {
  Tensor t({1});
  EXPECT_EQ(t.numel(), 1);
  EXPECT_EQ(t.nbytes(), 4);
}

TEST(TensorTest, LargeShape) {
  Tensor t({1024, 1152});  // Gemma-3 sized
  EXPECT_EQ(t.numel(), 1024 * 1152);
  EXPECT_EQ(t.nbytes(), static_cast<size_t>(1024 * 1152 * 4));
}

// ============================================================================
// DType sizes and names
// ============================================================================

TEST(TensorTest, DTypeSizes) {
  EXPECT_EQ(DTypeSize(DType::kFloat32), 4u);
  EXPECT_EQ(DTypeSize(DType::kFloat16), 2u);
  EXPECT_EQ(DTypeSize(DType::kInt8), 1u);
}

TEST(TensorTest, DTypeNames) {
  EXPECT_EQ(DTypeName(DType::kFloat32), "float32");
  EXPECT_EQ(DTypeName(DType::kFloat16), "float16");
  EXPECT_EQ(DTypeName(DType::kInt8), "int8");
}

TEST(TensorTest, NonDefaultDtype) {
  Tensor f16({4, 4}, DType::kFloat16);
  EXPECT_EQ(f16.dtype(), DType::kFloat16);
  EXPECT_EQ(f16.nbytes(), 16 * 2);  // 16 elements * 2 bytes

  Tensor i8({4, 4}, DType::kInt8);
  EXPECT_EQ(i8.dtype(), DType::kInt8);
  EXPECT_EQ(i8.nbytes(), 16 * 1);  // 16 elements * 1 byte
}

// ============================================================================
// Memory allocation
// ============================================================================

TEST(TensorTest, DataIsAllocated) {
  Tensor t({2, 3});
  EXPECT_NE(t.data_ptr(), nullptr);
  EXPECT_NE(t.data<float>(), nullptr);
  EXPECT_EQ(t.nbytes(), 24u);
}

TEST(TensorTest, ZeroInitialized) {
  Tensor t({4, 4});
  const float* p = t.data<float>();
  for (int64_t i = 0; i < t.numel(); ++i) {
    EXPECT_FLOAT_EQ(p[i], 0.0f) << "Element " << i << " not zero";
  }
}

// ============================================================================
// FP32 — at() / set()
// ============================================================================

TEST(TensorTest, SetAndGet2D) {
  Tensor t({2, 3});
  t.set({0, 0}, 1.0f);
  t.set({0, 2}, 3.0f);
  t.set({1, 2}, 42.0f);
  EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(t.at({0, 2}), 3.0f);
  EXPECT_FLOAT_EQ(t.at({1, 2}), 42.0f);
  EXPECT_FLOAT_EQ(t.at({0, 1}), 0.0f);  // Untouched = zero
}

TEST(TensorTest, SetAndGet3D) {
  Tensor t({2, 3, 4});
  t.set({0, 0, 0}, 1.0f);
  t.set({1, 2, 3}, 99.0f);
  t.set({0, 1, 2}, -7.5f);
  EXPECT_FLOAT_EQ(t.at({0, 0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(t.at({1, 2, 3}), 99.0f);
  EXPECT_FLOAT_EQ(t.at({0, 1, 2}), -7.5f);
  EXPECT_FLOAT_EQ(t.at({0, 0, 1}), 0.0f);
}

TEST(TensorTest, SetAndGet1D) {
  Tensor t({5});
  for (int64_t i = 0; i < 5; ++i) {
    t.set({i}, static_cast<float>(i * 10));
  }
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(t.at({i}), static_cast<float>(i * 10));
  }
}

TEST(TensorTest, OverwriteValue) {
  Tensor t({2, 2});
  t.set({0, 0}, 1.0f);
  EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
  t.set({0, 0}, 999.0f);
  EXPECT_FLOAT_EQ(t.at({0, 0}), 999.0f);
}

TEST(TensorTest, NegativeAndFractionalValues) {
  Tensor t({3});
  t.set({0}, -100.5f);
  t.set({1}, 0.001f);
  t.set({2}, 1e10f);
  EXPECT_FLOAT_EQ(t.at({0}), -100.5f);
  EXPECT_FLOAT_EQ(t.at({1}), 0.001f);
  EXPECT_FLOAT_EQ(t.at({2}), 1e10f);
}

// ============================================================================
// flat_index correctness
// ============================================================================

TEST(TensorTest, FlatIndex2D) {
  Tensor t({2, 3});
  // Row-major: [0,0]=0, [0,1]=1, [0,2]=2, [1,0]=3, [1,1]=4, [1,2]=5
  EXPECT_EQ(t.flat_index({0, 0}), 0);
  EXPECT_EQ(t.flat_index({0, 2}), 2);
  EXPECT_EQ(t.flat_index({1, 0}), 3);
  EXPECT_EQ(t.flat_index({1, 2}), 5);
}

TEST(TensorTest, FlatIndex3D) {
  Tensor t({2, 3, 4});
  EXPECT_EQ(t.flat_index({0, 0, 0}), 0);
  EXPECT_EQ(t.flat_index({0, 0, 1}), 1);
  EXPECT_EQ(t.flat_index({0, 1, 0}), 4);
  EXPECT_EQ(t.flat_index({1, 0, 0}), 12);
  EXPECT_EQ(t.flat_index({1, 2, 3}), 23);  // Last element
}

// ============================================================================
// FP16 dtype — at() / set() convert through half-float
// ============================================================================

TEST(TensorTest, Float16SetAndGet) {
  Tensor t({4}, DType::kFloat16);
  t.set({0}, 1.0f);
  t.set({1}, -2.0f);
  t.set({2}, 0.5f);
  t.set({3}, 0.0f);

  EXPECT_FLOAT_EQ(t.at({0}), 1.0f);
  EXPECT_FLOAT_EQ(t.at({1}), -2.0f);
  EXPECT_FLOAT_EQ(t.at({2}), 0.5f);
  EXPECT_FLOAT_EQ(t.at({3}), 0.0f);
}

TEST(TensorTest, Float16LosesPrecision) {
  // FP16 has ~3 decimal digits of precision.
  // 1.001f should round — that's expected and correct.
  Tensor t({1}, DType::kFloat16);
  t.set({0}, 1.001f);
  float result = t.at({0});
  // Should be close but not exact
  EXPECT_NEAR(result, 1.001f, 0.002f);
}

TEST(TensorTest, Float16LargeValue) {
  // FP16 max is ~65504
  Tensor t({2}, DType::kFloat16);
  t.set({0}, 65504.0f);
  EXPECT_FLOAT_EQ(t.at({0}), 65504.0f);

  t.set({1}, 100.0f);
  EXPECT_FLOAT_EQ(t.at({1}), 100.0f);
}

// ============================================================================
// INT8 dtype — at() / set() cast through int8
// ============================================================================

TEST(TensorTest, Int8SetAndGet) {
  Tensor t({4}, DType::kInt8);
  t.set({0}, 42.0f);
  t.set({1}, -100.0f);
  t.set({2}, 0.0f);
  t.set({3}, 127.0f);

  EXPECT_FLOAT_EQ(t.at({0}), 42.0f);
  EXPECT_FLOAT_EQ(t.at({1}), -100.0f);
  EXPECT_FLOAT_EQ(t.at({2}), 0.0f);
  EXPECT_FLOAT_EQ(t.at({3}), 127.0f);
}

TEST(TensorTest, Int8Truncation) {
  // INT8 can only store integers in [-128, 127].
  // Fractional values get truncated (cast to int8_t).
  Tensor t({1}, DType::kInt8);
  t.set({0}, 3.7f);
  EXPECT_FLOAT_EQ(t.at({0}), 3.0f);  // Truncated, not rounded
}

TEST(TensorTest, Int8Range) {
  Tensor t({2}, DType::kInt8);
  t.set({0}, -128.0f);
  t.set({1}, 127.0f);
  EXPECT_FLOAT_EQ(t.at({0}), -128.0f);
  EXPECT_FLOAT_EQ(t.at({1}), 127.0f);
}

// ============================================================================
// data<T>() typed accessor
// ============================================================================

TEST(TensorTest, TypedDataAccessFloat32) {
  Tensor t({3});
  t.set({0}, 10.0f);
  t.set({1}, 20.0f);
  t.set({2}, 30.0f);

  const float* p = t.data<float>();
  EXPECT_FLOAT_EQ(p[0], 10.0f);
  EXPECT_FLOAT_EQ(p[1], 20.0f);
  EXPECT_FLOAT_EQ(p[2], 30.0f);
}

TEST(TensorTest, TypedDataMutate) {
  Tensor t({3});
  float* p = t.data<float>();
  p[0] = 100.0f;
  p[1] = 200.0f;
  p[2] = 300.0f;

  EXPECT_FLOAT_EQ(t.at({0}), 100.0f);
  EXPECT_FLOAT_EQ(t.at({1}), 200.0f);
  EXPECT_FLOAT_EQ(t.at({2}), 300.0f);
}

TEST(TensorTest, TypedDataInt8) {
  Tensor t({3}, DType::kInt8);
  int8_t* p = t.data<int8_t>();
  p[0] = 10;
  p[1] = -50;
  p[2] = 127;

  EXPECT_FLOAT_EQ(t.at({0}), 10.0f);
  EXPECT_FLOAT_EQ(t.at({1}), -50.0f);
  EXPECT_FLOAT_EQ(t.at({2}), 127.0f);
}

// ============================================================================
// to_string
// ============================================================================

TEST(TensorTest, ToString) {
  Tensor t({3, 4});
  std::string s = t.to_string();
  EXPECT_NE(s.find("3"), std::string::npos);
  EXPECT_NE(s.find("4"), std::string::npos);
  EXPECT_NE(s.find("float32"), std::string::npos);
}

TEST(TensorTest, ToStringInt8) {
  Tensor t({5}, DType::kInt8);
  std::string s = t.to_string();
  EXPECT_NE(s.find("int8"), std::string::npos);
  EXPECT_NE(s.find("5"), std::string::npos);
}

// ============================================================================
// Stress: fill an entire tensor and read back
// ============================================================================

TEST(TensorTest, FillAndReadback2D) {
  const int64_t rows = 8, cols = 8;
  Tensor t({rows, cols});

  // Write a unique value at every position
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      t.set({r, c}, static_cast<float>(r * 100 + c));
    }
  }

  // Read them all back
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      EXPECT_FLOAT_EQ(t.at({r, c}), static_cast<float>(r * 100 + c))
          << "Mismatch at [" << r << ", " << c << "]";
    }
  }
}

}  // namespace
}  // namespace ie
