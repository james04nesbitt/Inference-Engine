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
  EXPECT_EQ(t.numel(), 1); // Scalar = 1 element
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
  Tensor t({1024, 1152}); // Gemma-3 sized
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
  EXPECT_EQ(f16.nbytes(), 16 * 2); // 16 elements * 2 bytes

  Tensor i8({4, 4}, DType::kInt8);
  EXPECT_EQ(i8.dtype(), DType::kInt8);
  EXPECT_EQ(i8.nbytes(), 16 * 1); // 16 elements * 1 byte
}

// ============================================================================
// Memory allocation & alignment
// ============================================================================

TEST(TensorTest, DataIsAllocated) {
  Tensor t({2, 3});
  EXPECT_NE(t.data_ptr(), nullptr);
  EXPECT_NE(t.data<float>(), nullptr);
  EXPECT_EQ(t.nbytes(), 24u);
}

TEST(TensorTest, ZeroInitialized) {
  Tensor t({4, 4});
  const float *p = t.data<float>();
  for (int64_t i = 0; i < t.numel(); ++i) {
    EXPECT_FLOAT_EQ(p[i], 0.0f) << "Element " << i << " not zero";
  }
}

TEST(TensorTest, AlignedAllocation) {
  // Constructor should produce 64-byte aligned data for SIMD
  Tensor t({64});
  uintptr_t addr = reinterpret_cast<uintptr_t>(t.data<float>());
  EXPECT_EQ(addr % 64, 0u) << "data<float>() not 64-byte aligned";
}

TEST(TensorTest, AlignedAllocationAllDtypes) {
  Tensor f32({256});
  Tensor f16({256}, DType::kFloat16);
  Tensor i8({256}, DType::kInt8);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(f32.data<float>()) % 64, 0u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(f16.data<uint16_t>()) % 64, 0u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(i8.data<int8_t>()) % 64, 0u);
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
  EXPECT_FLOAT_EQ(t.at({0, 1}), 0.0f); // Untouched = zero
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
  EXPECT_EQ(t.flat_index({1, 2, 3}), 23); // Last element
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
  EXPECT_FLOAT_EQ(t.at({0}), 3.0f); // Truncated, not rounded
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

  const float *p = t.data<float>();
  EXPECT_FLOAT_EQ(p[0], 10.0f);
  EXPECT_FLOAT_EQ(p[1], 20.0f);
  EXPECT_FLOAT_EQ(p[2], 30.0f);
}

TEST(TensorTest, TypedDataMutate) {
  Tensor t({3});
  float *p = t.data<float>();
  p[0] = 100.0f;
  p[1] = 200.0f;
  p[2] = 300.0f;

  EXPECT_FLOAT_EQ(t.at({0}), 100.0f);
  EXPECT_FLOAT_EQ(t.at({1}), 200.0f);
  EXPECT_FLOAT_EQ(t.at({2}), 300.0f);
}

TEST(TensorTest, TypedDataInt8) {
  Tensor t({3}, DType::kInt8);
  int8_t *p = t.data<int8_t>();
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

// ============================================================================
// size(dim) — PyTorch-style accessor
// ============================================================================

TEST(TensorTest, SizePositiveDim) {
  Tensor t({2, 3, 4});
  EXPECT_EQ(t.size(0), 2);
  EXPECT_EQ(t.size(1), 3);
  EXPECT_EQ(t.size(2), 4);
}

TEST(TensorTest, SizeNegativeDim) {
  Tensor t({2, 3, 4});
  EXPECT_EQ(t.size(-1), 4);
  EXPECT_EQ(t.size(-2), 3);
  EXPECT_EQ(t.size(-3), 2);
}

TEST(TensorTest, SizeOutOfRange) {
  Tensor t({2, 3});
  EXPECT_THROW(t.size(2), std::runtime_error);
  EXPECT_THROW(t.size(-3), std::runtime_error);
}

// ============================================================================
// shape_equals
// ============================================================================

TEST(TensorTest, ShapeEqualsVector) {
  Tensor t({2, 3, 4});
  EXPECT_TRUE(t.shape_equals({2, 3, 4}));
  EXPECT_FALSE(t.shape_equals({2, 3}));
  EXPECT_FALSE(t.shape_equals({2, 3, 5}));
}

TEST(TensorTest, ShapeEqualsTensor) {
  Tensor a({2, 3, 4});
  Tensor b({2, 3, 4});
  Tensor c({2, 3, 5});
  EXPECT_TRUE(a.shape_equals(b));
  EXPECT_FALSE(a.shape_equals(c));
}

// ============================================================================
// is_contiguous
// ============================================================================

TEST(TensorTest, FreshTensorIsContiguous) {
  Tensor t({3, 4, 5});
  EXPECT_TRUE(t.is_contiguous());
}

TEST(TensorTest, TransposeIsNotContiguous) {
  Tensor t({3, 4});
  t.set({0, 0}, 1.0f);
  t.set({0, 1}, 2.0f);
  Tensor tt = t.transpose(0, 1);
  EXPECT_FALSE(tt.is_contiguous());
}

// ============================================================================
// view / reshape
// ============================================================================

TEST(TensorTest, ViewFlattens) {
  Tensor t({2, 3});
  for (int64_t i = 0; i < 6; ++i) {
    t.data<float>()[i] = static_cast<float>(i);
  }
  auto v = t.view({6});
  ASSERT_TRUE(v.has_value());
  EXPECT_TRUE(v->shape_equals({6}));
  EXPECT_FLOAT_EQ(v->at({0}), 0.0f);
  EXPECT_FLOAT_EQ(v->at({5}), 5.0f);
}

TEST(TensorTest, ViewIncompatibleFails) {
  Tensor t({2, 3});
  auto v = t.view({7}); // 7 != 6
  EXPECT_FALSE(v.has_value());
}

TEST(TensorTest, ReshapeContiguous) {
  Tensor t({2, 3});
  for (int64_t i = 0; i < 6; ++i) {
    t.data<float>()[i] = static_cast<float>(i);
  }
  Tensor r = t.reshape({3, 2});
  EXPECT_TRUE(r.shape_equals({3, 2}));
  EXPECT_FLOAT_EQ(r.at({0, 0}), 0.0f);
  EXPECT_FLOAT_EQ(r.at({2, 1}), 5.0f);
}

TEST(TensorTest, ReshapeNonContiguous) {
  // transpose then reshape forces a copy
  Tensor t({2, 3});
  for (int64_t i = 0; i < 6; ++i) {
    t.data<float>()[i] = static_cast<float>(i);
  }
  Tensor tt = t.transpose(0, 1); // [3, 2], non-contiguous
  ASSERT_FALSE(tt.is_contiguous());
  Tensor r = tt.reshape({6}); // forces copy
  EXPECT_TRUE(r.shape_equals({6}));
  EXPECT_TRUE(r.is_contiguous());
  // Transposed data: row 0 of tt = col 0 of t = {0, 3}
  EXPECT_FLOAT_EQ(r.at({0}), 0.0f);
  EXPECT_FLOAT_EQ(r.at({1}), 3.0f);
}

// ============================================================================
// transpose / permute
// ============================================================================

TEST(TensorTest, TransposeSharesData) {
  Tensor t({2, 3});
  t.set({0, 1}, 42.0f);
  Tensor tt = t.transpose(0, 1);
  EXPECT_TRUE(tt.shape_equals({3, 2}));
  // t[0,1] == tt[1,0] because they share data
  EXPECT_FLOAT_EQ(tt.at({1, 0}), 42.0f);
}

TEST(TensorTest, PermuteReorders) {
  Tensor t({2, 3, 4});
  t.set({1, 2, 3}, 99.0f);
  Tensor p = t.permute({2, 0, 1});
  EXPECT_TRUE(p.shape_equals({4, 2, 3}));
  EXPECT_FLOAT_EQ(p.at({3, 1, 2}), 99.0f);
}

// ============================================================================
// slice — zero-copy sub-tensor view
// ============================================================================

TEST(TensorTest, SliceBasic) {
  Tensor t({4, 3});
  for (int64_t r = 0; r < 4; ++r) {
    for (int64_t c = 0; c < 3; ++c) {
      t.set({r, c}, static_cast<float>(r * 10 + c));
    }
  }
  Tensor s = t.slice(0, 1, 3); // rows 1..2
  EXPECT_TRUE(s.shape_equals({2, 3}));
  EXPECT_FLOAT_EQ(s.at({0, 0}), 10.0f);
  EXPECT_FLOAT_EQ(s.at({0, 2}), 12.0f);
  EXPECT_FLOAT_EQ(s.at({1, 0}), 20.0f);
  EXPECT_FLOAT_EQ(s.at({1, 2}), 22.0f);
}

TEST(TensorTest, SliceSharesData) {
  Tensor t({4});
  t.set({0}, 10.0f);
  t.set({1}, 20.0f);
  t.set({2}, 30.0f);
  t.set({3}, 40.0f);

  Tensor s = t.slice(0, 1, 3); // [20, 30]
  EXPECT_TRUE(s.shape_equals({2}));
  EXPECT_FLOAT_EQ(s.at({0}), 20.0f);
  EXPECT_FLOAT_EQ(s.at({1}), 30.0f);

  // Mutate through slice, verify original is affected
  s.set({0}, 999.0f);
  EXPECT_FLOAT_EQ(t.at({1}), 999.0f);
}

TEST(TensorTest, SliceDim1) {
  Tensor t({3, 4});
  for (int64_t r = 0; r < 3; ++r) {
    for (int64_t c = 0; c < 4; ++c) {
      t.set({r, c}, static_cast<float>(r * 10 + c));
    }
  }
  Tensor s = t.slice(1, 1, 3); // cols 1..2
  EXPECT_TRUE(s.shape_equals({3, 2}));
  EXPECT_FLOAT_EQ(s.at({0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(s.at({0, 1}), 2.0f);
  EXPECT_FLOAT_EQ(s.at({2, 0}), 21.0f);
}

TEST(TensorTest, SliceInvalidThrows) {
  Tensor t({4, 3});
  EXPECT_THROW(t.slice(0, -1, 2), std::runtime_error);
  EXPECT_THROW(t.slice(0, 3, 2), std::runtime_error); // start > end
  EXPECT_THROW(t.slice(0, 0, 5), std::runtime_error); // end > shape
  EXPECT_THROW(t.slice(2, 0, 1), std::runtime_error); // dim out of range
}

// ============================================================================
// fill — scalar fill respecting dtype
// ============================================================================

TEST(TensorTest, FillFloat32) {
  Tensor t({3, 4});
  t.fill(7.5f);
  for (int64_t i = 0; i < t.numel(); ++i) {
    EXPECT_FLOAT_EQ(t.data<float>()[i], 7.5f);
  }
}

TEST(TensorTest, FillFloat16) {
  Tensor t({4}, DType::kFloat16);
  t.fill(2.0f);
  for (int64_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(t.at({i}), 2.0f);
  }
}

TEST(TensorTest, FillInt8) {
  Tensor t({4}, DType::kInt8);
  t.fill(42.0f);
  for (int64_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(t.at({i}), 42.0f);
  }
}

TEST(TensorTest, FillNegInfinity) {
  // FlashAttention uses fill(-INFINITY) for m accumulators
  Tensor t({4});
  t.fill(-std::numeric_limits<float>::infinity());
  for (int64_t i = 0; i < 4; ++i) {
    EXPECT_TRUE(std::isinf(t.data<float>()[i]));
    EXPECT_LT(t.data<float>()[i], 0.0f);
  }
}

TEST(TensorTest, FillNonContiguous) {
  // Transpose creates a non-contiguous view, fill must use strides
  Tensor t({3, 4});
  Tensor tt = t.transpose(0, 1); // [4, 3], non-contiguous
  ASSERT_FALSE(tt.is_contiguous());
  tt.fill(5.0f);
  for (int64_t r = 0; r < 4; ++r) {
    for (int64_t c = 0; c < 3; ++c) {
      EXPECT_FLOAT_EQ(tt.at({r, c}), 5.0f);
    }
  }
}

// ============================================================================
// contiguous — make a C-order copy
// ============================================================================

TEST(TensorTest, ContiguousNoOp) {
  Tensor t({3, 4});
  t.fill(1.0f);
  Tensor c = t.contiguous();
  EXPECT_TRUE(c.is_contiguous());
  // Should share data (no copy needed)
  EXPECT_EQ(c.data<float>(), t.data<float>());
}

TEST(TensorTest, ContiguousAfterTranspose) {
  Tensor t({2, 3});
  for (int64_t i = 0; i < 6; ++i) {
    t.data<float>()[i] = static_cast<float>(i);
  }
  Tensor tt = t.transpose(0, 1); // [3, 2], non-contiguous
  ASSERT_FALSE(tt.is_contiguous());

  Tensor c = tt.contiguous();
  EXPECT_TRUE(c.is_contiguous());
  EXPECT_TRUE(c.shape_equals({3, 2}));
  // Verify data is correct: tt[r,c] = t[c,r]
  EXPECT_FLOAT_EQ(c.at({0, 0}), 0.0f); // t[0,0]
  EXPECT_FLOAT_EQ(c.at({0, 1}), 3.0f); // t[1,0]
  EXPECT_FLOAT_EQ(c.at({1, 0}), 1.0f); // t[0,1]
  EXPECT_FLOAT_EQ(c.at({2, 1}), 5.0f); // t[1,2]

  // Should NOT share data
  EXPECT_NE(c.data<float>(), tt.data<float>());
}

// ============================================================================
// Static Factories — from_buffer, from_vector, full, zeros, ones
// ============================================================================

TEST(TensorTest, FromBuffer) {
  // Simulate GGUF: allocate raw buffer, wrap as tensor
  const int64_t n = 4;
  auto buf = std::shared_ptr<uint8_t[]>(new uint8_t[n * sizeof(float)]);
  float *p = reinterpret_cast<float *>(buf.get());
  p[0] = 1.0f;
  p[1] = 2.0f;
  p[2] = 3.0f;
  p[3] = 4.0f;

  Tensor t = Tensor::from_buffer(buf, {n}, DType::kFloat32);
  EXPECT_TRUE(t.shape_equals({4}));
  EXPECT_FLOAT_EQ(t.at({0}), 1.0f);
  EXPECT_FLOAT_EQ(t.at({3}), 4.0f);
  // Should share the buffer (zero-copy)
  EXPECT_EQ(t.data<float>(), p);
}

TEST(TensorTest, FromBufferF16) {
  auto buf = std::shared_ptr<uint8_t[]>(new uint8_t[4 * sizeof(uint16_t)]);
  Tensor t = Tensor::from_buffer(buf, {2, 2}, DType::kFloat16);
  EXPECT_TRUE(t.shape_equals({2, 2}));
  EXPECT_EQ(t.dtype(), DType::kFloat16);
  EXPECT_EQ(t.nbytes(), 4 * 2);
}

TEST(TensorTest, FromVector) {
  std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  Tensor t = Tensor::from_vector(v);
  EXPECT_TRUE(t.shape_equals({5}));
  EXPECT_EQ(t.dtype(), DType::kFloat32);
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(t.at({i}), v[i]);
  }
}

TEST(TensorTest, FromVectorEmpty) {
  std::vector<float> v = {};
  Tensor t = Tensor::from_vector(v);
  EXPECT_TRUE(t.shape_equals({0}));
  EXPECT_EQ(t.numel(), 0);
}

TEST(TensorTest, Full) {
  Tensor t = Tensor::full({3, 4}, 7.0f);
  EXPECT_TRUE(t.shape_equals({3, 4}));
  for (int64_t i = 0; i < t.numel(); ++i) {
    EXPECT_FLOAT_EQ(t.data<float>()[i], 7.0f);
  }
}

TEST(TensorTest, FullNegInf) {
  Tensor t = Tensor::full({8}, -std::numeric_limits<float>::infinity());
  for (int64_t i = 0; i < 8; ++i) {
    EXPECT_TRUE(std::isinf(t.data<float>()[i]));
  }
}

TEST(TensorTest, Zeros) {
  Tensor t = Tensor::zeros({2, 3});
  EXPECT_TRUE(t.shape_equals({2, 3}));
  for (int64_t i = 0; i < t.numel(); ++i) {
    EXPECT_FLOAT_EQ(t.data<float>()[i], 0.0f);
  }
}

TEST(TensorTest, Ones) {
  Tensor t = Tensor::ones({2, 3});
  EXPECT_TRUE(t.shape_equals({2, 3}));
  for (int64_t i = 0; i < t.numel(); ++i) {
    EXPECT_FLOAT_EQ(t.data<float>()[i], 1.0f);
  }
}

TEST(TensorTest, OnesF16) {
  Tensor t = Tensor::ones({4}, DType::kFloat16);
  EXPECT_EQ(t.dtype(), DType::kFloat16);
  for (int64_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(t.at({i}), 1.0f);
  }
}

// ============================================================================
// cat — concatenation along a dimension
// ============================================================================

TEST(TensorTest, CatDim0) {
  Tensor a({2, 3});
  Tensor b({3, 3});
  a.fill(1.0f);
  b.fill(2.0f);

  Tensor c = Tensor::cat({a, b}, 0);
  EXPECT_TRUE(c.shape_equals({5, 3}));
  EXPECT_FLOAT_EQ(c.at({0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(c.at({1, 2}), 1.0f);
  EXPECT_FLOAT_EQ(c.at({2, 0}), 2.0f);
  EXPECT_FLOAT_EQ(c.at({4, 2}), 2.0f);
}

TEST(TensorTest, CatDim1) {
  Tensor a({2, 3});
  Tensor b({2, 4});
  a.fill(10.0f);
  b.fill(20.0f);

  Tensor c = Tensor::cat({a, b}, 1);
  EXPECT_TRUE(c.shape_equals({2, 7}));
  EXPECT_FLOAT_EQ(c.at({0, 0}), 10.0f);
  EXPECT_FLOAT_EQ(c.at({0, 2}), 10.0f);
  EXPECT_FLOAT_EQ(c.at({0, 3}), 20.0f);
  EXPECT_FLOAT_EQ(c.at({1, 6}), 20.0f);
}

TEST(TensorTest, CatThreeTensors) {
  Tensor a({1, 4});
  Tensor b({1, 4});
  Tensor c({1, 4});
  a.fill(1.0f);
  b.fill(2.0f);
  c.fill(3.0f);

  Tensor result = Tensor::cat({a, b, c}, 0);
  EXPECT_TRUE(result.shape_equals({3, 4}));
  EXPECT_FLOAT_EQ(result.at({0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(result.at({1, 0}), 2.0f);
  EXPECT_FLOAT_EQ(result.at({2, 0}), 3.0f);
}

TEST(TensorTest, CatEmptyThrows) {
  EXPECT_THROW(Tensor::cat({}, 0), std::runtime_error);
}

TEST(TensorTest, CatMismatchedShapeThrows) {
  Tensor a({2, 3});
  Tensor b({2, 4});
  // dim=0 requires all dims except 0 to match — but dim 1 differs (3 vs 4)
  EXPECT_THROW(Tensor::cat({a, b}, 0), std::runtime_error);
}

// ============================================================================
// to — dtype conversion
// ============================================================================

TEST(TensorTest, ToSameDtypeIsView) {
  Tensor t({4});
  t.fill(5.0f);
  Tensor t2 = t.to(DType::kFloat32);
  EXPECT_EQ(t2.dtype(), DType::kFloat32);
  // Should share data
  EXPECT_EQ(t2.data<float>(), t.data<float>());
}

TEST(TensorTest, ToF32ToF16) {
  Tensor t({4});
  t.set({0}, 1.0f);
  t.set({1}, 2.0f);
  t.set({2}, -3.0f);
  t.set({3}, 0.5f);

  Tensor h = t.to(DType::kFloat16);
  EXPECT_EQ(h.dtype(), DType::kFloat16);
  EXPECT_TRUE(h.shape_equals({4}));
  EXPECT_FLOAT_EQ(h.at({0}), 1.0f);
  EXPECT_FLOAT_EQ(h.at({1}), 2.0f);
  EXPECT_FLOAT_EQ(h.at({2}), -3.0f);
  EXPECT_FLOAT_EQ(h.at({3}), 0.5f);
}

TEST(TensorTest, ToF16ToF32) {
  Tensor h({3}, DType::kFloat16);
  h.set({0}, 10.0f);
  h.set({1}, -5.0f);
  h.set({2}, 0.0f);

  Tensor f = h.to(DType::kFloat32);
  EXPECT_EQ(f.dtype(), DType::kFloat32);
  EXPECT_FLOAT_EQ(f.at({0}), 10.0f);
  EXPECT_FLOAT_EQ(f.at({1}), -5.0f);
  EXPECT_FLOAT_EQ(f.at({2}), 0.0f);
}

TEST(TensorTest, ToF32ToInt8) {
  Tensor t({3});
  t.set({0}, 42.0f);
  t.set({1}, -100.0f);
  t.set({2}, 3.7f);

  Tensor i = t.to(DType::kInt8);
  EXPECT_EQ(i.dtype(), DType::kInt8);
  EXPECT_FLOAT_EQ(i.at({0}), 42.0f);
  EXPECT_FLOAT_EQ(i.at({1}), -100.0f);
  EXPECT_FLOAT_EQ(i.at({2}), 3.0f); // truncated
}

// ============================================================================
// clone — deep copy
// ============================================================================

TEST(TensorTest, CloneIsIndependent) {
  Tensor t({3});
  t.set({0}, 1.0f);
  t.set({1}, 2.0f);
  t.set({2}, 3.0f);

  Tensor c = t.clone();
  EXPECT_TRUE(c.shape_equals({3}));
  EXPECT_FLOAT_EQ(c.at({0}), 1.0f);
  EXPECT_FLOAT_EQ(c.at({2}), 3.0f);

  // Mutate clone, original unaffected
  c.set({0}, 999.0f);
  EXPECT_FLOAT_EQ(t.at({0}), 1.0f);
  EXPECT_FLOAT_EQ(c.at({0}), 999.0f);
}

TEST(TensorTest, CloneNonContiguous) {
  Tensor t({2, 3});
  for (int64_t r = 0; r < 2; ++r)
    for (int64_t c = 0; c < 3; ++c)
      t.set({r, c}, static_cast<float>(r * 10 + c));

  Tensor tt = t.transpose(0, 1); // non-contiguous
  Tensor c = tt.clone();
  EXPECT_TRUE(c.is_contiguous());
  EXPECT_TRUE(c.shape_equals({3, 2}));
  EXPECT_FLOAT_EQ(c.at({0, 0}), 0.0f);
  EXPECT_FLOAT_EQ(c.at({0, 1}), 10.0f);
  EXPECT_FLOAT_EQ(c.at({2, 1}), 12.0f);
}

TEST(TensorTest, CloneSliced) {
  Tensor t({4});
  for (int64_t i = 0; i < 4; ++i)
    t.set({i}, static_cast<float>(i * 10));

  Tensor s = t.slice(0, 1, 3); // [10, 20]
  Tensor c = s.clone();
  EXPECT_TRUE(c.shape_equals({2}));
  EXPECT_FLOAT_EQ(c.at({0}), 10.0f);
  EXPECT_FLOAT_EQ(c.at({1}), 20.0f);

  // Independent from original
  c.set({0}, 999.0f);
  EXPECT_FLOAT_EQ(t.at({1}), 10.0f);
}

// ============================================================================
// select — index a dimension, reducing ndim by 1
// ============================================================================

TEST(TensorTest, SelectDim0) {
  // Simulate embedding lookup: [vocab=3, dim=4]
  Tensor table({3, 4});
  for (int64_t v = 0; v < 3; ++v)
    for (int64_t d = 0; d < 4; ++d)
      table.set({v, d}, static_cast<float>(v * 100 + d));

  Tensor row1 = table.select(0, 1);
  EXPECT_TRUE(row1.shape_equals({4}));
  EXPECT_FLOAT_EQ(row1.at({0}), 100.0f);
  EXPECT_FLOAT_EQ(row1.at({3}), 103.0f);
}

TEST(TensorTest, SelectDim1) {
  Tensor t({3, 4});
  for (int64_t r = 0; r < 3; ++r)
    for (int64_t c = 0; c < 4; ++c)
      t.set({r, c}, static_cast<float>(r * 10 + c));

  Tensor col2 = t.select(1, 2);
  EXPECT_TRUE(col2.shape_equals({3}));
  EXPECT_FLOAT_EQ(col2.at({0}), 2.0f);
  EXPECT_FLOAT_EQ(col2.at({1}), 12.0f);
  EXPECT_FLOAT_EQ(col2.at({2}), 22.0f);
}

TEST(TensorTest, SelectSharesData) {
  Tensor t({3, 4});
  t.set({1, 2}, 42.0f);
  Tensor s = t.select(0, 1);
  EXPECT_FLOAT_EQ(s.at({2}), 42.0f);

  // Mutate via select, original changes
  s.set({2}, 999.0f);
  EXPECT_FLOAT_EQ(t.at({1, 2}), 999.0f);
}

TEST(TensorTest, SelectNegativeDim) {
  Tensor t({2, 3, 4});
  Tensor s = t.select(-1, 2); // select last dim, index 2
  EXPECT_TRUE(s.shape_equals({2, 3}));
}

TEST(TensorTest, SelectOutOfRange) {
  Tensor t({3, 4});
  EXPECT_THROW(t.select(0, 3), std::runtime_error);
  EXPECT_THROW(t.select(0, -1), std::runtime_error);
  EXPECT_THROW(t.select(2, 0), std::runtime_error);
}

// ============================================================================
// unsqueeze — insert a size-1 dimension
// ============================================================================

TEST(TensorTest, UnsqueezeDim0) {
  Tensor t({3, 4});
  Tensor u = t.unsqueeze(0);
  EXPECT_TRUE(u.shape_equals({1, 3, 4}));
  EXPECT_EQ(u.ndim(), 3);
}

TEST(TensorTest, UnsqueezeDim1) {
  Tensor t({3, 4});
  Tensor u = t.unsqueeze(1);
  EXPECT_TRUE(u.shape_equals({3, 1, 4}));
}

TEST(TensorTest, UnsqueezeEnd) {
  Tensor t({3, 4});
  Tensor u = t.unsqueeze(2);
  EXPECT_TRUE(u.shape_equals({3, 4, 1}));
}

TEST(TensorTest, UnsqueezeNegative) {
  Tensor t({3, 4});
  Tensor u = t.unsqueeze(-1); // same as unsqueeze(2)
  EXPECT_TRUE(u.shape_equals({3, 4, 1}));
}

TEST(TensorTest, UnsqueezeSharesData) {
  Tensor t({3});
  t.set({0}, 42.0f);
  Tensor u = t.unsqueeze(0);
  EXPECT_FLOAT_EQ(u.at({0, 0}), 42.0f);
  u.set({0, 0}, 999.0f);
  EXPECT_FLOAT_EQ(t.at({0}), 999.0f);
}

TEST(TensorTest, UnsqueezeOutOfRange) {
  Tensor t({3, 4});
  EXPECT_THROW(t.unsqueeze(3), std::runtime_error); // max valid is 2
  EXPECT_THROW(t.unsqueeze(-4), std::runtime_error);
}

// ============================================================================
// squeeze — remove a size-1 dimension
// ============================================================================

TEST(TensorTest, SqueezeDim0) {
  Tensor t({1, 3, 4});
  Tensor s = t.squeeze(0);
  EXPECT_TRUE(s.shape_equals({3, 4}));
}

TEST(TensorTest, SqueezeDim1) {
  Tensor t({3, 1, 4});
  Tensor s = t.squeeze(1);
  EXPECT_TRUE(s.shape_equals({3, 4}));
}

TEST(TensorTest, SqueezeSharesData) {
  Tensor t({1, 4});
  t.set({0, 2}, 42.0f);
  Tensor s = t.squeeze(0);
  EXPECT_FLOAT_EQ(s.at({2}), 42.0f);
  s.set({2}, 999.0f);
  EXPECT_FLOAT_EQ(t.at({0, 2}), 999.0f);
}

TEST(TensorTest, SqueezeNonOneThrows) {
  Tensor t({3, 4});
  EXPECT_THROW(t.squeeze(0), std::runtime_error); // shape[0] = 3 ≠ 1
}

TEST(TensorTest, UnsqueezeSqueezeRoundTrip) {
  Tensor t({3, 4});
  t.fill(5.0f);
  Tensor u = t.unsqueeze(1);
  EXPECT_TRUE(u.shape_equals({3, 1, 4}));
  Tensor s = u.squeeze(1);
  EXPECT_TRUE(s.shape_equals({3, 4}));
  EXPECT_FLOAT_EQ(s.at({0, 0}), 5.0f);
  // Data pointer should be same through the round trip
  EXPECT_EQ(s.data<float>(), t.data<float>());
}

// ============================================================================
// repeat — tile data along a dimension
// ============================================================================

TEST(TensorTest, RepeatDim0) {
  Tensor t({2, 3});
  for (int64_t r = 0; r < 2; ++r)
    for (int64_t c = 0; c < 3; ++c)
      t.set({r, c}, static_cast<float>(r * 10 + c));

  Tensor r = t.repeat(0, 3); // [6, 3]
  EXPECT_TRUE(r.shape_equals({6, 3}));
  // First copy
  EXPECT_FLOAT_EQ(r.at({0, 0}), 0.0f);
  EXPECT_FLOAT_EQ(r.at({1, 2}), 12.0f);
  // Second copy
  EXPECT_FLOAT_EQ(r.at({2, 0}), 0.0f);
  EXPECT_FLOAT_EQ(r.at({3, 2}), 12.0f);
  // Third copy
  EXPECT_FLOAT_EQ(r.at({4, 0}), 0.0f);
  EXPECT_FLOAT_EQ(r.at({5, 2}), 12.0f);
}

TEST(TensorTest, RepeatDim1) {
  // GQA: K is [seq, 1, head_dim], repeat dim=1 by num_heads
  Tensor t({2, 1, 4});
  t.fill(3.0f);
  Tensor r = t.repeat(1, 4); // [2, 4, 4]
  EXPECT_TRUE(r.shape_equals({2, 4, 4}));
  // All values should be 3.0
  for (int64_t a = 0; a < 2; ++a)
    for (int64_t b = 0; b < 4; ++b)
      for (int64_t c = 0; c < 4; ++c)
        EXPECT_FLOAT_EQ(r.at({a, b, c}), 3.0f);
}

TEST(TensorTest, RepeatOnce) {
  Tensor t({3});
  t.set({0}, 1.0f);
  t.set({1}, 2.0f);
  t.set({2}, 3.0f);
  Tensor r = t.repeat(0, 1); // no-op repeat
  EXPECT_TRUE(r.shape_equals({3}));
  EXPECT_FLOAT_EQ(r.at({0}), 1.0f);
  EXPECT_FLOAT_EQ(r.at({2}), 3.0f);
}

TEST(TensorTest, RepeatInvalidThrows) {
  Tensor t({3, 4});
  EXPECT_THROW(t.repeat(0, 0), std::runtime_error); // times < 1
  EXPECT_THROW(t.repeat(2, 2), std::runtime_error); // dim out of range
}

// ============================================================================
// Quantization metadata
// ============================================================================

TEST(TensorTest, NotQuantizedByDefault) {
  Tensor t({4, 8}, DType::kInt8);
  EXPECT_FALSE(t.is_quantized());
  EXPECT_TRUE(t.scales().empty());
  EXPECT_TRUE(t.zero_points().empty());
}

TEST(TensorTest, SetQuantizationParams) {
  Tensor t({4, 8}, DType::kInt8);
  std::vector<float> scales = {0.1f, 0.2f, 0.3f, 0.4f};
  std::vector<int32_t> zps = {0, 1, 0, -1};

  t.set_quantization_params(scales, zps);
  EXPECT_TRUE(t.is_quantized());
  EXPECT_EQ(t.scales().size(), 4u);
  EXPECT_EQ(t.zero_points().size(), 4u);
  EXPECT_FLOAT_EQ(t.scales()[0], 0.1f);
  EXPECT_EQ(t.zero_points()[3], -1);
}

TEST(TensorTest, QuantParamsMismatchThrows) {
  Tensor t({4, 8}, DType::kInt8);
  // scales.size() != zero_points.size()
  EXPECT_THROW(t.set_quantization_params({0.1f, 0.2f}, {0, 1, 2}),
               std::runtime_error);
}

TEST(TensorTest, QuantParamsWrongSizeThrows) {
  Tensor t({4, 8}, DType::kInt8);
  // scales.size() should equal shape[0] = 4
  EXPECT_THROW(t.set_quantization_params({0.1f, 0.2f}, {0, 1}),
               std::runtime_error);
}

// ============================================================================
// Offset propagation — views preserve offset correctly
// ============================================================================

TEST(TensorTest, SliceThenSelect) {
  // Compound operations: slice rows, then select a row
  Tensor t({4, 3});
  for (int64_t r = 0; r < 4; ++r)
    for (int64_t c = 0; c < 3; ++c)
      t.set({r, c}, static_cast<float>(r * 10 + c));

  Tensor s = t.slice(0, 1, 3); // rows 1-2 → [2, 3]
  Tensor row = s.select(0, 1); // row 1 of slice = row 2 of original → [3]
  EXPECT_TRUE(row.shape_equals({3}));
  EXPECT_FLOAT_EQ(row.at({0}), 20.0f);
  EXPECT_FLOAT_EQ(row.at({1}), 21.0f);
  EXPECT_FLOAT_EQ(row.at({2}), 22.0f);
}

TEST(TensorTest, SliceThenSlice) {
  Tensor t({10});
  for (int64_t i = 0; i < 10; ++i)
    t.set({i}, static_cast<float>(i));

  Tensor s1 = t.slice(0, 2, 8);  // [2,3,4,5,6,7]
  Tensor s2 = s1.slice(0, 1, 4); // [3,4,5]
  EXPECT_TRUE(s2.shape_equals({3}));
  EXPECT_FLOAT_EQ(s2.at({0}), 3.0f);
  EXPECT_FLOAT_EQ(s2.at({2}), 5.0f);
}

TEST(TensorTest, SelectThenIndex) {
  Tensor t({2, 3, 4});
  t.set({1, 2, 3}, 42.0f);
  Tensor s = t.select(0, 1); // [3, 4]
  EXPECT_FLOAT_EQ(s.at({2, 3}), 42.0f);
}

// ============================================================================
// Comprehensive: compose multiple ops
// ============================================================================

TEST(TensorTest, TransposeContiguousClone) {
  Tensor t({2, 3});
  for (int64_t i = 0; i < 6; ++i)
    t.data<float>()[i] = static_cast<float>(i);

  Tensor tt = t.transpose(0, 1); // non-contiguous view
  Tensor c = tt.contiguous();    // copy to contiguous
  Tensor cl = c.clone();         // deep copy

  // All should have same values
  EXPECT_FLOAT_EQ(cl.at({0, 0}), 0.0f); // t[0,0]
  EXPECT_FLOAT_EQ(cl.at({1, 0}), 1.0f); // t[0,1]
  EXPECT_FLOAT_EQ(cl.at({2, 0}), 2.0f); // t[0,2]

  // Clone should be independent
  cl.set({0, 0}, 999.0f);
  EXPECT_FLOAT_EQ(c.at({0, 0}), 0.0f);
}

TEST(TensorTest, FromVectorSliceSelectRoundTrip) {
  // Create embedding table from vectors
  Tensor t({3, 4});
  for (int64_t r = 0; r < 3; ++r)
    for (int64_t c = 0; c < 4; ++c)
      t.set({r, c}, static_cast<float>(r * 100 + c));

  // Select token 1's embedding
  Tensor embed = t.select(0, 1);
  EXPECT_TRUE(embed.shape_equals({4}));
  EXPECT_FLOAT_EQ(embed.at({0}), 100.0f);
  EXPECT_FLOAT_EQ(embed.at({3}), 103.0f);

  // Unsqueeze for batch dim
  Tensor batched = embed.unsqueeze(0);
  EXPECT_TRUE(batched.shape_equals({1, 4}));
  EXPECT_FLOAT_EQ(batched.at({0, 2}), 102.0f);

  // Squeeze back
  Tensor unbatched = batched.squeeze(0);
  EXPECT_TRUE(unbatched.shape_equals({4}));
}

TEST(TensorTest, CatSliceRoundTrip) {
  // Simulate KV cache: append then slice back
  Tensor cached_k = Tensor::full({3, 4}, 1.0f);
  Tensor new_k = Tensor::full({1, 4}, 2.0f);

  Tensor combined = Tensor::cat({cached_k, new_k}, 0); // [4, 4]
  EXPECT_TRUE(combined.shape_equals({4, 4}));

  // Slice out the original cached portion
  Tensor original = combined.slice(0, 0, 3);
  EXPECT_TRUE(original.shape_equals({3, 4}));
  EXPECT_FLOAT_EQ(original.at({0, 0}), 1.0f);

  // Slice out the new portion
  Tensor appended = combined.slice(0, 3, 4);
  EXPECT_TRUE(appended.shape_equals({1, 4}));
  EXPECT_FLOAT_EQ(appended.at({0, 0}), 2.0f);
}

} // namespace
} // namespace ie
