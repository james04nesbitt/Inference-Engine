#include "engine/tensor/tensor.h"

#include <gtest/gtest.h>

namespace ie {
namespace {

TEST(TensorTest, DefaultConstruction) {
  Tensor t;
  EXPECT_EQ(t.ndim(), 0);
  EXPECT_EQ(t.numel(), 1);
  EXPECT_EQ(t.dtype(), DType::kFloat32);
}

TEST(TensorTest, ShapedConstruction) {
  Tensor t({3, 4, 5});
  EXPECT_EQ(t.ndim(), 3);
  EXPECT_EQ(t.shape()[0], 3);
  EXPECT_EQ(t.shape()[1], 4);
  EXPECT_EQ(t.shape()[2], 5);
  EXPECT_EQ(t.numel(), 60);
  EXPECT_EQ(t.nbytes(), 60 * sizeof(float));
  EXPECT_EQ(t.dtype(), DType::kFloat32);
}

TEST(TensorTest, ZerosFactory) {
  Tensor t = Tensor::Zeros({2, 3});
  EXPECT_EQ(t.numel(), 6);
  // All values should be zero.
  const float* data = t.data<float>();
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(data[i], 0.0f);
  }
}

TEST(TensorTest, OnesFactory) {
  Tensor t = Tensor::Ones({2, 3});
  EXPECT_EQ(t.numel(), 6);
  const float* data = t.data<float>();
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(data[i], 1.0f);
  }
}

TEST(TensorTest, FullFactory) {
  Tensor t = Tensor::Full({4}, 3.14f);
  EXPECT_EQ(t.numel(), 4);
  const float* data = t.data<float>();
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(data[i], 3.14f);
  }
}

TEST(TensorTest, DTypeSizes) {
  EXPECT_EQ(DTypeSize(DType::kFloat32), 4u);
  EXPECT_EQ(DTypeSize(DType::kFloat16), 2u);
  EXPECT_EQ(DTypeSize(DType::kBFloat16), 2u);
  EXPECT_EQ(DTypeSize(DType::kInt32), 4u);
  EXPECT_EQ(DTypeSize(DType::kInt8), 1u);
}

TEST(TensorTest, ToString) {
  Tensor t({3, 4});
  std::string s = t.to_string();
  EXPECT_NE(s.find("3"), std::string::npos);
  EXPECT_NE(s.find("4"), std::string::npos);
  EXPECT_NE(s.find("float32"), std::string::npos);
}

TEST(TensorTest, Strides) {
  Tensor t({2, 3, 4}, DType::kFloat32);
  // For a contiguous float32 tensor [2,3,4]:
  //   strides should be [48, 16, 4] (in bytes)
  EXPECT_EQ(t.strides().size(), 3u);
  EXPECT_EQ(t.strides()[2], 4);   // innermost: 1 float = 4 bytes
  EXPECT_EQ(t.strides()[1], 16);  // 4 floats = 16 bytes
  EXPECT_EQ(t.strides()[0], 48);  // 3*4 floats = 48 bytes
}

// ============================================================================
// TODO: Add more tests as you implement features!
//
// Suggested test cases:
//   - at() / set() element access
//   - reshape() preserves data
//   - transpose() swaps dimensions
//   - slice() returns correct view
//   - contiguous() copies non-contiguous tensor
//   - Different dtypes (int8, float16)
// ============================================================================

}  // namespace
}  // namespace ie
