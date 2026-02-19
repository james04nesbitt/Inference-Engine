#include "engine/tensor/tensor.h"

#include <gtest/gtest.h>

namespace ie {
namespace {

// ============================================================================
// These tests are intentionally minimal. As you implement the Tensor class,
// ADD MORE TESTS. Writing tests is a great way to verify you understand
// what each function should do.
//
// Run tests with: bazel test //engine/tensor:tensor_test
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
  EXPECT_EQ(t.numel(), 60);             // 3 * 4 * 5
  EXPECT_EQ(t.nbytes(), 60 * 4);        // 60 floats * 4 bytes each
  EXPECT_EQ(t.dtype(), DType::kFloat32);
}

TEST(TensorTest, DTypeSizes) {
  EXPECT_EQ(DTypeSize(DType::kFloat32), 4u);
  EXPECT_EQ(DTypeSize(DType::kFloat16), 2u);
  EXPECT_EQ(DTypeSize(DType::kInt8), 1u);
}

TEST(TensorTest, ToString) {
  Tensor t({3, 4});
  std::string s = t.to_string();
  EXPECT_NE(s.find("3"), std::string::npos);
  EXPECT_NE(s.find("4"), std::string::npos);
  EXPECT_NE(s.find("float32"), std::string::npos);
}

// ============================================================================
// YOUR TESTS — uncomment these as you implement features!
// ============================================================================

// TEST(TensorTest, DataIsAllocated) {
//   Tensor t({2, 3});
//   EXPECT_NE(t.data_ptr(), nullptr);
//   EXPECT_EQ(t.nbytes(), 24u);  // 6 floats * 4 bytes
// }

// TEST(TensorTest, SetAndGet) {
//   Tensor t({2, 3});
//   t.set({0, 0}, 1.0f);
//   t.set({1, 2}, 42.0f);
//   EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
//   EXPECT_FLOAT_EQ(t.at({1, 2}), 42.0f);
//   EXPECT_FLOAT_EQ(t.at({0, 1}), 0.0f);  // Should be zero-initialized
// }

// TEST(TensorTest, ThreeDimensional) {
//   Tensor t({2, 3, 4});
//   t.set({1, 2, 3}, 99.0f);
//   EXPECT_FLOAT_EQ(t.at({1, 2, 3}), 99.0f);
//   EXPECT_FLOAT_EQ(t.at({0, 0, 0}), 0.0f);
// }

}  // namespace
}  // namespace ie
