#include "engine/attention/kv_cache.h"

#include <vector>

#include "gtest/gtest.h"

namespace ie {
namespace {

class KVCacheManagerTest : public ::testing::Test {
protected:
  // Small cache for testing: 2 layers, 1 KV head, 4-dim heads, 10 blocks of 4.
  static constexpr int32_t kNumLayers = 2;
  static constexpr int32_t kNumKVHeads = 1;
  static constexpr int32_t kHeadDim = 4;
  static constexpr int32_t kMaxBlocks = 10;
  static constexpr int32_t kBlockSize = 4;

  KVCacheManager CreateManager() {
    return KVCacheManager(kNumLayers, kNumKVHeads, kHeadDim, kMaxBlocks,
                          kBlockSize);
  }

  // Create a test KV tensor [num_kv_heads, head_dim] filled with val.
  Tensor MakeKV(float val) {
    Tensor t({kNumKVHeads, kHeadDim});
    t.fill(val);
    return t;
  }
};

TEST_F(KVCacheManagerTest, InitialState) {
  auto mgr = CreateManager();
  EXPECT_EQ(mgr.NumFreeBlocks(), kMaxBlocks);
}

TEST_F(KVCacheManagerTest, AllocateAndFree) {
  auto mgr = CreateManager();

  int64_t seq_id = mgr.AllocateSequence();
  // Each sequence allocates one block per layer.
  EXPECT_EQ(mgr.NumFreeBlocks(), kMaxBlocks - kNumLayers);

  mgr.FreeSequence(seq_id);
  EXPECT_EQ(mgr.NumFreeBlocks(), kMaxBlocks);
}

TEST_F(KVCacheManagerTest, MultipleSequences) {
  auto mgr = CreateManager();

  int64_t s1 = mgr.AllocateSequence();
  int64_t s2 = mgr.AllocateSequence();
  EXPECT_NE(s1, s2);
  EXPECT_EQ(mgr.NumFreeBlocks(), kMaxBlocks - 2 * kNumLayers);

  mgr.FreeSequence(s1);
  EXPECT_EQ(mgr.NumFreeBlocks(), kMaxBlocks - kNumLayers);

  mgr.FreeSequence(s2);
  EXPECT_EQ(mgr.NumFreeBlocks(), kMaxBlocks);
}

TEST_F(KVCacheManagerTest, AppendAndRetrieve) {
  auto mgr = CreateManager();
  int64_t seq_id = mgr.AllocateSequence();

  // Append 3 tokens to layer 0.
  mgr.AppendToken(seq_id, 0, MakeKV(1.0f), MakeKV(10.0f));
  mgr.AppendToken(seq_id, 0, MakeKV(2.0f), MakeKV(20.0f));
  mgr.AppendToken(seq_id, 0, MakeKV(3.0f), MakeKV(30.0f));

  EXPECT_EQ(mgr.SequenceLength(seq_id), 3);

  // Retrieve keys: should be [3, 1, 4].
  Tensor keys = mgr.GetKeys(seq_id, 0);
  EXPECT_EQ(keys.size(0), 3);
  EXPECT_EQ(keys.size(1), kNumKVHeads);
  EXPECT_EQ(keys.size(2), kHeadDim);

  // token 0 keys should be all 1.0.
  EXPECT_NEAR(keys.at({0, 0, 0}), 1.0f, 1e-5f);
  // token 1 keys should be all 2.0.
  EXPECT_NEAR(keys.at({1, 0, 0}), 2.0f, 1e-5f);
  // token 2 keys should be all 3.0.
  EXPECT_NEAR(keys.at({2, 0, 0}), 3.0f, 1e-5f);

  // Retrieve values.
  Tensor values = mgr.GetValues(seq_id, 0);
  EXPECT_NEAR(values.at({0, 0, 0}), 10.0f, 1e-5f);
  EXPECT_NEAR(values.at({1, 0, 0}), 20.0f, 1e-5f);
  EXPECT_NEAR(values.at({2, 0, 0}), 30.0f, 1e-5f);

  mgr.FreeSequence(seq_id);
}

TEST_F(KVCacheManagerTest, BlockOverflowAllocatesNew) {
  auto mgr = CreateManager();
  int64_t seq_id = mgr.AllocateSequence();

  int32_t initial_free = mgr.NumFreeBlocks();

  // Append block_size tokens — should fill the first block.
  for (int32_t i = 0; i < kBlockSize; ++i) {
    mgr.AppendToken(seq_id, 0, MakeKV(static_cast<float>(i)), MakeKV(0.0f));
  }
  EXPECT_EQ(mgr.SequenceLength(seq_id), kBlockSize);

  // Append one more — should allocate a new block.
  mgr.AppendToken(seq_id, 0, MakeKV(100.0f), MakeKV(0.0f));
  EXPECT_EQ(mgr.NumFreeBlocks(), initial_free - 1);
  EXPECT_EQ(mgr.SequenceLength(seq_id), kBlockSize + 1);

  // Verify all data is retrievable.
  Tensor keys = mgr.GetKeys(seq_id, 0);
  EXPECT_EQ(keys.size(0), kBlockSize + 1);
  EXPECT_NEAR(keys.at({kBlockSize, 0, 0}), 100.0f, 1e-5f);

  mgr.FreeSequence(seq_id);
}

TEST_F(KVCacheManagerTest, FreeSequenceThrowsForUnknown) {
  auto mgr = CreateManager();
  EXPECT_THROW(mgr.FreeSequence(999), std::runtime_error);
}

} // namespace
} // namespace ie
