#include "engine/compute/thread_pool.h"

#include <atomic>
#include <vector>

#include <gtest/gtest.h>

namespace ie {
namespace compute {
namespace {

TEST(ThreadPoolTest, SubmitExecutes) {
  ThreadPool pool(2);
  std::atomic<bool> done{false};
  auto future = pool.Submit([&done]() {
    done = true;
  });
  future.get();
  EXPECT_TRUE(done);
}

TEST(ThreadPoolTest, HardwareConcurrency) {
  ThreadPool pool(0);
  EXPECT_GE(pool.NumThreads(), 1);
}

TEST(ThreadPoolTest, ParallelForCorrectness) {
  ThreadPool pool(4);
  const int64_t n = 1000;
  std::vector<int> data(n, 0);

  pool.ParallelFor(n, [&](int64_t i) {
    data[i] = i * 2;
  });

  for (int64_t i = 0; i < n; ++i) {
    EXPECT_EQ(data[i], i * 2);
  }
}

TEST(ThreadPoolTest, ParallelForSmallCount) {
  ThreadPool pool(4);
  const int64_t n = 2; // smaller than num_threads
  std::vector<int> data(n, 0);

  pool.ParallelFor(n, [&](int64_t i) {
    data[i] = i * 2;
  });

  EXPECT_EQ(data[0], 0);
  EXPECT_EQ(data[1], 2);
}

TEST(ThreadPoolTest, ParallelForZeroCount) {
  ThreadPool pool(4);
  // Should not crash or hang
  pool.ParallelFor(0, [](int64_t i) {});
  EXPECT_TRUE(true);
}

TEST(ThreadPoolTest, ParallelForLargeCountMultipleThreads) {
  ThreadPool pool(8);
  const int64_t n = 100000;
  std::vector<std::atomic<int>> counts(n);
  for (int i = 0; i < n; ++i) counts[i] = 0;

  pool.ParallelFor(n, [&](int64_t i) {
    counts[i]++;
  });

  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(counts[i].load(), 1);
  }
}

}  // namespace
}  // namespace compute
}  // namespace ie
