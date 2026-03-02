#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace ie {
namespace compute {

// ============================================================================
// ThreadPool — Parallel execution for compute kernels.
//
// Used to parallelize GEMM across the M dimension, or to run multiple
// transformer layers concurrently when there are no data dependencies.
//
// Design decisions for you to make:
//   - Fixed pool vs work-stealing?
//   - std::jthread vs std::thread?
//   - Condition variable vs spinlock for low-latency wakeup?
//   - Task granularity: per-tile? per-row? per-layer?
//
// For inference, latency matters more than throughput, so you want
// minimal wake-up overhead. Consider spinning briefly before sleeping.
// ============================================================================
class ThreadPool {
 public:
  // Create a thread pool with `num_threads` workers.
  // If num_threads == 0, uses hardware_concurrency().
  explicit ThreadPool(size_t num_threads = 0);

  ~ThreadPool();

  // Submit a task for execution. Returns a future for the result.
  std::future<void> Submit(std::function<void()> task);

  // Execute `func(i)` for i in [0, count) in parallel across the pool.
  // Blocks until all iterations complete.
  // This is the primary interface for parallelizing compute kernels.
  //
  // Example: Parallelize GEMM rows
  //   pool.ParallelFor(M, [&](int64_t i) {
  //     // Compute row i of the output matrix
  //     for (int64_t j = 0; j < N; ++j) {
  //       float sum = 0;
  //       for (int64_t k = 0; k < K; ++k)
  //         sum += a[i*K+k] * b[k*N+j];
  //       c[i*N+j] = sum;
  //     }
  //   });
  //
  void ParallelFor(int64_t count, std::function<void(int64_t)> func);

  // Number of worker threads in the pool.
  size_t NumThreads() const { return workers_.size(); }

 private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable cv_;
  
  // Track tasks queued vs completed to avoid waiting on empty futures inline
  std::atomic<int> active_tasks_{0};
  std::atomic<bool> stop_{false};
};

}  // namespace compute
}  // namespace ie
