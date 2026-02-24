#include "engine/compute/thread_pool.h"

#include <stdexcept>

namespace ie {
namespace compute {

ThreadPool::ThreadPool(size_t num_threads) {
  if (num_threads == 0) {
    num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;  // Fallback
  }

  // TODO: Launch worker threads
  //
  // Each worker should:
  //   1. Lock the mutex
  //   2. Wait on the condition variable until a task is available or stop_
  //   3. Pop a task from the queue
  //   4. Unlock and execute the task
  //   5. Repeat
  //
  // Use std::jthread so threads auto-join on destruction.
  //
  // workers_.reserve(num_threads);
  // for (size_t i = 0; i < num_threads; ++i) {
  //   workers_.emplace_back([this](std::stop_token st) {
  //     while (!st.stop_requested()) {
  //       std::function<void()> task;
  //       {
  //         std::unique_lock lock(mutex_);
  //         cv_.wait(lock, [&] { return stop_ || !tasks_.empty(); });
  //         if (stop_ && tasks_.empty()) return;
  //         task = std::move(tasks_.front());
  //         tasks_.pop();
  //       }
  //       task();
  //     }
  //   });
  // }
}

ThreadPool::~ThreadPool() {
  // TODO: Signal workers to stop and join
  //   {
  //     std::unique_lock lock(mutex_);
  //     stop_ = true;
  //   }
  //   cv_.notify_all();
  //   // std::jthread auto-joins, but we need to request stop
  //   for (auto& w : workers_) w.request_stop();
}

std::future<void> ThreadPool::Submit(std::function<void()> task) {
  // TODO: Wrap task in a packaged_task, push to queue, notify a worker
  throw std::runtime_error("ThreadPool::Submit not implemented yet");
}

void ThreadPool::ParallelFor(int64_t count,
                             std::function<void(int64_t)> func) {
  // TODO: Divide [0, count) into chunks and submit each chunk
  //
  // Simple strategy:
  //   chunk_size = count / num_threads (round up)
  //   for each chunk: Submit([&, start, end] {
  //     for (i = start; i < end; ++i) func(i);
  //   })
  //   Wait for all futures
  //
  throw std::runtime_error("ThreadPool::ParallelFor not implemented yet");
}

}  // namespace compute
}  // namespace ie
