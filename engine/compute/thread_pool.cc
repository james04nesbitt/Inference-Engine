#include "engine/compute/thread_pool.h"

#include <stdexcept>

namespace ie {
namespace compute {

ThreadPool::ThreadPool(size_t num_threads) {
  if (num_threads == 0) {
    num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;  // Fallback
  }

  workers_.reserve(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    workers_.emplace_back([this]() {
      while (!stop_.load(std::memory_order_acquire)) {
        std::function<void()> task;

        // Brief spin loop for low latency wakeup before hard sleeping
        bool found_task = false;
        for (int spin_count = 0; spin_count < 1000; ++spin_count) {
           if (stop_.load(std::memory_order_relaxed)) return;
           std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
           if (lock.owns_lock() && !tasks_.empty()) {
             task = std::move(tasks_.front());
             tasks_.pop();
             found_task = true;
             break;
           }
        }

        if (!found_task) {
           std::unique_lock<std::mutex> lock(mutex_);
           cv_.wait(lock, [this]() {
             return stop_.load(std::memory_order_relaxed) || !tasks_.empty();
           });

           if (stop_.load(std::memory_order_relaxed) && tasks_.empty()) return;

           if (!tasks_.empty()) {
             task = std::move(tasks_.front());
             tasks_.pop();
           }
        }

        if (task) {
          task();
          active_tasks_.fetch_sub(1, std::memory_order_release);
        }
      }
    });
  }
}

ThreadPool::~ThreadPool() {
  stop_.store(true, std::memory_order_release);
  cv_.notify_all();
  for (auto& w : workers_) {
    if (w.joinable()) {
       w.join();
    }
  }
}

std::future<void> ThreadPool::Submit(std::function<void()> task) {
  active_tasks_.fetch_add(1, std::memory_order_release);
  auto packaged = std::make_shared<std::packaged_task<void()>>(std::move(task));
  std::future<void> res = packaged->get_future();
  {
    std::unique_lock<std::mutex> lock(mutex_);
    tasks_.emplace([packaged]() { (*packaged)(); });
  }
  cv_.notify_one();
  return res;
}

void ThreadPool::ParallelFor(int64_t count,
                             std::function<void(int64_t)> func) {
  if (count <= 0) return;
  std::atomic<int64_t> current_idx{0};

  size_t num_threads = workers_.size();
  if (num_threads == 0) {
    for (int64_t i = 0; i < count; ++i) {
      func(i);
    }
    return;
  }

  // Use atomic index assignment to dynamically distribute work load to waiting threads
  std::vector<std::future<void>> futures;
  futures.reserve(num_threads);

  for (size_t t = 0; t < num_threads; ++t) {
    futures.push_back(Submit([&, count]() {
      while (true) {
        int64_t idx = current_idx.fetch_add(1, std::memory_order_relaxed);
        if (idx >= count) break;
        func(idx);
      }
    }));
  }

  for (auto& f : futures) {
    f.get();
  }
}

}  // namespace compute
}  // namespace ie
