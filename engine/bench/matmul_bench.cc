#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include "engine/compute/simd_kernels.h"
#include "engine/compute/thread_pool.h"

// ============================================================================
// GEMM/GEMV/Dot Product Benchmarks
//
// Simple chrono-based benchmarks measuring GFLOPS for SIMD kernels.
// Compares naive scalar vs SIMD vs SIMD+threaded matmul.
//
// Run with: bazel run //engine/bench:matmul_bench
// Release:  bazel run --config=release //engine/bench:matmul_bench
// AVX2:     bazel run --config=release --config=avx2
// //engine/bench:matmul_bench
// ============================================================================

static double BenchmarkGemm(const char *label, int64_t M, int64_t N, int64_t K,
                            int warmup, int iters,
                            void (*fn)(const float *, const float *, float *,
                                       int64_t, int64_t, int64_t)) {
  std::vector<float> a(M * K, 0.5f);
  std::vector<float> b(K * N, 0.5f);
  std::vector<float> c(M * N, 0.0f);

  // Warmup
  for (int i = 0; i < warmup; ++i) {
    std::memset(c.data(), 0, c.size() * sizeof(float));
    fn(a.data(), b.data(), c.data(), M, N, K);
  }

  // Timed iterations
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    std::memset(c.data(), 0, c.size() * sizeof(float));
    fn(a.data(), b.data(), c.data(), M, N, K);
  }
  auto end = std::chrono::high_resolution_clock::now();

  double secs = std::chrono::duration<double>(end - start).count();
  double avg_ms = (secs / iters) * 1000.0;
  double flops = 2.0 * M * N * K; // multiply + add per element
  double gflops = (flops * iters) / secs / 1e9;

  std::cout << std::setw(30) << label << "  " << std::setw(5) << M << " x "
            << std::setw(5) << K << " x " << std::setw(5) << N << "  "
            << std::fixed << std::setprecision(2) << std::setw(8) << avg_ms
            << " ms  " << std::setw(8) << gflops << " GFLOPS" << std::endl;
  return gflops;
}

// Naive scalar GEMM baseline (i,k,j order for fairness).
static void NaiveGemm(const float *a, const float *b, float *c, int64_t M,
                      int64_t N, int64_t K) {
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t k = 0; k < K; ++k) {
      float a_ik = a[i * K + k];
      for (int64_t j = 0; j < N; ++j) {
        c[i * N + j] += a_ik * b[k * N + j];
      }
    }
  }
}

// SIMD GEMM (single-threaded)
static void SimdGemmWrapper(const float *a, const float *b, float *c, int64_t M,
                            int64_t N, int64_t K) {
  ie::compute::SimdGemm(a, b, c, M, N, K);
}

// SIMD GEMM + multi-threaded M-partition
static ie::compute::ThreadPool g_pool(0);

static void ThreadedSimdGemm(const float *a, const float *b, float *c,
                             int64_t M, int64_t N, int64_t K) {
  int64_t n_threads = static_cast<int64_t>(g_pool.NumThreads());
  if (n_threads < 1)
    n_threads = 1;

  g_pool.ParallelFor(n_threads, [&](int64_t tid) {
    int64_t rows_per_thread = (M + n_threads - 1) / n_threads;
    int64_t start = tid * rows_per_thread;
    int64_t end = std::min(start + rows_per_thread, M);
    if (start >= M)
      return;
    ie::compute::SimdGemm(a + start * K, b, c + start * N, end - start, N, K);
  });
}

// Dot product benchmark
static void BenchmarkDotProduct(int64_t N, int warmup, int iters) {
  std::vector<float> a(N, 0.5f);
  std::vector<float> b(N, 0.5f);
  float result = 0;

  for (int i = 0; i < warmup; ++i) {
    result = ie::compute::SimdDotProduct(a.data(), b.data(), N);
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    result = ie::compute::SimdDotProduct(a.data(), b.data(), N);
  }
  auto end = std::chrono::high_resolution_clock::now();
  (void)result;

  double secs = std::chrono::duration<double>(end - start).count();
  double avg_ns = (secs / iters) * 1e9;
  double gflops = (2.0 * N * iters) / secs / 1e9;

  std::cout << std::setw(30) << "SimdDotProduct" << "  N=" << std::setw(6) << N
            << "  " << std::fixed << std::setprecision(0) << std::setw(8)
            << avg_ns << " ns  " << std::setprecision(2) << std::setw(8)
            << gflops << " GFLOPS" << std::endl;
}

int main() {
  std::cout << "============================================" << std::endl;
  std::cout << "  Inference Engine Kernel Benchmarks" << std::endl;
  std::cout << "  Threads: " << g_pool.NumThreads() << std::endl;
  std::cout << "============================================" << std::endl;
  std::cout << std::endl;

  // --- Dot Product ---
  std::cout << "--- Dot Product ---" << std::endl;
  BenchmarkDotProduct(256, 10, 10000);
  BenchmarkDotProduct(1024, 10, 10000);
  BenchmarkDotProduct(1152, 10, 10000); // Gemma embed_dim
  BenchmarkDotProduct(6912, 10, 5000);  // Gemma FFN dim
  std::cout << std::endl;

  // --- GEMM: Small (Attention projections) ---
  std::cout << "--- GEMM: Small Matrices ---" << std::endl;
  struct {
    int64_t M, N, K;
    int iters;
  } small_sizes[] = {
      {4, 256, 1152, 50},   // Single-head Q projection
      {16, 1152, 1152, 20}, // Multi-token attention
  };
  for (auto &s : small_sizes) {
    BenchmarkGemm("Naive (scalar)", s.M, s.N, s.K, 2, s.iters, NaiveGemm);
    BenchmarkGemm("SIMD (Highway)", s.M, s.N, s.K, 2, s.iters, SimdGemmWrapper);
    std::cout << std::endl;
  }

  // --- GEMM: Medium (FFN layers) ---
  std::cout << "--- GEMM: Medium Matrices (FFN) ---" << std::endl;
  struct {
    int64_t M, N, K;
    int iters;
  } med_sizes[] = {
      {1, 6912, 1152, 50},  // Single-token FFN gate/up
      {1, 1152, 6912, 50},  // Single-token FFN down
      {16, 6912, 1152, 10}, // Multi-token FFN
  };
  for (auto &s : med_sizes) {
    BenchmarkGemm("Naive (scalar)", s.M, s.N, s.K, 2, s.iters, NaiveGemm);
    BenchmarkGemm("SIMD (Highway)", s.M, s.N, s.K, 2, s.iters, SimdGemmWrapper);
    if (s.M >= 4) {
      BenchmarkGemm("SIMD + Threaded", s.M, s.N, s.K, 2, s.iters,
                    ThreadedSimdGemm);
    }
    std::cout << std::endl;
  }

  // --- GEMM: Large (Square) ---
  std::cout << "--- GEMM: Large Square Matrices ---" << std::endl;
  int64_t large_sizes[] = {256, 512, 1024};
  for (auto N : large_sizes) {
    int iters = (N <= 512) ? 10 : 3;
    BenchmarkGemm("Naive (scalar)", N, N, N, 1, iters, NaiveGemm);
    BenchmarkGemm("SIMD (Highway)", N, N, N, 1, iters, SimdGemmWrapper);
    BenchmarkGemm("SIMD + Threaded", N, N, N, 1, iters, ThreadedSimdGemm);
    std::cout << std::endl;
  }

  return 0;
}
