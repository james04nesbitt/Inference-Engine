#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include "engine/compute/simd_kernels.h"
#include "engine/compute/thread_pool.h"

// ============================================================================
// SIMD Compute Benchmark Suite
//
// Measures GFLOPS, speedup ratios, and bandwidth for all SIMD kernels.
// Compares naive scalar vs Highway SIMD vs SIMD+threaded implementations.
//
// Run with: bazel run --config=release //engine/bench:simd_bench
// AVX2:     bazel run --config=release --config=avx2 //engine/bench:simd_bench
// ============================================================================

static ie::compute::ThreadPool g_pool(0);

// --- Utility: formatted table output ---

static void PrintHeader(const char *section) {
  std::cout << std::endl;
  std::cout << "--- " << section << " ---" << std::endl;
  std::cout << std::setw(28) << "Kernel" << "  " << std::setw(20) << "Size"
            << "  " << std::setw(10) << "Time" << "  " << std::setw(10)
            << "GFLOPS" << "  " << std::setw(10) << "Speedup" << std::endl;
  std::cout << std::string(84, '-') << std::endl;
}

static void PrintBandwidthHeader(const char *section) {
  std::cout << std::endl;
  std::cout << "--- " << section << " ---" << std::endl;
  std::cout << std::setw(28) << "Kernel" << "  " << std::setw(20) << "Size"
            << "  " << std::setw(10) << "Time" << "  " << std::setw(10)
            << "GB/s" << std::endl;
  std::cout << std::string(74, '-') << std::endl;
}

// ============================================================================
// GEMM Benchmarks
// ============================================================================

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

struct GemmResult {
  double avg_ms;
  double gflops;
};

static GemmResult BenchmarkGemm(const char *label, int64_t M, int64_t N,
                                int64_t K, int warmup, int iters,
                                void (*fn)(const float *, const float *,
                                           float *, int64_t, int64_t, int64_t),
                                double baseline_ms = -1.0) {
  std::vector<float> a(M * K, 0.5f);
  std::vector<float> b(K * N, 0.5f);
  std::vector<float> c(M * N, 0.0f);

  for (int i = 0; i < warmup; ++i) {
    std::memset(c.data(), 0, c.size() * sizeof(float));
    fn(a.data(), b.data(), c.data(), M, N, K);
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    std::memset(c.data(), 0, c.size() * sizeof(float));
    fn(a.data(), b.data(), c.data(), M, N, K);
  }
  auto end = std::chrono::high_resolution_clock::now();

  double secs = std::chrono::duration<double>(end - start).count();
  double avg_ms = (secs / iters) * 1000.0;
  double flops = 2.0 * M * N * K;
  double gflops = (flops * iters) / secs / 1e9;

  // Format size string.
  char size_str[32];
  snprintf(size_str, sizeof(size_str), "%lldx%lldx%lld", (long long)M,
           (long long)K, (long long)N);

  // Compute speedup.
  double speedup = (baseline_ms > 0.0) ? (baseline_ms / avg_ms) : 1.0;
  const char *speedup_str = (baseline_ms > 0.0) ? "" : "(baseline)";

  std::cout << std::setw(28) << label << "  " << std::setw(20) << size_str
            << "  " << std::fixed << std::setprecision(2) << std::setw(8)
            << avg_ms << "ms" << "  " << std::setw(8) << gflops << "  ";
  if (baseline_ms > 0.0) {
    std::cout << std::setw(7) << speedup << "x";
  } else {
    std::cout << std::setw(10) << speedup_str;
  }
  std::cout << std::endl;

  return {avg_ms, gflops};
}

static void SimdGemmWrapper(const float *a, const float *b, float *c, int64_t M,
                            int64_t N, int64_t K) {
  ie::compute::SimdGemm(a, b, c, M, N, K);
}

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

// ============================================================================
// GEMV Benchmarks
// ============================================================================

static void NaiveGemv(const float *a, const float *x, float *y, int64_t M,
                      int64_t K) {
  for (int64_t i = 0; i < M; ++i) {
    float sum = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
      sum += a[i * K + k] * x[k];
    }
    y[i] = sum;
  }
}

static GemmResult BenchmarkGemv(const char *label, int64_t M, int64_t K,
                                int warmup, int iters,
                                void (*fn)(const float *, const float *,
                                           float *, int64_t, int64_t),
                                double baseline_ms = -1.0) {
  std::vector<float> a(M * K, 0.5f);
  std::vector<float> x(K, 0.5f);
  std::vector<float> y(M, 0.0f);

  for (int i = 0; i < warmup; ++i) {
    fn(a.data(), x.data(), y.data(), M, K);
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    fn(a.data(), x.data(), y.data(), M, K);
  }
  auto end = std::chrono::high_resolution_clock::now();

  double secs = std::chrono::duration<double>(end - start).count();
  double avg_ms = (secs / iters) * 1000.0;
  double flops = 2.0 * M * K;
  double gflops = (flops * iters) / secs / 1e9;

  char size_str[32];
  snprintf(size_str, sizeof(size_str), "%lldx%lld", (long long)M, (long long)K);

  double speedup = (baseline_ms > 0.0) ? (baseline_ms / avg_ms) : 1.0;
  const char *speedup_str = (baseline_ms > 0.0) ? "" : "(baseline)";

  std::cout << std::setw(28) << label << "  " << std::setw(20) << size_str
            << "  " << std::fixed << std::setprecision(2) << std::setw(8)
            << avg_ms << "ms" << "  " << std::setw(8) << gflops << "  ";
  if (baseline_ms > 0.0) {
    std::cout << std::setw(7) << speedup << "x";
  } else {
    std::cout << std::setw(10) << speedup_str;
  }
  std::cout << std::endl;

  return {avg_ms, gflops};
}

static void SimdGemvWrapper(const float *a, const float *x, float *y, int64_t M,
                            int64_t K) {
  ie::compute::SimdGemv(a, x, y, M, K);
}

// ============================================================================
// Dot Product Benchmarks
// ============================================================================

static GemmResult BenchmarkDotProduct(const char *label, int64_t N, int warmup,
                                      int iters, double baseline_ms = -1.0) {
  std::vector<float> a(N, 0.5f);
  std::vector<float> b(N, 0.5f);
  volatile float result = 0;

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
  double avg_ms = (secs / iters) * 1000.0;
  double gflops = (2.0 * N * iters) / secs / 1e9;

  char size_str[32];
  snprintf(size_str, sizeof(size_str), "N=%lld", (long long)N);

  std::cout << std::setw(28) << label << "  " << std::setw(20) << size_str
            << "  " << std::fixed << std::setprecision(4) << std::setw(8)
            << avg_ms << "ms" << "  " << std::setprecision(2) << std::setw(8)
            << gflops << "  " << std::setw(10) << "" << std::endl;

  return {avg_ms, gflops};
}

// ============================================================================
// Element-wise Bandwidth Benchmarks
// ============================================================================

static void BenchmarkElementwise(const char *label, int64_t N, int warmup,
                                 int iters,
                                 void (*fn)(const float *, const float *,
                                            float *, int64_t)) {
  std::vector<float> a(N), b(N), c(N);
  for (int64_t i = 0; i < N; ++i) {
    a[i] = static_cast<float>(i) * 0.001f;
    b[i] = static_cast<float>(N - i) * 0.001f;
  }

  for (int i = 0; i < warmup; ++i) {
    fn(a.data(), b.data(), c.data(), N);
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    fn(a.data(), b.data(), c.data(), N);
  }
  auto end = std::chrono::high_resolution_clock::now();

  double secs = std::chrono::duration<double>(end - start).count();
  double avg_ms = (secs / iters) * 1000.0;
  // 3 arrays * N elements * 4 bytes (2 read + 1 write)
  double bytes = 3.0 * N * sizeof(float);
  double gb_s = (bytes * iters) / secs / 1e9;

  char size_str[32];
  snprintf(size_str, sizeof(size_str), "N=%lld", (long long)N);

  std::cout << std::setw(28) << label << "  " << std::setw(20) << size_str
            << "  " << std::fixed << std::setprecision(4) << std::setw(8)
            << avg_ms << "ms" << "  " << std::setprecision(2) << std::setw(8)
            << gb_s << std::endl;
}

static void BenchmarkUnary(const char *label, int64_t N, int warmup, int iters,
                           void (*fn)(const float *, float *, int64_t)) {
  std::vector<float> a(N), c(N);
  for (int64_t i = 0; i < N; ++i) {
    a[i] = static_cast<float>(i) * 0.001f;
  }

  for (int i = 0; i < warmup; ++i) {
    fn(a.data(), c.data(), N);
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    fn(a.data(), c.data(), N);
  }
  auto end = std::chrono::high_resolution_clock::now();

  double secs = std::chrono::duration<double>(end - start).count();
  double avg_ms = (secs / iters) * 1000.0;
  // 2 arrays * N elements * 4 bytes (1 read + 1 write)
  double bytes = 2.0 * N * sizeof(float);
  double gb_s = (bytes * iters) / secs / 1e9;

  char size_str[32];
  snprintf(size_str, sizeof(size_str), "N=%lld", (long long)N);

  std::cout << std::setw(28) << label << "  " << std::setw(20) << size_str
            << "  " << std::fixed << std::setprecision(4) << std::setw(8)
            << avg_ms << "ms" << "  " << std::setprecision(2) << std::setw(8)
            << gb_s << std::endl;
}

static void BenchmarkRmsNorm(const char *label, int64_t N, int warmup,
                             int iters) {
  std::vector<float> input(N), weight(N), output(N);
  for (int64_t i = 0; i < N; ++i) {
    input[i] = static_cast<float>(i) * 0.001f;
    weight[i] = 1.0f;
  }

  for (int i = 0; i < warmup; ++i) {
    ie::compute::SimdRmsNorm(input.data(), weight.data(), output.data(), N,
                             1e-6f);
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    ie::compute::SimdRmsNorm(input.data(), weight.data(), output.data(), N,
                             1e-6f);
  }
  auto end = std::chrono::high_resolution_clock::now();

  double secs = std::chrono::duration<double>(end - start).count();
  double avg_ms = (secs / iters) * 1000.0;
  // 3 arrays: input (read) + weight (read) + output (write)
  double bytes = 3.0 * N * sizeof(float);
  double gb_s = (bytes * iters) / secs / 1e9;

  char size_str[32];
  snprintf(size_str, sizeof(size_str), "N=%lld", (long long)N);

  std::cout << std::setw(28) << label << "  " << std::setw(20) << size_str
            << "  " << std::fixed << std::setprecision(4) << std::setw(8)
            << avg_ms << "ms" << "  " << std::setprecision(2) << std::setw(8)
            << gb_s << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main() {
  std::cout << "============================================================"
            << std::endl;
  std::cout << "  Inference Engine — SIMD Compute Benchmark Suite" << std::endl;
  std::cout << "  Threads: " << g_pool.NumThreads() << std::endl;
  std::cout << "============================================================"
            << std::endl;

  // ---- GEMM: Gemma-3 FFN shapes (single-token decode) ----
  {
    PrintHeader("GEMM: FFN Shapes (Single-Token Decode)");
    struct {
      int64_t M, N, K;
      int iters;
      const char *desc;
    } sizes[] = {
        {1, 6912, 1152, 100, "FFN gate/up"},
        {1, 1152, 6912, 100, "FFN down"},
    };
    for (auto &s : sizes) {
      std::cout << "  " << s.desc << ":" << std::endl;
      auto naive =
          BenchmarkGemm("Naive (scalar)", s.M, s.N, s.K, 3, s.iters, NaiveGemm);
      BenchmarkGemm("SIMD (Highway)", s.M, s.N, s.K, 3, s.iters,
                    SimdGemmWrapper, naive.avg_ms);
      std::cout << std::endl;
    }
  }

  // ---- GEMM: Attention projection shapes ----
  {
    PrintHeader("GEMM: Attention Projections");
    struct {
      int64_t M, N, K;
      int iters;
      const char *desc;
    } sizes[] = {
        {1, 1024, 1152, 100, "Q/K/V projection (1 token)"},
        {16, 1024, 1152, 30, "Q/K/V projection (16 tokens)"},
        {16, 1152, 1152, 30, "Output projection (16 tokens)"},
    };
    for (auto &s : sizes) {
      std::cout << "  " << s.desc << ":" << std::endl;
      auto naive =
          BenchmarkGemm("Naive (scalar)", s.M, s.N, s.K, 2, s.iters, NaiveGemm);
      auto simd = BenchmarkGemm("SIMD (Highway)", s.M, s.N, s.K, 2, s.iters,
                                SimdGemmWrapper, naive.avg_ms);
      if (s.M >= 4) {
        BenchmarkGemm("SIMD + Threaded", s.M, s.N, s.K, 2, s.iters,
                      ThreadedSimdGemm, naive.avg_ms);
      }
      std::cout << std::endl;
    }
  }

  // ---- GEMM: Large square matrices (stress test) ----
  {
    PrintHeader("GEMM: Large Square (Stress Test)");
    int64_t sizes[] = {256, 512, 1024};
    for (auto N : sizes) {
      int iters = (N <= 512) ? 10 : 3;
      auto naive =
          BenchmarkGemm("Naive (scalar)", N, N, N, 1, iters, NaiveGemm);
      auto simd = BenchmarkGemm("SIMD (Highway)", N, N, N, 1, iters,
                                SimdGemmWrapper, naive.avg_ms);
      BenchmarkGemm("SIMD + Threaded", N, N, N, 1, iters, ThreadedSimdGemm,
                    naive.avg_ms);
      std::cout << std::endl;
    }
  }

  // ---- GEMV: Single-token decode (M=1 matmul) ----
  {
    PrintHeader("GEMV: Single-Token Decode");
    struct {
      int64_t M, K;
      int iters;
      const char *desc;
    } sizes[] = {
        {1152, 6912, 200, "FFN down projection"},
        {6912, 1152, 200, "FFN gate/up projection"},
        {1024, 1152, 200, "Attention projection"},
    };
    for (auto &s : sizes) {
      std::cout << "  " << s.desc << ":" << std::endl;
      auto naive =
          BenchmarkGemv("Naive (scalar)", s.M, s.K, 5, s.iters, NaiveGemv);
      BenchmarkGemv("SIMD (Highway)", s.M, s.K, 5, s.iters, SimdGemvWrapper,
                    naive.avg_ms);
      std::cout << std::endl;
    }
  }

  // ---- Dot Product ----
  {
    PrintHeader("Dot Product");
    BenchmarkDotProduct("SimdDotProduct", 256, 10, 50000);
    BenchmarkDotProduct("SimdDotProduct", 1152, 10, 50000);
    BenchmarkDotProduct("SimdDotProduct", 6912, 10, 20000);
  }

  // ---- Element-wise Bandwidth ----
  {
    int64_t N = 1152 * 26; // embed_dim * num_layers (typical activation size)
    PrintBandwidthHeader("Element-wise Ops (Bandwidth)");
    BenchmarkElementwise("SimdAdd", N, 10, 10000, ie::compute::SimdAdd);
    BenchmarkElementwise("SimdMul", N, 10, 10000, ie::compute::SimdMul);
    BenchmarkUnary("SimdSilu", N, 10, 10000, ie::compute::SimdSilu);
    BenchmarkUnary("SimdSoftmax", N, 10, 10000, ie::compute::SimdSoftmax);
    BenchmarkRmsNorm("SimdRmsNorm", N, 10, 10000);
  }

  // ---- Summary ----
  std::cout << std::endl;
  std::cout << "============================================================"
            << std::endl;
  std::cout << "  Benchmark complete." << std::endl;
  std::cout << "============================================================"
            << std::endl;

  return 0;
}
