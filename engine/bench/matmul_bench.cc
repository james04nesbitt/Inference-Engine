#include <benchmark/benchmark.h>

// ============================================================================
// GEMM/GEMV Benchmarks
//
// Use these benchmarks to measure the impact of your optimizations:
//   1. Naive i,j,k loop
//   2. Cache-friendly i,k,j reorder
//   3. Tiled GEMM
//   4. SIMD-accelerated (Highway)
//   5. Multi-threaded
//
// Run with: bazel run --config=release //engine/bench:matmul_bench
//
// Typical output:
//   BM_Gemm_Naive/1024        X ms
//   BM_Gemm_Tiled/1024        Y ms    (should see ~3-5x improvement)
//   BM_Gemm_SIMD/1024         Z ms    (should see ~4-8x over tiled)
// ============================================================================

// TODO: Include your SIMD kernel headers when ready
// #include "engine/compute/simd_kernels.h"

static void BM_Gemm_Naive(benchmark::State& state) {
  const int64_t N = state.range(0);

  // Allocate matrices
  std::vector<float> a(N * N, 1.0f);
  std::vector<float> b(N * N, 1.0f);
  std::vector<float> c(N * N, 0.0f);

  for (auto _ : state) {
    // TODO: Replace with your GEMM implementation
    // SimdGemm(a.data(), b.data(), c.data(), N, N, N);

    // Placeholder: naive matmul
    for (int64_t i = 0; i < N; ++i)
      for (int64_t k = 0; k < N; ++k)
        for (int64_t j = 0; j < N; ++j)
          c[i * N + j] += a[i * N + k] * b[k * N + j];

    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }

  // Report FLOPS: 2 * N^3 (multiply + add per element)
  state.SetItemsProcessed(state.iterations() * 2 * N * N * N);
  state.SetBytesProcessed(state.iterations() * 3 * N * N * sizeof(float));
}

static void BM_Gemv(benchmark::State& state) {
  const int64_t M = state.range(0);
  const int64_t K = state.range(0);

  std::vector<float> a(M * K, 1.0f);
  std::vector<float> x(K, 1.0f);
  std::vector<float> y(M, 0.0f);

  for (auto _ : state) {
    // TODO: Replace with SimdGemv
    for (int64_t i = 0; i < M; ++i) {
      y[i] = 0;
      for (int64_t k = 0; k < K; ++k)
        y[i] += a[i * K + k] * x[k];
    }
    benchmark::DoNotOptimize(y.data());
  }

  state.SetItemsProcessed(state.iterations() * 2 * M * K);
}

static void BM_DotProduct(benchmark::State& state) {
  const int64_t N = state.range(0);

  std::vector<float> a(N, 1.0f);
  std::vector<float> b(N, 1.0f);

  for (auto _ : state) {
    float sum = 0;
    for (int64_t i = 0; i < N; ++i)
      sum += a[i] * b[i];
    benchmark::DoNotOptimize(sum);
  }

  state.SetItemsProcessed(state.iterations() * 2 * N);
}

// Register benchmarks with relevant sizes
// 1152 = Gemma embed_dim, 6912 = FFN hidden_dim
BENCHMARK(BM_Gemm_Naive)->Arg(256)->Arg(512)->Arg(1024)->Arg(1152);
BENCHMARK(BM_Gemv)->Arg(256)->Arg(1024)->Arg(1152)->Arg(6912);
BENCHMARK(BM_DotProduct)->Arg(256)->Arg(1024)->Arg(6912);

BENCHMARK_MAIN();
