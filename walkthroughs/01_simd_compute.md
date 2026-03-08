# SIMD Compute & High-Performance Kernels

## What This Does

The `engine/compute/` and `engine/ops/` directories contain the performance-critical compute kernels that transformer inference runs on. Since LLM inference is ~90% matrix multiplication, the matmul kernel dominates runtime. We built a multi-stage optimization pipeline from naive loops to SIMD-vectorized, multi-threaded GEMM.

## Architecture

```
ops/ops.cc               ← High-level tensor operations (matmul, softmax, etc.)
    ↓ delegates to
compute/simd_kernels.cc  ← Highway SIMD kernels (platform-adaptive)
compute/thread_pool.cc   ← Lock-free thread pool for parallelism
```

## Key Files

| File | Purpose |
|------|---------|
| [simd_kernels.h](../engine/compute/simd_kernels.h) | SIMD kernel declarations |
| [simd_kernels.cc](../engine/compute/simd_kernels.cc) | Highway SIMD implementations |
| [thread_pool.h](../engine/compute/thread_pool.h) | Thread pool with `ParallelFor` |
| [ops.cc](../engine/ops/ops.cc) | High-level ops that call SIMD kernels |

## GEMM Optimization Stages

The matmul went through 5 optimization stages, each building on the previous:

### 1. Naive (i,j,k)
```cpp
for (i) for (j) for (k) C[i][j] += A[i][k] * B[k][j];
```
**Problem:** Accesses B in column-major order → cache misses on every `B[k][j]`.

### 2. Cache-Friendly (i,k,j)
```cpp
for (i) for (k) for (j) C[i][j] += A[i][k] * B[k][j];
```
**Why:** Swapping the inner two loops makes B access sequential. The value `A[i][k]` is loaded once and reused across the entire j-loop. This alone gives ~2-3x speedup from better cache utilization.

### 3. Tiled (64×64 blocks)
```cpp
for (i in tiles of 64) for (k in tiles of 64) for (j in tiles of 64)
    // process tile
```
**Why:** A 64×64 float32 tile = 16KB. Two tiles fit comfortably in L1 cache (32-48KB). The A-tile stays resident in L2 while we sweep across B-tiles. This maximizes data reuse and minimizes cache evictions.

### 4. SIMD (Google Highway)
```cpp
hn::ScalableTag<float> d;
auto sum = hn::Zero(d);
for (k in steps of Lanes(d))
    sum = hn::MulAdd(hn::Load(d, &A[k]), hn::Load(d, &B[k]), sum);
```
**Why we chose Highway over raw intrinsics:**
- **Portability:** Same code compiles to AVX2 (8 floats), AVX-512 (16 floats), or NEON (4 floats) depending on the target CPU. No `#ifdef` spaghetti.
- **`MulAdd` fusion:** Fused multiply-add is a single instruction on modern CPUs (FMA3/FMA4), doing 2 FLOPs per cycle instead of 1.
- **`ReduceSum`:** Horizontal reduction is notoriously tricky with raw intrinsics. Highway handles the cross-lane shuffle pattern.

### 5. Multi-threaded (ThreadPool)
```cpp
pool.ParallelFor(M, [&](int64_t start, int64_t end) {
    SimdGemm(A + start*K, B, C + start*N, end-start, N, K);
});
```
**Why M-dimension partitioning:** Each thread gets a contiguous chunk of output rows. There are no write conflicts (each thread writes to different rows of C), so no synchronization or atomics needed.

## Other SIMD Kernels

All accelerated with Highway:

| Kernel | What It Does | Why SIMD Matters |
|--------|-------------|------------------|
| `SimdDotProduct` | Inner product of two vectors | Core building block for GEMM/GEMV |
| `SimdGemv` | Matrix-vector multiply | Used during single-token decode (seq_len=1) |
| `SimdSoftmax` | Numerically stable softmax | `MaxOfLanes` → subtract → `Exp` → `ReduceSum` → divide |
| `SimdRmsNorm` | RMS layer normalization | Vectorized square-sum + reciprocal sqrt |
| `SimdSilu` | SiLU activation (x·σ(x)) | Vectorized `1/(1+exp(-x))` |
| `SimdMul` / `SimdAdd` | Element-wise ops | Trivially vectorizable, 4-16x speedup |

## ThreadPool Design

**Why a custom pool instead of `std::async` or OpenMP:**
- `std::async` creates threads per call (overhead kills latency)
- OpenMP is a whole separate compiler extension with runtime dependencies
- Our pool pre-spawns workers and uses condition variable wakeup — minimal latency for fine-grained parallelism

**Key method: `ParallelFor(count, func)`** — divides `count` iterations evenly across threads, blocks until all complete. This is the exact pattern needed for GEMM row partitioning.

## Architectural Choice: Why Not Just Use BLAS?

We *could* link MKL or OpenBLAS for matmul. We didn't because:
1. **Learning:** This is a from-scratch engine; writing GEMM teaches cache hierarchy, SIMD, and parallelism
2. **Control:** We can fuse operations (e.g., matmul + bias + activation) without library boundaries
3. **Portability:** No external runtime dependencies beyond Highway (header-only)
4. **Quantization:** Custom kernels let us do mixed-precision GEMM (FP32 × INT8) which BLAS doesn't support
