# Inference Engine

A standalone, high-performance C++20 inference runtime built from scratch for the Gemma-3 1B model. Engineered for maximum throughput and low latency, this engine leverages SIMD hardware acceleration, advanced memory management, and state-of-the-art attention mechanisms to deliver highly optimized LLM inference.

## Key Features

* **End-to-End Generation**: Full transformer forward pass implementation including Grouped-Query Attention (GQA), SwiGLU FFN, RoPE, and RMSNorm. Includes a GGUF binary weight loader, BPE tokenizer, and autoregressive generation with greedy, top-k, and top-p sampling.
* **SIMD-Accelerated Compute**: Custom GEMM/GEMV kernels optimized with Google Highway supporting AVX2, AVX-512, and NEON. Achieves up to 4x speedup over naive FP32 baselines through cache-aware tiling (64x64 blocks), software prefetching, and multi-threaded row partitioning via a custom thread pool.
* **Optimized Attention Memory**: Integrates FlashAttention (tiled Q×K with online softmax) and a PagedAttention-style KV cache manager. Features block-level allocation, per-sequence page tables, and copy-on-write support to reduce attention memory complexity to O(N) and boost multi-sequence throughput under heavy load.
* **INT8 KV Cache Quantization**: Outlier-aware INT8 quantization with per-channel scaling and mixed-precision handling (FP16 for outlier channels, INT8 for the rest). Reduces the KV cache memory footprint by ~50% with less than 0.5% perplexity degradation.
* **Continuous Batching**: Round-robin batch scheduler with dynamic request management, per-sequence paged KV cache allocation, and streaming token callbacks for concurrent multi-request serving.


## Architecture

```
├── engine/
│   ├── tensor/           # Custom Tensor class (shape, strides, memory)
│   ├── ops/              # Math kernels (matmul, softmax, RMSNorm, RoPE, SiLU)
│   ├── compute/          # Accelerated backends
│   │   ├── simd_kernels  # Highway SIMD kernels (AVX2/AVX-512/NEON)
│   │   └── thread_pool   # Lock-free thread pool (std::jthread)
│   ├── attention/        # Advanced attention mechanisms
│   │   ├── kv_cache      # PagedAttention KV cache manager
│   │   └── flash_attention # FlashAttention (tiled + fused)
│   ├── quantize/         # INT8 quantization (per-channel, outlier-aware)
│   ├── gguf/             # GGUF file parser (binary I/O, weight loading)
│   ├── tokenizer/        # BPE tokenizer (encode/decode text ↔ token IDs)
│   ├── model/            # Transformer layers (Attention, FFN, GemmaModel)
│   ├── scheduler/        # Continuous batching (BatchScheduler)
│   ├── bench/            # Microbenchmarks (GEMM, GEMV, dot product)
│   ├── engine.h/cc       # Top-level inference engine (load, generate)
│   └── main.cc           # CLI entry point
└── model/                # Model weights (GGUF files, gitignored)
```

## Build & Run

```bash
# Build everything
bazel build //...

# Run tests
bazel test //...

# Run inference (CLI)
bazel run //engine:inference -- --model model/gemma-3-1b-it-f16.gguf --prompt "Hello"

# Interactive mode
bazel run //engine:inference -- --model model/gemma-3-1b-it-f16.gguf --interactive

# With sampling options
bazel run //engine:inference -- --model model/gemma-3-1b-it-f16.gguf --prompt "Hello" --temperature 0.8 --top-k 50

# Release build (with optimizations)
bazel build --config=release //...

# AVX2/native SIMD build
bazel build --config=avx2 //...

# Run GEMM benchmarks
bazel run --config=release //engine/bench:matmul_bench
```

## Tech Stack

- **C++20** with modern idioms (RAII, move semantics, `std::variant`, `std::shared_ptr`, `std::jthread`)
- **Bazel** (bzlmod) build system
- **Google Highway** for portable SIMD intrinsics
- **GoogleTest** for unit testing
- **Google Benchmark** for kernel microbenchmarking
- **Abseil** for utilities (`absl::Span`, `absl::StatusOr`)

---

## Performance Benchmarks

The engine has been heavily optimized for throughput and latency, leveraging MSVC AVX2 compilation, locked memory caching, and zero-allocation hot paths.

### End-To-End Inference Throughput (Gemma-3 1B)
- **Prefill Speed:** ~9.3 tokens/sec (Time-to-First-Token: 440ms for 4 tokens, 1727ms for 16 tokens)
- **Decode Speed:** **7.63 tokens/sec**
- **Per-Layer Average Latency:** 5.04 ms
- **Full Forward Pass:** 131.0 ms

### SIMD Compute & Matmul
Highway SIMD + Threaded kernels show massive speedups over scalar baselines.
- **Large GEMM (1024x1024x1024):** 21.67 ms (99.09 GFLOPS) — *27.9x Speedup*
- **Q/K/V Projection (16x1152x1024):** 0.49 ms (77.53 GFLOPS) — *22x Speedup*

### FlashAttention & INT8 KV Cache
- **Memory Footprint (SeqLen 2048):**
  - FP32: 104.00 MB
  - INT8: **26.13 MB** *(74.9% Savings)*
- **Flash vs Naive Latency (SeqLen 512):** Flash is 1.1x faster with a 3.0x peak memory reduction.
- **KV Append Overhead:** 4.5 µs/token (220k tokens/sec)

---

## Gemma 3 1B Architecture Reference

| Parameter | Value |
|-----------|-------|
| Layers | 26 |
| Embedding dim | 1152 |
| Attention heads | 4 |
| KV heads (GQA) | 1 |
| Head dim | 256 |
| FFN hidden dim | 6912 |
| Vocab size | 262,144 |
| Max seq length | 32,768 |

## Resources

- [GGUF Spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Reference implementation
- [Gemma 3 Technical Report](https://ai.google.dev/gemma)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoPE Paper](https://arxiv.org/abs/2104.09864)
- [GQA Paper](https://arxiv.org/abs/2305.13245)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135) — Memory-efficient attention
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) — Improved tiling
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180) — Paged KV cache (vLLM)
- [SmoothQuant Paper](https://arxiv.org/abs/2211.10438) — Outlier-aware quantization
- [LLM.int8() Paper](https://arxiv.org/abs/2208.07339) — Mixed-precision inference
- [Google Highway](https://github.com/google/highway) — Portable SIMD
