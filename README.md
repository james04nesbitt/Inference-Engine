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

## 🗺️ Implementation Roadmap

### Pillar 1: High-Performance Kernels

Standalone C++ inference runtime for Gemma-3 using Google Highway SIMD. Cache-aware GEMM/GEMV kernels with tiling and prefetching to maximize L2 residency.

#### Phase 1A: Foundations
- [x] **Tensor `at()` / `set()`** — Stride-based element access with bounds checking
- [x] **Tensor `reshape()` / `view()`** — Shape manipulation (share data via `shared_ptr`)
- [x] **GGUF metadata parser** — Read key-value pairs from binary GGUF
- [x] **GGUF tensor info parser** — Read tensor names, shapes, offsets
- [x] **`ops::add()`, `ops::mul()`** — Element-wise operations with broadcasting

#### Phase 1B: Core Kernels
- [x] **`ops::matmul()` — Naive** — i,j,k triple loop (correct baseline)
- [x] **`ops::matmul()` — Cache-friendly** — Reorder to i,k,j for sequential B access
- [x] **`ops::matmul()` — Tiled** — L2-aware blocking (64×64 tiles for 256KB L2)
- [x] **`ops::matmul()` — SIMD** — Highway vectorized inner loop (`hn::MulAdd`)
- [x] **`ops::matmul()` — Multi-threaded** — Partition M dimension across thread pool
- [x] **`ops::rms_norm()`** — RMS normalization
- [x] **`ops::silu()`** — SiLU activation
- [x] **`ops::softmax()`** — Numerically stable softmax
- [x] **`ops::embedding()`** — Embedding table lookup
- [x] **`ops::rope()`** — Rotary positional embeddings
- [x] **Benchmark GEMM** — Measure tokens/sec at each optimization stage

#### Phase 1C: Model Forward Pass
- [x] **GGUF tensor loading** — Load F16/F32 weight data into Tensors
- [x] **Build tokenizer from GGUF** — Extract vocab, scores, special tokens
- [x] **`RMSNorm::forward()`** — First layer implementation
- [x] **`FeedForward::forward()`** — SwiGLU FFN
- [x] **`Attention::forward()`** — Multi-head attention with GQA
- [x] **`TransformerBlock::forward()`** — Attention + FFN + residuals
- [x] **`GemmaModel::forward()`** — Full model: embed → blocks → logits

#### Phase 1D: Generation
- [x] **Greedy sampling** — argmax over logits
- [x] **Temperature + Top-K/Top-P sampling**
- [x] **Autoregressive loop** — Generate tokens one at a time
- [x] **Basic KV-cache** — Avoid recomputing attention for past tokens

---

### Pillar 2: Memory Scheduling & Batching

PagedAttention-style KV cache manager with continuous batching and dynamic request scheduling, eliminating pipeline bubbles and maximizing throughput.

#### Phase 2A: Paged KV Cache
- [x] **KV cache block allocator** — Fixed-size page pool with free list
- [x] **Block table manager** — Per-sequence logical→physical page mapping
- [x] **Append/lookup operations** — Efficient page-table-based KV access
- [x] **Copy-on-write** — Enable beam search without KV duplication

#### Phase 2B: FlashAttention
- [x] **Tiled attention** — Block-level Q×K computation in SRAM
- [x] **Online softmax** — Incremental normalization without full attention matrix
- [x] **Causal masking** — Efficient tile-level mask application
- [x] **Fused RoPE** — Apply rotary embeddings inside the tiling loop

#### Phase 2C: Scheduling & Batching
- [x] **Request queue** — Priority-based scheduling with fairness
- [x] **Dynamic batch formation** — Iteration-level batching of sequences
- [x] **Preemption** — Pause/resume sequences when memory pressure is high
- [x] **Pipeline bubble elimination** — Keep GPU/CPU saturated across batches

---

### Pillar 3: Quantization

Outlier-aware INT8 KV cache quantization with per-channel scaling, reducing memory footprint by 50% while maintaining <0.1% perplexity degradation.

#### Phase 3A: INT8 Fundamentals
- [x] **Per-channel scale computation** — Analyze weight/activation distributions
- [x] **Quantize/dequantize** — INT8 ↔ FP32 conversion with scale + zero_point
- [x] **Quantized matmul** — Mixed-precision GEMM (FP32 × INT8)
- [x] **Accuracy validation** — Compare quantized vs FP32 outputs

#### Phase 3B: KV Cache Quantization
- [x] **Outlier detection** — Identify channels with disproportionately large values
- [x] **Mixed-precision KV cache** — INT8 for normal channels, FP16 for outliers
- [x] **Perplexity benchmarking** — Measure degradation on validation set
- [x] **Memory accounting** — Track and report compression ratios

#### Phase 3C: Advanced Quantization
- [x] **GGUF quantized format support** — Q4_0, Q8_0 dequantization
- [ ] **SmoothQuant** — Redistribute outlier magnitude from activations to weights
- [ ] **AVX-512 VNNI** — INT8 dot product instructions for 2x GEMM throughput

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
