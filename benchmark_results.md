# Inference Engine Benchmark Results

Here are the collected benchmark numbers for the engine components, running with fully optimized AVX2 SIMD instructions and cached weight transpositions.

## 1. End-To-End Inference Throughput
Through massive optimization of memory allocations (eliminating 15M+ vector allocations and 700MB+ copies *per token*) and enabling MSVC AVX2 compilation, inference speed has increased **>120x**.

- **Prefill Throughput (Time-to-First-Token)**
  - Prompt Len 4: 440.2 ms
  - Prompt Len 16: 1727.4 ms
  - Speed: **~9.3 tokens/sec**

- **Decode Throughput (Tokens/sec)**
  - Speed: **7.63 tokens/sec**
  - Per-Layer Average Latency: **5.04 ms**
  - Full Forward Pass: **131.0 ms**

## 2. SIMD Compute & Matmul

The SIMD + Threaded kernels show massive speedups over the naive baseline.

- **Large GEMM (1024x1024x1024):** 
  - Naive: 604.61 ms (3.55 GFLOPS)
  - SIMD (Highway): 116.45 ms (18.44 GFLOPS)
  - **SIMD + Threaded:** 21.67 ms (**99.09 GFLOPS**) — **27.9x Speedup**

- **Q/K/V Projection (16 tokens: 16x1152x1024):**
  - Naive: 10.74 ms
  - **SIMD + Threaded:** 0.49 ms (**77.53 GFLOPS**) — **22x Speedup**

## 3. FlashAttention

FlashAttention reduces memory footprint significantly, especially at larger sequence lengths, and speeds up computation.

- **Sequence Length 512:**
  - Naive Latency: 318.02 ms
  - **Flash Latency:** 287.98 ms (**1.10x Speedup**)
  - Naive Peak Memory: 6.00 MB
  - **Flash Peak Memory:** 2.00 MB (**3.0x Reduction**)
- **Correctness:** Max error vs Naive is `2.98e-08`.

## 4. Paged KV Cache

The KV Cache Manager handles sequences with minimal overhead.

- **Sequence Allocation:** 4.43 µs/seq (225k allocs/sec)
- **Single-Sequence Append:** 4.5 µs/token (220k tokens/sec)
- **Multi-Sequence (16 seqs, 64 tokens):** 14.77 ms (69k tokens/sec)

## 5. INT8 Quantization

Quantization yields massive memory savings for the KV Cache.

- **KV Cache Memory Footprint (Gemma-3 1B, 2048 Tokens):**
  - FP32: 104.00 MB
  - **INT8: 26.13 MB**
  - **Savings: 74.9% (77.87 MB)**
