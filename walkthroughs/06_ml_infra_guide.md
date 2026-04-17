# ML Infrastructure: Profiling and Economics

## Overview
Understanding how an inference engine works line-by-line is important, but as an ML Infra Engineer, your job is to view the entire engine as a machine where **Latency** and **Throughput** are the primary outputs, constrained by **Compute** and **Memory Bandwidth**.

This guide synthesizes the code into practical infrastructural profiling.

## Key Inference Metrics

When evaluating a deployment of this inference engine, track these core metrics:

1. **TTFT (Time-to-First-Token)**  
   *What is it:* The time from when the user sends a prompt until the first word is generated.  
   *Engine Phase:* **Prefill**. The model processes `Prompt_Length` tokens simultaneously.  
   *Hardware Bound:* **Compute-bound**. It depends heavily on how fast the CPU/GPU can do Matrix Multiplications (GEMM).  
   *Optimizations:* SIMD vectorization, multi-threading (`ThreadPool`), AVX-512.

2. **TPOT (Time-per-Output-Token)**  
   *What is it:* The time taken to generate each subsequent token.  
   *Engine Phase:* **Decode**. The model processes exactly `1` token, reading from the KV Cache.  
   *Hardware Bound:* **Memory-bound**. It requires loading the entire model's weights from DRAM to the CPU/GPU registers to perform a matrix-vector multiply (GEMV). The speed limit is the memory bus (e.g., DDR5 bandwidth at ~80 GB/s).  
   *Optimizations:* Model Quantization (loading fewer bytes), KV Cache compression/PagedAttention, Continuous Batching.

3. **Total Latency**  
   *Formula:* `Latency = TTFT + (TPOT × (Generated_Tokens - 1))`

4. **Throughput (Tokens per Second)**  
   *What is it:* The total number of tokens the server can generate across all concurrent users.  
   *Why it matters:* This determines Cost. High throughput means you can cram more users onto the same server, amortizing the hardware cost.

## Determining Your Bottleneck (Roofline)

If you modify the engine and want to measure the impact, use the provided `inference_bench` tool:
```bash
bazel run --config=release //engine/bench:inference_bench
```

This acts as a mini-profiler. If your **Prefill Tok/sec** is low, but your **Decode Tokens/sec** is high, you have a compute problem. You should look at `engine/compute/simd_kernels.cc` to see if your AVX instructions are compiling correctly.

If your **Decode Tokens/sec** is low, you are starved for Memory Bandwidth. In this engine:
- FP32 1B Decode: ~10 Tokens/Sec (loads ~4.4GB per step)
- INT8 1B Decode: ~40 Tokens/Sec (loads ~1.1GB per step)

If you see decode speeds below these ranges on modern hardware, your quantization scales might be failing, or you are accidentally materializing tensors rather than streaming them into L1 cache blocks.

## Handling Hardware Costs

When deploying an Inference Engine to production, memory dictates cost.

### 1. Weights Memory
A model's weights must stay resident in RAM.
- **Rule of Thumb:** `Params × Precision_Bytes = VRAM_Required`.
- Gemma-3 1B in FP32 = 1.15 Billion × 4 bytes = 4.6 GB.
- Gemma-3 1B in INT8 = 1.15 Billion × 1 byte = 1.15 GB.

### 2. KV Cache Memory
Every active sequence stores its history in the KV Cache.
- **Single Token Size:** `2 × Layers × KV_Heads × Head_Dim × Bytes`.
- If you use `batch_scheduler.cc` to manage 100 concurrent requests, you might need 2-5 GB of dedicated KV Cache RAM. If the system starts crashing (OOM), you must tune `max_blocks` in the KVCacheManager down.

**The Economic Trade-off:** 
You want to maximize batch size to increase Throughput. But increasing batch size linearly increases KV Cache RAM usage. Once the KV Cache fills out your remaining RAM (after Model Weights are loaded), you hit the wall. You cannot add more users.
This is why techniques like GQA (Grouped Query Attention) exist in Gemma models—it significantly shrinks the KV Cache size by making 4 attention heads share a single K and V state.

## Where to Investigate Next
To become highly proficient in this codebase:
1. Try adding a new configuration constraint in `engine/scheduler/batch_scheduler.cc` to reject requests if `active_sequences` * `average_length` exceeds KV Cache limits.
2. Read the `.bazelrc` release flags to understand how the compiler links the high-performance math operators to the target architecture.
