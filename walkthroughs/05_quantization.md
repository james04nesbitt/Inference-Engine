# Quantization

## What This Does

The `engine/quantize/` directory implements INT8 quantization for reducing memory footprint by ~50% with minimal accuracy loss. The key innovation is **outlier-aware KV cache quantization**: channels with disproportionately large values stay in high precision while the rest are quantized.

## Architecture

```
quantize.h / quantize.cc
  ├── ComputeQuantParams()  — analyze tensor → per-channel scales
  ├── QuantizeInt8()         — FP32 → INT8 with scaling
  ├── DequantizeInt8()       — INT8 → FP32 reconstruction
  ├── QuantizeKVCache()      — outlier-aware mixed-precision
  └── QuantizedMatmul()      — FP32 × INT8 on-the-fly GEMM

GGUF Q8_0/Q4_0 support (in gguf_loader.cc):
  └── Dequantize block-quantized weights at load time
```

## Key Files

| File | Purpose |
|------|---------|
| [quantize.h](../engine/quantize/quantize.h) | Quantization API + data structures |
| [quantize.cc](../engine/quantize/quantize.cc) | All quantization implementations |
| [gguf_loader.cc](../engine/gguf/gguf_loader.cc) | GGUF Q8_0/Q4_0 dequantization |

---

## Per-Channel INT8 Quantization

### The Math

Quantization maps a float value to an 8-bit integer:

```
quantized_value = clamp(round(float_value / scale), -128, 127)
dequantized_value = quantized_value × scale
```

The `scale` determines the range: `scale = max(|values|) / 127`.

### Why Per-Channel (Not Per-Tensor)?

**Per-tensor** uses one scale for the entire tensor. If one channel has values in [-100, 100] and another in [-0.01, 0.01], the small channel gets quantized to {0, 0, 0, ...} — all information is lost.

**Per-channel** uses one scale per output channel. Each channel uses its full INT8 range independently.

```
Channel 0: values in [-100, 100] → scale = 100/127 ≈ 0.787
Channel 1: values in [-0.01, 0.01] → scale = 0.01/127 ≈ 0.0000787
```

**Trade-off:** Per-channel needs to store `num_channels` scales (typically 256-1152 floats = 1-4.5KB). Negligible overhead for 50% memory savings on the actual data.

### Symmetric vs Asymmetric

We use **symmetric** quantization (zero_point = 0):
```
quantized = round(value / scale)     // symmetric
quantized = round(value / scale) + zero_point  // asymmetric
```

**Why symmetric:** Transformer weights and activations are roughly centered around zero. Asymmetric adds a bias correction term to every computation, which complicates the matmul kernel for marginal benefit.

---

## Outlier-Aware KV Cache Quantization

### The Problem

Transformer activations have a well-documented phenomenon: a small number of channels (~1-5%) have values 10-100x larger than the rest. These are called **outlier channels**. If you quantize them with the same scale as normal channels, either:
- The outliers clip (lose information) → accuracy degrades
- Everything else gets a tiny scale (wasteful range) → quantization noise increases

### The Solution: Mixed-Precision

```
1. Compute per-channel absolute max values
2. Find the median absolute max across all channels
3. Channels with absmax > threshold × median are "outliers"
4. Normal channels → INT8 (per-channel scaled)
5. Outlier channels → kept in original FP32 precision
```

**Default threshold = 6.0×** — empirically, channels with >6× the median magnitude are outliers. This typically identifies 1-3% of channels.

### Memory Accounting

```
Normal channels: INT8 = 1 byte per value (was 4 bytes) → 75% reduction
Outlier channels: still FP32 = 4 bytes per value → 0% reduction

Overall: ~97% of channels at 75% reduction + ~3% unchanged
        ≈ 50% total memory reduction
```

The `QuantizedTensor` struct tracks compressed bytes and reports compression ratio:
```cpp
size_t CompressedBytes();   // actual memory used
float CompressionRatio();   // original_size / compressed_size
```

### Why This Preserves Accuracy

The core insight from the LLM.int8() paper: outlier channels carry disproportionate information. By keeping them in full precision, we preserve the most important signal. The 97% of normal channels can tolerate INT8's ~0.4% relative error because they're small-magnitude and somewhat redundant.

---

## Quantized Matmul

### What: FP32 × INT8 GEMM

```
C[i][j] = Σ_k  A[i][k] × dequant(B_int8[k][j])
        = Σ_k  A[i][k] × (B_int8[k][j] × scale[j])
        = scale[j] × Σ_k  A[i][k] × B_int8[k][j]
```

**Key optimization:** We factor out `scale[j]` and compute the integer dot product first, then multiply by the scale once. This avoids per-element dequantization.

### Why On-the-Fly Dequantization?

Two approaches:
1. **Materialize:** Dequantize entire B to FP32, then do normal GEMM → uses full FP32 memory (defeats the purpose)
2. **On-the-fly:** Dequantize within the GEMM inner loop → stays at INT8 memory, FP32 compute

We use on-the-fly. The INT8→FP32 conversion is a single instruction on modern CPUs (`vcvtdq2ps` on AVX2), so the overhead is negligible compared to the memory bandwidth savings.

---

## GGUF Quantized Format Support

The GGUF file format supports block-quantized types that we dequantize at model load time:

### Q8_0 (8-bit block quantization)
```
Block of 32 values:
  - 1 FP16 scale factor (2 bytes)
  - 32 INT8 values (32 bytes)
  Total: 34 bytes for 32 values = 1.0625 bytes/value
```

### Q4_0 (4-bit block quantization)
```
Block of 32 values:
  - 1 FP16 scale factor (2 bytes)
  - 32 4-bit values packed into 16 bytes
  Total: 18 bytes for 32 values = 0.5625 bytes/value
```

**Why dequantize at load time?** For this engine, we prioritize inference speed over model loading speed. Dequantizing to FP32 at load time lets us use our optimized FP32 GEMM kernels without per-inference dequantization overhead. A production engine might keep weights quantized and use specialized INT4/INT8 GEMM kernels, but that adds significant kernel complexity.

## Architectural Choices Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Quantization granularity | Per-channel | Best accuracy/overhead trade-off |
| Symmetry | Symmetric (zero_point=0) | Simpler matmul, minimal accuracy loss |
| Outlier handling | Mixed-precision (INT8 + FP32) | Preserves critical channels, ~50% savings |
| GGUF quant types | Dequant at load time | Simpler inference path, reuses FP32 kernels |
| QuantizedMatmul | On-the-fly dequant | Preserves memory savings during compute |

## Cost & Infrastructure Perspective

### Memory-Bound Economics
As an ML Infra Engineer, the single most important math equation you will use is calculating your inference engine's "Roofline". 
During Decode, the time taken is:
`Decode Time = Max(FLOPs / Compute_Bandwidth, Bytes / Memory_Bandwidth)`

For an FP32 1B Model (~4.4 GB) on a processor with 50 GB/s bandwidth:
`Decode Time = 4.4 GB / 50 GB/s = 88 ms per token (11.3 Tokens/sec)`
If you quantize to INT8 (1.1 GB):
`Decode Time = 1.1 GB / 50 GB/s = 22 ms per token (45.4 Tokens/sec)`

You get a 4x latency reduction literally for free because the execution was starving for bytes. 

### W8A16 vs W8A8 True Limits
- **W8A16** (Weight 8-bit, Activation FP16/FP32): This is what we implement in `QuantizedMatmul`. Weights are loaded as INT8 (lowering memory bandwidth demands), but the dot-product math is done in FP32. This is extremely effective for the Decode phase.
- **W8A8** (Both Weights and Activations 8-bit): Used in heavily Compute-bound scenarios (like the Prefill phase of massive prompts). Since our primary bottleneck for local execution is memory (Decode phase), W8A16 gets us 95% of the performance gains without the harsh accuracy penalty and complex activation scaling factors of W8A8.

## Developer Guide: Adding a New GGUF Quantization Format

Suppose you want to support downloading a `Q4_K` model from Hugging Face instead of the basic `Q8_0` format.

### Step 1: Understand the Block Structure
GGUF `Q4_K` (K-quants) uses sub-blocks. 256 values are grouped together, with a global scale, and then further subdivided into smaller blocks with their own local scales/mins.

### Step 2: Add the Enum
In `engine/gguf/gguf_loader.h`, add the struct for the GGUF spec:
```cpp
#pragma pack(push, 1)
struct BlockQ4_K { // Exact specs from llama.cpp
    uint8_t scales[12];
    uint8_t qs[128];
    // etc... based on K-quant specification
};
#pragma pack(pop)
```

### Step 3: Implement the Dequantize Kernel
In `engine/gguf/gguf_loader.cc`, write the kernel to turn these packed 4-bit values back into FP32 during load time:
```cpp
void DequantizeQ4_K(const void* src, float* dst, int64_t num_elements) {
    const BlockQ4_K* blocks = static_cast<const BlockQ4_K*>(src);
    int num_blocks = num_elements / 256; // Q4_K block size
    for (int i = 0; i < num_blocks; ++i) {
        // ... unpack the 4-bit values and multiply by the two-level scales... 
        // dst[i * 256 + j] = ...
    }
}
```

### Step 4: Hook it up to the loader
Inside `ParseTensorData()` in the same file:
```cpp
if (type == 12) { // GGUF type ID for Q4_K
    DequantizeQ4_K(raw_data, t.data<float>(), num_elements);
    return t; // Successfully returned an FP32 contiguous representation
}
```
*Note:* Since we dequantize at load time, you don't need to write new Matrix Math operations! The existing `ops::matmul` will just process the resultant FP32 tensor normally.
