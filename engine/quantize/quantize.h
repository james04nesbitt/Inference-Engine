#pragma once

#include <cstdint>
#include <vector>

#include "engine/tensor/tensor.h"

namespace ie {

// ============================================================================
// INT8 Quantization — 50% Memory Reduction
//
// Quantization maps floating-point values to lower-precision integers.
// For INT8: each float is mapped to an 8-bit integer [-128, 127] using:
//
//   quantized = clamp(round(float_val / scale) + zero_point, -128, 127)
//   dequantized = (quantized - zero_point) * scale
//
// Quantization granularity:
//   - Per-tensor: one scale for the entire tensor (simplest, least accurate)
//   - Per-channel: one scale per output channel (good balance)
//   - Per-token: one scale per token position (best for KV cache)
//
// Outlier-aware quantization:
//   Some transformer activations have outlier channels with 10-100x larger
//   magnitude than the rest. Naively quantizing these wastes most of the
//   INT8 range on a few outliers. Solutions:
//   1. Per-channel scaling (different scale per channel)
//   2. Mixed-precision: keep outlier channels in FP16, quantize the rest
//   3. SmoothQuant: redistribute outlier magnitude from activations to weights
//
// Papers to read:
//   - "SmoothQuant" (Xiao et al., 2022): https://arxiv.org/abs/2211.10438
//   - "LLM.int8()" (Dettmers et al., 2022): https://arxiv.org/abs/2208.07339
// ============================================================================

// Per-channel quantization parameters.
struct QuantizationParams {
  std::vector<float> scales;       // One scale per channel
  std::vector<int32_t> zero_points; // One zero-point per channel (often 0 for symmetric)
  int32_t num_channels;
  bool symmetric;                   // If true, zero_point is always 0
};

// A quantized tensor: INT8 data + metadata for dequantization.
struct QuantizedTensor {
  Tensor data;                      // DType::kInt8
  QuantizationParams params;
  std::vector<int64_t> original_shape;

  // Size in bytes (for memory accounting)
  size_t CompressedBytes() const;

  // Compression ratio vs FP32
  float CompressionRatio() const;
};

// --- Quantization Functions ---

// Compute per-channel quantization parameters.
// Analyzes the value distribution along `channel_dim` to find optimal scale.
//
// TODO: Implement
//   For each channel c:
//     max_val = max(abs(tensor[..., c, ...]))
//     scale = max_val / 127.0
//     zero_point = 0 (for symmetric)
//
QuantizationParams ComputeQuantParams(const Tensor& tensor,
                                       int32_t channel_dim,
                                       bool symmetric = true);

// Quantize a tensor to INT8 using the given parameters.
//
// TODO: Implement
//   For each element:
//     channel_idx = ... (depends on channel_dim)
//     quantized = clamp(round(value / scales[channel_idx]), -128, 127)
//
QuantizedTensor QuantizeInt8(const Tensor& tensor,
                             const QuantizationParams& params,
                             int32_t channel_dim);

// Dequantize back to float32.
//
// TODO: Implement
//   For each element:
//     value = (int8_val - zero_point) * scale
//
Tensor DequantizeInt8(const QuantizedTensor& qtensor);

// --- Outlier-Aware KV Cache Quantization ---

// Quantize KV cache tensors with outlier detection.
// Channels with values > outlier_threshold * median are kept in FP16.
//
// This is the key to maintaining <0.1% perplexity degradation while
// getting 50% memory reduction.
//
// TODO: Implement
//   1. Compute per-channel absmax
//   2. Identify outlier channels (absmax > threshold * median_absmax)
//   3. Quantize normal channels to INT8
//   4. Keep outlier channels in FP16
//   5. Return a mixed-precision representation
//
QuantizedTensor QuantizeKVCache(const Tensor& keys_or_values,
                                float outlier_threshold = 6.0f);

// --- Matmul with Quantized Operands ---

// Compute C = A_fp32 @ B_int8 (mixed-precision GEMM)
// Dequantizes B on-the-fly during computation to avoid materializing FP32 B.
//
// TODO: Implement (advanced — this is where the real savings happen)
//
Tensor QuantizedMatmul(const Tensor& a, const QuantizedTensor& b);

}  // namespace ie
