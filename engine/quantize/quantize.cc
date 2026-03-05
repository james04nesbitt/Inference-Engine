#include "engine/quantize/quantize.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace ie {

size_t QuantizedTensor::CompressedBytes() const {
  return data.nbytes() + params.scales.size() * sizeof(float) +
         params.zero_points.size() * sizeof(int32_t);
}

float QuantizedTensor::CompressionRatio() const {
  size_t original_bytes = 1;
  for (auto dim : original_shape)
    original_bytes *= dim;
  original_bytes *= sizeof(float); // Assuming FP32 original
  return static_cast<float>(original_bytes) /
         static_cast<float>(CompressedBytes());
}

// ============================================================================
// ComputeQuantParams — Per-channel quantization parameter computation
// ============================================================================
QuantizationParams ComputeQuantParams(const Tensor &tensor, int32_t channel_dim,
                                      bool symmetric) {
  if (tensor.ndim() == 0) {
    throw std::runtime_error("ComputeQuantParams: empty tensor");
  }

  // Normalize negative dim.
  int32_t ndim = static_cast<int32_t>(tensor.ndim());
  if (channel_dim < 0)
    channel_dim += ndim;
  if (channel_dim < 0 || channel_dim >= ndim) {
    throw std::runtime_error("ComputeQuantParams: invalid channel_dim");
  }

  int32_t num_channels = static_cast<int32_t>(tensor.size(channel_dim));
  int64_t total_elements = tensor.numel();
  int64_t elements_per_channel = total_elements / num_channels;

  // Compute stride for the channel dimension.
  int64_t channel_stride = 1;
  for (int32_t d = channel_dim + 1; d < ndim; ++d) {
    channel_stride *= tensor.size(d);
  }
  int64_t outer_stride = channel_stride * num_channels;

  // Work on contiguous data.
  Tensor t = tensor.contiguous();
  const float *data = t.data<float>();

  QuantizationParams params;
  params.num_channels = num_channels;
  params.symmetric = symmetric;
  params.scales.resize(num_channels);
  params.zero_points.resize(num_channels, 0);

  if (symmetric) {
    // For each channel, find the max absolute value.
    for (int32_t c = 0; c < num_channels; ++c) {
      float max_abs = 0.0f;
      // Iterate over all elements belonging to this channel.
      for (int64_t outer = 0; outer < total_elements / outer_stride; ++outer) {
        for (int64_t inner = 0; inner < channel_stride; ++inner) {
          int64_t idx = outer * outer_stride + c * channel_stride + inner;
          float val = std::abs(data[idx]);
          if (val > max_abs)
            max_abs = val;
        }
      }
      // scale = max_abs / 127; avoid division by zero.
      params.scales[c] = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
      params.zero_points[c] = 0;
    }
  } else {
    // Asymmetric: find min and max per channel.
    for (int32_t c = 0; c < num_channels; ++c) {
      float min_val = data[c * channel_stride];
      float max_val = min_val;
      for (int64_t outer = 0; outer < total_elements / outer_stride; ++outer) {
        for (int64_t inner = 0; inner < channel_stride; ++inner) {
          int64_t idx = outer * outer_stride + c * channel_stride + inner;
          float val = data[idx];
          if (val < min_val)
            min_val = val;
          if (val > max_val)
            max_val = val;
        }
      }
      float range = max_val - min_val;
      params.scales[c] = (range > 0.0f) ? (range / 255.0f) : 1.0f;
      params.zero_points[c] =
          static_cast<int32_t>(std::round(-min_val / params.scales[c]));
    }
  }

  return params;
}

// ============================================================================
// QuantizeInt8 — FP32 → INT8 with per-channel scaling
// ============================================================================
QuantizedTensor QuantizeInt8(const Tensor &tensor,
                             const QuantizationParams &params,
                             int32_t channel_dim) {
  int32_t ndim = static_cast<int32_t>(tensor.ndim());
  if (channel_dim < 0)
    channel_dim += ndim;

  Tensor t = tensor.contiguous();
  const float *src = t.data<float>();
  int64_t total = tensor.numel();

  // Compute channel stride.
  int64_t channel_stride = 1;
  for (int32_t d = channel_dim + 1; d < ndim; ++d) {
    channel_stride *= tensor.size(d);
  }
  int64_t outer_stride = channel_stride * tensor.size(channel_dim);

  // Allocate INT8 output tensor.
  Tensor out(tensor.shape(), DType::kInt8);
  int8_t *dst = out.data<int8_t>();

  for (int64_t i = 0; i < total; ++i) {
    // Determine which channel this element belongs to.
    int32_t channel_idx =
        static_cast<int32_t>((i % outer_stride) / channel_stride);
    float scale = params.scales[channel_idx];
    int32_t zp = params.zero_points[channel_idx];

    // Quantize: round(value / scale) + zero_point, clamped to [-128, 127].
    int32_t q = static_cast<int32_t>(std::round(src[i] / scale)) + zp;
    q = std::max(-128, std::min(127, q));
    dst[i] = static_cast<int8_t>(q);
  }

  QuantizedTensor result;
  result.data = std::move(out);
  result.params = params;
  result.original_shape = tensor.shape();
  return result;
}

// ============================================================================
// DequantizeInt8 — INT8 → FP32 reconstruction
// ============================================================================
Tensor DequantizeInt8(const QuantizedTensor &qtensor) {
  const int8_t *src = qtensor.data.data<int8_t>();
  Tensor out(qtensor.original_shape, DType::kFloat32);
  float *dst = out.data<float>();

  int64_t total = qtensor.data.numel();
  int32_t num_channels = qtensor.params.num_channels;

  // Determine channel stride from original shape.
  // The channel dimension is assumed to be the first dimension (dim 0).
  int64_t channel_stride = total / num_channels;

  for (int64_t i = 0; i < total; ++i) {
    int32_t channel_idx = static_cast<int32_t>(i / channel_stride);
    if (channel_idx >= num_channels)
      channel_idx = num_channels - 1;

    float scale = qtensor.params.scales[channel_idx];
    int32_t zp = qtensor.params.zero_points[channel_idx];

    dst[i] = (static_cast<float>(src[i]) - static_cast<float>(zp)) * scale;
  }

  return out;
}

// ============================================================================
// QuantizeKVCache — Outlier-aware mixed-precision quantization
// ============================================================================
QuantizedTensor QuantizeKVCache(const Tensor &keys_or_values,
                                float outlier_threshold) {
  // KV cache shape: [seq_len, num_kv_heads, head_dim]
  // We quantize per-channel along the last dimension (head_dim).
  // Outlier channels (abs > threshold * median) are handled specially.

  Tensor t = keys_or_values.contiguous();
  const float *data = t.data<float>();

  int32_t ndim = static_cast<int32_t>(t.ndim());
  int32_t last_dim = static_cast<int32_t>(t.size(ndim - 1));
  int64_t total = t.numel();
  int64_t num_rows = total / last_dim;

  // Step 1: Compute per-channel absolute max.
  std::vector<float> absmax(last_dim, 0.0f);
  for (int64_t row = 0; row < num_rows; ++row) {
    for (int32_t c = 0; c < last_dim; ++c) {
      float val = std::abs(data[row * last_dim + c]);
      if (val > absmax[c])
        absmax[c] = val;
    }
  }

  // Step 2: Compute median of absmax values to identify outliers.
  std::vector<float> sorted_absmax = absmax;
  std::sort(sorted_absmax.begin(), sorted_absmax.end());
  float median = sorted_absmax[last_dim / 2];

  // Step 3: Identify outlier channels.
  std::vector<bool> is_outlier(last_dim, false);
  int32_t num_outliers = 0;
  for (int32_t c = 0; c < last_dim; ++c) {
    if (absmax[c] > outlier_threshold * median) {
      is_outlier[c] = true;
      num_outliers++;
    }
  }

  // Step 4: Compute per-channel quantization params (excluding outliers).
  // For outlier channels, scale is set to absmax so they're in the valid range
  // but with reduced precision. In a real implementation, these would be
  // stored in FP16 separately.
  QuantizationParams params;
  params.num_channels = last_dim;
  params.symmetric = true;
  params.scales.resize(last_dim);
  params.zero_points.resize(last_dim, 0);

  for (int32_t c = 0; c < last_dim; ++c) {
    params.scales[c] = (absmax[c] > 0.0f) ? (absmax[c] / 127.0f) : 1.0f;
  }

  // Step 5: Quantize all channels to INT8.
  // (In a production system, outlier channels would be kept in FP16
  // separately.)
  return QuantizeInt8(t, params, ndim - 1);
}

// ============================================================================
// QuantizedMatmul — Mixed-precision GEMM with on-the-fly dequantization
// ============================================================================
// C = A_fp32 @ B_int8 (dequantized on-the-fly)
// C[i][j] = scale[j] * sum_k A[i][k] * B_int8[k][j]
Tensor QuantizedMatmul(const Tensor &a, const QuantizedTensor &b) {
  // a: [M, K] float32
  // b: [K, N] int8 (quantized along dim 0, i.e., K channels)
  if (a.ndim() != 2 || b.data.ndim() != 2) {
    throw std::runtime_error("QuantizedMatmul: inputs must be 2D");
  }

  int64_t M = a.size(0);
  int64_t K = a.size(1);
  int64_t N = b.data.size(1);

  if (K != b.data.size(0)) {
    throw std::runtime_error("QuantizedMatmul: inner dimensions must match");
  }

  Tensor a_c = a.contiguous();
  const float *a_data = a_c.data<float>();
  const int8_t *b_data = b.data.data<int8_t>();

  Tensor out({M, N}, DType::kFloat32);
  float *c_data = out.data<float>();

  // Determine how scales map. Per-channel quantization along dim 0 means
  // each row of B has its own scale. But for matmul, we want per-column
  // (output channel) scaling. So we treat it as:
  // C[i][j] = sum_k A[i][k] * ((B_int8[k][j] - zp[k]) * scale[k])
  //
  // For symmetric quantization (zp=0):
  // C[i][j] = sum_k A[i][k] * B_int8[k][j] * scale[k]

  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        float scale = b.params.scales[k % b.params.num_channels];
        int32_t zp = b.params.zero_points[k % b.params.num_channels];
        float b_val =
            (static_cast<float>(b_data[k * N + j]) - static_cast<float>(zp)) *
            scale;
        sum += a_data[i * K + k] * b_val;
      }
      c_data[i * N + j] = sum;
    }
  }

  return out;
}

} // namespace ie
