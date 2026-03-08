#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "engine/quantize/quantize.h"
#include "engine/tensor/tensor.h"

// ============================================================================
// INT8 Quantization Benchmark Suite
//
// Measures quantization throughput, compression ratios, round-trip accuracy,
// and quantized matmul performance vs FP32.
//
// Run with: bazel run --config=release //engine/bench:quantize_bench
// ============================================================================

// Helper: fill tensor with random data, optionally with outlier channels.
static void FillRandom(ie::Tensor &t, std::mt19937 &rng,
                       float outlier_fraction = 0.0f,
                       float outlier_magnitude = 50.0f) {
  float *data = t.data<float>();
  int64_t total = t.numel();
  int64_t last_dim = t.size(t.ndim() - 1);
  int64_t num_rows = total / last_dim;

  std::normal_distribution<float> normal(0.0f, 1.0f);

  // Determine outlier channels.
  int32_t num_outlier = static_cast<int32_t>(last_dim * outlier_fraction);
  std::vector<bool> is_outlier(last_dim, false);
  for (int32_t c = 0; c < num_outlier; ++c) {
    is_outlier[c] = true; // First N channels are outliers.
  }

  for (int64_t row = 0; row < num_rows; ++row) {
    for (int64_t col = 0; col < last_dim; ++col) {
      float val = normal(rng);
      if (is_outlier[col]) {
        val *= outlier_magnitude;
      }
      data[row * last_dim + col] = val;
    }
  }
}

// Naive FP32 matmul for comparison.
static ie::Tensor NaiveMatmul(const ie::Tensor &a, const ie::Tensor &b) {
  int64_t M = a.size(0);
  int64_t K = a.size(1);
  int64_t N = b.size(1);

  ie::Tensor c({M, N});
  const float *a_data = a.data<float>();
  const float *b_data = b.data<float>();
  float *c_data = c.data<float>();
  std::memset(c_data, 0, M * N * sizeof(float));

  for (int64_t i = 0; i < M; ++i) {
    for (int64_t k = 0; k < K; ++k) {
      float a_ik = a_data[i * K + k];
      for (int64_t j = 0; j < N; ++j) {
        c_data[i * N + j] += a_ik * b_data[k * N + j];
      }
    }
  }
  return c;
}

int main() {
  std::cout << "============================================================"
            << std::endl;
  std::cout << "  Inference Engine — INT8 Quantization Benchmark Suite"
            << std::endl;
  std::cout << "============================================================"
            << std::endl;

  std::mt19937 rng(42);

  // ---- Quantization/Dequantization Throughput ----
  {
    std::cout << std::endl;
    std::cout << "--- Quantize/Dequantize Throughput ---" << std::endl;
    std::cout << std::setw(20) << "Shape" << "  " << std::setw(12)
              << "Quant (ms)" << "  " << std::setw(12) << "Dequant (ms)"
              << "  " << std::setw(12) << "Quant GB/s" << "  " << std::setw(12)
              << "Dequant GB/s" << std::endl;
    std::cout << std::string(74, '-') << std::endl;

    struct {
      std::vector<int64_t> shape;
      const char *label;
    } shapes[] = {
        {{128, 1, 256}, "128x1x256"},   {{256, 1, 256}, "256x1x256"},
        {{512, 1, 256}, "512x1x256"},   {{1024, 1, 256}, "1024x1x256"},
        {{1024, 4, 256}, "1024x4x256"},
    };

    for (auto &s : shapes) {
      ie::Tensor t(s.shape);
      FillRandom(t, rng);

      int32_t channel_dim = static_cast<int32_t>(s.shape.size()) - 1;
      auto params = ie::ComputeQuantParams(t, channel_dim, true);

      const int iters = 50;

      // Quantize.
      auto start = std::chrono::high_resolution_clock::now();
      ie::QuantizedTensor qt;
      for (int i = 0; i < iters; ++i) {
        qt = ie::QuantizeInt8(t, params, channel_dim);
      }
      auto end = std::chrono::high_resolution_clock::now();
      double q_ms =
          std::chrono::duration<double, std::milli>(end - start).count() /
          iters;
      double q_gb_s = (t.nbytes() / 1e9) / (q_ms / 1000.0);

      // Dequantize.
      start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < iters; ++i) {
        ie::DequantizeInt8(qt);
      }
      end = std::chrono::high_resolution_clock::now();
      double dq_ms =
          std::chrono::duration<double, std::milli>(end - start).count() /
          iters;
      double dq_gb_s = (t.nbytes() / 1e9) / (dq_ms / 1000.0);

      std::cout << std::setw(20) << s.label << "  " << std::fixed
                << std::setprecision(3) << std::setw(10) << q_ms << "ms"
                << "  " << std::setw(10) << dq_ms << "ms" << "  "
                << std::setprecision(2) << std::setw(10) << q_gb_s << "  "
                << std::setw(10) << dq_gb_s << std::endl;
    }
  }

  // ---- KV Cache Compression Ratios ----
  {
    std::cout << std::endl;
    std::cout << "--- KV Cache Compression Ratios ---" << std::endl;
    std::cout << std::setw(20) << "Shape" << "  " << std::setw(14)
              << "FP32 (KB)" << "  " << std::setw(14) << "INT8 (KB)" << "  "
              << std::setw(14) << "Ratio" << "  " << std::setw(14)
              << "Savings %" << std::endl;
    std::cout << std::string(82, '-') << std::endl;

    struct {
      std::vector<int64_t> shape;
      const char *label;
    } shapes[] = {
        {{128, 1, 256}, "128x1x256"},
        {{256, 1, 256}, "256x1x256"},
        {{512, 1, 256}, "512x1x256"},
        {{1024, 1, 256}, "1024x1x256"},
    };

    for (auto &s : shapes) {
      ie::Tensor t(s.shape);
      FillRandom(t, rng);

      auto qt = ie::QuantizeKVCache(t, 6.0f);
      double fp32_kb = t.nbytes() / 1024.0;
      double int8_kb = qt.CompressedBytes() / 1024.0;
      double ratio = qt.CompressionRatio();
      double savings = (1.0 - int8_kb / fp32_kb) * 100.0;

      std::cout << std::setw(20) << s.label << "  " << std::fixed
                << std::setprecision(1) << std::setw(12) << fp32_kb << "KB"
                << "  " << std::setw(12) << int8_kb << "KB" << "  "
                << std::setprecision(2) << std::setw(12) << ratio << "x"
                << "  " << std::setprecision(1) << std::setw(12) << savings
                << "%" << std::endl;
    }
  }

  // ---- Outlier-Aware Quantization Accuracy ----
  {
    std::cout << std::endl;
    std::cout << "--- Outlier-Aware Quantization Accuracy ---" << std::endl;
    std::cout << std::setw(16) << "Outlier %" << "  " << std::setw(14)
              << "Max Error" << "  " << std::setw(14) << "Mean Error" << "  "
              << std::setw(14) << "RMSE" << "  " << std::setw(14) << "Savings %"
              << std::endl;
    std::cout << std::string(78, '-') << std::endl;

    float outlier_fracs[] = {0.0f, 0.02f, 0.05f, 0.10f, 0.20f};
    ie::Tensor t({512, 1, 256});

    for (auto frac : outlier_fracs) {
      FillRandom(t, rng, frac, 50.0f);

      auto qt = ie::QuantizeKVCache(t, 6.0f);
      ie::Tensor reconstructed = ie::DequantizeInt8(qt);

      const float *orig = t.data<float>();
      const float *recon = reconstructed.data<float>();
      int64_t total = t.numel();

      double max_err = 0.0;
      double sum_err = 0.0;
      double sum_sq_err = 0.0;
      for (int64_t i = 0; i < total; ++i) {
        double err = std::abs(orig[i] - recon[i]);
        max_err = std::max(max_err, err);
        sum_err += err;
        sum_sq_err += err * err;
      }
      double mean_err = sum_err / total;
      double rmse = std::sqrt(sum_sq_err / total);
      double fp32_kb = t.nbytes() / 1024.0;
      double int8_kb = qt.CompressedBytes() / 1024.0;
      double savings = (1.0 - int8_kb / fp32_kb) * 100.0;

      char frac_str[16];
      snprintf(frac_str, sizeof(frac_str), "%.0f%%", frac * 100);

      std::cout << std::setw(16) << frac_str << "  " << std::scientific
                << std::setprecision(3) << std::setw(14) << max_err << "  "
                << std::setw(14) << mean_err << "  " << std::setw(14) << rmse
                << "  " << std::fixed << std::setprecision(1) << std::setw(12)
                << savings << "%" << std::endl;
    }
  }

  // ---- Quantized Matmul vs FP32 Matmul ----
  {
    std::cout << std::endl;
    std::cout << "--- Quantized Matmul vs FP32 Matmul ---" << std::endl;
    std::cout << std::setw(16) << "Shape" << "  " << std::setw(12)
              << "FP32 (ms)" << "  " << std::setw(12) << "INT8 (ms)" << "  "
              << std::setw(10) << "Speedup" << "  " << std::setw(14)
              << "Max Error" << "  " << std::setw(14) << "Mean Error"
              << std::endl;
    std::cout << std::string(84, '-') << std::endl;

    struct {
      int64_t M, K, N;
      const char *label;
    } sizes[] = {
        {1, 1152, 6912, "1x1152x6912"},
        {1, 6912, 1152, "1x6912x1152"},
        {16, 1152, 1152, "16x1152x1152"},
    };

    for (auto &s : sizes) {
      ie::Tensor a({s.M, s.K});
      ie::Tensor b({s.K, s.N});
      {
        float *a_data = a.data<float>();
        float *b_data = b.data<float>();
        std::normal_distribution<float> dist(0.0f, 0.1f);
        for (int64_t i = 0; i < a.numel(); ++i)
          a_data[i] = dist(rng);
        for (int64_t i = 0; i < b.numel(); ++i)
          b_data[i] = dist(rng);
      }

      // Quantize B.
      auto params = ie::ComputeQuantParams(b, 0, true);
      auto qb = ie::QuantizeInt8(b, params, 0);

      const int iters = 20;

      // FP32 matmul.
      auto start = std::chrono::high_resolution_clock::now();
      ie::Tensor c_fp32;
      for (int i = 0; i < iters; ++i) {
        c_fp32 = NaiveMatmul(a, b);
      }
      auto end = std::chrono::high_resolution_clock::now();
      double fp32_ms =
          std::chrono::duration<double, std::milli>(end - start).count() /
          iters;

      // INT8 matmul.
      start = std::chrono::high_resolution_clock::now();
      ie::Tensor c_int8;
      for (int i = 0; i < iters; ++i) {
        c_int8 = ie::QuantizedMatmul(a, qb);
      }
      end = std::chrono::high_resolution_clock::now();
      double int8_ms =
          std::chrono::duration<double, std::milli>(end - start).count() /
          iters;

      // Accuracy comparison.
      const float *fp32_data = c_fp32.data<float>();
      const float *int8_data = c_int8.data<float>();
      int64_t total = c_fp32.numel();
      double max_err = 0.0;
      double sum_err = 0.0;
      for (int64_t i = 0; i < total; ++i) {
        double err = std::abs(fp32_data[i] - int8_data[i]);
        max_err = std::max(max_err, err);
        sum_err += err;
      }
      double mean_err = sum_err / total;

      double speedup = fp32_ms / int8_ms;

      std::cout << std::setw(16) << s.label << "  " << std::fixed
                << std::setprecision(2) << std::setw(10) << fp32_ms << "ms"
                << "  " << std::setw(10) << int8_ms << "ms" << "  "
                << std::setw(8) << speedup << "x" << "  " << std::scientific
                << std::setprecision(3) << std::setw(14) << max_err << "  "
                << std::setw(14) << mean_err << std::endl;
    }
  }

  // ---- Memory Footprint Summary ----
  {
    std::cout << std::endl;
    std::cout << "--- KV Cache Memory Footprint (Full Model) ---" << std::endl;

    // Gemma-3 1B: 26 layers, 1 KV head, 256 head_dim
    const int32_t num_layers = 26;
    const int32_t num_kv_heads = 1;
    const int32_t head_dim = 256;

    // Per token per layer: K + V = 2 * 1 * 256 * sizeof(float) = 2048 bytes
    // Per token (all layers): 2048 * 26 = 53,248 bytes
    int64_t per_token_bytes = 2 * num_kv_heads * head_dim * sizeof(float);
    int64_t per_token_all_layers = per_token_bytes * num_layers;

    std::cout << std::endl;
    std::cout << "  Gemma-3 1B KV dimensions:" << std::endl;
    std::cout << "    Layers:       " << num_layers << std::endl;
    std::cout << "    KV heads:     " << num_kv_heads << std::endl;
    std::cout << "    Head dim:     " << head_dim << std::endl;
    std::cout << "    Per-token KV: " << per_token_all_layers << " bytes ("
              << std::fixed << std::setprecision(1)
              << (per_token_all_layers / 1024.0) << " KB)" << std::endl;

    std::cout << std::endl;
    std::cout << std::setw(12) << "SeqLen" << "  " << std::setw(14)
              << "FP32 (MB)" << "  " << std::setw(14) << "INT8 (MB)" << "  "
              << std::setw(14) << "Savings (MB)" << "  " << std::setw(12)
              << "Savings %" << std::endl;
    std::cout << std::string(72, '-') << std::endl;

    int64_t seq_lens[] = {128, 256, 512, 1024, 2048, 4096};
    for (auto seq_len : seq_lens) {
      double fp32_mb = static_cast<double>(per_token_all_layers) * seq_len /
                       (1024.0 * 1024.0);
      // INT8: 1 byte per element + scales overhead (~0.5% overhead)
      double int8_mb = fp32_mb * (1.0 / 4.0) * 1.005;
      double savings_mb = fp32_mb - int8_mb;
      double savings_pct = (savings_mb / fp32_mb) * 100.0;

      std::cout << std::setw(12) << seq_len << "  " << std::fixed
                << std::setprecision(2) << std::setw(12) << fp32_mb << "MB"
                << "  " << std::setw(12) << int8_mb << "MB" << "  "
                << std::setw(12) << savings_mb << "MB" << "  "
                << std::setprecision(1) << std::setw(10) << savings_pct << "%"
                << std::endl;
    }
  }

  std::cout << std::endl;
  std::cout << "============================================================"
            << std::endl;
  std::cout << "  Benchmark complete." << std::endl;
  std::cout << "============================================================"
            << std::endl;

  return 0;
}
