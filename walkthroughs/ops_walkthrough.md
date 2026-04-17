# Ops Implementation Walkthrough

## Overview

This document covers the implementation of all math kernels in `engine/ops/ops.cc` — the core computational building blocks for transformer inference.

## Architecture

```
ops.h (declarations)  →  ops.cc (implementations)  →  ops_test.cc (31 tests)
         ↓
   Tensor class (data<T>(), contiguous(), shape, strides)
```

All ops follow the same pattern:
1. Validate input shapes
2. Make inputs contiguous (required for `data<T>()` pointer access)
3. Allocate output tensor
4. Perform computation using raw float pointers (fast path)
5. Return result

---

## Operations Reference

### Element-wise: `add()`, `mul()`

**What**: Pointwise arithmetic on tensors of identical shape.

**Bug fix**: The original code called `a.contiguous()` but discarded the return value. `contiguous()` returns a *new* tensor — it doesn't mutate in-place. Fixed by capturing:
```cpp
// BEFORE (bug): a.contiguous();  // return value discarded!
// AFTER (fix):
Tensor a_c = a.contiguous();
```

**Usage**:
```cpp
Tensor a = Tensor::from_vector({1.0f, 2.0f, 3.0f});
Tensor b = Tensor::from_vector({4.0f, 5.0f, 6.0f});
Tensor c = ops::add(a, b);  // [5, 7, 9]
Tensor d = ops::mul(a, b);  // [4, 10, 18]
```

---

### Matrix Multiply: `matmul()`

**What**: `C[M,N] = A[M,K] × B[K,N]` — the most performance-critical kernel.

**Strategy**: Cache-friendly **i,k,j loop order** instead of naive i,j,k.

```
Naive (i,j,k): B is accessed column-wise → cache misses on every B access
                ┌─────────┐
                │ B[0][j] │  ← jumps by N elements each k iteration
                │ B[1][j] │
                │ B[2][j] │
                └─────────┘

Optimized (i,k,j): B is accessed row-wise → sequential, cache-friendly
                ┌─────────────────────┐
                │ B[k][0] B[k][1] ... │  ← stride-1, fills cache lines
                └─────────────────────┘
```

The inner loop broadcasts `A[i][k]` across an entire row of B:
```cpp
for (int64_t i = 0; i < M; ++i) {
    for (int64_t k = 0; k < K; ++k) {
        const float a_ik = A[i * K + k];  // hoisted into register
        for (int64_t j = 0; j < N; ++j) {
            C[i * N + j] += a_ik * B[k * N + j];  // both stride-1
        }
    }
}
```

**Performance**: ~3-5× faster than naive i,j,k for large matrices.

**Next optimization stages** (not yet implemented):
| Stage | Technique | Expected Speedup |
|-------|-----------|-------------------|
| 1 ✅ | Cache-friendly loop order (i,k,j) | 3-5× |
| 2 | L2-aware tiling (64×64 blocks) | 2-3× |
| 3 | SIMD (Highway `hn::MulAdd`) | 4-8× |
| 4 | Multi-threaded (partition M dim) | N_cores× |

**Usage**:
```cpp
Tensor a({2, 3});  // 2×3 matrix
Tensor b({3, 4});  // 3×4 matrix
Tensor c = ops::matmul(a, b);  // 2×4 result
```

---

### RMS Normalization: `rms_norm()`

**What**: `output = (x / rms) × weight` where `rms = √(mean(x²) + ε)`

Used by Gemma instead of LayerNorm (simpler, no mean subtraction).

**Implementation detail**: Operates on the last dimension. For 2D+ inputs, each "row" is normalized independently. Uses `1/rms` multiplication instead of division for efficiency.

**Usage**:
```cpp
Tensor x({batch, seq_len, embed_dim});  // input features
Tensor weight({embed_dim});              // learnable scale
Tensor normed = ops::rms_norm(x, weight);
```

---

### Activations: `silu()`, `gelu()`

**SiLU** (Sigmoid Linear Unit): `silu(x) = x × σ(x) = x / (1 + e^(-x))`
- Used in Gemma's SwiGLU FFN: `FFN(x) = silu(W_gate @ x) * (W_up @ x)`

**GeLU** (Gaussian Error Linear Unit): `gelu(x) = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`
- Uses the tanh approximation (same as PyTorch)

**Usage**:
```cpp
Tensor gate = ops::matmul(x, w_gate);
Tensor up = ops::matmul(x, w_up);
Tensor activated = ops::mul(ops::silu(gate), up);  // SwiGLU
```

---

### Softmax: `softmax()`

**What**: `softmax(x)_i = exp(x_i - max) / Σ exp(x_j - max)`

**Key**: Numerically stable — subtracts max before exponentiating to prevent overflow. Without this, `exp(1000)` = infinity.

Supports arbitrary dimensions (not just last dim). For attention scores in transformers, this is always along the sequence dimension.

**Usage**:
```cpp
Tensor scores({batch, n_heads, seq_len, seq_len});  // attention scores
Tensor probs = ops::softmax(scores, -1);  // along last dim (default)
```

---

### Rotary Positional Embedding: `rope()`

**What**: Encodes position information by rotating pairs of dimensions in Q/K vectors.

For each pair (2i, 2i+1) at position `pos`:
```
θ = pos × freq_base^(-2i/head_dim)
x'[2i]   = x[2i]·cos(θ) - x[2i+1]·sin(θ)
x'[2i+1] = x[2i]·sin(θ) + x[2i+1]·cos(θ)
```

**Optimization**: Inverse frequencies are precomputed once per call (O(head_dim/2) computation amortized across all batch × seq × heads).

**Key property**: Rotation preserves vector norms — our tests verify `||x|| = ||RoPE(x)||`.

**Usage**:
```cpp
Tensor q({batch, seq_len, n_heads, head_dim});  // query vectors
Tensor positions({batch, seq_len});               // [0, 1, 2, ...]
Tensor q_rotated = ops::rope(q, positions);       // position-encoded
```

---

### Embedding: `embedding()`

**What**: Table lookup — maps integer token IDs to dense vectors.

**Optimization**: Uses `memcpy` for row copies instead of element-wise loops. For contiguous float data, this is significantly faster (memcpy can use SIMD internally).

**Usage**:
```cpp
Tensor table({vocab_size, embed_dim});            // 262144 × 1152 for Gemma-3
Tensor tokens = Tensor::from_vector({42.0f, 7.0f, 100.0f});
Tensor embeddings = ops::embedding(table, tokens); // [3, 1152]
```

---

## Multi-Dtype Support

All ops support **FP32, FP16, and INT8** tensors via upcast/downcast (Option A):

```cpp
DType out_dtype = x.dtype();                    // remember original dtype
Tensor x_f = x.to(DType::kFloat32).contiguous(); // upcast to FP32
// ... all computation happens in FP32 ...
return out.to(out_dtype);                        // convert back
```

**Why this approach?**
- **Simplicity**: One code path handles all dtypes — no template specialization needed
- **Accuracy**: FP32 computation avoids FP16 overflow/underflow during intermediate steps (e.g., `exp()` in softmax)
- **Correctness**: The Tensor class `.to()` already handles all conversions via `HalfToFloat`/`FloatToHalf`
- **Real-world match**: This is exactly what PyTorch does on CPU for most ops

**When to upgrade to Option B** (per-dtype kernels):
Once Highway SIMD is enabled, write specialized FP16 kernels — Highway has native `float16_t` support and can process 16 FP16 values per AVX-512 instruction vs 8 FP32 values.

---

## Test Summary

All 38 tests pass:

| Operation | Tests | What's Covered |
|-----------|-------|----------------|
| `add` | 3 + 1 FP16 | basic, 2D, shape mismatch, FP16 round-trip |
| `mul` | 2 + 1 FP16 | basic, shape mismatch, FP16 round-trip |
| `matmul` | 5 + 1 FP16 | identity multiply, known 2×2, non-square, dim mismatch, non-2D, FP16 |
| `rms_norm` | 4 + 1 FP16 | basic, weight scaling, 2D, weight mismatch, FP16 |
| `silu` | 2 + 1 FP16 | zero, known values, FP16 |
| `gelu` | 2 | zero, known values |
| `softmax` | 6 + 1 FP16 | uniform, sums-to-1, monotonic, numerical stability, 2D, FP16 |
| `rope` | 4 | norm preservation, position-0 identity, odd head_dim, batch mismatch |
| `embedding` | 3 + 1 FP16 | basic lookup, out-of-bounds, negative index, FP16 table |


```bash
# Run all ops tests
bazel test //engine/ops:ops_test

# Run with verbose output
bazel test //engine/ops:ops_test --test_output=all
```

---

## How These Ops Connect in the Forward Pass

```
tokens → embedding() → x
                        ↓
              ┌─── TransformerBlock (×26) ───┐
              │                               │
              │  x → rms_norm() → Attention   │
              │       ↓                       │
              │  Q,K,V = matmul(x, W_q/k/v)  │
              │  Q,K = rope(Q, positions),    │
              │        rope(K, positions)      │
              │  scores = matmul(Q, K^T)      │
              │  probs = softmax(scores)      │
              │  attn = matmul(probs, V)      │
              │  x = x + matmul(attn, W_o)    │
              │       ↓                       │
              │  x → rms_norm() → FFN         │
              │  gate = matmul(x, W_gate)     │
              │  up = matmul(x, W_up)         │
              │  x = x + matmul(              │
              │    silu(gate) * up, W_down)    │
              └───────────────────────────────┘
                        ↓
              logits = matmul(x, W_embed^T)
              probs = softmax(logits)
```

## Developer Guide: How to Add a New Op

If you need to implement a new mathematical operation (e.g., `GELU` or `Sigmoid`), follow this exact recipe:

### 1. Declare in `ops.h`
Add the function signature. It should almost always take `const Tensor&` and return a new `Tensor`.
```cpp
// In engine/ops/ops.h
Tensor sigmoid(const Tensor& a);
```

### 2. Boilerplate Setup in `ops.cc`
Every operation must start by ensuring the data is contiguous (laid out linearly in memory) and allocating the output tensor.
```cpp
Tensor sigmoid(const Tensor& a) {
    // 1. Force contiguous layout
    Tensor a_c = a.contiguous();
    
    // 2. Remember original dtype, upcast to Float32 for math
    DType out_dtype = a_c.dtype();
    Tensor a_f = a_c.to(DType::kFloat32);
    
    // 3. Allocate output tensor
    Tensor out(a_f.shape(), DType::kFloat32);
    
    // 4. Get raw float pointers
    const float* in_ptr = a_f.data<float>();
    float* out_ptr = out.data<float>();
    int64_t n = a_f.numel();
    
    // 5. The Math Loop  (or call to simd_kernels.h!)
    for (int64_t i = 0; i < n; ++i) {
        out_ptr[i] = 1.0f / (1.0f + std::exp(-in_ptr[i]));
    }
    
    // 6. Return, downcasting back to original format if necessary
    return out.to(out_dtype);
}
```

### 3. Add to `ops_test.cc`
You MUST test your operation against known values. Add a test case using `EXPECT_NEAR` for floating point comparisons:
```cpp
TEST(OpsTest, Sigmoid) {
    Tensor a = Tensor::from_vector({0.0f, 2.0f, -2.0f});
    Tensor out = ops::sigmoid(a);
    
    EXPECT_NEAR(out.data<float>()[0], 0.5f, 1e-4);
    EXPECT_NEAR(out.data<float>()[1], 0.88079f, 1e-4);
    EXPECT_NEAR(out.data<float>()[2], 0.11920f, 1e-4);
}
```

### Why we upcast everything to FP32 inside Ops
You will notice the pattern: `a.to(DType::kFloat32)`. 
While the weights might be INT8 (`Q8_0`) or FP16 coming off the disk or out of the KV Cache, doing intermediary math (like `exp()` in softmax or the Sigmoid function) in low precision causes catastrophic numerical instability. 
1. `std::exp(-x)` can easily underflow FP16.
2. Accumulating large dot-products in FP16 will overflow the maximum value of `65504`.
By upcasting to FP32, doing the math, and converting back, we pay a tiny compute overhead in exchange for rock-solid stability.

### 4. Compilation Units and Headers
You'll notice we declare `ops.h` and implement in `ops.cc`. In industry, keeping your `#include` directives minimal in the `.h` file is critical. Every time a `.h` file is included in another file, the C++ compiler essentially copy-pastes its contents. If you put heavy `#include <vector>` or `#include "engine.h"` in your header, any file that includes your header also compiles that heavy code. This causes compile times to explode (scaling to hours on large projects).

**C++ Best Practice:** Always put your heavy `#includes` in the `.cc` implementation file, and use "forward declarations" (e.g., `class Tensor;`) in your header file whenever possible.
