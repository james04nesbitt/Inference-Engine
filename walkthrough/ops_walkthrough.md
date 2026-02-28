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

## Test Summary

All 31 tests pass:

| Operation | Tests | What's Covered |
|-----------|-------|----------------|
| `add` | 3 | basic, 2D, shape mismatch |
| `mul` | 2 | basic, shape mismatch |
| `matmul` | 5 | identity multiply, known 2×2, non-square, inner dim mismatch, non-2D throw |
| `rms_norm` | 4 | basic, weight scaling, 2D (row-wise), weight size mismatch |
| `silu` | 2 | zero, known values |
| `gelu` | 2 | zero, known values |
| `softmax` | 6 | uniform, sums-to-1, monotonic, numerical stability, 2D dim=1, 2D dim=0 |
| `rope` | 4 | norm preservation, position-0 identity, odd head_dim throw, batch mismatch throw |
| `embedding` | 3 | basic lookup, out-of-bounds throw, negative index throw |

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
