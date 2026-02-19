# 🚀 Inference Engine

A from-scratch C++ inference engine for open-weight large language models, starting with **Gemma 3 1B**.

Built to learn modern C++, ML systems engineering, and extensible software design.

## Architecture

```
engine/
├── tensor/          # Custom Tensor class (shape, strides, memory management)
├── ops/             # Math kernels (matmul, softmax, RMSNorm, RoPE, SiLU)
├── gguf/            # GGUF file parser (binary I/O, model weight loading)
├── tokenizer/       # BPE tokenizer (encode/decode text ↔ token IDs)
├── model/           # Transformer layers (Attention, FFN, TransformerBlock)
├── engine.h/cc      # Top-level inference engine (load, generate)
└── main.cc          # CLI entry point
```

## Build & Run

```bash
# Build everything
bazel build //...

# Run tests
bazel test //...

# Run inference
bazel run //engine:inference -- --model model/gemma-3-1b-it-f16.gguf --prompt "Hello"
```

## Tech Stack

- **C++20** with modern idioms (RAII, move semantics, `std::variant`, `std::shared_ptr`)
- **Bazel** (bzlmod) build system
- **GoogleTest** for unit testing
- **Abseil** for utilities (`absl::Span`, `absl::StatusOr`)

---

## 🗺️ Implementation Roadmap

### Phase 1: Foundations *(Start Here)*
- [ ] **Tensor `at()` / `set()`** — Stride-based element access with bounds checking
- [ ] **Tensor `reshape()` / `view()`** — Shape manipulation
- [ ] **GGUF metadata parser** — Read key-value pairs from binary GGUF
- [ ] **GGUF tensor info parser** — Read tensor names, shapes, offsets
- [ ] **`ops::add()`, `ops::mul()`** — Element-wise operations with broadcasting

### Phase 2: Core Kernels *(The Important Stuff)*
- [ ] **`ops::matmul()`** — Matrix multiplication (naive → tiled → SIMD)
- [ ] **`ops::rms_norm()`** — RMS normalization
- [ ] **`ops::silu()`** — SiLU activation  
- [ ] **`ops::softmax()`** — Numerically stable softmax
- [ ] **`ops::embedding()`** — Embedding table lookup
- [ ] **`ops::rope()`** — Rotary positional embeddings

### Phase 3: Model Forward Pass
- [ ] **GGUF tensor loading** — Load F16/F32 weight data into Tensors
- [ ] **Build tokenizer from GGUF** — Extract vocab, scores, special tokens
- [ ] **`RMSNorm::forward()`** — First layer implementation
- [ ] **`FeedForward::forward()`** — SwiGLU FFN
- [ ] **`Attention::forward()`** — Multi-head attention with GQA
- [ ] **`TransformerBlock::forward()`** — Attention + FFN + residuals
- [ ] **`GemmaModel::forward()`** — Full model: embed → blocks → logits

### Phase 4: Generation
- [ ] **Greedy sampling** — argmax over logits
- [ ] **Temperature + Top-K/Top-P sampling**
- [ ] **Autoregressive loop** — Generate tokens one at a time
- [ ] **KV-cache** — Avoid recomputing attention for past tokens

### Phase 5: Optimization & Extensibility
- [ ] **SIMD kernels** — AVX2/AVX-512 for matmul and element-wise ops
- [ ] **Multi-threading** — Parallelize across transformer layers
- [ ] **Quantization support** — Q4_0, Q4_K_M, Q8_0 dequantization
- [ ] **Memory mapping** — `mmap` for zero-copy model loading
- [ ] **KV-cache optimization** — Paged attention, sliding window
- [ ] **Benchmarking** — tokens/sec measurement and comparison

### Phase 6: Production Features *(Stretch Goals)*
- [ ] **gRPC serving** — Serve the model over a network
- [ ] **Batched inference** — Process multiple requests concurrently
- [ ] **Multiple architectures** — Support Llama, Mistral, Phi, etc.
- [ ] **Speculative decoding** — Use a smaller model to draft tokens
- [ ] **Flash Attention** — Memory-efficient attention algorithm

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
