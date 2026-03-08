# Model Architecture & Forward Pass

## What This Does

The `engine/model/` directory implements the full Gemma-3 1B transformer architecture. Given a sequence of token IDs, it produces logits (probability scores) over the vocabulary for next-token prediction. The `engine/engine.cc` wraps this with tokenization, sampling, and the autoregressive generation loop.

## Architecture

```
main.cc (CLI)
  └── InferenceEngine
        ├── LoadModel()  → GGUF parser → BuildTokenizer() + BuildModel()
        └── Generate()   → Encode → Prefill → Decode loop → Decode
                               ↓
                          GemmaModel::forward()
                            ├── Embedding lookup + √d scaling
                            ├── 26× TransformerBlock::forward()
                            │     ├── RMSNorm → Attention → residual
                            │     └── RMSNorm → FeedForward → residual
                            ├── Final RMSNorm
                            └── Logits (weight-tying with embedding)
```

## Key Files

| File | Purpose |
|------|---------|
| [config.h](../engine/model/config.h) | `GemmaConfig` — model hyperparameters |
| [layers.h](../engine/model/layers.h) | Layer class declarations |
| [layers.cc](../engine/model/layers.cc) | Forward pass implementations |
| [engine.cc](../engine/engine.cc) | Model loading + generation loop |

## Gemma-3 1B Specifics

| Hyperparameter | Value | Implication |
|----------------|-------|-------------|
| Layers | 26 | Deep but narrow — fast per-layer, many iterations |
| Embed dim | 1152 | Relatively small — fits in cache well |
| Attention heads | 4 | Very few heads, each has head_dim=256 |
| KV heads | 1 | **GQA ratio = 4:1** — massive KV cache savings |
| Head dim | 256 | Large heads improve representation quality |
| FFN hidden | 6912 | 6x expansion ratio (6912/1152) |
| Vocab | 262,144 | Very large vocabulary — embedding is 1.1GB in FP32 |

## Component Deep Dives

### Grouped-Query Attention (GQA)

**What:** Instead of each attention head having its own K and V projections (Multi-Head Attention), GQA shares K/V across groups of Q heads. Gemma-3 1B uses 4 Q heads sharing 1 K/V head.

**Why this matters:**
- KV cache memory is proportional to `num_kv_heads × head_dim × seq_len`
- With GQA (1 KV head vs 4), the KV cache is **4x smaller**
- At 32K context length: 26 layers × 1 head × 256 dim × 32K tokens × 2 (K+V) × 4 bytes = **~410MB** instead of 1.6GB

**Implementation:** In `Attention::forward`, we project Q with `num_heads=4` dimensions but K/V with `num_kv_heads=1`. Before attention, we expand K/V via `Tensor::repeat()` to match Q's head count. This avoids storing duplicated KV data while keeping the attention math correct.

### SwiGLU Feed-Forward Network

**What:** The FFN uses three weight matrices instead of the standard two:
```
gate = SiLU(x @ W_gate)
up   = x @ W_up
out  = (gate ⊙ up) @ W_down
```

**Why SwiGLU over standard FFN:**
- The gating mechanism (element-wise multiply of two projections) lets the network learn which features to pass through
- SiLU (Sigmoid Linear Unit) is smoother than ReLU, avoiding dead neurons
- Empirically shown to improve model quality at the same parameter count (PaLM, LLaMA, Gemma all use this)

**Trade-off:** 3 matrices means 50% more FFN parameters than a 2-matrix FFN, but Gemma compensates by using fewer attention heads.

### RoPE (Rotary Position Embeddings)

**What:** Instead of adding position information to the input, RoPE encodes positions by rotating Q and K vectors in 2D subspaces. Position `p` rotates by angle `p·θ` where `θ` is a frequency that varies per dimension.

**Why RoPE over learned/sinusoidal embeddings:**
- **Relative position awareness:** The dot product Q·K naturally encodes the *distance* between positions, not absolute position
- **Extrapolation:** Can handle sequences longer than training length (with techniques like NTK scaling)
- **No extra parameters:** Position encoding is computed, not learned

**Implementation:** We apply RoPE to Q and K after projection but before attention. The `ops::rope()` function takes a position tensor and rotates pairs of dimensions using sin/cos.

### Weight Tying

**What:** The output projection (logits = hidden @ W_vocab) reuses the token embedding matrix transposed, rather than having a separate output weight.

**Why:** The vocabulary is 262K tokens × 1152 dims = ~1.1GB. Weight tying saves this entire matrix, which is significant for a 1B parameter model. It also provides regularization — the model learns embeddings that work for both input and output.

### Embedding Scaling

**What:** After embedding lookup, we multiply by √(embed_dim):
```cpp
float scale = sqrt(config.embed_dim);  // √1152 ≈ 33.9
x *= scale;
```

**Why:** This is a Gemma-specific design choice. It up-scales the embeddings to match the magnitude of the residual stream after many layers of attention and FFN. Without this, early-layer embeddings would be dwarfed by later activations.

## Generation Pipeline

### Prefill Phase
Process the entire prompt in one forward pass. All tokens attend to each other (with causal masking), populating the KV cache for all positions simultaneously.

**Why batch prefill:** Processing N tokens at once lets us use GEMM (matrix-matrix multiply), which is compute-bound and well-optimized. Processing one at a time would use GEMV (matrix-vector), which is memory-bound on modern hardware.

### Decode Phase
Generate one token at a time. Each new token:
1. Forward pass with just the new token (seq_len=1)
2. KV cache provides the context from all previous tokens
3. Sample from logits → get next token ID → decode to text

**Why autoregressive:** Transformer language models predict one token at a time, conditioned on all previous tokens. The KV cache avoids recomputing attention for past tokens, making each decode step O(1) in compute (just one new token) instead of O(N).

## Sampling Strategies

| Strategy | When to Use |
|----------|-------------|
| **Greedy** (argmax) | Deterministic output, factual queries |
| **Top-K** (k=40, temp=0.8) | Creative text with bounded randomness |
| **Top-P / Nucleus** (p=0.9) | Adaptive vocabulary — uses fewer tokens when model is confident |

Temperature scales logits before softmax: higher = more random, lower = more deterministic.
