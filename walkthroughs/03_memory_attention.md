# Memory Management: FlashAttention & Paged KV Cache

## What This Does

The `engine/attention/` directory solves the two biggest memory problems in LLM inference:

1. **FlashAttention** — Computes attention without materializing the full N×N attention matrix, reducing memory from O(N²) to O(N)
2. **Paged KV Cache** — Manages KV cache memory using OS-style virtual memory (pages, page tables, free lists), eliminating waste from pre-allocation

Together, these enable efficient long-context inference and multi-sequence serving.

## Key Files

| File | Purpose |
|------|---------|
| [flash_attention.h](../engine/attention/flash_attention.h) | FlashAttention API |
| [flash_attention.cc](../engine/attention/flash_attention.cc) | Tiled attention with online softmax |
| [kv_cache.h](../engine/attention/kv_cache.h) | `KVCacheManager` — paged block allocator |
| [kv_cache.cc](../engine/attention/kv_cache.cc) | Block pool, page tables, alloc/free |

---

## FlashAttention

### The Problem

Standard attention computes:
```
Attention(Q, K, V) = softmax(Q @ K^T / √d) @ V
```

The matrix `Q @ K^T` has shape [seq_len, seq_len]. At 32K context length, that's 32K × 32K × 4 bytes = **4GB** just for one attention matrix. With 26 layers × 4 heads = 104 such matrices during a prefill pass.

### The Solution: Tiled Computation

FlashAttention never materializes the full attention matrix. Instead, it processes Q×K in tiles:

```
For each tile of Q (block of query rows):
    For each tile of K, V (block of key/value rows):
        1. Compute partial_attn = Q_tile @ K_tile^T   ← small matrix, fits in L1
        2. Update running softmax statistics (max, sum)
        3. Accumulate output: O_tile += softmax(partial_attn) @ V_tile
    Rescale O_tile with final softmax denominator
```

**Memory complexity:** O(N) instead of O(N²) — we only store the output matrix, not the intermediate attention scores.

### Online Softmax

The key algorithmic insight: softmax can be computed *incrementally* without seeing all values at once.

Standard softmax requires two passes:
1. Find global max (for numerical stability)
2. Compute exp(x - max) / sum

Online softmax maintains running statistics:
```
For each new block of scores:
    new_max = max(running_max, block_max)
    rescale = exp(running_max - new_max)       // correction factor
    running_sum = running_sum * rescale + block_sum
    output = output * rescale + new_contribution
```

**Why this matters:** We never need to store all attention scores simultaneously. Each tile is computed, used, and discarded. This is what makes O(N) memory possible.

### Causal Masking

During prefill, token `i` should only attend to tokens `0..i` (not future tokens). We apply causal masking at the tile level:

```cpp
if (causal && kv_start + j > q_start + i) {
    score = -infinity;  // masked out
}
```

**Why tile-level masking:** Checking the mask per-element within a tile is simpler than trying to skip entire tiles. The branch predictor handles it well since the mask is spatially coherent (block of -inf in the upper-right triangle).

### Architectural Choice: Why Our Own FlashAttention?

We implemented FlashAttention from scratch instead of using a library because:
1. **Integration:** Our attention layer directly calls `flash_attention(Q, K, V, scale, causal)` with our Tensor type — no format conversion
2. **Simplicity:** The CPU implementation is ~100 lines of straightforward C++. The GPU version is where the real complexity lies (shared memory, warp-level primitives). For CPU, the tiling still helps by keeping working sets in L1/L2
3. **Fusion potential:** We can fuse RoPE, bias, or masking directly into the tiling loop

---

## Paged KV Cache (PagedAttention)

### The Problem

Traditional KV caches pre-allocate a contiguous buffer per sequence for the maximum possible length:
```
Buffer per sequence = max_seq_len × num_kv_heads × head_dim × 2(K+V) × sizeof(float)
                    = 32768 × 1 × 256 × 2 × 4 = 64MB per sequence
```

If you're serving 8 concurrent sequences, that's **512MB** allocated even if most sequences are only 100 tokens long. This is internal fragmentation — the same problem that motivated virtual memory in operating systems.

### The Solution: Virtual Memory for KV Cache

PagedAttention (from the vLLM paper) applies the same solution: **paging**.

```
Physical memory:  [Block 0][Block 1][Block 2][Block 3][Block 4][Block 5]...
                      ↑                  ↑         ↑
Page table seq 0:  [0] ───────────────  [2] ──── [4]
Page table seq 1:     [1] ────────── [3]
Free list: [5, 6, 7, ...]
```

Each block holds `block_size` tokens (default: 16) worth of K and V data. Sequences only allocate blocks as they grow.

### Key Components

**Block Pool:** Pre-allocated array of fixed-size KV blocks. Each block stores `[block_size, num_kv_heads, head_dim]` for keys and values separately.

**Free List:** Stack of available block indices. Allocation = pop, deallocation = push. O(1) both ways.

**Block Table:** Per-sequence, per-layer mapping from logical block index to physical block index:
```
block_tables_[seq_id][layer_idx] = {physical_block_0, physical_block_1, ...}
```

**AppendToken:** Write one token's K/V into the current block's next slot. If full, pop a new block from the free list and append to the page table.

**GetKeys/GetValues:** Walk the page table, copy data from each block into a contiguous output tensor. This reconstruction is the cost we pay for flexible allocation.

### Memory Savings

| Scenario | Traditional | Paged |
|----------|------------|-------|
| 1 seq, 100 tokens | 64MB allocated | 0.4MB allocated |
| 8 seqs, avg 200 tokens | 512MB | 6.4MB |
| 8 seqs, max 32K | 512MB | 512MB (same at max) |

The savings are proportional to how much shorter actual sequences are vs the maximum.

### Architectural Choices

**Why block_size = 16?**
- Too small → too many blocks, page table overhead
- Too large → internal fragmentation in the last block
- 16 tokens × 1 head × 256 dim × 4 bytes = 16KB per half-block — fits nicely in L1 cache

**Why separate K and V in each block?**
- K and V are accessed at different times during attention (K for scores, V for weighted sum)
- Separating them lets each fit independently in cache lines

**Why not use `std::allocator` or a general-purpose pool?**
- KV blocks have a fixed, known size — no need for size classes or coalescing
- The free list is simpler and faster than `malloc`/`free` for fixed-size allocations
- We need per-sequence accounting (which blocks belong to which sequence) that a general allocator doesn't provide

**Copy-on-Write (for beam search):**
- Multiple beam hypotheses can share KV blocks until one diverges
- Reference counting on blocks enables this without full duplication

## Cost & Infrastructure Perspective

### KV Cache Sizing Math
As an ML Infra Engineer, determining how much memory to reserve for the KV cache dictates how many concurrent users you can serve.
The formula for a **single token's KV cache** in FP32 is:
`Bytes per token = 2 (K and V) × num_layers × num_kv_heads × head_dim × 4 bytes`

For Gemma-3 1B:
`2 × 26 × 1 × 256 × 4 = 53,248 bytes (~53 KB per token)`

To target `X` concurrent users generating an average of `1,024` tokens:
`Total Cache Size = 53 KB × 1,024 × X`
Serving 100 users requires ~5.4 GB of RAM just for the KV Cache.

### Hardware Page Size vs KV Page Size
Why block_size=16?
In standard OS virtual memory, a page is typically 4KB.
Our KV "page" for 16 tokens is: `16 tokens × 26 layers × 1 KV head × 256 dim × 4 bytes × 2 (K+V) = 851,968 bytes (~852 KB)` across the whole network. 
Per layer, per K/V tensor, a 16-token block is `16 × 256 × 4 = 16 KB`. This aligns perfectly with filling the L1 Data Cache (typically 32KB-48KB) sequentially, maximizing SIMD loading bandwidth without caching eviction. 

## Developer Guide: How to Experiment with Attention

If you want to modify attention (e.g., implement Sliding Window Attention or a custom penalty):

### Step 1: Hook into `flash_attention.cc`
All attention math happens in `engine/attention/flash_attention.cc`. Let's say you want to add a distance penalty (ALiBi-style):
1. Navigate to the inner `Q_tile @ K_tile` loop.
2. After the dot product, apply your penalty:
```cpp
// Inside the k_idx loop in flash_attention:
float distance = std::abs(q_idx - k_idx);
float penalty = distance * penalty_slope;
score -= penalty; // Modify the tile score before online softmax
```

### Step 2: Extracting Attention Maps
If you want to visualize where the model is looking, you need to abandon Flash Attention (which throws away the raw scores).
You can write a simple debug attention function:
```cpp
Tensor debug_attention(Tensor Q, Tensor K, Tensor V) {
    // 1. Matmul: scores = Q @ K^T
    Tensor scores = ops::matmul(Q, ops::transpose(K));
    
    // 2. You can now save 'scores' to disk or print it
    
    // 3. Softmax & multiply by V
    Tensor probs = ops::softmax(scores);
    return ops::matmul(probs, V);
}
```
Swap out the `flash_attention` call in `engine/model/layers.cc` for `debug_attention`, recompile `bazel run //engine:inference`, and you can inspect the attention maps.
