#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "engine/tensor/tensor.h"

namespace ie {

// ============================================================================
// PagedAttention KV Cache Manager
//
// Traditional KV caches pre-allocate a contiguous buffer per sequence,
// wasting memory when sequences have different lengths. PagedAttention
// (from vLLM) solves this by splitting the KV cache into fixed-size pages
// ("blocks") and using a block table to map logical positions to physical
// pages — just like OS virtual memory!
//
// Key concepts:
//   - Block: a fixed-size chunk of KV data (e.g., 16 tokens worth)
//   - Block Table: per-sequence mapping from logical block index to physical
//   - Free List: pool of available blocks for allocation
//   - Copy-on-Write: enables beam search without duplicating KV data
//
// Papers to read:
//   - "Efficient Memory Management for Large Language Model Serving with
//     PagedAttention" (Kwon et al., 2023): https://arxiv.org/abs/2309.06180
// ============================================================================

// Number of token positions stored in each KV cache block.
constexpr int32_t kDefaultBlockSize = 16;

// A single physical block of KV cache memory.
// Stores block_size tokens worth of keys and values for one layer.
struct KVCacheBlock {
  Tensor keys;    // [block_size, num_kv_heads, head_dim]
  Tensor values;  // [block_size, num_kv_heads, head_dim]
  int32_t num_filled = 0;  // How many token slots are used
};

// Manages the pool of KV cache blocks across all sequences and layers.
//
// This is the memory allocator for attention. It:
//   1. Pre-allocates a pool of blocks at startup
//   2. Assigns blocks to sequences as they grow
//   3. Reclaims blocks when sequences finish
//   4. Supports copy-on-write for beam search
//
class KVCacheManager {
 public:
  // Create a cache manager for the given model dimensions.
  //
  // Parameters:
  //   num_layers:    number of transformer layers (each needs its own KV cache)
  //   num_kv_heads:  number of key/value heads (GQA)
  //   head_dim:      dimension per head
  //   max_blocks:    total physical blocks in the pool
  //   block_size:    number of tokens per block
  //
  KVCacheManager(int32_t num_layers, int32_t num_kv_heads, int32_t head_dim,
                 int32_t max_blocks, int32_t block_size = kDefaultBlockSize);

  ~KVCacheManager() = default;

  // Allocate blocks for a new sequence. Returns a sequence handle ID.
  // TODO: Implement — assign initial block(s), create block table entry
  int64_t AllocateSequence();

  // Free all blocks associated with a sequence.
  // TODO: Implement — return blocks to free list
  void FreeSequence(int64_t seq_id);

  // Append a new token's KV data to a sequence's cache.
  // If the current block is full, allocate a new one.
  // TODO: Implement
  void AppendToken(int64_t seq_id, int32_t layer_idx,
                   const Tensor& key, const Tensor& value);

  // Get all cached keys/values for a sequence at a given layer.
  // This reconstructs the full KV history from the page table.
  // TODO: Implement
  Tensor GetKeys(int64_t seq_id, int32_t layer_idx) const;
  Tensor GetValues(int64_t seq_id, int32_t layer_idx) const;

  // Number of free blocks remaining.
  int32_t NumFreeBlocks() const;

  // Total number of tokens cached for a sequence.
  int32_t SequenceLength(int64_t seq_id) const;

 private:
  int32_t num_layers_;
  int32_t num_kv_heads_;
  int32_t head_dim_;
  int32_t block_size_;

  // Physical block pool: all pre-allocated blocks
  std::vector<KVCacheBlock> block_pool_;

  // Free list: indices into block_pool_ of available blocks
  std::vector<int32_t> free_blocks_;

  // Block table: seq_id -> layer_idx -> list of physical block indices
  // block_tables_[seq_id][layer_idx] = {block_idx_0, block_idx_1, ...}
  std::unordered_map<int64_t, std::vector<std::vector<int32_t>>> block_tables_;

  int64_t next_seq_id_ = 0;
};

}  // namespace ie
