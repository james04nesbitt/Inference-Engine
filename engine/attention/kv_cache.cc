#include "engine/attention/kv_cache.h"

#include <stdexcept>

namespace ie {

KVCacheManager::KVCacheManager(int32_t num_layers, int32_t num_kv_heads,
                               int32_t head_dim, int32_t max_blocks,
                               int32_t block_size)
    : num_layers_(num_layers),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      block_size_(block_size) {
  // TODO: Pre-allocate the block pool
  //
  // For each block:
  //   block.keys = Tensor({block_size, num_kv_heads, head_dim});
  //   block.values = Tensor({block_size, num_kv_heads, head_dim});
  //
  // Initialize the free list with all block indices.
  //
  // Memory calculation:
  //   Each block = 2 * block_size * num_kv_heads * head_dim * sizeof(float)
  //   For Gemma 3 1B: 2 * 16 * 1 * 256 * 4 = 32 KB per block
  //   1000 blocks = ~32 MB total KV cache
}

int64_t KVCacheManager::AllocateSequence() {
  // TODO: Assign initial blocks for a new sequence
  //
  // 1. Assign a new seq_id
  // 2. Create an empty block table entry for all layers
  // 3. Allocate one initial block per layer from the free list
  // 4. Return the seq_id
  //
  throw std::runtime_error("AllocateSequence not implemented yet");
}

void KVCacheManager::FreeSequence(int64_t seq_id) {
  // TODO: Return all blocks for this sequence to the free list
  //
  // 1. Look up block_tables_[seq_id]
  // 2. For each layer, return all block indices to free_blocks_
  // 3. Remove the block table entry
  //
  throw std::runtime_error("FreeSequence not implemented yet");
}

void KVCacheManager::AppendToken(int64_t seq_id, int32_t layer_idx,
                                 const Tensor& key, const Tensor& value) {
  // TODO: Append a single token's K/V to the cache
  //
  // 1. Find the last block for this sequence + layer
  // 2. If the block is full (num_filled == block_size_):
  //    a. Allocate a new block from the free list
  //    b. Add its index to the block table
  // 3. Copy key/value data into the block at position num_filled
  // 4. Increment num_filled
  //
  throw std::runtime_error("AppendToken not implemented yet");
}

Tensor KVCacheManager::GetKeys(int64_t seq_id, int32_t layer_idx) const {
  // TODO: Reconstruct contiguous keys from paged blocks
  //
  // 1. Look up block_tables_[seq_id][layer_idx]
  // 2. Allocate output tensor [total_tokens, num_kv_heads, head_dim]
  // 3. Copy data from each block into the output
  //
  throw std::runtime_error("GetKeys not implemented yet");
}

Tensor KVCacheManager::GetValues(int64_t seq_id, int32_t layer_idx) const {
  // TODO: Same as GetKeys but for values
  throw std::runtime_error("GetValues not implemented yet");
}

int32_t KVCacheManager::NumFreeBlocks() const {
  return static_cast<int32_t>(free_blocks_.size());
}

int32_t KVCacheManager::SequenceLength(int64_t seq_id) const {
  // TODO: Calculate total tokens across all blocks for this sequence
  throw std::runtime_error("SequenceLength not implemented yet");
}

}  // namespace ie
