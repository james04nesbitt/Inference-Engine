#include "engine/attention/kv_cache.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ie {

KVCacheManager::KVCacheManager(int32_t num_layers, int32_t num_kv_heads,
                               int32_t head_dim, int32_t max_blocks,
                               int32_t block_size)
    : num_layers_(num_layers), num_kv_heads_(num_kv_heads), head_dim_(head_dim),
      block_size_(block_size) {
  // Pre-allocate the block pool.
  // Each block stores block_size tokens of K and V data for one layer.
  //   keys:   [block_size, num_kv_heads, head_dim]
  //   values: [block_size, num_kv_heads, head_dim]
  //
  // Memory per block = 2 * block_size * num_kv_heads * head_dim * sizeof(float)
  // For Gemma 3 1B: 2 * 16 * 1 * 256 * 4 = 32 KB per block
  block_pool_.resize(max_blocks);
  for (int32_t i = 0; i < max_blocks; ++i) {
    block_pool_[i].keys = Tensor({block_size, num_kv_heads, head_dim});
    block_pool_[i].values = Tensor({block_size, num_kv_heads, head_dim});
    block_pool_[i].num_filled = 0;
    free_blocks_.push_back(i);
  }
}

int64_t KVCacheManager::AllocateSequence() {
  int64_t seq_id = next_seq_id_++;

  // Create block table: one list of block indices per layer.
  auto &table = block_tables_[seq_id];
  table.resize(num_layers_);

  // Allocate one initial block per layer.
  for (int32_t layer = 0; layer < num_layers_; ++layer) {
    if (free_blocks_.empty()) {
      throw std::runtime_error(
          "KVCacheManager: out of free blocks during AllocateSequence");
    }
    int32_t block_idx = free_blocks_.back();
    free_blocks_.pop_back();
    block_pool_[block_idx].num_filled = 0;
    table[layer].push_back(block_idx);
  }

  return seq_id;
}

void KVCacheManager::FreeSequence(int64_t seq_id) {
  auto it = block_tables_.find(seq_id);
  if (it == block_tables_.end()) {
    throw std::runtime_error("KVCacheManager: unknown sequence ID");
  }

  // Return all blocks for this sequence to the free list.
  for (auto &layer_blocks : it->second) {
    for (int32_t block_idx : layer_blocks) {
      block_pool_[block_idx].num_filled = 0;
      free_blocks_.push_back(block_idx);
    }
  }
  block_tables_.erase(it);
}

void KVCacheManager::AppendToken(int64_t seq_id, int32_t layer_idx,
                                 const Tensor &key, const Tensor &value) {
  auto it = block_tables_.find(seq_id);
  if (it == block_tables_.end()) {
    throw std::runtime_error("KVCacheManager: unknown sequence ID");
  }

  auto &layer_blocks = it->second[layer_idx];
  int32_t last_block_idx = layer_blocks.back();
  KVCacheBlock &block = block_pool_[last_block_idx];

  // If the current block is full, allocate a new one.
  if (block.num_filled >= block_size_) {
    if (free_blocks_.empty()) {
      throw std::runtime_error(
          "KVCacheManager: out of free blocks during AppendToken");
    }
    int32_t new_block_idx = free_blocks_.back();
    free_blocks_.pop_back();
    block_pool_[new_block_idx].num_filled = 0;
    layer_blocks.push_back(new_block_idx);
    last_block_idx = new_block_idx;
  }

  KVCacheBlock &target_block = block_pool_[last_block_idx];
  int32_t slot = target_block.num_filled;

  // Copy key data into the block at position slot.
  // key shape is [num_kv_heads, head_dim], copy into block.keys[slot, :, :]
  const float *key_src = key.data<float>();
  float *key_dst =
      target_block.keys.data<float>() + slot * num_kv_heads_ * head_dim_;
  std::memcpy(key_dst, key_src, num_kv_heads_ * head_dim_ * sizeof(float));

  // Copy value data.
  const float *val_src = value.data<float>();
  float *val_dst =
      target_block.values.data<float>() + slot * num_kv_heads_ * head_dim_;
  std::memcpy(val_dst, val_src, num_kv_heads_ * head_dim_ * sizeof(float));

  target_block.num_filled++;
}

Tensor KVCacheManager::GetKeys(int64_t seq_id, int32_t layer_idx) const {
  auto it = block_tables_.find(seq_id);
  if (it == block_tables_.end()) {
    throw std::runtime_error("KVCacheManager: unknown sequence ID");
  }

  const auto &layer_blocks = it->second[layer_idx];

  // Count total tokens across all blocks.
  int32_t total_tokens = 0;
  for (int32_t block_idx : layer_blocks) {
    total_tokens += block_pool_[block_idx].num_filled;
  }

  // Allocate output tensor [total_tokens, num_kv_heads, head_dim].
  Tensor output({total_tokens, num_kv_heads_, head_dim_});
  float *out_ptr = output.data<float>();

  // Copy data from each block into contiguous output.
  int32_t tokens_copied = 0;
  for (int32_t block_idx : layer_blocks) {
    const KVCacheBlock &block = block_pool_[block_idx];
    int32_t n = block.num_filled;
    if (n <= 0)
      continue;

    const float *src = block.keys.data<float>();
    std::memcpy(out_ptr + tokens_copied * num_kv_heads_ * head_dim_, src,
                n * num_kv_heads_ * head_dim_ * sizeof(float));
    tokens_copied += n;
  }

  return output;
}

Tensor KVCacheManager::GetValues(int64_t seq_id, int32_t layer_idx) const {
  auto it = block_tables_.find(seq_id);
  if (it == block_tables_.end()) {
    throw std::runtime_error("KVCacheManager: unknown sequence ID");
  }

  const auto &layer_blocks = it->second[layer_idx];

  int32_t total_tokens = 0;
  for (int32_t block_idx : layer_blocks) {
    total_tokens += block_pool_[block_idx].num_filled;
  }

  Tensor output({total_tokens, num_kv_heads_, head_dim_});
  float *out_ptr = output.data<float>();

  int32_t tokens_copied = 0;
  for (int32_t block_idx : layer_blocks) {
    const KVCacheBlock &block = block_pool_[block_idx];
    int32_t n = block.num_filled;
    if (n <= 0)
      continue;

    const float *src = block.values.data<float>();
    std::memcpy(out_ptr + tokens_copied * num_kv_heads_ * head_dim_, src,
                n * num_kv_heads_ * head_dim_ * sizeof(float));
    tokens_copied += n;
  }

  return output;
}

int32_t KVCacheManager::NumFreeBlocks() const {
  return static_cast<int32_t>(free_blocks_.size());
}

int32_t KVCacheManager::SequenceLength(int64_t seq_id) const {
  auto it = block_tables_.find(seq_id);
  if (it == block_tables_.end()) {
    throw std::runtime_error("KVCacheManager: unknown sequence ID");
  }

  // Sum filled slots across all blocks in layer 0 (all layers have same count).
  int32_t total = 0;
  const auto &layer_blocks = it->second[0];
  for (int32_t block_idx : layer_blocks) {
    total += block_pool_[block_idx].num_filled;
  }
  return total;
}

} // namespace ie
