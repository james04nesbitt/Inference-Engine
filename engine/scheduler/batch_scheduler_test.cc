#include "engine/scheduler/batch_scheduler.h"

#include <string>
#include <vector>

#include "engine/attention/kv_cache.h"
#include "engine/model/config.h"
#include "engine/model/layers.h"
#include "engine/tokenizer/bpe_tokenizer.h"
#include "gtest/gtest.h"

namespace ie {
namespace {

// ============================================================================
// Helper: Create a tiny model + tokenizer for testing the scheduler.
// ============================================================================

class BatchSchedulerTest : public ::testing::Test {
protected:
  static constexpr int32_t kEmbedDim = 4;
  static constexpr int32_t kNumHeads = 1;
  static constexpr int32_t kNumKvHeads = 1;
  static constexpr int32_t kHeadDim = 4;
  static constexpr int32_t kHiddenDim = 8;
  static constexpr int32_t kNumLayers = 1;
  static constexpr int32_t kVocabSize = 10;

  GemmaConfig MakeConfig() {
    GemmaConfig config;
    config.embed_dim = kEmbedDim;
    config.num_heads = kNumHeads;
    config.num_kv_heads = kNumKvHeads;
    config.head_dim = kHeadDim;
    config.hidden_dim = kHiddenDim;
    config.num_layers = kNumLayers;
    config.vocab_size = kVocabSize;
    config.max_seq_len = 64;
    config.rms_norm_eps = 1e-6f;
    config.rope_theta_local = 10000.0f;
    config.rope_theta_global = 1000000.0f;
    return config;
  }

  std::unique_ptr<GemmaModel> MakeModel(const GemmaConfig &config) {
    Tensor token_embedding = Tensor::full(
        {static_cast<int64_t>(kVocabSize), static_cast<int64_t>(kEmbedDim)},
        0.1f);

    std::vector<TransformerBlock> layers;
    for (int32_t i = 0; i < kNumLayers; ++i) {
      RMSNorm attn_norm(Tensor::ones({kEmbedDim}), config.rms_norm_eps);
      // Pre-transposed weights: [in, out] layout for matmul(x, W)
      // Q/K/V projections: input=embed_dim, output varies
      Attention attn(config, i,
                     Tensor::full({kEmbedDim, static_cast<int64_t>(kNumHeads * kHeadDim)}, 0.1f),  // wq_t [embed, q_dim]
                     Tensor::full({kEmbedDim, static_cast<int64_t>(kNumKvHeads * kHeadDim)}, 0.1f), // wk_t [embed, kv_dim]
                     Tensor::full({kEmbedDim, static_cast<int64_t>(kNumKvHeads * kHeadDim)}, 0.1f), // wv_t [embed, kv_dim]
                     Tensor::full({static_cast<int64_t>(kNumHeads * kHeadDim), kEmbedDim}, 0.1f));  // wo_t [q_dim, embed]
      RMSNorm post_attn_norm(Tensor::zeros({kEmbedDim}), config.rms_norm_eps);
      RMSNorm ffn_norm(Tensor::ones({kEmbedDim}), config.rms_norm_eps);
      // FFN weights: gate/up [embed, hidden], down [hidden, embed]
      FeedForward ffn(Tensor::full({kEmbedDim, kHiddenDim}, 0.1f),
                      Tensor::full({kEmbedDim, kHiddenDim}, 0.1f),
                      Tensor::full({kHiddenDim, kEmbedDim}, 0.1f));
      RMSNorm post_ffn_norm(Tensor::zeros({kEmbedDim}), config.rms_norm_eps);

      layers.emplace_back(config, i, std::move(attn_norm), std::move(attn),
                          std::move(post_attn_norm), std::move(ffn_norm),
                          std::move(ffn), std::move(post_ffn_norm));
    }

    RMSNorm final_norm(Tensor::ones({kEmbedDim}), config.rms_norm_eps);
    // Create a transposed embedding for logit projection.
    Tensor embed_t = token_embedding.transpose(0, 1).contiguous();
    return std::make_unique<GemmaModel>(config, std::move(token_embedding),
                                        std::move(layers),
                                        std::move(final_norm),
                                        std::move(embed_t));
  }

  std::unique_ptr<BPETokenizer> MakeTokenizer() {
    // Tiny vocab: tokens 0-9 map to strings "0" through "9".
    std::vector<std::string> vocab;
    std::vector<float> scores;
    for (int i = 0; i < kVocabSize; ++i) {
      vocab.push_back(std::to_string(i));
      scores.push_back(0.0f);
    }
    return std::make_unique<BPETokenizer>(vocab, scores,
                                          /*bos_id=*/2, /*eos_id=*/1,
                                          /*pad_id=*/0);
  }
};

TEST_F(BatchSchedulerTest, SingleRequest) {
  auto config = MakeConfig();
  auto model = MakeModel(config);
  auto tokenizer = MakeTokenizer();

  KVCacheManager kv_cache(kNumLayers, kNumKvHeads, kHeadDim,
                          /*max_blocks=*/50, /*block_size=*/16);

  BatchScheduler scheduler(*model, *tokenizer, kv_cache);

  std::string result;
  Request req;
  req.prompt = "3";
  req.max_tokens = 5;
  req.on_complete = [&result](const std::string &text) { result = text; };

  scheduler.AddRequest(std::move(req));
  EXPECT_EQ(scheduler.NumPending(), 1);
  EXPECT_EQ(scheduler.NumActive(), 0);

  scheduler.Run();

  EXPECT_EQ(scheduler.NumPending(), 0);
  EXPECT_EQ(scheduler.NumActive(), 0);
  EXPECT_EQ(scheduler.NumCompleted(), 1);
  EXPECT_FALSE(result.empty());
}

TEST_F(BatchSchedulerTest, MultipleRequests) {
  auto config = MakeConfig();
  auto model = MakeModel(config);
  auto tokenizer = MakeTokenizer();

  KVCacheManager kv_cache(kNumLayers, kNumKvHeads, kHeadDim,
                          /*max_blocks=*/100, /*block_size=*/16);

  BatchScheduler scheduler(*model, *tokenizer, kv_cache);

  int completed = 0;
  std::vector<std::string> results(3);

  for (int i = 0; i < 3; ++i) {
    Request req;
    req.prompt = std::to_string(i + 3);
    req.max_tokens = 3;
    req.on_complete = [&results, &completed, i](const std::string &text) {
      results[i] = text;
      completed++;
    };
    scheduler.AddRequest(std::move(req));
  }

  EXPECT_EQ(scheduler.NumPending(), 3);

  scheduler.Run();

  EXPECT_EQ(completed, 3);
  EXPECT_EQ(scheduler.NumCompleted(), 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_FALSE(results[i].empty()) << "Result " << i << " is empty";
  }
}

TEST_F(BatchSchedulerTest, StreamingCallback) {
  auto config = MakeConfig();
  auto model = MakeModel(config);
  auto tokenizer = MakeTokenizer();

  KVCacheManager kv_cache(kNumLayers, kNumKvHeads, kHeadDim,
                          /*max_blocks=*/50, /*block_size=*/16);

  BatchScheduler scheduler(*model, *tokenizer, kv_cache);

  int stream_count = 0;
  Request req;
  req.prompt = "3";
  req.max_tokens = 4;
  req.on_token = [&stream_count](const std::string &) { stream_count++; };

  scheduler.AddRequest(std::move(req));
  scheduler.Run();

  // Should have received token callbacks (at least 1 from prefill + decode).
  EXPECT_GT(stream_count, 0);
}

TEST_F(BatchSchedulerTest, StepByStep) {
  auto config = MakeConfig();
  auto model = MakeModel(config);
  auto tokenizer = MakeTokenizer();

  KVCacheManager kv_cache(kNumLayers, kNumKvHeads, kHeadDim,
                          /*max_blocks=*/50, /*block_size=*/16);

  BatchScheduler scheduler(*model, *tokenizer, kv_cache);

  Request req;
  req.prompt = "3";
  req.max_tokens = 3;
  scheduler.AddRequest(std::move(req));

  // Step manually.
  int32_t active = scheduler.Step(); // Should prefill + decode first token.
  EXPECT_GE(active, 0);

  // Keep stepping until done.
  while (active > 0) {
    active = scheduler.Step();
  }

  EXPECT_EQ(scheduler.NumCompleted(), 1);
}

} // namespace
} // namespace ie
