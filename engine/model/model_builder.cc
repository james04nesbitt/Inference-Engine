#include "engine/model/model_builder.h"

#include <iostream>
#include <stdexcept>
#include <string>

#include "engine/tokenizer/bpe_tokenizer.h"

namespace ie {

// ============================================================================
// BuildGemmaModel — Reads config from GGUF metadata, loads all weights,
// constructs transformer layers, and creates the KV cache.
// ============================================================================
ModelBundle BuildGemmaModel(GGUFFile &gguf) {
  ModelBundle bundle;
  auto &config = bundle.config;

  // Read model architecture name to determine key prefix.
  // Gemma 3 GGUFs use "gemma3." prefix; some older ones use "gemma.".
  std::string arch = gguf.GetString("general.architecture", "gemma3");

  auto getInt = [&](const std::string &suffix, int64_t def) -> int64_t {
    int64_t val = gguf.GetInt(arch + "." + suffix, -1);
    if (val != -1)
      return val;
    return gguf.GetInt("gemma." + suffix, def);
  };
  auto getFloat = [&](const std::string &suffix, float def) -> float {
    float val = gguf.GetFloat(arch + "." + suffix, -1.0f);
    if (val != -1.0f)
      return val;
    return gguf.GetFloat("gemma." + suffix, def);
  };

  config.num_layers = static_cast<int32_t>(getInt("block_count", 26));
  config.embed_dim = static_cast<int32_t>(getInt("embedding_length", 1152));
  config.num_heads = static_cast<int32_t>(getInt("attention.head_count", 4));
  config.num_kv_heads =
      static_cast<int32_t>(getInt("attention.head_count_kv", 1));
  config.head_dim = static_cast<int32_t>(getInt("attention.key_length", 256));
  config.hidden_dim =
      static_cast<int32_t>(getInt("feed_forward_length", 6912));
  config.vocab_size = static_cast<int32_t>(getInt("vocab_size", 262144));
  config.rms_norm_eps = getFloat("attention.layer_norm_rms_epsilon", 1e-6f);
  config.rope_theta_global = gguf.GetFloat(
      arch + ".rope.freq_base", gguf.GetFloat("gemma.rope.freq_base", 1e6f));
  config.rope_theta_local = 10000.0f;
  config.sliding_window =
      static_cast<int32_t>(getInt("attention.sliding_window", 512));
  config.max_seq_len =
      static_cast<int32_t>(getInt("context_length", 32768));

  std::cout << "\nModel config:"
            << "\n  layers:       " << config.num_layers
            << "\n  embed_dim:    " << config.embed_dim
            << "\n  num_heads:    " << config.num_heads
            << "\n  num_kv_heads: " << config.num_kv_heads
            << "\n  head_dim:     " << config.head_dim
            << "\n  hidden_dim:   " << config.hidden_dim
            << "\n  vocab_size:   " << config.vocab_size
            << "\n  max_seq_len:  " << config.max_seq_len
            << "\n  rope_global:  " << config.rope_theta_global
            << "\n  rope_local:   " << config.rope_theta_local
            << "\n  sliding_win:  " << config.sliding_window << std::endl;

  // Helper: load a 2D weight and transpose it to F32.
  // BF16→F32 conversion happens in .to(kFloat32), transpose in .transpose().
  auto loadTransposed = [&](const std::string &name) -> Tensor {
    Tensor t = gguf.LoadTensor(name);
    return t.transpose(0, 1).to(DType::kFloat32).contiguous();
  };

  Tensor token_embedding =
      gguf.LoadTensor("token_embd.weight").to(DType::kFloat32).contiguous();
  // Pre-transpose embedding for logit projection: [vocab, embed] → [embed,
  // vocab]
  Tensor embed_t = loadTransposed("token_embd.weight");

  std::vector<TransformerBlock> layers;
  layers.reserve(config.num_layers);

  for (int32_t i = 0; i < config.num_layers; ++i) {
    std::string prefix = "blk." + std::to_string(i) + ".";

    // Load weight matrices: BF16 → transpose → F32 contiguous.
    Tensor wq_t = loadTransposed(prefix + "attn_q.weight");
    Tensor wk_t = loadTransposed(prefix + "attn_k.weight");
    Tensor wv_t = loadTransposed(prefix + "attn_v.weight");
    Tensor wo_t = loadTransposed(prefix + "attn_output.weight");

    Tensor q_norm_w;
    if (gguf.GetTensorInfo(prefix + "attn_q_norm.weight")) {
      q_norm_w = gguf.LoadTensor(prefix + "attn_q_norm.weight")
                     .to(DType::kFloat32)
                     .contiguous();
    }
    Tensor k_norm_w;
    if (gguf.GetTensorInfo(prefix + "attn_k_norm.weight")) {
      k_norm_w = gguf.LoadTensor(prefix + "attn_k_norm.weight")
                     .to(DType::kFloat32)
                     .contiguous();
    }

    Tensor gate_t = loadTransposed(prefix + "ffn_gate.weight");
    Tensor up_t = loadTransposed(prefix + "ffn_up.weight");
    Tensor down_t = loadTransposed(prefix + "ffn_down.weight");

    Tensor attn_norm_w = gguf.LoadTensor(prefix + "attn_norm.weight")
                             .to(DType::kFloat32)
                             .contiguous();
    Tensor post_attn_norm_w =
        gguf.LoadTensor(prefix + "post_attention_norm.weight")
            .to(DType::kFloat32)
            .contiguous();

    Tensor ffn_norm_w = gguf.LoadTensor(prefix + "ffn_norm.weight")
                            .to(DType::kFloat32)
                            .contiguous();
    Tensor post_ffn_norm_w = gguf.LoadTensor(prefix + "post_ffw_norm.weight")
                                 .to(DType::kFloat32)
                                 .contiguous();

    RMSNorm attn_norm(std::move(attn_norm_w), config.rms_norm_eps);
    Attention attn(config, i, std::move(wq_t), std::move(wk_t),
                   std::move(wv_t), std::move(wo_t), std::move(q_norm_w),
                   std::move(k_norm_w));
    RMSNorm post_attn_norm(std::move(post_attn_norm_w), config.rms_norm_eps);
    RMSNorm ffn_norm(std::move(ffn_norm_w), config.rms_norm_eps);
    FeedForward ffn(std::move(gate_t), std::move(up_t), std::move(down_t));
    RMSNorm post_ffn_norm(std::move(post_ffn_norm_w), config.rms_norm_eps);

    layers.emplace_back(config, i, std::move(attn_norm), std::move(attn),
                        std::move(post_attn_norm), std::move(ffn_norm),
                        std::move(ffn), std::move(post_ffn_norm));

    std::cout << "  Loaded block " << i << std::endl;
  }

  Tensor final_norm_w =
      gguf.LoadTensor("output_norm.weight").to(DType::kFloat32).contiguous();
  RMSNorm final_norm(std::move(final_norm_w), config.rms_norm_eps);

  bundle.model = std::make_unique<GemmaModel>(
      config, std::move(token_embedding), std::move(layers),
      std::move(final_norm), std::move(embed_t));

  // Create the paged KV cache.
  int32_t blocks_per_seq =
      (config.max_seq_len + kDefaultBlockSize - 1) / kDefaultBlockSize;
  // Allocate enough blocks for a few concurrent sequences.
  int32_t max_blocks =
      blocks_per_seq * config.num_layers * 4 + config.num_layers;

  bundle.kv_cache = std::make_unique<KVCacheManager>(
      config.num_layers, config.num_kv_heads, config.head_dim, max_blocks,
      kDefaultBlockSize);

  std::cout << "Model built successfully (" << config.num_layers << " layers, "
            << max_blocks << " KV cache blocks)" << std::endl;
  return bundle;
}

// ============================================================================
// BuildTokenizer — Reads vocab, scores, and special token IDs from GGUF.
// ============================================================================
std::unique_ptr<Tokenizer> BuildTokenizer(GGUFFile &gguf) {
  auto *tokens_val = gguf.GetMetadata("tokenizer.ggml.tokens");
  if (!tokens_val) {
    std::cerr << "BuildTokenizer: missing tokenizer.ggml.tokens" << std::endl;
    return nullptr;
  }
  auto *vocab = std::get_if<std::vector<std::string>>(tokens_val);
  if (!vocab) {
    std::cerr << "BuildTokenizer: tokenizer.ggml.tokens is not a string array"
              << std::endl;
    return nullptr;
  }

  auto *scores_val = gguf.GetMetadata("tokenizer.ggml.scores");
  if (!scores_val) {
    std::cerr << "BuildTokenizer: missing tokenizer.ggml.scores" << std::endl;
    return nullptr;
  }
  auto *scores = std::get_if<std::vector<float>>(scores_val);
  if (!scores) {
    std::cerr << "BuildTokenizer: tokenizer.ggml.scores is not a float array"
              << std::endl;
    return nullptr;
  }

  int32_t bos_id =
      static_cast<int32_t>(gguf.GetInt("tokenizer.ggml.bos_token_id", 2));
  int32_t eos_id =
      static_cast<int32_t>(gguf.GetInt("tokenizer.ggml.eos_token_id", 1));
  int32_t pad_id =
      static_cast<int32_t>(gguf.GetInt("tokenizer.ggml.padding_token_id", 0));

  auto tokenizer =
      std::make_unique<BPETokenizer>(*vocab, *scores, bos_id, eos_id, pad_id);

  std::cout << "Tokenizer built: " << tokenizer->VocabSize() << " tokens"
            << ", BOS=" << bos_id << ", EOS=" << eos_id << ", PAD=" << pad_id
            << std::endl;
  return tokenizer;
}

// ============================================================================
// ApplyGemma3ChatTemplate — Wraps user text in the Gemma 3 IT format.
//
// The Gemma 3 chat template from the GGUF is a Jinja2 template.
// For single-turn user prompts we hardcode the simplified output:
//
//   <start_of_turn>user
//   {message}<end_of_turn>
//   <start_of_turn>model
//
// Note: BOS is added separately by the generation code.
// ============================================================================
std::string ApplyGemma3ChatTemplate(const std::string &user_message) {
  return "<start_of_turn>user\n" + user_message +
         "<end_of_turn>\n<start_of_turn>model\n";
}

}  // namespace ie
