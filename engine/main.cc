#include <iostream>
#include <string>

#include "engine/engine.h"

void PrintUsage() {
  std::cout
      << "Inference Engine — Gemma-3 GGUF Model Runner\n"
      << "\n"
      << "Usage:\n"
      << "  inference --model <path> [options]\n"
      << "\n"
      << "Options:\n"
      << "  --model <path>       Path to GGUF model file (required)\n"
      << "  --prompt <text>      Input prompt (default: \"Hello, I am\")\n"
      << "  --max-tokens <n>     Max tokens to generate (default: 128)\n"
      << "  --temperature <f>    Sampling temperature (default: 1.0)\n"
      << "  --top-k <n>          Top-K sampling (default: 40)\n"
      << "  --top-p <f>          Top-P / nucleus sampling (default: 0.9)\n"
      << "  --greedy             Use greedy sampling (default)\n"
      << "  --interactive        Interactive multi-turn mode\n"
      << "  --help               Show this help\n"
      << std::endl;
}

int main(int argc, char *argv[]) {
  std::string model_path;
  std::string prompt = "Hello, I am";
  int32_t max_tokens = 128;
  bool interactive = false;

  ie::SamplingConfig sampling;
  sampling.strategy = ie::SamplingStrategy::kGreedy;

  // Parse command-line arguments.
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      model_path = argv[++i];
    } else if (arg == "--prompt" && i + 1 < argc) {
      prompt = argv[++i];
    } else if (arg == "--max-tokens" && i + 1 < argc) {
      max_tokens = std::stoi(argv[++i]);
    } else if (arg == "--temperature" && i + 1 < argc) {
      sampling.temperature = std::stof(argv[++i]);
      if (sampling.strategy == ie::SamplingStrategy::kGreedy) {
        sampling.strategy = ie::SamplingStrategy::kTopK;
      }
    } else if (arg == "--top-k" && i + 1 < argc) {
      sampling.top_k = std::stoi(argv[++i]);
      sampling.strategy = ie::SamplingStrategy::kTopK;
    } else if (arg == "--top-p" && i + 1 < argc) {
      sampling.top_p = std::stof(argv[++i]);
      sampling.strategy = ie::SamplingStrategy::kTopP;
    } else if (arg == "--greedy") {
      sampling.strategy = ie::SamplingStrategy::kGreedy;
    } else if (arg == "--interactive") {
      interactive = true;
    } else if (arg == "--help") {
      PrintUsage();
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      PrintUsage();
      return 1;
    }
  }

  if (model_path.empty()) {
    std::cerr << "Error: --model path is required\n" << std::endl;
    PrintUsage();
    return 1;
  }

  // Load model.
  ie::InferenceEngine engine;
  if (!engine.LoadModel(model_path)) {
    std::cerr << "Failed to load model from: " << model_path << std::endl;
    return 1;
  }

  // Token streaming callback: print each token to stdout as it's generated.
  auto stream_token = [](const std::string &token) {
    std::cout << token << std::flush;
  };

  if (interactive) {
    // ====================================================================
    // Interactive mode: multi-turn prompt loop
    // ====================================================================
    std::cout << "\n=== Interactive Mode ===\n"
              << "Type your prompt and press Enter. Type 'quit' to exit.\n"
              << std::endl;

    while (true) {
      std::cout << "> " << std::flush;
      std::string line;
      if (!std::getline(std::cin, line) || line == "quit" || line == "exit") {
        std::cout << "\nGoodbye!" << std::endl;
        break;
      }
      if (line.empty())
        continue;

      // Clear KV caches between prompts.
      engine.ClearCache();

      std::cout << "\n";
      try {
        engine.GenerateStreaming(line, max_tokens, sampling, stream_token);
      } catch (const std::runtime_error &e) {
        std::cerr << "\n[Error: " << e.what() << "]" << std::endl;
      }
      std::cout << "\n" << std::endl;
    }
  } else {
    // ====================================================================
    // Single-shot mode
    // ====================================================================
    std::cout << "\nPrompt: " << prompt << std::endl;
    std::cout << "Max tokens: " << max_tokens << "\n" << std::endl;

    try {
      engine.GenerateStreaming(prompt, max_tokens, sampling, stream_token);
      std::cout << std::endl;
    } catch (const std::runtime_error &e) {
      std::cerr << "\n[Error: " << e.what() << "]" << std::endl;
      return 1;
    }
  }

  return 0;
}
