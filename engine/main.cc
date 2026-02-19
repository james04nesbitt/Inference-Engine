#include <iostream>
#include <string>

#include "engine/engine.h"

void PrintUsage() {
  std::cout << "Usage: inference --model <path> [--prompt <text>] "
               "[--max-tokens <n>]"
            << std::endl;
}

int main(int argc, char* argv[]) {
  std::string model_path;
  std::string prompt = "Hello, I am";
  int32_t max_tokens = 128;

  // Parse command-line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      model_path = argv[++i];
    } else if (arg == "--prompt" && i + 1 < argc) {
      prompt = argv[++i];
    } else if (arg == "--max-tokens" && i + 1 < argc) {
      max_tokens = std::stoi(argv[++i]);
    } else if (arg == "--help") {
      PrintUsage();
      return 0;
    }
  }

  if (model_path.empty()) {
    std::cerr << "Error: --model path is required" << std::endl;
    PrintUsage();
    return 1;
  }

  // Create and run the inference engine
  ie::InferenceEngine engine;

  if (!engine.LoadModel(model_path)) {
    std::cerr << "Failed to load model from: " << model_path << std::endl;
    return 1;
  }

  std::cout << "\nPrompt: " << prompt << std::endl;
  std::cout << "Max tokens: " << max_tokens << std::endl;
  std::cout << "\n--- Generation ---" << std::endl;

  try {
    std::string output = engine.Generate(prompt, max_tokens);
    std::cout << output << std::endl;
  } catch (const std::runtime_error& e) {
    std::cout << "[Generation not yet implemented: " << e.what() << "]"
              << std::endl;
    std::cout << "\nNext step: Implement the GGUF metadata parser, then "
                 "the tensor loader, then the forward pass!"
              << std::endl;
  }

  return 0;
}
