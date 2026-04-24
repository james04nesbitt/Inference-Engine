#include "engine/gguf/gguf_loader.h"

#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  std::string model_path =
      "C:/Users/james/Coding/Projects/Inference-Engine/bazel-inference-engine/"
      "model/gemma-3-1b-it-BF16.gguf";
  bool debug = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--debug" || arg == "-d") {
      debug = true;
    } else {
      model_path = arg;
    }
  }

  std::cout << "Loading GGUF file: " << model_path << std::endl;

  ie::GGUFFile gguf;
  if (!gguf.Open(model_path, debug)) {
    std::cerr << "Failed to open GGUF file!" << std::endl;
    return 1;
  }

  if (!debug) {
    // Print the file summary if debug wasn't already printed
    gguf.PrintSummary();
  }

  // List all tensor names
  auto names = gguf.TensorNames();
  std::cout << "\nTotal tensors: " << names.size() << std::endl;

  // Try loading the first tensor
  if (!names.empty()) {
    const std::string &first_name = names[0];
    std::cout << "\n--- Loading tensor: \"" << first_name << "\" ---"
              << std::endl;
    try {
      ie::Tensor t = gguf.LoadTensor(first_name);
      std::cout << "  " << t.to_string() << std::endl;
      std::cout << "  nbytes: " << t.nbytes() << std::endl;
      std::cout << "  is_contiguous: " << (t.is_contiguous() ? "yes" : "no")
                << std::endl;

      // Print first few values (via .at() which handles BF16→F32 conversion)
      int64_t n = std::min(t.numel(), int64_t(8));
      std::cout << "  first " << n << " values: [";
      auto flat = t.view({t.numel()});
      if (flat.has_value()) {
        for (int64_t i = 0; i < n; ++i) {
          if (i > 0)
            std::cout << ", ";
          std::cout << flat->at({i});
        }
      }
      std::cout << "]" << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "  Error loading tensor: " << e.what() << std::endl;
    }
  }

  // Try loading the embedding table specifically (if it exists)
  const std::string embed_name = "token_embd.weight";
  if (gguf.GetTensorInfo(embed_name)) {
    std::cout << "\n--- Loading \"" << embed_name << "\" ---" << std::endl;
    try {
      ie::Tensor emb = gguf.LoadTensor(embed_name);
      std::cout << "  " << emb.to_string() << std::endl;

      // Print a few values from the embedding table
      if (emb.ndim() == 2) {
        std::cout << "  emb[0, 0..3]: ";
        for (int64_t i = 0; i < std::min(emb.size(1), int64_t(4)); ++i) {
          std::cout << emb.at({0, i}) << " ";
        }
        std::cout << std::endl;

        std::cout << "  emb[1, 0..3]: ";
        for (int64_t i = 0; i < std::min(emb.size(1), int64_t(4)); ++i) {
          std::cout << emb.at({1, i}) << " ";
        }
        std::cout << std::endl;

        // Demonstrate BF16→F32 conversion pipeline
        std::cout << "\n  Converting BF16 embedding to F32..." << std::endl;
        ie::Tensor emb_f32 = emb.to(ie::DType::kFloat32);
        std::cout << "  " << emb_f32.to_string() << std::endl;
      }
    } catch (const std::exception &e) {
      std::cerr << "  Error: " << e.what() << std::endl;
    }
  }

  std::cout << "\nDone!" << std::endl;
  return 0;
}
