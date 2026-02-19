#include "engine/model/layers.h"

#include <stdexcept>

namespace ie {

Tensor RMSNorm::forward(const Tensor& x) const {
  // TODO: Implement RMS normalization — start here!
  throw std::runtime_error("RMSNorm::forward not implemented yet");
}

Tensor Attention::forward(const Tensor& x, int32_t start_pos) const {
  // TODO: Implement multi-head attention with GQA
  throw std::runtime_error("Attention::forward not implemented yet");
}

Tensor FeedForward::forward(const Tensor& x) const {
  // TODO: Implement SwiGLU feed-forward
  throw std::runtime_error("FeedForward::forward not implemented yet");
}

Tensor TransformerBlock::forward(const Tensor& x, int32_t start_pos) const {
  // TODO: Implement attention + FFN with residual connections
  throw std::runtime_error("TransformerBlock::forward not implemented yet");
}

Tensor GemmaModel::forward(const Tensor& tokens, int32_t start_pos) const {
  // TODO: Implement full model forward pass
  throw std::runtime_error("GemmaModel::forward not implemented yet");
}

}  // namespace ie
