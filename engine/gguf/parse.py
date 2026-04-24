from gguf_parser import GGUFParser

# Monkey-patch GGUFParser to support BF16 (type 30)
GGUFParser.TENSOR_TYPES[30] = "BF16"

parser = GGUFParser("C:\\Users\\james\\Coding\\Projects\\Inference-Engine\\bazel-inference-engine\\model\\gemma-3-1b-it-BF16.gguf")
parser.parse()
parser.print()