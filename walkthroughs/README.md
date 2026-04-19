# Inference Engine: Master Curriculum

Welcome to the documentation. Because this project spans raw metal (AVX-512 memory alignment) all the way up to high-level system architecture (Continuous Batching logic), jumping into a random file can quickly feel overwhelming and jumbled.

To maximize your understanding as an ML Infrastructure Engineer transitioning from university-level C++, **read the walkthroughs in the following exact order**.

---

### Phase 1: The Foundations (C++ and Hardware)
Before you look at a single Neural Network layer, you must understand how data fundamentally exists and moves inside the engine.

1. **`07_deep_dive_tensors.md`**
   * **Why read this first?** This is your bridge from University C++ to Industry C++. It explains how the 4GB `.gguf` file gets mapped into RAM via pointers, and what a `Tensor` actually is under the hood (Strides, `std::shared_ptr` allocation, and RAII).
2. **`08_deep_dive_compute_quantization.md`** 
   * **Why read this next?** Now that you have Tensors, you need to understand how the CPU manipulates them. This explains the specific macros we use to talk to AVX CPU registers, and exactly how `#pragma pack` and bit-shifting are used to read 4-bit numbers without compiler padding.

### Phase 2: Operations and Quantization (The Math)
Now that you understand the C++ foundations, it's time to see how the math represents an ML model.

3. **`ops_walkthrough.md`**
   * **Why read this?** This maps the high-level Math. It explains that all intermediary Math ops (like `exp()` in `softmax`) are converted to `FP32` so you don't encounter catastrophic precision limits. It dictates how to implement a new high-level operation.
4. **`01_simd_compute.md`**
   * **Why read this?** This explains how `ops::matmul` delegates its work out to concurrent cores using modern C++ thread pools (`std::function` and lambdas) and loops.
5. **`05_quantization.md`**
   * **Why read this?** This connects the 4-bit math from Phase 1 to actual Economics. It explains *why* outlier-aware quantization saves 50% memory and explicitly explains how the `LoadTensor` functionality decompresses the weights seamlessly at execution.

### Phase 3: The Model Architecture
You now know how Tensors move, how they compute, and how they compress. It is finally time to wire them together.

6. **`02_model_architecture.md`**
   * **Why read this?** This outlines the complete `Gemma-3 1B` flow from Token in, to Text out. It breaks down the memory ownership (`std::unique_ptr`) and how `const` correctness keeps the execution safe.

### Phase 4: Scaling and Infrastructure
You have a working model. Now, you need to serve it to hundreds of concurrent users without the server crashing.

7. **`03_memory_attention.md`**
   * **Why read this?** Generating sequences requires memory. Since calling `new` inside a hot loop is fatal for servers, this file explains how the `KVCacheManager` pre-allocates everything instantly using an object pool, and how FlashAttention scales.
8. **`04_continuous_batching.md`**
   * **Why read this?** The pinnacle of the architecture. This wraps everything together by scheduling multiple users simultaneously, using asynchronous streaming callbacks to pass text back to the real world.
9. **`06_ml_infra_guide.md`**
   * **Why read this last?** This is your operational dashboard. It pulls back and looks at the Engine as an abstract machine with Latency (TTFT) and Throughput outputs, defining how CPU FLOPs and DDR5 memory bandwidth constrain your economic scaling.

---

### Pro-Tip for Reading
When you open a walkthrough, **split your IDE vertically**. Put the markdown file on the left, and the actual `.cc` file being referenced on the right. Verify the text against the actual code line-by-line.
