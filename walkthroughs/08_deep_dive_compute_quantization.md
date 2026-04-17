# Code Deep Dive: Compute, SIMD, & Quantization Logic

This guide dives into the C++ design patterns handling the actual Math. As a heavy C++ developer modifying this engine, you must be extremely comfortable with preprocessor macros mapping to SIMD registers, memory strides in matrix loops, and bitwise math.

---

## 1. Zero-Cost Abstractions: Highway SIMD Macros
*File: `engine/compute/simd_kernels.cc`*

Directly writing AVX-512 intrinsic code (`_mm512_add_ps()`) would lock the entire inference engine to modern Intel CPUs, preventing it from running on ARM Macs or older AMD processors. 

Instead of writing 4 different versions of the code with `#ifdef`, we use Google Highway's specific C++ macro configuration.

### The Macro Setup
```cpp
// 1. Clear state
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "engine/compute/simd_kernels.cc"

// 2. Multi-Target Generation Expansion
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

// 3. The implementation MUST be inside this exact namespace
namespace hn = hwy::HWY_NAMESPACE;
```
**C++ Design Insight:** The `#include "hwy/foreach_target.h"` is essentially a code-generator. It includes `simd_kernels.cc` *multiple times* during compilation. For each inclusion pass, the compiler sets a different target (SSE4, AVX2, AVX-512) and re-compiles the code.
`hwy::HWY_NAMESPACE` dynamically expands to the target being compiled (e.g., `namespace N_AVX2`).

At runtime, this code executes dynamically:
```cpp
HWY_EXPORT(SimdGemmImpl); // Exposed to the outside world
void SimdGemm(const float *a, const float *b, float *c, int64_t M, int64_t N, int64_t K) {
  // Looks up the CPU capabilities, calls the correct specialized pointer
  HWY_DYNAMIC_DISPATCH(SimdGemmImpl)(a, b, c, M, N, K);
}
```

---

## 2. Unrolling the Matrix Multiply (GEMM)
*File: `engine/compute/simd_kernels.cc`*

In C++, writing nested loops correctly determines if your code executes in 1 millisecond or 100 milliseconds due to the **L1 Data Cache**.

### The Inner Loop
```cpp
// i, k, j ordering (b_row and c_row accessed sequentially)
for (int64_t k = 0; k < K; ++k) {
  const auto a_ik = hn::Set(d, a[i * K + k]); // Broadcast 1 float into all SIMD lanes
  const float *b_row = b + k * N;
  float *c_row = c + i * N;

  int64_t j = 0;
  for (; j + lanes <= N; j += lanes) {
    auto cv = hn::Load(d, c_row + j); // Load N outputs
    auto bv = hn::Load(d, b_row + j); // Load N weights
    cv = hn::MulAdd(a_ik, bv, cv);    // cv = a_ik * bv + cv
    hn::Store(cv, d, c_row + j);      // Store N outputs
  }
}
```

**C++ Design Explanation:**
1. **Pointers Over Indexing:** Inside the hot `k` loop, we precompute `const float* b_row = b + k * N`. By iterating `j` directly on this pointer, the compiler avoids evaluating `(i*N*K + j)` index multiplication bounds checking at every cycle.
2. **Fused Multiply-Add (`MulAdd`):** The hardware CPU instruction `vfmadd231ps` performs $A * B + C$ simultaneously in a single clock cycle. This literally doubles the FLOPs/sec of the engine versus doing multiplication, waiting, and doing addition.
3. **Register Spilling:** Notice how `cv` is loaded, manipulated, and stored. We do not make function calls or declare heavy objects inside the loop because doing so would cause the compiler to spill these registers to the stack.

---

## 3. Quantization Bit-Math and Struct Packing
*File: `engine/gguf/gguf_loader.cc`*

Quantization fundamentally forces 32-bit `float` mathematical concepts into tightly packed arrays of bits.

### Ensuring Memory Layout (`#pragma pack`)
If you define a structure in C++:
```cpp
struct BlockQ8 {
    uint16_t scale; // 2 bytes
    int8_t qs[32];  // 32 bytes
};
```
The compiler standard naturally adds padding to ensure the total size aligns to computer word boundaries (e.g., jumping from 34 to 36 bytes). We *cannot* allow this when parsing raw arrays of binary numbers straight off a disk or out of `mmap`.
To stop the C++ compiler from padding structs:
```cpp
#pragma pack(push, 1) // Disable padding, align to 1 byte
struct BlockQ4_K { ... };
#pragma pack(pop)     // Re-enable normal C++ rules
```

### Unpacking Q4_0 Nibbles
The `Q4_0` integer format stores *two* 4-bit numbers in a single 8-bit block (a nibble). If we have `uint8_t byte = 0xAB` (1010 1011):
- High bit is `A` (1010, decimal 10)
- Low bit is `B` (1011, decimal 11)

To unpack this in C++ safely:
```cpp
// Inside DequantizeQ4_0
uint8_t byte = nibbles[i];

// 1. Low Nibble: Mask out the top 4 bits using bitwise AND (0x0F = 0000 1111)
// byte & 0x0F -> 0x0B (11)
int8_t lo = static_cast<int8_t>(byte & 0x0F) - 8;

// 2. High Nibble: Shift the top 4 bits down
// byte >> 4 -> 0x0A (10)
int8_t hi = static_cast<int8_t>(byte >> 4) - 8;
```
**Why `- 8`?**
A 4-bit unsigned number ranges from `0` to `15`. We want values centered around zero to act as multiplier coefficients in linear regressions. `15 - 8 = 7` and `0 - 8 = -8`. So the values now safely range from `[-8, 7]`.

Finally, we hit the scale vector:
```cpp
dst[b * 32 + i * 2] = scale * static_cast<float>(lo);
dst[b * 32 + i * 2 + 1] = scale * static_cast<float>(hi);
```
**C++ Design Caveat:** The `static_cast<float>` is required *before* multiplying the scale. You must convert the integer to floating point in register bounds, otherwise you are doing implicit type demotion algorithms that might throw warnings or behave strangely on specific x86 architectures.
