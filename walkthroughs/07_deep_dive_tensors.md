# Code Deep Dive: Tensors & Memory Architecture

To successfully modify and extend this Inference Engine in C++, you must understand how memory is allocated, aligned, and accessed. This guide tears down the foundational abstractions in `gguf_loader.cc` and `tensor.cc`, heavily emphasizing the *C++ design choices* made to guarantee performance.

---

## 1. Zero-Copy Loading via `mmap`
*File: `engine/gguf/gguf_loader.cc`*

Loading a 4.4GB model file into RAM historically involved `std::ifstream::read()`. The C++ design choice here was to completely abandon stream IO in favor of OS-level memory mapping (`mmap` on POSIX, `MapViewOfFile` on Windows).

### C++ Paradigm: Let the OS do the work
```cpp
// From gguf_loader.cc
void *ptr = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
mapped_data_ = static_cast<const uint8_t *>(ptr);
```

**Why this matters:**
Using `std::ifstream::read()` copies data from the hard drive into the OS page cache, and *then* copies it again into your C++ user-space `std::vector`. That double-copy destroys latency.
By using `mmap(..., MAP_PRIVATE)`, we ask the OS to map the file directly into our process's virtual address space.
1. **Zero-Copy:** We get a raw `uint8_t*` pointer to the file layout without allocating an extra 4.4GB buffer.
2. **Lazy Paging:** The OS loads 4KB pages into RAM only when the CPU actually dereferences the pointer.
3. **OOM Safety:** If the OS runs out of RAM, it can just evict clean pages belonging to the model rather than crashing out of memory.

---

## 2. The `Tensor` Class: Custom RAII and SIMD Alignment
*File: `engine/tensor/tensor.cc`*

A `Tensor` in C++ needs to manage its lifespan automatically (RAII). A naive approach would use `std::vector<float>`, but this engine chose a highly specific smart pointer.

### The Allocation Design
```cpp
// Inside Tensor::Tensor(shape, dtype)
constexpr size_t kAlignment = 64; // Crucial for SIMD
size_t alloc_size = (nbytes() + kAlignment - 1) & ~(kAlignment - 1);

void *raw = std::aligned_alloc(kAlignment, alloc_size);

// Custom Deleter inside a shared_ptr
data_ = std::shared_ptr<uint8_t[]>(
    static_cast<uint8_t *>(raw),
    [](uint8_t *p) { std::free(p); }
);
```

**Why not `std::vector`?**
1. **SIMD Alignment:** Modern CPUs (specifically AVX-512) load memory into 64-byte vector registers. If a pointer is not a multiple of 64, the CPU must issue *two* cache line loads instead of one. `std::vector` guarantees standard alignment (usually 8 or 16 bytes), which causes massive performance penalties for Highway SIMD. We *must* use `std::aligned_alloc(64, ...)` to hit the compute roofline.
2. **Type Erasure:** A Tensor might hold `float`, `uint16_t` (FP16), or `int8_t`. Instead of writing complicated template logic (`Tensor<float>`), we store type-erased raw bytes `std::shared_ptr<uint8_t[]>`.

### RAII via Custom Deleters
Because we allocate with `std::aligned_alloc/std::free` instead of `new/delete`, we must provide a Custom Deleter to the `std::shared_ptr`. When the last `Tensor` referencing this block goes out of scope, the lambda `[](uint8_t *p){ std::free(p); }` executes automatically. No memory leaks.

---

## 3. Strides and Flat Indexing
*File: `engine/tensor/tensor.cc`*

Memory in C++ is strictly 1-Dimensional. A 3D tensor `[Batch, Seq, Dim]` is an abstraction. To map `[b, s, d]` to a flat `uint8_t[]` index, we use **Strides**.

```cpp
// Computing strides backwards based on shape
for (int64_t i = shape.size() - 2; i >= 0; --i) {
    strides_[i] = strides_[i + 1] * shape_[i + 1];
}

// Flat Index resolution
int64_t Tensor::flat_index(const std::vector<int64_t> &indices) const {
    int64_t idx = 0;
    for (int64_t i = 0; i < ndim(); ++i) {
        idx += indices[i] * strides_[i];
    }
    return idx;
}
```
**C++ Design Insight:** Notice how we iterate through indices and multiply by strides. This enables **Views**.

---

## 4. Views vs. Clones (Copy-on-Write semantics)
*File: `engine/tensor/tensor.cc`*

Deep copying memory is catastrophic for performance. The `Tensor` class is built around "Views"—operations that mutate the *shape* and *strides* of the tensor without ever copying the underlying `data_` pointer.

```cpp
Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
  auto new_shape = shape_;
  auto new_strides = strides_;
  
  // Swap the metadata, NOT the data
  std::swap(new_shape[dim0], new_shape[dim1]);
  std::swap(new_strides[dim0], new_strides[dim1]);
  
  // Return a new Tensor object that shares the same shared_ptr<uint8_t[]>
  return Tensor(new_shape, new_strides, dtype_, data_, offset_);
}
```

Because `data_` is a `std::shared_ptr`, creating this new `Tensor` simply increments the reference count. O(1) complexity. 
However, look at `is_contiguous()`:
```cpp
bool Tensor::is_contiguous() const {
  return strides_ == ComputeStrides(shape_);
}
```
If you transpose a matrix, its strides no longer match its shape linearly. It is mathematically "Non-contiguous". 
You **cannot** extract a raw pointer `data<float>()` from a non-contiguous tensor to feed into SIMD (it would read the wrong values). You must call `.contiguous()`, which forces an expensive `memcpy` into a brand new, correctly aligned `std::shared_ptr` allocation.

**Key Rule for Modifying Engine Code:**
When you write an Op, *always* do this first:
```cpp
Tensor a_c = a.contiguous(); // Fails fast or allocates memory if shape is messy
float* ptr = a_c.data<float>(); // Safe for raw pointer SIMD math
```
