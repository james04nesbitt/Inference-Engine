# Tensor Implementation Walkthrough

## Bugs Fixed

| Bug | Location | What was wrong |
|-----|----------|---------------|
| [fill()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#280-304) non-contiguous | [tensor.cc](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc) | Was reading `this->at()` and writing the *read* value instead of writing the `val` parameter |
| [contiguous()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#309-333) dtype | [tensor.cc](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc) | Always wrote to `data<float>()` regardless of actual dtype — broken for F16/INT8 |
| `data<T>()` offset | [tensor.h](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.h) | Didn't account for `offset_`, so sliced views returned wrong pointers |
| [data_ptr()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#95-98) offset | [tensor.cc](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc) | Same as above — raw pointer ignored `offset_` |
| [reshape()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#197-224) dtype | [tensor.cc](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc) | Non-contiguous copy path wrote to `data<float>()` for all dtypes |
| [view()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#183-196)/[transpose()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#225-233)/[permute()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#234-244) offset | [tensor.cc](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc) | Weren't propagating `offset_` to the returned view |

## All Implementations

### Offset-Aware Data Access
- `data<T>()` → `reinterpret_cast<T*>(data_.get()) + offset_`
- [data_ptr()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#95-98) → `data_.get() + offset_ * DTypeSize(dtype_)`
- **Why different?** `data<T>()` works in *element* units (T-sized), while [data_ptr()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#95-98) works in *byte* units since it returns `void*`

### Static Factories
| Method | Implementation |
|--------|---------------|
| [from_buffer()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#361-366) | Wraps existing `shared_ptr` + computes strides → zero-copy |
| [from_vector()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#367-372) | Allocates 1D F32 tensor + `memcpy` from vector |
| [full()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#373-378) | Allocates + calls [fill()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#280-304) |
| [zeros()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#379-383) | Just [Tensor(shape, dtype)](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.h#200-204) — constructor memsets to 0 |
| [ones()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#384-387) | Delegates to [full(shape, 1.0f)](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#373-378) |
| [cat()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#388-437) | Validates shapes/dtypes, allocates output, copies via [at()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#154-166)/[set()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#167-182) |

### Methods
| Method | Zero-copy? | Key detail |
|--------|-----------|------------|
| [size(dim)](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#442-450) | N/A | Supports negative dims (`dim + ndim()`) |
| [to(dtype)](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#455-475) | If same dtype + contiguous | Converts via [at()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#154-166)/[set()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#167-182) which handle all dtype pairs |
| [clone()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#480-492) | No (always copies) | Fast path: `memcpy` if contiguous+no offset |
| [select(dim, idx)](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#497-524) | ✅ view | Removes dimension, adjusts offset |
| [unsqueeze(dim)](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#525-550) | ✅ view | Inserts size-1 dim with computed stride |
| [squeeze(dim)](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#551-576) | ✅ view | Removes size-1 dim, throws if `shape[dim] != 1` |
| [repeat(dim, n)](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#577-607) | No (copies) | Tiles data by writing each element `n` times |
| [shape_equals()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#612-615) | N/A | Simple `==` on shape vectors |
| [set_quantization_params()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#340-352) | N/A | Validates `scales.size() == zero_points.size() == shape[0]` |

## Key Details for Downstream Modules

> [!IMPORTANT]
> **Offset is in elements, not bytes.** `data<T>()` adds `offset_` to the typed pointer, meaning `offset_` counts in elements of the *underlying storage type*. This is correct because [flat_index()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#146-153) also returns element offsets via strides.

> [!TIP]
> **Zero-copy methods** ([slice](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#265-275), [select](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#497-524), [unsqueeze](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#525-550), [squeeze](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#551-576), [transpose](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#225-233), [permute](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#234-244)) all share the underlying [data_](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#95-98) buffer. Mutating through one view affects all views. Use [clone()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#480-492) if you need an independent copy.

> [!NOTE]
> **[cat()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#388-437) and [repeat()](file:///c:/Users/james/Coding/Projects/Inference-Engine/engine/tensor/tensor.cc#577-607) are O(n) copies** — they allocate new tensors. For KV cache appending, this is fine for a learning project but a production system would use pre-allocated ring buffers.

## Verification

```
bazel test //engine/tensor:tensor_test → PASSED (all 29 tests)
```
