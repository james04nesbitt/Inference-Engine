# Continuous Batching

## What This Does

The `engine/scheduler/` directory implements a **BatchScheduler** that serves multiple generation requests concurrently. Instead of processing requests one at a time (wasting model weight loading), the scheduler interleaves decode steps across sequences, sharing the model and KV cache infrastructure.

## Architecture

```
InferenceEngine::GenerateBatch(prompts)
    └── BatchScheduler
          ├── Pending Queue (FIFO of incoming requests)
          ├── Active Sequences (in-flight, each with own seq_id + KV pages)
          │
          │  PrefillNext():   Pending → Active (tokenize, prefill, allocate KV)
          │  Step():          One decode token per active sequence (round-robin)
          │  Reap finished:   Free KV blocks, fire completion callback
          │
          └── Run():  loop { PrefillNext(); Step(); } until all done
```

## Key Files

| File | Purpose |
|------|---------|
| [batch_scheduler.h](../engine/scheduler/batch_scheduler.h) | `Request`, `ActiveSequence`, `BatchScheduler` |
| [batch_scheduler.cc](../engine/scheduler/batch_scheduler.cc) | Scheduling loop implementation |

## How It Works

### Request Lifecycle

```
1. AddRequest(prompt, max_tokens, callbacks)
   → Request enters pending queue

2. PrefillNext()
   → Tokenize prompt
   → AllocateSequence() in KV cache (get unique seq_id)
   → Run full forward pass with all prompt tokens (prefill)
   → Sample first output token
   → Move to active list

3. Step() — repeated until done
   → For each active sequence:
       - Forward pass with last token (decode, seq_len=1)
       - Sample next token
       - Call on_token callback (streaming)
       - Check: EOS or max_tokens reached? → mark finished
   → Reap finished sequences:
       - Call on_complete callback with full text
       - FreeSequence() → return KV blocks to pool

4. Run() = loop PrefillNext() + Step() until queues empty
```

### Why Round-Robin Scheduling?

Each `Step()` call processes one decode token for every active sequence. This design was chosen because:

1. **Fairness:** No sequence starves. Every active request makes progress every step.
2. **Simplicity:** No priority queue, no preemption logic, no starvation prevention needed.
3. **KV cache efficiency:** All sequences' KV cache blocks stay warm in memory across consecutive steps.
4. **CPU-appropriate:** On CPU, the dominant cost is loading model weights from DRAM. Round-robin ensures we amortize this load across all sequences per step.

### Why Not True Batched Forward Pass?

In GPU inference engines (like vLLM), multiple sequences are batched into a single forward pass — their tokens are concatenated and processed together through shared matmuls.

We chose sequential-per-sequence on CPU because:
- **Variable-length attention:** Handling multiple sequences with different KV cache lengths in one attention kernel requires complex index arithmetic or padding
- **CPU cache locality:** Processing one sequence at a time keeps that sequence's KV cache hot in L2. Batching would thrash the cache across sequences
- **The bottleneck is different:** GPU inference is compute-bound (batch larger = better GPU utilization). CPU inference is memory-bandwidth-bound (model weights dominate), and we already load weights once per `Step()` regardless

### Paged KV Cache Synergy

The BatchScheduler works hand-in-hand with the paged KV cache:

| Event | KV Cache Action |
|-------|----------------|
| New request | `AllocateSequence()` → fresh page table |
| Each token | `AppendToken()` → fills current block, auto-allocates next |
| Request done | `FreeSequence()` → all blocks returned to free list |

Because pages are tiny (16 tokens each), there's minimal wasted memory even with many concurrent short sequences. A traditional pre-allocated cache would reserve `max_seq_len` per sequence regardless of actual usage.

### Streaming Callbacks

Each request can optionally provide:
- `on_token(string)` — called with each generated token as it's decoded (enables streaming output)
- `on_complete(string)` — called with the full generated text when the sequence finishes

This design lets callers process output in real-time without polling.

## Architectural Choice: Scheduler Owns Nothing

The `BatchScheduler` takes references to `GemmaModel`, `Tokenizer`, and `KVCacheManager` — it owns none of them. This means:
- The engine can use the model/cache for single-request mode without the scheduler
- Multiple schedulers could share the same model (e.g., for A/B testing different configs)
- Testing is easy: create tiny model + cache + tokenizer, pass to scheduler
