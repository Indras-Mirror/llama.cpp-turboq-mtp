# DeepSeek Task Handoff — TBQ4_0 KV Cache Optimization

**Last updated:** 2026-05-08

## What You're Working On

You're optimizing the TBQ4_0 (TurboQuant 4-bit) CUDA kernels in a llama.cpp fork that runs Qwen3.6-27B with MTP (Multi-Token Prediction) speculative decoding on an RTX 4090 24GB.

**The problem**: TBQ4 KV cache gives near-lossless quality (4.25 bits per value with FWHT rotation) but is ~40% slower than plain Q4_0 KV during generation. The bottleneck is the inverse Fast Walsh-Hadamard Transform (FWHT) that runs as a **separate kernel** writing F32 to global memory before flash attention reads it back.

**Your goal**: Optimize the FWHT butterfly and add sparse V dequant. Claude is handling the harder kernel fusion work separately.

## YOUR TASK STATUS
- **D1 (Warp-Shuffle FWHT):** ✅ COMPLETE — tbq4-cuda.cuh:146-210
- **D2 (Sparse V Dequant):** ✅ COMPLETE — tbq4-sparse-v.cuh (39 lines)
- **D3 (Block-Diagonal Hadamard):** ✅ COMPLETE — tbq4-cuda.cuh:220-379

## NEXT ASKS FROM CLAUDE
1. **Review `fattn-mma-tbq4.cuh`** — verify the tile loader centroid lookup (`d_tbq4_centroids[nibble] * norm`) is correct relative to how block_tbq4_0 stores data (rotated domain, no FWHT needed in loader)
2. **Update COLLAB_NOTES.md** — add final status notes for D1/D2/D3 with any gotchas
3. **Think about D2 integration** — how to wire sparse-V threshold into the fused V tile loading path in fattn-mma-f16.cuh

## Coordination

- **Joint notes**: `COLLAB_NOTES.md` in this repo — add timestamped notes when you complete tasks or hit blockers
- **Claude's tasks**: Fusing TBQ4 dequant into flash attention (fattn-mma-f16.cuh), rotated-domain K attention
- **Your tasks**: D1 (warp-shuffle FWHT), D2 (sparse V dequant), D3 (block-diagonal Hadamard)
- **Don't modify**: `fattn-mma-f16.cuh`, `fattn-common.cuh`, `fattn-tile.cuh` — Claude is working on those
- **Your files**: `tbq4-cuda.cuh` (primary), `cpy.cu` (where dequant is called), new files if needed

---

## TASK D1: Warp-Shuffle FWHT Butterfly

**File**: `ggml/src/ggml-cuda/tbq4-cuda.cuh`, lines 142-180 (`k_tbq4_dequant_full`)

**Current implementation problems**:
1. Uses shared memory `buf[128]` with 7 `__syncthreads()` barriers (one per butterfly stage)
2. Only 64 of 128 threads are active during the butterfly loop (line 167: `if (tid < 64)`)
3. Each sync barrier is ~5-20 cycles of wasted time on sm_89

**What to do**:

Replace the shared-memory butterfly with warp shuffles for stages 0-4 (stride 1 through 16), keeping shared memory only for the cross-warp stages 5-6 (stride 32, 64).

**Implementation approach**:
```
Thread layout: 128 threads = 4 warps of 32
Each thread owns 1 element (tid maps to element index)

Stages 0-4 (stride 1,2,4,8,16): All within a warp
  - Use __shfl_xor_sync(0xFFFFFFFF, val, stride) to exchange butterfly partners
  - Partner of thread t at stride h: t ^ h
  - Butterfly: a = val + partner, b = val - partner
  - Thread keeps 'a' if (tid % (2*h)) < h, else keeps 'b'
  - No sync needed — warp-level shuffle is implicit sync

Stages 5-6 (stride 32, 64): Cross-warp, needs shared memory
  - Write to shared memory, __syncthreads()
  - Read partner from shared memory
  - Only 2 syncs instead of 7
```

**Key math**: The butterfly at each stage does:
```
for each pair (j, j+h) where j is the lower index:
    a = buf[j] + buf[j+h]
    b = buf[j] - buf[j+h]
    buf[j] = a
    buf[j+h] = b
```

With warp shuffles, thread `t` gets its partner's value via `__shfl_xor_sync(mask, my_val, h)`. Then:
- If `(t & h) == 0` (lower partner): `result = my_val + partner_val`
- If `(t & h) != 0` (upper partner): `result = partner_val - my_val`

**Also fix**: Make all 128 threads active during the butterfly. Currently threads 64-127 sit idle. Each thread should handle its own element through all stages.

**Test**: After changes, the dequantized output must be bit-identical to the current implementation. Add a debug assertion that compares old vs new output for the first few blocks.

**Expected speedup**: ~30-40% faster `k_tbq4_dequant_full` kernel execution.

---

## TASK D2: Sparse V Dequant (Attention-Aware Skip)

**Concept**: Flash attention computes softmax weights (from Q*K) before accumulating V. At long context, 90%+ of attention weights are near-zero. Skip V dequant for positions with negligible attention weight.

**Reference**: TheTom's implementation got +22.8% decode speed at 32K context with PPL unchanged.
- Reddit: "Skipping 90% of KV dequant work" (r/LocalLLaMA, 847 upvotes)
- Fork: `reference/llama-cpp-turboquant-cuda/ggml/src/ggml-cuda/turbo-sink.cuh` (16 lines)

**Where to implement**: This goes in the attention path, NOT in `tbq4-cuda.cuh` itself. The sparse skip happens at the flash attention level — after softmax scores are computed but before V accumulation.

**However**, since Claude is working on the flash attention files, implement this as a **standalone utility** that can be plugged in:

Create `ggml/src/ggml-cuda/tbq4-sparse-v.cuh`:
```cpp
// Given attention weights (post-softmax) and a threshold,
// return a bitmask of which V positions to actually dequant.
// Positions below threshold get zero V contribution.
```

**Threshold selection**: The attention weights after softmax sum to 1.0. A good threshold is `1.0 / (4 * seq_len)` — positions contributing less than 0.25x the average attention get skipped.

**Key constraint**: This only helps at long context (>8K). At short context, most positions have significant weight. Add a `seq_len > threshold` guard.

**Reference to study**: Read `reference/llama-cpp-turboquant-cuda/ggml/src/ggml-cuda/turbo-sink.cuh` — it's only 16 lines and shows the exact approach.

---

## TASK D3: Block-Diagonal Hadamard Option

**Concept**: Replace the full 128-point FWHT with 4x 32-point block-diagonal transforms. Each 32-point block fits entirely within a single warp (32 threads), eliminating ALL `__syncthreads()` barriers.

**Quality tradeoff**: SAW-INT4 (Together AI + Tri Dao, arXiv:2604.19157) shows block-diagonal Hadamard achieves nearly identical quality to full FWHT for KV cache quantization.

**Implementation**: Add this as an **alternative mode** in `tbq4-cuda.cuh`, not a replacement. Add a `#define TBQ4_BLOCK_DIAGONAL` or runtime flag.

**What changes**:
1. Sign arrays `d_tbq4_wht_s1[128]` and `d_tbq4_wht_s2[128]` → need new sign arrays for 32-point blocks
2. `tbq4_fwht_128()` → `tbq4_fwht_32()` called 4 times on 32-element sub-blocks
3. `k_tbq4_dequant_full` → 32 threads per block-diagonal sub-block, fully within one warp
4. Quantization path (`quantize_f32_tbq4_0_block`) also needs matching 4x32 forward transform

**Important**: This creates an incompatible block format. GGUFs quantized with full 128-point FWHT CANNOT be dequantized with 32-point block-diagonal, and vice versa. This would need a new type identifier (e.g., `GGML_TYPE_TBQ4_BD32`).

**Lower priority than D1 and D2** — do this last, and only if time permits. The format incompatibility makes it a bigger change.

---

## Key Files You Need to Know

```
ggml/src/ggml-cuda/
├── tbq4-cuda.cuh          ← YOUR PRIMARY FILE (188 lines)
│   ├── Lines 10-15: Lloyd-Max centroids (16 values for N(0,1/√128))
│   ├── Lines 24-33: FWHT sign arrays s1[128], s2[128]
│   ├── Lines 37-48: tbq4_fwht_128() — in-register butterfly
│   ├── Lines 51-62: rotate_forward / rotate_inverse — sign + FWHT + sign
│   ├── Lines 99-124: quantize_f32_tbq4_0_block — quantize path (set_rows)
│   ├── Lines 129-138: dequantize_tbq4_0 — per-element, NO FWHT (get_rows)
│   └── Lines 142-188: k_tbq4_dequant_full — THE BOTTLENECK KERNEL
│                       128 threads, shared memory, 7 syncs
│
├── cpy.cu:554-559         ← Where k_tbq4_dequant_full is called from
├── set-rows.cu:324-325    ← Where quantize is called from
├── fattn-common.cuh       ← Flash attention common (DON'T MODIFY — Claude's territory)
├── fattn-tile.cuh         ← Flash attention tile kernel (DON'T MODIFY)
├── fattn-mma-f16.cuh      ← Flash attention MMA kernel (DON'T MODIFY)
└── turbo-sink.cuh         ← Reference: spiritbuun's sparse V (in reference/ dir)
```

## Reference Repos (already cloned)

```
reference/
├── llama-cpp-turboquant-cuda/   ← spiritbuun's fork with fused turbo attention
│   └── ggml/src/ggml-cuda/
│       ├── fattn-mma-turbo.cuh  ← Fused turbo4 attention (reference for Claude)
│       ├── turbo-quant-cuda.cuh ← Their quantize/dequant (1356 lines, very complete)
│       ├── turbo-wht.cuh        ← Their WHT implementation
│       └── turbo-sink.cuh       ← Sparse V dequant (16 lines — read this!)
├── AXELRAM/                     ← Rotated-domain attention (Python, theoretical reference)
│   └── axelram/rotation/hadamard.py  ← Hadamard transform implementations
└── saw-int4/                    ← Block-diagonal Hadamard (paper reference)
    └── tools/fit_kv_centroids.py
```

## TBQ4_0 Block Format

```c
// From ggml-common.h
typedef struct {
    ggml_half d;      // 2 bytes: norm correction factor
    uint8_t qs[64];   // 64 bytes: 128 x 4-bit indices, packed in pairs
} block_tbq4_0;       // Total: 66 bytes per 128 elements
// QK_TBQ4 = 128 (elements per block)
```

Each 4-bit index maps to one of 16 Lloyd-Max centroids for N(0, 1/√128). The data is stored in the **FWHT-rotated domain** — to get original values, you must apply inverse rotation: `s2 → FWHT → s1` then multiply by norm `d`.

## How to Build and Test

```bash
cd /home/mal/AI/llama.cpp-mtp

# Build
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc)

# Quick test (perplexity — validates correctness)
build/bin/llama-perplexity \
  -m /media/Crucial1TB/models/Qwen3.6-Heretic-MTP/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q4_K_M.gguf \
  -f /home/mal/AI/llama.cpp-mtp/test_prompt.txt \
  -c 4096 -ngl 99 -ctk tbq4_0 -ctv tbq4_0

# Speed test (server + curl, see COLLAB_NOTES.md for full commands)
```

## Important Constraints

1. **Don't break the quantize path** — `quantize_f32_tbq4_0_block` must produce identical output
2. **Don't break get_rows** — `dequantize_tbq4_0` (per-element, no FWHT) is used for non-attention paths
3. **The dequant_full kernel output must be bit-identical** for D1 (warp-shuffle is just an optimization, not a math change)
4. **For D3 (block-diagonal)**, output WILL differ — that's a new format, keep it behind a flag
5. **sm_89 target** — RTX 4090 is compute capability 8.9, use appropriate warp shuffle intrinsics
6. **Update COLLAB_NOTES.md** when you complete a task or hit a blocker
