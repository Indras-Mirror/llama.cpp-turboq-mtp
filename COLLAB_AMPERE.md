# COLLAB NOTES — Ampere sm_86 Garbage Output Bug

**Branch:** `fix/ampere-sm86-garbage`  
**GitHub Issue:** Indras-Mirror/llama.cpp-mtp#4  
**Date:** 2026-05-10  
**Status:** OPEN — needs sm_86 testing (no 3090/A5000 available locally)

---

## Summary

All Ampere GPUs (sm_86 / GA10x) produce garbled/infinite output with this fork. Ada Lovelace (sm_89 / RTX 4090) works fine. Bug is definitively in this fork's kernel changes — same GGUF + same hardware works correctly on upstream turboquant and upstream llama.cpp.

## Reproduction (cnndabbler's A/B tests)

**Hardware:** RTX A5000 24 GB (sm_86, Linux, driver 535.x, CUDA 12.1)

| # | Model | Config | Result |
|---|-------|--------|--------|
| 1 | MTP-grafted Qwen3.6-27B | FA on + Q4_0 KV + MTP | `!!!!!!!!!!` total garbage |
| 2 | Same GGUF | No FA, FP16 KV | First token OK → gradual drift |
| 3 | Vanilla Qwen3.6-27B (no MTP) | This fork | `//////////////` garbage |
| 4 | Same vanilla GGUF | Upstream turboquant | Clean output ✅ |

## Hypotheses (cnndabbler's analysis)

Two independent bugs, both sm_86-specific:

1. **Fused TBQ4 FA** (`fattn-mma-tbq4.cuh`, `fattn-mma-tbq4-launch.cuh`) — total numerical garbage on Ampere. Likely an Ada-specific shared-memory layout or `cp.async` assumption from RTX 4090 (AD102) tuning.

2. **Modified MMA path / shared-tensor linking** — affects non-FA paths too, small per-layer drift compounding over 64 layers.

## AD102 vs GA102 differences that could matter

| Feature | AD102 (4090, sm_89) | GA102 (3090, sm_86) |
|---------|---------------------|---------------------|
| Shared memory | 128 KB/SM | 128 KB/SM (same) |
| L1 cache | 128 KB/SM | 128 KB/SM (same) |
| `cp.async` | Supported | Supported |
| Warp scheduler | 4 per SM | 4 per SM (same) |
| FP8 cores | Yes | **No** |
| TMA | Yes | **No** |
| `movmatrix.sync.aligned` | Supported | Supported (same PTX) |
| SM count | 128 | 82 |
| Clock speed | Higher | Lower |

The architectures are very similar in the features we use. The most likely difference: **L1 cache behavior** or **shared memory bank conflict patterns** differ between AD102 and GA102 when using the specific access patterns in our TBQ4 tile loader.

## Known Related Bug

**BUG 6** in the main `COLLAB_NOTES.md`: "misaligned address" crash in the fused TBQ4 kernel. This was observed on sm_89 (4090) during nstages=2 work. On sm_86 it may corrupt output silently instead of crashing.

## Files To Investigate

### Primary suspects (fused TBQ4 FA — Test #1 & #3 garbage):

| File | What It Does | sm_86 Risk |
|------|-------------|------------|
| `ggml/src/ggml-cuda/fattn-mma-tbq4.cuh` | TBQ4 tile loader, centroid×norm dequant inline | 🔴 HIGH |
| `ggml/src/ggml-cuda/fattn-mma-tbq4-launch.cuh` | Template launcher, shmem calculation | 🔴 HIGH |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | Modified — TBQ4 guards in iter function (4 locations) | 🟡 MED |
| `ggml/src/ggml-cuda/fattn.cu` | TBQ4 dispatch + rotation kernel calls | 🟡 MED |

### Secondary suspects (shared tensor linking — Test #2 drift):

| File | What It Does | sm_86 Risk |
|------|-------------|------------|
| `src/models/qwen35_mtp.cpp` | MTP tensor sharing (`link_shared_tensors`) | 🟡 MED |
| `src/models/qwen35moe_mtp.cpp` | MoE MTP tensor sharing | 🟡 MED |
| `src/llama-model.cpp` | Model loading, partial_load for MTP | 🟢 LOW |

### TBQ4 dequant kernels (could affect non-FA paths):

| File | What It Does | sm_86 Risk |
|------|-------------|------------|
| `ggml/src/ggml-cuda/tbq4-cuda.cuh` | FWHT, quantize, dequant, full-block dequant | 🟡 MED |
| `ggml/src/ggml-cuda/cpy.cu` | TBQ4→F32/F16 dequant | 🟡 MED |
| `ggml/src/ggml-cuda/set-rows.cu` | TBQ4 SET_ROWS quantize | 🟡 MED |

## Approach

### Phase 1 — Diagnose (isolate which kernel path)

1. **Add diagnostic flag**: `GGML_CUDA_TBQ4_DISABLE` env var that forces F16 KV path even when `-ctk tbq4_0 -ctv tbq4_0` is passed. If garbage disappears → TBQ4 kernel is culprit. If garbage remains → something deeper.

2. **sm_86 guard**: Add `#if __CUDA_ARCH__ >= 890` around the TBQ4 fused FA dispatch in `fattn.cu`. On sm_86, always take the GPU-dequant-to-F16 path (not fused). If that fixes it → confirmed fused TBQ4 FA issue.

3. **Compare shmem layouts**: Dump the shared memory usage of our TBQ4 tile loader vs the standard F16 tile loader at runtime on both sm_86 and sm_89. Look for bank conflicts or alignment issues.

### Phase 2 — Fix

Once isolated:
- If TBQ4 fused FA → fix the tile loader in `fattn-mma-tbq4.cuh` for sm_86
- If shared tensor linking → fix `qwen35_mtp.cpp` tensor sharing
- If dequant → fix `tbq4-cuda.cuh` FWHT or quantize kernels

### Phase 3 — Verify

- cnndabbler's A/B test suite across all 4 configs
- PPL comparison vs upstream turboquant
- Decode speed benchmark

## How To Help (for peers)

1. **If you have an Ampere GPU** (3090, A5000, A6000, A4000, 3070, 3080):
   - Build this branch: `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`
   - Run with any Qwen3.6 GGUF (even un-grafted, no MTP needed)
   - Test with `--flash-attn on` vs `--flash-attn off`
   - Report: garbage or clean output?

2. **If you have CUDA debugging experience**:
   - Check `fattn-mma-tbq4.cuh` for shared memory bank conflicts
   - Compare the TBQ4 tile loader's memory access pattern against the F16 loader
   - Check if any `__syncthreads()` patterns differ between our fork and upstream

3. **If you have both Ampere + Ada GPUs**:
   - Run identical configs on both, compare token-by-token output
   - Log the kernel launch parameters on both architectures
   - The first divergent token tells us exactly which layer/kernel misbehaves

## Build & Test

```bash
cd /home/mal/AI/llama.cpp-mtp-fixes
git checkout fix/ampere-sm86-garbage

# Build for sm_86 (Ampere)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j$(nproc) --target llama-server

# Test (replace model path with any Qwen3.6 GGUF)
./build/bin/llama-server \
  -m /path/to/any-qwen3.6.gguf \
  --port 8098 -c 4096 --flash-attn on --mlock -t 8 -ngl 99 \
  --parallel 1 -b 2048 -ub 32 \
  -ctk q4_0 -ctv q4_0 --jinja --temp 0 --seed 3407

# Then test via API:
curl -s http://localhost:8098/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is the capital of France?","max_tokens":20,"temperature":0}'
# Expected: "Paris" — if you get "/////////" or "!!!!!!!!!" → bug reproduced
```

## Contact

- **GitHub:** @Indras-Mirror (issue #4)
- **Relay:** `quetza-codetl` (QuetzaCodetl session)
- **Thread:** `tbq4-coordination`
