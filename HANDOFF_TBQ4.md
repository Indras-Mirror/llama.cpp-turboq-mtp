# HANDOFF: TBQ4_0 Fused Flash Attention

**Date:** 2026-05-08 (Session 2)  
**Repo:** `/home/mal/AI/llama.cpp-mtp` (fork: `github.com/Indras-Mirror/llama.cpp-mtp`, branch `master`)  
**Goal:** Fuse TBQ4_0 dequant into flash attention — target 90+ tok/s at 200K on Qwen3.6-27B

---

## TL;DR — What's Working

| Component | Status |
|-----------|--------|
| GPU-side TBQ4 dequant on 27B at 200K | ✅ 85 t/s @ 256, 66 t/s @ 512 |
| Build (zero errors/warnings) | ✅ |
| llama-graph.cpp conditional pass-through | ✅ FIXED (both K and V) |
| Stride unit fix (half2→bytes) | ✅ FIXED |
| Template dispatch fix (ncols2=1 dead code) | ✅ FIXED |
| Fused kernel activation `<256,256,4,8>` | ✅ ACTIVATES |
| D=256 kernel extension (rotation, templates, dispatch) | ✅ Compiled |
| Fused kernel TEST on 27B | ❌ "misaligned address" crash |
| 7B TBQ4 support | ❌ nb1=264 alignment (deferred) |

---

## All Bugs Found (Chronological)

### BUG 1: llama-graph.cpp cast (Gemma — FIXED commit 5617cad)
**File:** `src/llama-graph.cpp:1952`
**Symptom:** Fused kernel never activated, CPU heap leak, 57 tok/s at 135K
**Code:** `tbq_attn_type = use_flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32`
**Fix:** Conditional — skip cast when `per_head_dim == 128 || per_head_dim == 256`

### BUG 2: fattn-mma-f16.cuh stride units (Gemma — FIXED commit 801b5df)
**Files:** `ggml/src/ggml-cuda/fattn-mma-f16.cuh:572, :910`
**Symptom:** "CUDA error: misaligned address" crash in TBQ4 tile loader
**Code:** TBQ4 loader called with `stride_K` (half2 units, e.g. 128) but loader expects BYTES (needs 512)
**Fix:** `const int stride_K_bytes = stride_K * int(sizeof(half2));` — applied to both K and V

### BUG 3: 7B model nb1=264 alignment (UNFIXED, deferred)
4-head models produce 8-byte aligned stride → VEC dispatch fails. Fix: padding in KV cache allocator. Not needed for 27B target.

### BUG 4: V-side D=256 pass-through missing (DeepSeek — FIXED commit fe1e6d6)
**File:** `src/llama-graph.cpp:1987`
**Symptom:** Fused kernel never activated on 27B (38.75 t/s vs 85 t/s baseline)
**Code:** `const bool use_tbq4_fused = use_flash_attn && per_head_dim == 128;` (V only allowed D=128)
**K-side:** `per_head_dim == 128 || per_head_dim == 256` — D=256 WAS allowed for K
**Fix:** Changed V-side to `per_head_dim == 128 || per_head_dim == 256`

### BUG 5: ncols2=1 dispatch to dead code (Gemma — FIXED locally, not pushed yet)
**File:** `ggml/src/ggml-cuda/fattn.cu` — `switch_ncols1` function
**Symptom:** When `use_gqa_opt=false`, ncols2=1 → dispatches to `<8,1>` (ncols=8). Volta MMA guard kills ncols<32 with `NO_DEVICE_CODE; return;`
**Fix:** Restricted 8/ncols2 branch to Turing-only; Ada falls through to ncols1=32 path

### BUG 6: "misaligned address" in fused kernel (ONGOING)
**Symptom:** Kernel compiles and activates `<256,256,4,8>`. printf at entry fires. Crash inside kernel body during first decode.
**Config:** ncols=32, nthreads=128, nbatch_fa=32, nbatch_K2/V2=128, Q_in_reg=true, nstages=0, shmem=33792
**Ruled out:** Q rotation, stride fix, template dispatch, shmem overflow, ldmatrix alignment, TBQ4 GMEM reads
**Active:** Zero-fill K tile test to isolate K loader vs MMA path

---

## MTP Performance Context
- Qwen3.6-27B MTP on 3090 Ti (q8 KV): ~47 tok/s (Reddit reference)
- Our GPU dequant TBQ4 at 200K: 66 t/s @ 512 — already 40% faster
- Fused kernel target: 90+ t/s
- Prefill reduced ~30% with MTP (known)
- MTP + vision currently broken (known)

---

## Key Files Changed

| File | Change | Author |
|------|--------|--------|
| `src/llama-graph.cpp:1957-1991` | Conditional TBQ4 pass-through (K + V) | Gemma + DeepSeek |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh:570-574,907-913` | stride_K/V * sizeof(half2) fix | Gemma |
| `ggml/src/ggml-cuda/fattn-mma-tbq4.cuh` | Q rotation + output rotation D=256 | DeepSeek |
| `ggml/src/ggml-cuda/fattn.cu:422-432,547-616` | Dispatch D=128+256 + ncols fix | DeepSeek + Gemma |
| `ggml/src/ggml-cuda/fattn-mma-tbq4-launch.cuh` | TBQ4 MMA launcher (nstages=0) | Claude |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh:520-521,562-574` | TBQ4 type hooks | Claude |
| `ggml/src/ggml-cuda/tbq4-cuda.cuh` | Warp-shuffle FWHT + block-diag | DeepSeek |
| `ggml/src/ggml-cuda/tbq4-sparse-v.cuh` | Sparse V utility | DeepSeek |
| `ggml/src/ggml-cuda/cpy.cu:554-559` | TBQ4→F32 CUDA dequant | Claude |
| `ggml/src/ggml-cuda/set-rows.cu:324-326` | TBQ4 CUDA quantize | Claude |
| 4× template instance files | D=128 + D=256 MMA kernels | Claude + DeepSeek |

---

## Build
```bash
cd /home/mal/AI/llama.cpp-mtp
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc) --target llama-server
# Force recompile after header changes:
touch ggml/src/ggml-cuda/fattn-mma-f16.cuh && cmake --build build -j$(nproc) --target llama-server
```

## Test Server (27B with fused kernel)
```bash
fuser -k 8096/tcp
cd /home/mal/AI/llama.cpp-mtp
build/bin/llama-server \
  -m /media/Crucial1TB/models/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P-MTP-slim.gguf \
  --port 8096 -c 200000 --flash-attn on --mlock -t 8 --poll 0 -ngl 99 \
  --parallel 1 --spec-type mtp --spec-draft-n-max 3 -b 2048 -ub 32 \
  -ctk tbq4_0 -ctv tbq4_0 --jinja --temp 0.6 --seed 3407
```

## Debug Commands
```bash
# Fallback to Q4_0 KV (verify model loads without TBQ4):
-ctk q4_0 -ctv q4_0

# Without warmup:
--no-warmup

# GPU memcheck:
compute-sanitizer --tool memcheck build/bin/llama-server ...

# Small context for fast iteration:
-c 4096
```

## Peers
- **quetzacodetl** (Gemma/Claude): Testing, dispatching, instrumentation
- **quetzacodetl-2** (DeepSeek): Kernel verification, tile analysis, documentation
- **Relay thread:** `tbq4-coordination`

## Full Session Log
`COLLAB_NOTES.md`
