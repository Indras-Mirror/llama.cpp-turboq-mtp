# TBQ4_0 Fused Flash Attention — Collaboration Log

**Goal:** Eliminate TBQ4_0 KV cache overhead (~40% slower than Q4_0) by fusing dequant into flash attention.
**Target:** 90+ tok/s at 200K context on RTX 4090 24GB with Qwen3.6-27B + MTP.
**Repo:** `/home/mal/AI/llama.cpp-mtp` (fork: `github.com/Indras-Mirror/llama.cpp-mtp`, branch `master`)
**Hardware:** RTX 4090 24GB (sm_89 Ada Lovelace)

## Performance Summary

| Config | Context | 256 tok | 512 tok | 1024 tok | Accept | VRAM |
|--------|---------|---------|---------|----------|--------|------|
| Q4_0 KV baseline | 135K | ~95 | 91 t/s | 76 t/s | 64% | 21 GB |
| TBQ4 original (CPU dequant) | 135K | ~60 | 57 t/s | 42 t/s | 62% | 19 GB |
| **TBQ4 GPU dequant (working)** | **200K** | **85.0 t/s** | **66.4 t/s** | **63.4 t/s** | 57-75% | **23.6 GB** |
| TBQ4 fused kernel (target) | 200K | ??? | ??? | ??? | ??? | 23.6 GB |

## Bugs Found & Fixed

### BUG 1: llama-graph.cpp cast (Gemma — FIXED)
**File:** `src/llama-graph.cpp:1952` (originally)
**Symptom:** Fused kernel never activated, CPU heap leak, 57 tok/s at 135K
**Code:** `tbq_attn_type = use_flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32`
**Effect:** Cast TBQ4→F16 before flash_attn. No CUDA TBQ4→F16 path → CPU dequant → PCIe bottleneck.
**Fix:** Conditional — skip cast when `per_head_dim == 128 || per_head_dim == 256` (fused kernel handles raw TBQ4). Otherwise TBQ4→F32→F16 on GPU.

### BUG 2: fattn-mma-f16.cuh stride units (Gemma — FIXED)
**Files:** `ggml/src/ggml-cuda/fattn-mma-f16.cuh:572, :910`
**Symptom:** "CUDA error: misaligned address" crash in TBQ4 tile loader
**Code:** TBQ4 loader called with `stride_K` (half2 units, e.g. 128) but loader expects BYTES (needs 512)
**Effect:** Tile loader strides by 128 bytes between GMEM rows instead of 512 → reads garbage → ldmatrix faults
**Fix:** `const int stride_K_bytes = stride_K * int(sizeof(half2));` — multiply by 4 to convert half2→bytes. Applied to both K (:572) and V (:910) tile loads.

### BUG 3: 7B model nb1=264 alignment (UNFIXED, separate issue)
**Symptom:** Qwen2.5-7B crashes with TBQ4 even WITHOUT fused kernel (F32 dequant path)
**Cause:** 7B has n_head_kv=4, D=128 → nb1 = 4 heads × 66 bytes/block = 264 → 8-byte aligned (not 16-byte). 27B has nb1=528 → 16-byte aligned → works fine.
**Status:** Deferred. 27B is the target. Fix would be padding in KV cache allocator.

## Architecture: How the Fused Kernel Works

### Rotated-Domain Attention (key insight)
Hadamard matrix H is orthonormal (H^T = H). So:
- `Q·K = rotate_forward(Q)^T · K_stored` (attention in rotated domain)
- `output = rotate_inverse(sum(α_i · V_stored_i · norm_i))`

This eliminates ALL FWHT from the inner attention loop. K and V tile loaders only do `centroid[nibble] * norm` lookup — no FWHT. FWHT runs exactly twice: once on Q (rotate_forward), once on output (rotate_inverse).

### Files

| File | Purpose | Status |
|------|---------|--------|
| `src/llama-graph.cpp:1957-1991` | Conditional TBQ4 pass-through | ✅ FIXED |
| `ggml/src/ggml-cuda/fattn-mma-tbq4.cuh` | TBQ4 tile loader + Q/output rotation | ✅ D=128+256 |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh:562-574,907-913` | TBQ4 hooks in MMA kernel | ✅ Stride FIXED |
| `ggml/src/ggml-cuda/fattn-mma-tbq4-launch.cuh` | TBQ4 MMA launcher | ✅ |
| `ggml/src/ggml-cuda/fattn.cu:421-432,547-616` | Dispatch + kernel selection | ✅ D=128+256 |
| `ggml/src/ggml-cuda/tbq4-cuda.cuh` | Warp-shuffle FWHT + block-diag FWHT | ✅ |
| `ggml/src/ggml-cuda/tbq4-sparse-v.cuh` | Sparse V dequant utility | ✅ (not integrated) |
| `ggml/src/ggml-cuda/cpy.cu:554-559` | TBQ4→F32 CUDA dequant | ✅ |
| `ggml/src/ggml-cuda/set-rows.cu:324-326` | TBQ4 CUDA quantize | ✅ |
| Template instances: `fattn-mma-tbq4-instance-ncols2_{1,2,4,8}.cu` | D=128 + D=256 kernels | ✅ |

### DeepSeek's Contributions (peer: quetzacodetl-2)
- **D1:** Warp-shuffle FWHT butterfly (tbq4-cuda.cuh:146-210) — 7→3 barriers, 57% reduction
- **D2:** Sparse V dequant utility (tbq4-sparse-v.cuh) — attention-aware skip
- **D3:** Block-diagonal Hadamard 4×32 (tbq4-cuda.cuh:220-379) — format-breaking, env var
- **D=256 extension:** Q rotation 2-block loop, output rotation 2-block loop, template instances for DKQ=256, generic ncols switch, dispatch update

## Current State (2026-05-08 end of session)

### What Works
- ✅ Build: zero errors, all template instances compile
- ✅ GPU-side TBQ4 dequant on 27B at 200K (85 t/s @ 256, 66 t/s @ 512)
- ✅ llama-graph.cpp conditional pass-through fix
- ✅ Stride unit fix (half2→bytes) applied to both K and V tile loads
- ✅ D=256 re-enabled in graph builder (per_head_dim 128 OR 256)
- ✅ Pushed to GitHub: `Indras-Mirror/llama.cpp-mtp` master

### TEST NEEDED (interrupted)
The fused D=256 kernel on the 27B was about to be tested when context window was refreshed. Server command:
```bash
fuser -k 8096/tcp
cd /home/mal/AI/llama.cpp-mtp
build/bin/llama-server \
  -m /media/Crucial1TB/models/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P-MTP-slim.gguf \
  --port 8096 -c 200000 --flash-attn on --mlock -t 8 --poll 0 -ngl 99 \
  --parallel 1 --spec-type mtp --spec-draft-n-max 3 -b 2048 -ub 32 \
  -ctk tbq4_0 -ctv tbq4_0 --jinja --temp 0.6 --seed 3407
```
Expected: server loads clean (no "misaligned address" crash). If it works, benchmark and compare to 66 t/s GPU dequant baseline.

### If Fused Kernel Crashes
1. Re-enable stride debug: add fprintf to `fattn-mma-tbq4-launch.cuh` showing K/V nb values
2. Check if the right ncols template is selected (gqa_ratio for Qwen3.6 = 6 → ncols2=8 path)
3. Verify D=256 kernel actually selected: add fprintf to `ggml_cuda_flash_attn_ext_mma_tbq4()`
4. Try with Q4_0 KV first to confirm model loads (baseline sanity check)

## Build
```bash
cd /home/mal/AI/llama.cpp-mtp
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc) --target llama-server
# If only headers changed, touch to force recompile:
touch ggml/src/ggml-cuda/fattn-mma-f16.cuh && cmake --build build -j$(nproc) --target llama-server
```

## Peer Coordination
**Relay:** MCP tool `mcp__quetza-relay__relay_ask` to `quetzacodetl-2` (DeepSeek)
**Protocol:** Use `relay_ask` for NEW messages (not relay_reply — ask_ids expire quickly)
**DeepSeek's current focus:** Tile layout / ldmatrix analysis, D=256 kernel activation verification
