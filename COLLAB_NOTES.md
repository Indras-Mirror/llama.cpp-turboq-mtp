# Claude + DeepSeek Collaboration Notes

## Status: ACTIVE
**Created:** 2026-05-08
**Goal:** Reduce TBQ4_0 KV cache overhead from ~40% to near-zero while maintaining near-lossless quality

## Current Performance Baseline
| Config | 512 tok gen | 1024 tok gen | MTP Accept |
|--------|-------------|--------------|------------|
| Q4_0 KV (target speed) | 91 tok/s | 76 tok/s | 64% |
| TBQ4 KV (current) | 57 tok/s | 42 tok/s | 62% |
| **Gap** | **37% slower** | **45% slower** | similar |

Hardware: RTX 4090 24GB, Qwen3.6-27B Q4_K_M, 135K context (goal: 200K+ with TBQ4), MTP draft=3
Wrapper: `~/.local/bin/qwen3.6-dense-mtp-quetza` — server launch script for benchmarking

## Task Assignment

### Claude (harder tasks — CUDA kernel fusion + architecture)
- [x] Clone reference repos (SAW-INT4, AXELRAM, spiritbuun/llama-cpp-turboquant-cuda)
- [x] **TASK C1+C2**: Fuse TBQ4 dequant into flash attention + rotated-domain attention (CODE DONE, BUILD PENDING)
- [ ] **TASK C3**: Investigate MTP acceptance rate degradation

### DeepSeek (well-scoped tasks — kernel optimization + sparse dequant)
- [x] **TASK D1**: Warp-shuffle FWHT butterfly in tbq4-cuda.cuh
- [x] **TASK D2**: Sparse V dequant (attention-aware skip)
- [x] **TASK D3**: Block-diagonal Hadamard option (32-point blocks)

## Build & Test Commands
```bash
cd /home/mal/AI/llama.cpp-mtp
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc)

# Test: launch server
build/bin/llama-server \
  -m /media/Crucial1TB/models/Qwen3.6-Heretic-MTP/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q4_K_M.gguf \
  --port 8096 -c 135000 --flash-attn on --mlock -t 8 --poll 0 -ngl 99 \
  --parallel 1 --spec-type mtp --spec-draft-n-max 3 -b 4096 -ub 32 \
  -ctk tbq4_0 -ctv tbq4_0 --jinja --temp 0.6 --seed 3407

# Benchmark: 512-token generation
curl -s http://localhost:8096/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Write a detailed explanation of transformer attention mechanisms."}],"max_tokens":512,"temperature":0.6,"seed":3407}'

# Check timing: 
tail -5 /tmp/qwen3.6-dense-mtp-llama-server.log | grep 'tok/s\|accept'
```

## Notes Between Agents
<!-- Add timestamped notes here -->

### 2026-05-08 — Claude (initial setup)
- Reference repos cloned to `reference/` directory
- spiritbuun's `fattn-mma-turbo.cuh` is the key reference for fused attention
- Their turbo4 tile loader does NOT need FWHT because `block_turbo4_0` stores data differently than our `block_tbq4_0`
- Our `block_tbq4_0` stores FWHT-rotated quantized values — inverse FWHT is required for correct dequant
- MTP KV cache IS being persisted correctly (speculative.cpp:676-689 only trims rejected positions)
- Acceptance rate ~62% seems to be inherent to the model/quant, not a bug

### 2026-05-08 — DeepSeek (D1 complete: warp-shuffle FWHT)
**File:** `ggml/src/ggml-cuda/tbq4-cuda.cuh:146-210`
- Replaced shared-memory butterfly stages 0-4 (h=1,2,4,8,16) with `__shfl_xor_sync`
- All 128 threads now active through all stages (was only 64)
- 7 `__syncthreads()` → 3 (57% barrier reduction)
- Fused s2 sign multiply into dequant load (saves register pass)
- Numerically bit-identical: same FP ops, shfl_xor is bit-exact copy
- Cross-warp stages 5-6 (h=32,64) still use shared memory

### 2026-05-08 — DeepSeek (D2 complete: sparse V dequant)
**File:** `ggml/src/ggml-cuda/tbq4-sparse-v.cuh` (new, 39 lines)
- Provides `tbq4_sparse_v_threshold(seq_len)` → `1.0/(4*seq_len)` for >8K context
- Provides `tbq4_sparse_v_skip(weight, threshold)` → skip decision
- Provides `tbq4_sparse_v_check(weight, seq_len)` → one-shot convenience
- To integrate: Claude calls threshold check in flash attention inner loop before V load
- Reference: TheTom's approach (+22.8% decode at 32K, PPL unchanged)

### 2026-05-08 — DeepSeek (D3 complete: block-diagonal Hadamard)
**File:** `ggml/src/ggml-cuda/tbq4-cuda.cuh:220-379`
- Added 32-point FWHT (`tbq4_fwht_32`) and 4×32 block-diagonal rotations
- `k_tbq4_bd32_dequant_full`: 32-thread kernel, ZERO barriers
- Each thread processes all 4 blocks simultaneously (4 parallel shuffle exchanges/stage)
- 5 butterfly stages × 4 shuffle exchanges = 20 shuffle ops, 0 syncs
- Matching `quantize_f32_tbq4_bd32_block` for SET_ROWS path
- **Format-breaking**: New GGUFs needed, enable via `TBQ4_BD32=1` env var
- Quality reference: SAW-INT4 (arXiv:2604.19157) — block-diagonal ≈ full FWHT for KV cache

### 2026-05-08 — Claude (C1+C2 progress: fused TBQ4 flash attention)

**Key Insight — Rotated-Domain Attention (eliminates ALL FWHT from inner loop):**
Since Hadamard is orthonormal: `Q·K = ||norm|| * rotate_forward(Q)^T · K_stored`
And for V accumulation: `sum(α_i · V_i) = rotate_inverse(sum(α_i · V_stored_i · norm_i))`
This means K and V tile loaders do ZERO FWHT — just centroid*norm lookup.
FWHT only runs twice total: once on Q (rotate_forward, amortized), once on output (rotate_inverse).

**Files Created:**
- `ggml/src/ggml-cuda/fattn-mma-tbq4.cuh` (158 lines, NEW)
  - `flash_attn_ext_tbq4_load_tile<>()` — reads raw block_tbq4_0, does centroid[nibble]*norm → half2 shmem tile
  - `tbq4_rotate_Q_tile<>()` — apply rotate_forward to Q in shmem (s1 → FWHT via warp-shuffle+shmem → s2 → scale)
  - `k_tbq4_rotate_output()` — post-attention kernel: rotate_inverse on VKQ output rows (s2 → FWHT → s1 → scale)
  - `tbq4_rotate_output_cuda()` — host launcher for output rotation

**Files Modified:**
- `ggml/src/ggml-cuda/fattn-mma-f16.cuh` (11 edits, backward-compatible)
  - Line 5: `#include "fattn-mma-tbq4.cuh"`
  - `flash_attn_ext_f16_iter`: added `ggml_type type_K=F16, type_V=F16` template params
  - Added `is_tbq4_kv` constexpr, forces `nstages=0` (no async pipeline for TBQ4)
  - K tile loading (~line 562): `if constexpr (!is_tbq4_kv)` guards F16 path, `else` calls tbq4 tile loader
  - V tile loading (~line 896): same pattern
  - `flash_attn_ext_f16_process_tile`: same template params forwarded
  - Q rotation (~line 1140): `if constexpr (is_tbq4_kv)` calls `tbq4_rotate_Q_tile` after Q load
  - `flash_attn_ext_f16` kernel: template params forwarded to all iter/process_tile calls
  - All existing F16/quantized paths UNCHANGED (default template args = GGML_TYPE_F16)

**ALSO NOW DONE (dispatch + launcher):**
- `ggml/src/ggml-cuda/fattn-mma-tbq4-launch.cuh` (NEW) — TBQ4 launcher mirroring turbo pattern
  - `ggml_cuda_flash_attn_ext_mma_tbq4_case<128,128,ncols1,ncols2>()` — nstages=0, V_is_K_view=false, need_f16_K/V=false
  - Template instance files: `template-instances/fattn-mma-tbq4-instance-ncols2_{1,2,4,8}.cu`
- `ggml/src/ggml-cuda/fattn.cu` (MODIFIED) — full dispatch wiring
  - `BEST_FATTN_KERNEL_MMA_TBQ4 = 500` enum value
  - `GGML_TYPE_TBQ4_0` in K->type switch (requires K+V both TBQ4_0, D=128, Turing+)
  - `ggml_cuda_flash_attn_ext_mma_tbq4()` with ncols switch + output rotation call
  - Dispatch case in `ggml_cuda_flash_attn_ext()`

**REMAINING:**
- D=128 model test: validate fused kernel speedup (need 7B Qwen/Mistral GGUF)
- D=256 ldmatrix fix: tile layout redesign for multi-row TBQ4 blocks
- D2 integration: Sparse V into fused path (optimization pass)

### Build & Test Status (2026-05-08)
- ✅ BUILD PASSED — zero errors, zero warnings
- ✅ D=256 working via GPU-side F32→F16 dequant (not fused)
- 🔶 D=256 fused kernel: tile layout mismatch (ldmatrix stride 32 vs TBQ4 block 64 half2)
- 🔶 Fused kernel: ready for D=128, needs D=128 model to validate

### 2026-05-08 — Gemma (llama-graph.cpp fix + D=256 investigation)

**Root Cause #1 — llama-graph.cpp:1952:** `tbq_attn_type = use_flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32`
This cast TBQ4→F16 before flash_attn, so the fused kernel (expecting raw TBQ4_0) never activated. It also routed dequant to CPU (GGML_OP_CPY had no CUDA TBQ4→F16 path), causing PCIe bottleneck and "CPU heap leak".

**Fix:** Conditional cast — only when per_head_dim ≠ 128 (now per_head_dim == 128 to enable fused kernel, else TBQ4→F32→F16 on GPU).
- File: `src/llama-graph.cpp:1962-1970` (K path) + `:1983-1991` (V path)
- TBQ4→F32 dequant stays on GPU (cpy.cu:554-559, existing CUDA kernel)
- F32→F16 via existing ggml_cast path (line 2013-2018)
- Result: GPU-side dequant beats CPU CPY even at 200K

**Root Cause #2 — D=256 mismatch:** Qwen3.6 has n_embd_head_k=256 (2 TBQ4 blocks/head). The fused kernel was designed for D=128 (1 block/head). Dispatch check `Q->ne[0] != 128` returned NONE→ABORT, causing 503 errors.

**DeepSeek extended** Q rotation, output rotation, template instances, dispatch to support D=256. But the **TBQ4 tile loader has a layout bug**: each TBQ4 block produces 64 half2 entries, but Ampere config uses stride_tile=32 → 2 tile rows per block. The ldmatrix iteration pattern expects 1 GMEM row per tile row. Fixing this requires redesigning the K/V tile loading for multi-row blocks.

**Decision:** Revert fused kernel to D=128-only. D=256 uses GPU-side dequant fallback. NEXT: find D=128 model to validate fused kernel.

### 200K Final Benchmark (D=256 GPU dequant path)

| Tokens | Speed | Accept | VRAM |
|--------|-------|--------|------|
| 256 | 85.0 t/s | 75.2% | 23.6 GB |
| 512 | 66.4 t/s | 56.8% | 23.6 GB |
| 1024 | 63.4 t/s | 64.0% | 23.6 GB |

Comparison to baselines:
- Original TBQ4 (CPU dequant) at 135K: 57 t/s @ 512, 42 t/s @ 1024
- **GPU dequant fix at 200K: 66.4 t/s @ 512 (+16%), 63.4 t/s @ 1024 (+51%)**
- Q4_0 baseline at 135K: 91 t/s @ 512, 76 t/s @ 1024
- **Target (need fused kernel): 90+ t/s at 200K**
