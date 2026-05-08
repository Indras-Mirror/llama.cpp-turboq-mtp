# TBQ4 Fused Flash Attention — Full Session Handoff

**Date:** 2026-05-08
**Goal:** Reduce TBQ4_0 KV cache overhead from ~40% to near-zero, enabling 200K+ context with TBQ4 at near-Q4_0 speed
**Hardware:** RTX 4090 24GB (sm_89), Qwen3.6-27B Q4_K_M, 135K context (goal: 200K+), MTP draft=3
**Wrapper:** `~/.local/bin/qwen3.6-dense-mtp-quetza` — launches server with correct model/settings for benchmarking

## Performance Baseline

| Config | 512 tok gen | 1024 tok gen | MTP Accept |
|--------|-------------|--------------|------------|
| Q4_0 KV (target) | 91 tok/s | 76 tok/s | 64% |
| TBQ4 KV (current) | 57 tok/s | 42 tok/s | 62% |
| **Gap** | **37% slower** | **45% slower** | similar |

**Root cause:** TBQ4 dequant runs as a separate CUDA kernel → writes F32 to GMEM → flash attention reads it back. The FWHT butterfly (128-point, 7 sync barriers) is the bottleneck.

## The Solution: Rotated-Domain Fused Attention

**Key mathematical insight:** Since Hadamard is orthonormal, we can do attention entirely in the rotated domain:
- `Q·K = rotate_forward(Q)^T · (centroid[idx] * norm)` — no FWHT on K
- `sum(α·V) = rotate_inverse(sum(α · centroid[idx] * norm))` — no FWHT on V
- FWHT runs only TWICE total: once on Q (rotate_forward), once on output (rotate_inverse)

This eliminates the per-tile FWHT that was the performance killer.

---

## Task Status

### Claude Tasks

#### C1+C2: Fused TBQ4 Attention — 70% COMPLETE

**DONE:**

1. **`ggml/src/ggml-cuda/fattn-mma-tbq4.cuh`** (NEW, 158 lines)
   - `flash_attn_ext_tbq4_load_tile<D, stride_tile, nbatch_fa, nthreads, oob_check>()` — TBQ4 tile loader
     - Reads raw `block_tbq4_0` from GMEM (66 bytes: 2-byte norm + 64-byte packed 4-bit indices)
     - Does centroid lookup: `d_tbq4_centroids[nibble] * norm` → half2 shmem tile
     - NO FWHT — works in rotated domain
     - Template matches flash_attn_ext_f16 tile loader signature
   - `tbq4_rotate_Q_tile<DKQ, ncols, nwarps>()` — Q rotation (rotate_forward)
     - Reads half2 Q from shmem, unpacks to float
     - Multiplies by s1 sign array
     - 128-point FWHT: stages 0-4 via `__shfl_xor_sync`, stages 5-6 via shared memory (q_fwht_buf[128])
     - Multiplies by `inv_sqrt_128 * s2`, packs back to half2
     - Called ONCE per kernel invocation (amortized over all K/V tiles)
   - `k_tbq4_rotate_output()` — post-attention output rotation (rotate_inverse)
     - 128 threads per row, one row per block
     - s2 → FWHT (same warp-shuffle + shmem pattern) → s1 → scale
   - `tbq4_rotate_output_cuda()` — host launcher

2. **`ggml/src/ggml-cuda/fattn-mma-f16.cuh`** (MODIFIED, 11 edits)
   - All changes use template defaults `type_K=GGML_TYPE_F16, type_V=GGML_TYPE_F16` — existing paths untouched
   - Line 5: `#include "fattn-mma-tbq4.cuh"`
   - `flash_attn_ext_f16_iter` template: added `ggml_type type_K, type_V` params
     - `constexpr bool is_tbq4_kv = (type_K == GGML_TYPE_TBQ4_0 || type_V == GGML_TYPE_TBQ4_0)`
     - `nstages = is_tbq4_kv ? 0 : ...` (disable async pipeline — TBQ4 tile loader doesn't use cp.async)
     - K tile loading (~line 562): `if constexpr (!is_tbq4_kv) { /* F16 path */ } else { flash_attn_ext_tbq4_load_tile<...>() }`
     - V tile loading (~line 896): same pattern
   - `flash_attn_ext_f16_process_tile` template: same type_K/type_V params
     - Q rotation (~line 1140): `if constexpr (is_tbq4_kv) { tbq4_rotate_Q_tile<...>() }` after Q load
   - `flash_attn_ext_f16` kernel template: type_K/type_V forwarded to all iter/process_tile calls

**ALSO DONE (dispatch + launcher + instances):**

3. **`ggml/src/ggml-cuda/fattn-mma-tbq4-launch.cuh`** (NEW) — TBQ4 launcher
   - `ggml_cuda_flash_attn_ext_mma_tbq4_case<128,128,ncols1,ncols2>()` — mirrors turbo pattern
   - nstages=0, V_is_K_view=false, need_f16_K=false, need_f16_V=false
   - DECL macros for extern template declarations (D=128, ncols 8/16/32/64)

4. **`ggml/src/ggml-cuda/fattn.cu`** (MODIFIED) — dispatch wiring
   - `BEST_FATTN_KERNEL_MMA_TBQ4 = 500` enum value
   - `GGML_TYPE_TBQ4_0` case in K->type switch: requires both K+V TBQ4_0, D=128, Turing+
   - `ggml_cuda_flash_attn_ext_mma_tbq4_switch_ncols1<ncols2>()` — ncols routing
   - `ggml_cuda_flash_attn_ext_mma_tbq4()` — top-level with GQA logic + output rotation call
   - Added to main `ggml_cuda_flash_attn_ext()` switch

5. **Template instance files** (4 NEW .cu files)
   - `template-instances/fattn-mma-tbq4-instance-ncols2_{1,2,4,8}.cu`
   - Each instantiates 4 ncols1 variants (total 16 specializations for D=128)
   - Auto-picked up by CMake via `file(GLOB SRCS "template-instances/fattn-mma*.cu")`

6. **Output rotation** — wired into `ggml_cuda_flash_attn_ext_mma_tbq4()` after launch_fattn returns
   - `tbq4_rotate_output_cuda((float*)KQV->data, nrows, 128, stream)`

**BUILD STATUS: PASSED (2026-05-08 06:08 AEST)**
- Zero errors, zero warnings
- All 4 TBQ4 template instance files compiled and linked
- Binary: `build/bin/llama-server` (9.4MB)

**NEXT STEP: TEST**
- Launch server with TBQ4 fused flash attention and benchmark
- Compare tok/s against baseline (57 tok/s unfused, 91 tok/s Q4_0 target)
- Watch for: CUDA errors at runtime, incorrect output, crashes at long context

**REMAINING WORK:**
- C3: MTP acceptance rate investigation (lower priority)
- D2 integration: sparse V into fused attention path (optimization pass after correctness verified)

#### C3: MTP Acceptance Rate — NOT STARTED
- ~62% accept rate may be inherent to TBQ4 quantization noise, not a bug
- speculative.cpp:676-689 correctly trims only rejected positions
- Lower priority — focus on C1+C2 speed first

### DeepSeek Tasks (via Quetza Relay)

#### D1: Warp-Shuffle FWHT — COMPLETE
- File: `tbq4-cuda.cuh:146-210`
- Replaced shmem butterfly stages 0-4 with `__shfl_xor_sync`
- All 128 threads active (was 64), 7 barriers → 3
- Bit-identical output

#### D2: Sparse V Dequant — COMPLETE
- File: `tbq4-sparse-v.cuh` (NEW, 39 lines)
- `tbq4_sparse_v_threshold(seq_len)` → `1.0/(4*seq_len)` for >8K context
- `tbq4_sparse_v_check(weight, seq_len)` → skip decision
- Integration point: V tile loading branch in fattn-mma-f16.cuh (Claude's territory)

#### D3: Block-Diagonal Hadamard — COMPLETE
- File: `tbq4-cuda.cuh:220-379`
- 32-point FWHT, 4×32 block-diagonal rotations
- 32-thread kernel, ZERO sync barriers
- Format-breaking: new GGUFs needed, `TBQ4_BD32=1` env var
- Matching quantize path included

---

## File Map

```
ggml/src/ggml-cuda/
├── fattn-mma-tbq4.cuh    ← NEW: TBQ4 tile loader + Q/output rotation (Claude)
├── fattn-mma-f16.cuh     ← MODIFIED: template params + conditional loading (Claude)
├── fattn-mma-tbq4-launch.cuh ← NEW: TBQ4 launcher (nstages=0, no F16 conv) (Claude)
├── fattn.cu              ← MODIFIED: dispatch enum + type switch + ncols routing + output rotation (Claude)
├── fattn-common.cuh      ← READ ONLY: launch_fattn() at line 917
├── tbq4-cuda.cuh         ← MODIFIED: warp-shuffle FWHT + block-diagonal (DeepSeek D1+D3)
├── tbq4-sparse-v.cuh     ← NEW: sparse V utility (DeepSeek D2)
├── cpy.cu:554-559        ← Where standalone k_tbq4_dequant_full is called (bypass target)
└── template-instances/
    ├── fattn-mma-tbq4-instance-ncols2_1.cu  ← NEW: D=128 ncols1={8,16,32,64}
    ├── fattn-mma-tbq4-instance-ncols2_2.cu  ← NEW: D=128 ncols1={4,8,16,32}
    ├── fattn-mma-tbq4-instance-ncols2_4.cu  ← NEW: D=128 ncols1={2,4,8,16}
    └── fattn-mma-tbq4-instance-ncols2_8.cu  ← NEW: D=128 ncols1={1,2,4,8}

reference/
├── llama-cpp-turboquant-cuda/  ← spiritbuun's fork (key reference for launcher pattern)
│   └── ggml/src/ggml-cuda/
│       ├── fattn-mma-turbo.cuh ← Turbo4 launcher (reference for C1+C2 remaining work)
│       └── turbo-quant-cuda.cuh
├── AXELRAM/                    ← Rotated-domain theory
└── saw-int4/                   ← Block-diagonal Hadamard paper
```

## Build & Test

```bash
cd /home/mal/AI/llama.cpp-mtp
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc)

# Server test
build/bin/llama-server \
  -m /media/Crucial1TB/models/Qwen3.6-Heretic-MTP/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q4_K_M.gguf \
  --port 8096 -c 135000 --flash-attn on --mlock -t 8 --poll 0 -ngl 99 \
  --parallel 1 --spec-type mtp --spec-draft-n-max 3 -b 4096 -ub 32 \
  -ctk tbq4_0 -ctv tbq4_0 --jinja --temp 0.6 --seed 3407

# Benchmark: 512-token generation
curl -s http://localhost:8096/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Write a detailed explanation of transformer attention mechanisms."}],"max_tokens":512,"temperature":0.6,"seed":3407}'
```

## Coordination

- **COLLAB_NOTES.md** — timestamped progress log, both agents update
- **DEEPSEEK_HANDOFF.md** — DeepSeek-specific task details and file map
- **Quetza Relay** — `relay_peers` to find DeepSeek, `relay_ask` to message
- DeepSeek's peer name: `quetzacodetl`
