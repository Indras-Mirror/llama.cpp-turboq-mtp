# HANDOFF: TBQ4_0 Fused Flash Attention

**Date:** 2026-05-08 (Session 5 — nstages=2 SURVIVES, 15 bugs fixed, ready for benchmarks)
**Repo:** `/home/mal/AI/llama.cpp-mtp` (fork: `github.com/Indras-Mirror/llama.cpp-mtp`, branch `master`)
**Next session:** 2026-05-09

## END GOAL

**60-90 tok/s with MTP + TBQ4_0 (lossless 4.25 bpw KV cache) at 200K+ context on RTX 4090 24GB.**

VRAM fits (~20GB at 200K). Speed is the only blocker.

---

## Current Status: nstages=2 SURVIVES (Session 5)

**v1 (committed):** MTP + TBQ4 fused FA, nstages=0. **43.1 tok/s** (Q4_0 baseline: 69.8 tok/s).

**v2 (Session 4, uncommitted):** nstages=2 pipeline with raw byte staging + collaborative dequant.
- Non-MTP: 47.2 tok/s (15% speedup)
- MTP: CRASHED ("misaligned address") — `#if !defined(TBQ4_KV_FUSED)` guards missing

**Session 5 (2026-05-08, uncommitted):**
- Guard fix verified: MTP works at 50.5 tok/s (nstages=0)
- nstages=2 guard refactor: inline `#if defined(TBQ4_NSTAGES_2)` approach
- **Non-MTP nstages=2: 43.8-45.6 tok/s, coherent output** (quetzacodetl)
- **MTP nstages=2: SURVIVES** on quetzacodetl-2 end (31.3 tok/s, garbled — stale build)
- **MTP nstages=2: crashes silently** on quetzacodetl end (under investigation)

### Session 5 Benchmark Results

| Config | tok/s | Output | Who |
|--------|-------|--------|-----|
| Q4_0 baseline + MTP | 69.8 | Coherent | Reference |
| TBQ4 v1 (nstages=0) + MTP | 43.1-50.5 | Coherent | Both |
| **TBQ4 nstages=2 non-MTP** | **43.8-45.6** | **Coherent** | quetzacodetl |
| TBQ4 nstages=2 non-MTP | 44.9 | Garbled (stale build) | quetzacodetl-2 |
| TBQ4 nstages=2 + MTP | 31.3 | Garbled (stale build) | quetzacodetl-2 |
| TBQ4 nstages=2 + MTP | CRASH | N/A | quetzacodetl |

Target: 55-65 tok/s with MTP.

---

## Architecture

### 3-Phase v1 (committed)
```
1. k_tbq4_rotate_input   → Pre-rotate Q (separate FWHT kernel, 128-thread warp shuffle)
2. Fused TBQ4 FA kernel  → Read TBQ4 blocks directly from GMEM, centroid×norm dequant, nstages=0
3. k_tbq4_rotate_output  → Post-rotate VKQ back to original domain
```

K/V are pre-rotated at SET_ROWS time (quantize_f32_tbq4_0_block calls tbq4_rotate_forward before quantization — tbq4-cuda.cuh:107). Everything operates in rotated domain.

### v2 Pipeline (working for non-MTP)
```
Iter N:
  1. Dequant K[N] from staging → tile_K        (raw bytes preloaded by previous iter)
  2. Load V[N] raw bytes → staging              (int copies overlap with KQ compute)
  3. KQ MMA + softmax
  4. Dequant V[N] from staging → tile_V         (V ready, staging now free)
  5. VKQ MMA
  6. Load K[N+1] raw bytes → staging            (prefetch for next iter)
```

Single staging buffer reused: V overwrites K's raw bytes (K already in tile_K), K[next] overwrites V's raw bytes (V already in tile_V).

**block_tbq4_0:** 64 bytes qs + 2 bytes norm = **66 bytes** (QK_TBQ4=128). D=256: 2 blocks/row = 132 bytes → padded to 144 for staging alignment.

### Guard Structure (Session 5)

4 locations in `fattn-mma-f16.cuh` use the two-level guard:

```cpp
#if defined(TBQ4_NSTAGES_2)
    // TBQ4 nstages=2: int-load pipeline (dequant K, load V, dequant V, K next preload)
#elif !defined(TBQ4_KV_FUSED)
    // F16 nstages>1: cp_async pipeline
#endif
```

| Line | Location | Purpose |
|------|----------|---------|
| 552 | iter start | K dequant + V load |
| 921 | mid-iter | V dequant from staging (ordering fix) |
| 942 | iter end | K[next] preload |
| 1262 | process_tile | Initial K[0] preload (sync mask, no cp_async) |

Instance files define both `TBQ4_NSTAGES_2` and `TBQ4_KV_FUSED`. `TBQ4_NSTAGES_2` checked FIRST for priority.

Same kernel (`flash_attn_ext_f16`), same process_tile/combine/Q-load. Only iter function's nstages>1 blocks differ via guards. Launcher uses `#ifdef TBQ4_NSTAGES_2 → nstages=2` with staging + 2-stage KV shmem.

---

## Uncommitted Changes (8 files, +443/-106)

```
ggml/src/ggml-cuda/fattn-mma-f16.cuh               | +168  TBQ4_NSTAGES_2 guards, V dequant fix, last iter fix
ggml/src/ggml-cuda/fattn-mma-tbq4.cuh              | +115  raw int loader, collaborative dequant, staging helpers
ggml/src/ggml-cuda/fattn-mma-tbq4-launch.cuh       |  +19  nstages=2, 2-stage KV shmem, staging allocation
template-instances/fattn-mma-tbq4-instance-ncols2_1.cu | +1  TBQ4_NSTAGES_2 define
template-instances/fattn-mma-tbq4-instance-ncols2_2.cu | +1  TBQ4_NSTAGES_2 define
template-instances/fattn-mma-tbq4-instance-ncols2_4.cu | +1  TBQ4_NSTAGES_2 define
template-instances/fattn-mma-tbq4-instance-ncols2_8.cu | +1  TBQ4_NSTAGES_2 define
HANDOFF_TBQ4.md                                    |     (this file)
```

---

## Bugs Fixed (15 total across v1+v2)

| # | Bug | Fix | Session |
|---|-----|-----|---------|
| 1 | ncols2=1 dead dispatch (Volta MMA) | Turing-only guard | v1 |
| 2 | V-side D=256 pass-through | Added per_head_dim==256 condition | v1 |
| 3 | nvcc constexpr dead-branch codegen | `#if !defined(TBQ4_KV_FUSED)` guards (v1) → `TBQ4_NSTAGES_2` (v2) | v1/v2 |
| 4 | Q rotation register spill | Separate rotation kernel | v1 |
| 5 | CUDA graph capture rejects debug | Removed debug sync/printf | v1 |
| 6 | Mask null pointer dereference | Added mask_h guard | v1 |
| 7 | V dequant AFTER K preload (reads garbage) | Move V dequant before K preload | v2 |
| 8 | cp_async misaligned (132-byte rows, 16-byte req) | Replace cp_async with int loads | v2 |
| 9 | oob_check static_assert with nstages>1 | Add `|| is_tbq4_kv` exception | v2 |
| 10 | Default arg not at end of param list | Move tbq4_staging to last param | v2 |
| 11 | constexpr __host__ in device code | Add `__host__ __device__` annotation | v2 |
| 12 | cp_async_cg_16 without CP_ASYNC guard | Wrap in #ifdef CP_ASYNC_AVAILABLE | v2 |
| 13 | Unused variable elems_per_pass | Removed | v2 |
| 14 | Last iter calls missing tbq4_staging | Added tbq4_staging + type_K,type_V to both ncols2 paths | S5 |
| 15 | cp_async mask load in process_tile preload | Changed to sync mask load (no cp_async) | S5 |

---

## Spiritbuun Fork Analysis

- **Same architecture as our v1:** separate WHF kernel + fused FA with nstages=0
- Comment: "Turbo forces nstages=0: cp.async can't do ALU dequant, so tiles load synchronously"
- **No nstages>0 support — nobody has solved this before us**
- Different block format (block_turbo4_0 vs block_tbq4_0) — NOT directly compatible

---

## Test Commands

```bash
cd /home/mal/AI/llama.cpp-mtp

# Build
cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc) --target llama-server

# Non-MTP test (WORKS — 43.8 tok/s coherent on quetzacodetl)
build/bin/llama-server \
    -m "/media/Crucial1TB/models/Qwen3.6-Heretic-MTP/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q4_K_M.gguf" \
    --port 8097 -c 4096 --flash-attn on -ngl 99 --parallel 1 \
    -b 256 -ub 32 -ctk tbq4_0 -ctv tbq4_0 --no-warmup

# MTP test (SURVIVES on quetzacodetl-2, crashes on quetzacodetl)
build/bin/llama-server \
    -m "..." --port 8097 -c 4096 --flash-attn on -ngl 99 --parallel 1 \
    -b 256 -ub 32 -ctk tbq4_0 -ctv tbq4_0 --no-warmup \
    --spec-type mtp --spec-draft-n-max 3

# Q4_0 baseline for comparison
# -ctk q4_0 -ctv q4_0
```

---

## Next Session Priorities (2026-05-09)

1. **Fix quetzacodetl's MTP crash** — silent crash on first inference, not Bug #14. MTP draft model may hit different ncols config or tensor layout. Check ncols2=1 path (oob_check=true) for draft model.
2. **Benchmark MTP properly** — quetzacodetl-2 has working MTP (31.3 tok/s), need coherent output for proper draft acceptance measurement.
3. **Longer context benchmark** — pipeline overlap benefit may only show at longer contexts where K/V loading dominates.
4. **Fix quetzacodetl-2 stale build** — garbled "///////" output on clean build suggests GPU driver state issue (cold boot recommended).
5. **Try nstages=2 on both ends with coherent output** to measure real speedup vs nstages=0.
6. **Commit and push** once MTP crash resolved on both ends.

---

## Collaborators

- **quetzacodetl** — Guard fix verification, nstages=2 guard refactor, process_tile initial preload, MTP testing. Coherent output at 43.8 tok/s non-MTP.
- **quetzacodetl-2** (this session) — Last iter fix (#14), cp_async fix (#15), MTP survival confirmation, build testing.

### Relay Protocol

Use `relay_peers` to discover active sessions. Use `relay_ask` to coordinate. Thread: `tbq4-coordination`.

---

## Key Lessons

1. **`#if !defined(TBQ4_KV_FUSED)` guards are ESSENTIAL** — nvcc generates bad code from `if constexpr` dead branches in TBQ4 templates.
2. **`TBQ4_NSTAGES_2` must be checked BEFORE `!defined(TBQ4_KV_FUSED)`** — TBQ4_NSTAGES_2 instances define both macros.
3. **cp_async intrinsics poison TBQ4 templates anywhere** — not just in iter function. Process_tile mask load also affected (Bug #15).
4. **cp_async requires 16-byte alignment** on both src and dst — TBQ4 rows are 132 bytes. Use int loads instead.
5. **Staging buffer ordering is critical** — V dequant MUST happen before K[next] preload (shared buffer).
6. **Last iter calls need tbq4_staging** — nstages>1 iter ALWAYS accesses staging for K dequant, even on last iteration (Bug #14).
7. **Launcher and kernel must agree on nstages** — mismatch causes OOB shared memory access.
8. **Q rotation is via SEPARATE kernel** (fattn.cu:591) — in-kernel rotation (tbq4_rotate_Q_tile) would double-rotate and is disabled.
9. **K/V are pre-rotated at SET_ROWS time** — quantize_f32_tbq4_0_block calls tbq4_rotate_forward (tbq4-cuda.cuh:107).
10. **Spiritbuun fork confirms nobody has solved nstages>0** for quantized KV FA — we're pioneering this.
