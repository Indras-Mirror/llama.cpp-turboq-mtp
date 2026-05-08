# HANDOFF: TBQ4_0 Fused Flash Attention

**Date:** 2026-05-08 (Session 2 — COMPLETE)  
**Repo:** `/home/mal/AI/llama.cpp-mtp` (fork: `github.com/Indras-Mirror/llama.cpp-mtp`, branch `master`)  

## END GOAL

**60-90 tok/s with MTP + TBQ4_0 (lossless 4.25 bpw KV cache) at 200K+ context on RTX 4090 24GB.**

This enables maximum context with no quality loss vs FP16 KV cache, while keeping MTP speculative decoding active.

## Current Status: v1 COMPLETE (f35f29bcb)

**Working:** MTP + TBQ4 fused flash attention produces correct output at 49.5 tok/s.

### Architecture (v1)
```
1. k_tbq4_rotate_input   → Pre-rotate Q (separate FWHT kernel, isolates from FA register pressure)
2. Fused TBQ4 FA kernel  → Real TBQ4 K/V loaders, nstages=0 (synchronous), nvcc fixes applied
3. k_tbq4_rotate_output  → Post-rotate VKQ back to original domain
4. MTP speculative decode → --spec-type mtp --spec-draft-n-max 3
```

### Performance
| Config | tok/s |
|--------|-------|
| F16 baseline + MTP | ~85 |
| GPU dequant TBQ4 + F16 FA + MTP | ~66 |
| **Fused TBQ4 FA + MTP (v1)** | **49.5** |
| Fused TBQ4 FA no MTP (est.) | ~41 |

**Bottleneck:** nstages=0 — load and compute are serialized. F16 uses nstages=2 (double-buffered cp_async pipeline). TBQ4 dequant requires ALU (centroid lookup × norm), which can't be done via cp_async DMA.

**VRAM savings:** Eliminates F16 dequant buffer (~2GB at 200K context). Enables longer context at the cost of ~40% speed vs F16 baseline.

### Path to v2 (60-90 tok/s)

1. **nstages=1 with raw cp_async:** Copy raw TBQ4 blocks (66 bytes) via cp_async to staging buffer → sync ALU dequant from staging → target 55-65 tok/s
2. **nstages=2 double-buffer:** Full K/V preload + double-buffer → target 65-75 tok/s
3. **Column-group tile loader** (implemented, uncommitted): 100% thread utilization vs 25% → marginal improvement alone, but compounds with async loading
4. **Pre-computed centroid lookup tables:** Shift rotation to quantization time (spiritbuun's InnerQ approach) → eliminates Q rotation kernel overhead

### Bugs Fixed (6 total, ~10 hour session)

| # | Bug | Root Cause | Fix | Commit |
|---|-----|-----------|-----|--------|
| 1 | ncols2=1 dead dispatch | ncols<32 dispatched to Volta-only MMA code | Turing-only guard `ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING` | f35f29bcb |
| 2 | V-side D=256 pass-through | V fused path only checked `per_head_dim == 128` | Added `per_head_dim == 256` to condition | fe1e6d621 |
| 3 | nvcc constexpr dead-branch codegen | `if constexpr (nstages > 1)` generates bad code even when nstages=0 in TBQ4 template | `#if !defined(TBQ4_KV_FUSED)` removes blocks from TBQ4 compilation AST entirely | f35f29bcb |
| 4 | Q rotation register spill | Inlining FWHT rotation into FA kernel corrupts register allocation | Separate `k_tbq4_rotate_input` micro-kernel (128 threads, warp-shuffle FWHT) | f35f29bcb |
| 5 | CUDA graph capture rejects debug code | `cudaDeviceSynchronize`/`printf` not allowed during graph capture | Removed all debug synchronization and printfs | f35f29bcb |
| 6 | Mask null pointer dereference | Mask loader guard `ncols2 > 1` always true, enters with null mask_h | Added `mask_h &&` to guard condition | f35f29bcb |

### Files Changed (v1)
```
ggml/src/ggml-cuda/fattn-mma-f16.cuh               | 30 +++++++---
ggml/src/ggml-cuda/fattn-mma-tbq4.cuh              | 69 +++++++++++++++++++---
ggml/src/ggml-cuda/fattn.cu                        |  9 ++-
.../fattn-mma-tbq4-instance-ncols2_1.cu            |  1 +
.../fattn-mma-tbq4-instance-ncols2_2.cu            |  1 +
.../fattn-mma-tbq4-instance-ncols2_4.cu            |  1 +
.../fattn-mma-tbq4-instance-ncols2_8.cu            |  1 +
7 files, +94/-18 lines
```

### Key Technical Decisions

**Why #if instead of if constexpr guard:** nvcc processes dead branches in `if constexpr` for code generation purposes. The `flash_attn_ext_f16_load_tile` template instantiation inside the dead `if constexpr (nstages > 1)` block generates instructions that corrupt the parent kernel's register allocation, causing `cudaErrorMisalignedAddress`. The `#if !defined(TBQ4_KV_FUSED)` preprocessor approach completely removes the blocks from the AST before the compiler sees them.

**Why separate Q rotation kernel:** The `tbq4_rotate_Q_tile` function (128-point FWHT with warp shuffles and `__shared__` buffer), even as `__noinline__`, destabilizes register allocation in the parent FA kernel. Moving Q rotation to a completely separate kernel launch (`k_tbq4_rotate_input`) isolates the register usage and eliminates the spill alignment issue. Overhead is ~100μs per attention call — negligible.

**spiritbuun fork analysis:** The `spiritbuun/buun-llama-cpp` fork uses the same architecture (separate WHF kernel, nstages=0 for turbo, piggyback on F16 FA). It does NOT support MTP. Our v1 is ahead on MTP integration. Their InnerQ per-channel equalization and codebook extraction tools are potential v2 features.

### Working Tree (uncommitted)
- `fattn-mma-tbq4.cuh`: Optimized column-group tile loader (100% thread utilization vs 25%). Marginal standalone improvement, but compounds with nstages=1 async loading.

### Test Commands
```bash
# Build
cd /home/mal/AI/llama.cpp-mtp && cmake --build build -j$(nproc) --target llama-server

# Start server with MTP + TBQ4 (short context for quick test)
build/bin/llama-server \
    -m "/media/Crucial1TB/models/Qwen3.6-Heretic-MTP/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q4_K_M.gguf" \
    --port 8096 -c 4096 --flash-attn on -ngl 99 --parallel 1 \
    -b 256 -ub 32 -ctk tbq4_0 -ctv tbq4_0 \
    --spec-type mtp --spec-draft-n-max 3

# Quick inference test
curl -s http://localhost:8096/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 10, "temperature": 0}'

# Check MTP stats
grep "draft acceptance" /tmp/tbq4-*.log

# Full wrapper (135K context, MTP, TBQ4)
qwen3.6-dense-mtp-quetza --QKV=tbq4 --MTP=3

# Q4_0 baseline for comparison
qwen3.6-dense-mtp-quetza --QKV=q4 --MTP=3
```

### Collaborators
- `quetzacodetl` — Code analysis, relay coordination, fix design, testing
- `quetzacodetl-2` — Build/test on GPU, bisect debugging, commit/push
- `mal` — Code review, q_fwht_buf dynamic shmem fix, register spill analysis

### Next Session Priorities
1. Implement nstages=1 cp_async raw TBQ4 block loading (target 55-65 tok/s)
2. Benchmark at 200K context to verify VRAM usage
3. Profile to identify next bottleneck after nstages=1
4. Spiritbuun InnerQ integration evaluation
