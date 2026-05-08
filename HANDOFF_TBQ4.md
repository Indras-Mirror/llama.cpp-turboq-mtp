# HANDOFF: TBQ4_0 Fused Flash Attention — Final State

**Date:** 2026-05-08
**Repo:** `/home/mal/AI/llama.cpp-mtp` (fork: github.com/Indras-Mirror/llama.cpp-mtp)
**Branch:** master
**Session:** Gemma + Claude + DeepSeek collaboration

---

## What Works

### D=256 GPU-side Dequant (Production Path)

The fused kernel doesn't work for D=256 yet (tile layout issue). Instead, TBQ4→F32→F16 runs entirely on GPU:

**200K Context Benchmarks:**
| Tokens | Speed | Accept | VRAM |
|--------|-------|--------|------|
| 256 | 85.0 t/s | 75.2% | 23.6 GB |
| 512 | 66.4 t/s | 56.8% | 23.6 GB |
| 1024 | 63.4 t/s | 64.0% | 23.6 GB |

This beats the original CPU-side dequant (57 t/s at 135K) by keeping dequant on GPU.

### Fused MMA Kernel (D=128 ready, untested)

All pieces for the D=128 fused TBQ4 flash attention kernel are implemented:
- Q rotation (fattn-mma-tbq4.cuh): Warp-shuffle FWHT, per-Q-column
- Output rotation (fattn-mma-tbq4.cuh): Inverse FWHT on VKQ output
- K/V tile loader (fattn-mma-tbq4.cuh): Centroid*norm → half2 shmem
- MMA dispatch (fattn.cu): Kernel selection + launch
- Template instances: 4× ncols2 files for DKQ=128
- Graph fix (llama-graph.cpp): Conditional TBQ4 pass-through

Need a D=128 model to validate (e.g., Qwen2.5-7B, Mistral-7B).

## Key Fix Applied

### llama-graph.cpp:1952 — One-line root cause

**Bug:** `tbq_attn_type = use_flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32`
This cast TBQ4→F16 before flash_attn, causing:
1. Fused kernel never activated (K->type was F16, not TBQ4_0)
2. Dequant fell to CPU (no CUDA TBQ4→F16 path) → PCIe bottleneck + CPU heap leak

**Fix:** Conditional cast — skip when `per_head_dim == 128` (fused kernel handles raw TBQ4 natively), else TBQ4→F32→F16 on GPU.

## D=256 Fused Kernel — Blocked

Extended to D=256 but hits tile layout mismatch:
- TBQ4 block produces 64 half2 entries
- Ampere MMA config uses stride_tile=32 half2/tile-row
- Each block spans 2 tile rows, but ldmatrix expects 1:1 mapping
- Fix needs: multi-row block distribution tile loader

## Build & Test Commands

```bash
cd /home/mal/AI/llama.cpp-mtp
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc) --target llama-server

# 200K context server
build/bin/llama-server \
  -m /media/Crucial1TB/models/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P-MTP-slim.gguf \
  --port 8096 -c 200000 --flash-attn on --mlock -t 8 --poll 0 -ngl 99 \
  --parallel 1 --spec-type mtp --spec-draft-n-max 3 -b 2048 -ub 32 \
  -ctk tbq4_0 -ctv tbq4_0 --jinja --temp 0.6 --seed 3407

# Benchmark
curl -s http://localhost:8096/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"..."}],"max_tokens":512,"temperature":0,"seed":42}'
```

## Files Modified (18 files)

### Core Fix
- `src/llama-graph.cpp` — Conditional TBQ4 cast

### Fused Kernel
- `ggml/src/ggml-cuda/fattn-mma-tbq4.cuh` — Tile loader + rotation kernels
- `ggml/src/ggml-cuda/fattn-mma-tbq4-launch.cuh` — MMA launcher
- `ggml/src/ggml-cuda/fattn-mma-f16.cuh` — TBQ4 type hooks
- `ggml/src/ggml-cuda/fattn.cu` — Dispatch + kernel selection
- `ggml/src/ggml-cuda/template-instances/fattn-mma-tbq4-instance-ncols2_*.cu`

### CUDA Kernels
- `ggml/src/ggml-cuda/tbq4-cuda.cuh` — FWHT + Sparse V
- `ggml/src/ggml-cuda/tbq3-cuda.cuh` — 3-bit variant
- `ggml/src/ggml-cuda/tbq4-sparse-v.cuh` — Sparse V utility
- `ggml/src/ggml-cuda/cpy.cu` — Dequant dispatch
- `ggml/src/ggml-cuda/set-rows.cu` — Quantize dispatch
- `ggml/src/ggml-cuda/ggml-cuda.cu` — Type support checks

### Core Types
- `ggml/src/ggml-common.h` — Block struct definitions
- `ggml/src/ggml.c` — Type traits
- `ggml/src/ggml-turboq.c` — CPU reference
- `ggml/src/ggml-turboq-tables.h` — FWHT sign arrays, centroids

## Next Steps

1. **Download D=128 model** (Qwen2.5-7B) — validate fused kernel speedup
2. **Fix D=256 tile layout** — multi-row block distribution for ldmatrix
3. **Integrate D2 sparse-V** — attention-aware V dequant skip
4. **MTP acceptance rate** — investigate degradation at long context

## Coordination

See `COLLAB_NOTES.md` for detailed session-by-session notes.
