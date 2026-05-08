# HANDOFF PROMPT — New Context Window Quick-Start

You are taking over a TBQ4_0 fused flash attention project in `/home/mal/AI/llama.cpp-mtp`. Here's everything you need to know in one shot:

## What We're Doing
Fusing TBQ4_0 KV cache dequant into CUDA flash attention to eliminate ~40% overhead. Target: 90+ tok/s at 200K context on Qwen3.6-27B with MTP on RTX 4090 24GB.

## Current Status (CRITICAL — 2026-05-08 Session 2)

**Build:** CLEAN — zero errors, zero warnings
**GPU dequant path:** WORKING at 200K (85 t/s @ 256, 66 t/s @ 512) on 27B
**Fused kernel:** ACTIVATES with correct `<256,256,4,8>` template — but CRASHES with "misaligned address"
**7B model:** TBQ4 crashes (alignment issue nb1=264, deferred — not target)

## Four Bugs Found (3 Fixed, 1 Ongoing)

| # | File | Bug | Status |
|---|------|-----|--------|
| 1 | llama-graph.cpp:1952 | Cast TBQ4→F16 before FA (killed fused path) | ✅ FIXED |
| 2 | fattn-mma-f16.cuh:572,910 | K/V stride in half2 vs bytes (stride fix) | ✅ FIXED |
| 3 | fattn.cu dispatch | ncols2=1 dispatch to dead Volta MMA code | ✅ FIXED |
| 4 | llama-graph.cpp:1987 | V-side only allowed D=128 pass-through (no D=256) | ✅ FIXED |

## THE BLOCKER: "misaligned address" crash in fused kernel (BUG 5)

**Symptom:** Kernel compiles, activates `<256,256,4,8>`, printf at entry fires, then crashes inside kernel body.
**Config:** DKQ=256, DV=256, ncols1=4, ncols2=8, ncols=32, nthreads=128, nbatch_fa=32, nbatch_K2=128, nbatch_V2=128, Q_in_reg=true, nstages=0, shmem=33792

**What's been ruled out:**
- ❌ Q rotation (disabled → still crashes)
- ❌ Warmup-specific (--no-warmup → still crashes)
- ❌ Stride mismatch (already fixed)
- ❌ V-side pass-through (fixed, kernel activates)
- ❌ Wrong template instance (printf confirms <4,8>)
- ❌ Shmem overflow (33792 < 49152)
- ❌ ldmatrix alignment (all offsets verified 16B-aligned via static analysis)
- ❌ TBQ4 tile loader GMEM reads (K tensor layout verified: nb=[66,528,132,135168])

**Active investigation:** Zero-fill K tile test — if crash disappears with zeroed K tile, the TBQ4 loader's half2 output is the cause. If it still crashes, the V loader or MMA compute path has the bug.

**Next debug steps:**
1. Complete zero-fill K tile test
2. If K loader is culprit: check TBQ4 tile loader half2 value range (NaN/Inf?)
3. Try compute-sanitizer: `compute-sanitizer --tool memcheck build/bin/llama-server ...`
4. Try forcing nstages=0 with F16 KV at D=256 — if F16 also crashes, bug is in nstages=0 path, not TBQ4

## Architecture
Rotated-domain attention: K/V tile loaders do centroid*norm lookup (no FWHT). FWHT runs twice: once on Q (rotate_forward), once on output (rotate_inverse). Eliminates FWHT from inner attention loop.

## Model Info
Qwen3.6-27B: key_length=256, value_length=256, n_head=24, n_head_kv=4, gqa_ratio=6. D=256 heads confirmed.

## Build
```bash
cd /home/mal/AI/llama.cpp-mtp
git pull fork master
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc) --target llama-server
# Force recompile after header changes:
touch ggml/src/ggml-cuda/fattn-mma-f16.cuh && cmake --build build -j$(nproc) --target llama-server
```

## Test Server
```bash
fuser -k 8096/tcp
build/bin/llama-server \
  -m /media/Crucial1TB/models/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P-MTP-slim.gguf \
  --port 8096 -c 200000 --flash-attn on --mlock -t 8 --poll 0 -ngl 99 \
  --parallel 1 --spec-type mtp --spec-draft-n-max 3 -b 2048 -ub 32 \
  -ctk tbq4_0 -ctv tbq4_0 --jinja --temp 0.6 --seed 3407
```

## Peers
- **quetzacodetl** (Gemma/Claude): Running server, testing kernels, dispatching
- **quetzacodetl-2** (DeepSeek): Kernel verification, tile layout analysis, docs
- **Relay thread:** `tbq4-coordination`

## Full Docs
- `COLLAB_NOTES.md` — Complete session log with all findings
- `HANDOFF_TBQ4.md` — File-level reference with key line numbers

## GitHub
- **Fork:** `github.com/Indras-Mirror/llama.cpp-mtp` (branch `master`)
- **Remote:** `fork` → `https://github.com/Indras-Mirror/llama.cpp-mtp.git`
- **Push:** `git push fork master`
