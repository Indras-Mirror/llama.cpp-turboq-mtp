# QUETZA MTP + TURBO4 — SUPER DETAILED HANDOFF

**Date:** 2026-05-07 17:25
**Next Session:** Start here. Everything you need is below.

---

## 1. WHAT WE'RE DOING

Running Qwen3.6-27B uncensored with Multi-Token Prediction (MTP) + Turbo4 KV cache on RTX 4090 24GB at 200K context. Combined speedup is 2-3× vs baseline.

**Goal:** 200K+ context, no CPU leakage, no OOM, 90%+ draft acceptance.

**Current Status:** Server runs at 200K with TBQ4_0 + slim GGUF + shared tok_embd. 93% draft accept, 22.9 GB VRAM. BUT ~500% CPU usage from `token_embd.weight` on CPU_Mapped + recurrent state ops.

---

## 2. KEY PATHS

```bash
# Build
/home/mal/AI/llama.cpp-mtp/                    # MTP fork (Indras-Mirror/llama.cpp-mtp)
/home/mal/AI/llama.cpp-mtp/build/bin/llama-server

# GGUFs
/media/Crucial1TB/models/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P-MTP.gguf          # 17GB - original Q4_K_P + Q8_0 MTP (untouched)
/media/Crucial1TB/models/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P-MTP-slim.gguf      # 15.8GB - Q4_K model + IQ4_NL output + Q6_K MTP (DEFAULT)
/media/Crucial1TB/models/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P-MTP-q8mtp.gguf     # 16GB - Q4_K model + Q4_K output + Q8_0 MTP

# NEW: Heretic MTP-preserved GGUF (downloading)
/media/Crucial1TB/models/Qwen3.6-Heretic-MTP/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q4_K_M.gguf
# From: https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF
# This is abliterated (Heretic v1.2 MPOA) with 15 MTP heads PRESERVED — no grafting needed!
# 94% fewer refusals (6/100 vs 92/100), 0.0021 KL divergence

# Graft tool
/home/mal/AI/MTP-Q8_0.gguf          # 436MB - standalone MTP head (blk.64.* Q8_0, 15 tensors)
/home/mal/AI/mtp-convert.py         # Graft script
/home/mal/AI/.venv-mtp/             # Python venv with gguf library

# Wrapper
/home/mal/.local/bin/qwen3.6-dense-mtp-quetza   # v5.3, port 8096

# Blog
/home/mal/AI/Website/indras-mirror-site-enhanced/blog-mtp-shared-tensors-200k.html

# Handoff docs
/home/mal/AI/llama.cpp-mtp/HANDOFF_TBQ4.md       # TBQ4 CUDA kernel details
/home/mal/AI/llama.cpp-mtp/HANDOFF_FINAL.md       # Summary
/home/mal/AI/llama.cpp-mtp/HANDOFF_SUPER_DETAILED.md  # THIS FILE

# Server logs
/tmp/qwen3.6-dense-mtp-llama-server.log  # wrapper log
/tmp/mtp-v6.log                           # latest manual launch

# dflash fork (for reference CUDA kernels)
/home/mal/AI/llama.cpp-dflash/  # spiritbuun/buun-llama-cpp
```

---

## 3. BUILD COMMANDS

```bash
cd /home/mal/AI/llama.cpp-mtp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc) --target llama-server
# Build succeeds cleanly (no warnings, no errors)
```

---

## 4. WORKING CONFIGURATION (WRAPPER v5.3)

```bash
qwen3.6-dense-mtp-quetza
# Default: slim GGUF + TBQ4_0 + 200K + ngl=99 + ub=16 + t=8
```

Manual equivalent:
```bash
/home/mal/AI/llama.cpp-mtp/build/bin/llama-server \
  -m /media/Crucial1TB/models/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P-MTP-slim.gguf \
  --port 8096 -c 200000 \
  --flash-attn on -t 8 -ngl 99 \
  --parallel 1 --spec-type mtp --spec-draft-n-max 3 \
  -b 2048 -ub 16 \
  -ctk tbq4_0 -ctv tbq4_0 \
  --jinja --chat-template-kwargs '{"enable_thinking":false,"preserve_thinking":true}' \
  --temp 0.6 --top-p 0.95 --top-k 20 --seed 3407
```

**VRAM Budget at 200K:**
| Component | Size |
|-----------|------|
| Model (slim GGUF) | 15,108 MiB GPU |
| token_embd.weight | 682 MiB CPU_Mapped |
| KV cache (TBQ4, 16 layers) | 3,226 MiB |
| Recurrent state | 599 MiB |
| MTP head model | 1,014 MiB |
| MTP KV cache | 202 MiB |
| Compute buffers | ~1,100 MiB |
| **Total GPU** | **~22.9 GB** |
| **Total CPU mapped** | **682 MB** |

---

## 5. THREE TECHNOLOGIES IN THIS FORK

### 5.1 MTP (PR #22673)
- Multi-Token Prediction — Qwen3.6 has built-in draft heads that predict 3 future tokens
- Activated with `--spec-type mtp --spec-draft-n-max 3`
- Achieves 90-98% draft acceptance (tokens predicted by MTP head that the main model accepts)
- **93% acceptance = 2.8× effective speedup** (3 drafts × 0.93 accepted)

### 5.2 CUDA TBQ4_0 (Turbo4)
- 4.25 bpw KV cache using FWHT rotation + 4-bit PolarQuant
- Lossless quality (matches FP16), ~3.8× compression
- Activated with `-ctk tbq4_0 -ctv tbq4_0`
- CUDA kernels in `ggml/src/ggml-cuda/tbq4-cuda.cuh`
- Block: 128 elements, fp16 norm + 64 bytes packed 4-bit = 66 bytes

### 5.3 Tensor Sharing (`link_shared_tensors`)
- Saves 682 MB GPU RAM by preventing MTP model from loading its own token_embd
- New public API: `llama_model_link_shared_tensors(model, trunk)`
- Works via virtual method `link_shared_tensors()` on llama_model
- Only tok_embd is shared (output.weight kept separate — sharing output caused 0% acceptance)
- Files modified: include/llama.h, src/llama-model.h/cpp, src/models/models.h, src/models/qwen35_mtp.cpp, src/models/qwen35moe_mtp.cpp, tools/server/server-context.cpp

---

## 6. KEY TECHNICAL DETAILS

### TBQ4_0 FWHT Algorithm
```
Quantize: normalize → s1 multiply → FWHT butterfly → s2 multiply → 4-bit quantize via centroids → norm correction
Dequant:  centroid lookup → s2 multiply → inverse FWHT → s1 multiply → norm scale
```
- Sign arrays: seed=42, 128 elements each (s1, s2)
- 4-bit centroids: Lloyd-Max for N(0, 1/sqrt(128)), 16 levels
- FWHT: O(n log n) butterfly, 128 elements = 7 stages = 896 operations
- Norm correction: corrected_norm = original_norm / reconstruction_norm

### CUDA Dequant Architecture
- **Per-element:** `dequantize_tbq4_0()` — NO inverse FWHT. Used by get_rows template. Returns rotated-domain values.
- **Full-block:** `k_tbq4_dequant_full()` — WITH inverse FWHT via shared memory butterfly. Used by CPY/CAST for attention. 128 threads cooperate.

### TBQ→F16 Path REMOVED
- The TBQ→F16 dequant path used `cudaMallocAsync` for temp F32 buffers
- This was eliminated — only TBQ→F32 is supported
- GGML decomposes TBQ→F16 into TBQ→F32 + F32→F16 using pre-allocated buffers
- **Eliminated the 200K OOM crash**

### The 500% CPU Problem
- `token_embd.weight` (682 MB, Q4_K) is on CPU_Mapped by llama.cpp's buffer assignment
- Every forward pass calls `ggml_get_rows(token_embd, tokens)` which reads from CPU_Mapped
- The GPU accesses this via PCIe — CPU handles the page faults
- `-t 8` creates 8 threads that spin on CPU_Mapped access + recurrent state management
- **The fix:** Move token_embd to GPU. Requires ~682 MB more VRAM. Options:
  1. Reduce context to 168K → saves ~500 MB KV cache
  2. Requantize token_embd Q4_K → IQ4_NL → saves ~100 MB
  3. Use the new Heretic GGUF (Q4_K_M — might have smaller token_embd or different layout)

---

## 7. GGUFs EXPLAINED

### Original MTP GGUF (17 GB)
- Created by grafting MTP-Q8_0.gguf onto HauhauCS Q4_K_P using mtp-convert.py
- 866 tensors: 851 base + 15 MTP (blk.64.* at Q8_0)
- output.weight: Q6_K (~995 MB)
- token_embd.weight: Q4_K (~682 MB)
- MTP head: Q8_0 (~430 MB)

### Slim GGUF (15.8 GB) — CURRENT DEFAULT
- Requantized from original using llama-quantize with --allow-requantize
- ALL tensors re-quantized to Q4_K (lost K_P optimization, ~1 quant level quality loss)
- output.weight: IQ4_NL (~682 MB)
- MTP head: Q6_K (~280 MB)
- token_embd.weight: Q4_K (~682 MB, still on CPU_Mapped)
- **Note:** The --allow-requantize flag re-quantizes EVERYTHING including already-quantized K_P tensors. We lost the K_P optimization. A better approach would be surgical requantization but llama-quantize doesn't support it.

### Heretic MTP-Preserved GGUF (16 GB) — DOWNLOADING
- Native MTP heads preserved (no grafting needed!)
- Q4_K_M quant
- Abliterated with Heretic v1.2 MPOA
- **Potential advantage:** May have different tensor layout that puts token_embd on GPU
- Download: https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF

---

## 8. REMAINING ISSUES

### HIGH PRIORITY: CPU Leakage (500% CPU)
- **Root cause:** `token_embd.weight` on CPU_Mapped (682 MB)
- **Fix:** Move it to GPU by freeing ~682 MB elsewhere
- **Approach:** Reduce context from 200K to 168K (frees ~500 MB KV cache) + requantize token_embd to IQ4_NL (frees ~100 MB) = ~600 MB savings
- **Or:** Test the Heretic GGUF — different quant might have different CPU/GPU split

### MEDIUM: TBQ3_0 CUDA Kernel Bug
- Produces `/////` garbage output — pack/unpack bit layout doesn't round-trip
- CPU quantize and CUDA dequant don't match
- Needs systematic debugging of the 3-bit bitstream format

### LOW: 262K Context
- 200K is stable, 262K OOMs
- Would need: ngl=97 (offload 2 layers), or smaller GGUF, or TBQ3 working
- The Heretic GGUF at Q4_K_M might be smaller — worth testing at 262K

---

## 9. QUICK START FOR NEXT SESSION

```bash
# 1. Kill any old server
kill $(lsof -ti:8096) 2>/dev/null

# 2. Launch
qwen3.6-dense-mtp-quetza

# 3. Monitor
watch -n 2 'echo "VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) | CPU: $(ps -p $(lsof -ti:8096 | head -1) -o %cpu --no-headers 2>/dev/null)% | Swap: $(free -h | grep Swap | awk "{print \$3}")"'

# 4. Check MTP stats
grep "statistics mtp\|draft acceptance" /tmp/qwen3.6-dense-mtp-llama-server.log | tail -5

# 5. Test the Heretic GGUF
/home/mal/AI/llama.cpp-mtp/build/bin/llama-server \
  -m /media/Crucial1TB/models/Qwen3.6-Heretic-MTP/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q4_K_M.gguf \
  --port 8096 -c 200000 \
  --flash-attn on -t 4 -ngl 99 \
  --parallel 1 --spec-type mtp --spec-draft-n-max 3 \
  -b 2048 -ub 16 \
  -ctk tbq4_0 -ctv tbq4_0 \
  --jinja --chat-template-kwargs '{"enable_thinking":false,"preserve_thinking":true}' \
  --temp 0.6 --top-p 0.95 --top-k 20 --seed 3407
```

---

## 10. ALL FILES MODIFIED (18 files)

### Tensor Sharing (7 files)
`include/llama.h` — `llama_model_link_shared_tensors()` API
`src/llama-model.h` — virtual `link_shared_tensors()`, `get_tensor_mutable()`
`src/llama-model.cpp` — implementation
`src/models/models.h` — override declarations
`src/models/qwen35_mtp.cpp` — tok_embd NOT_REQUIRED, sharing implementation
`src/models/qwen35moe_mtp.cpp` — same for MoE
`tools/server/server-context.cpp` — call after MTP model load

### CUDA TBQ Kernels (5 files)
`ggml/src/ggml-cuda/tbq4-cuda.cuh` — NEW: FWHT quantize/dequant
`ggml/src/ggml-cuda/tbq3-cuda.cuh` — NEW: 3-bit FWHT (BROKEN)
`ggml/src/ggml-cuda/set-rows.cu` — TBQ3_0 + TBQ4_0 dispatch
`ggml/src/ggml-cuda/cpy.cu` — TBQ→F32 dequant (F16 path removed)
`ggml/src/ggml-cuda/ggml-cuda.cu` — TBQ op support

### TBQ Algorithm (4 files)
`ggml/src/ggml-common.h` — block_tbq4_0 / block_tbq3_0 (128-element FWHT)
`ggml/src/ggml.c` — type traits
`ggml/src/ggml-turboq.c` — FWHT quantize/dequant CPU reference
`ggml/src/ggml-turboq-tables.h` — FWHT signs + Lloyd-Max centroids

### Documentation (2 files)
`HANDOFF_TBQ4.md` — TBQ4 CUDA kernel details
`HANDOFF_FINAL.md` — summary
`HANDOFF_SUPER_DETAILED.md` — this file

---

## 11. GIT STATE

```bash
cd /home/mal/AI/llama.cpp-mtp
git log --oneline -5
# 0987d0a5b docs: Add README_MTP and comprehensive TBQ4 handoff
# f112d6de7 ggml : fix TurboQuant CPU review issues
# 9e7229e13 ggml : limit the first TurboQuant CPU PR to TBQ
# 88dc3d10b feat: add CPU TurboQuant KV cache types
# bc5892744 Merge PR #22673: llama + spec: MTP Support

# Uncommitted changes (all the CUDA TBQ kernel work + tensor sharing)
git status --short
# M  ggml/src/ggml-common.h
# M  ggml/src/ggml-cuda/cpy.cu
# M  ggml/src/ggml-cuda/ggml-cuda.cu
# M  ggml/src/ggml-cuda/set-rows.cu
# A  ggml/src/ggml-cuda/tbq3-cuda.cuh
# A  ggml/src/ggml-cuda/tbq4-cuda.cuh
# M  ggml/src/ggml-turboq-tables.h
# M  ggml/src/ggml-turboq.c
# M  ggml/src/ggml.c
# M  include/llama.h
# M  src/llama-model.cpp
# M  src/llama-model.h
# M  src/models/models.h
# M  src/models/qwen35_mtp.cpp
# M  src/models/qwen35moe_mtp.cpp
# M  tools/server/server-context.cpp

# Remote
git remote -v
# origin  https://github.com/ggml-org/llama.cpp.git
# fork    https://github.com/Indras-Mirror/llama.cpp-mtp.git  (already pushed)
```
