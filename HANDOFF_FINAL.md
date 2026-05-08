# MTP + Turbo4 + Shared Tensors — Complete Handoff

**Date:** 2026-05-07
**Repo:** https://github.com/Indras-Mirror/llama.cpp-mtp
**Branch:** master

---

## What We Built

A custom llama.cpp fork combining three technologies:

1. **MTP (Multi-Token Prediction)** — PR #22673, 2-3× generation speedup via Qwen3.6's built-in draft heads
2. **CUDA TBQ4_0 KV Cache** — FWHT-based TurboQuant, 4.25bpv, lossless quality, ~3.8× compression vs FP16
3. **Tensor Sharing** — `link_shared_tensors()` API saving 682 MB GPU RAM by preventing tok_embd duplication

**Result:** 200K context, 93% draft acceptance, 22.9 GB VRAM on RTX 4090 24GB with uncensored Qwen3.6-27B.

---

## Key Files Modified (18 files)

### Tensor Sharing Infrastructure

| File | Change |
|------|--------|
| `include/llama.h` | Added `llama_model_link_shared_tensors()` public API |
| `src/llama-model.h` | Added `virtual link_shared_tensors()` + `get_tensor_mutable()` |
| `src/llama-model.cpp` | Implemented `llama_model_link_shared_tensors()` |
| `src/models/models.h` | Added override declarations to `qwen35_mtp` and `qwen35moe_mtp` |
| `src/models/qwen35_mtp.cpp` | `tok_embd` → TENSOR_NOT_REQUIRED; `link_shared_tensors()` sets it from trunk |
| `src/models/qwen35moe_mtp.cpp` | Same for MoE variant |
| `tools/server/server-context.cpp` | Calls `llama_model_link_shared_tensors()` after MTP model load |

### CUDA TBQ4_0 Kernels

| File | Change |
|------|--------|
| `ggml/src/ggml-cuda/tbq4-cuda.cuh` | **NEW** — FWHT quantize/dequant CUDA kernels (header-only) |
| `ggml/src/ggml-cuda/tbq3-cuda.cuh` | **NEW** — 3-bit FWHT CUDA kernels (bug: pack/unpack mismatch) |
| `ggml/src/ggml-cuda/set-rows.cu` | Added TBQ3_0 + TBQ4_0 dispatch |
| `ggml/src/ggml-cuda/cpy.cu` | Added TBQ3_0 + TBQ4_0 → F32 dequant (F16 path REMOVED to eliminate cudaMallocAsync) |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Added TBQ types to SET_ROWS + CPY support |

### TBQ Algorithm (CPU + CUDA)

| File | Change |
|------|--------|
| `ggml/src/ggml-common.h` | `block_tbq4_0`: 128-element FWHT format; `block_tbq3_0`: same |
| `ggml/src/ggml.c` | Type traits: QK_TBQ4=128, QK_TBQ3=128 |
| `ggml/src/ggml-turboq.c` | FWHT quantize/dequant for TBQ4_0 + TBQ3_0 (CPU reference) |
| `ggml/src/ggml-turboq-tables.h` | FWHT sign arrays (seed=42), Lloyd-Max centroids for 4-bit + 3-bit |

---

## Stable Configuration (qwen3.6-dense-mtp-quetza v5.3)

```
Model:   Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P-MTP-slim.gguf (15.8 GB)
KV:      tbq4_0 (lossless, 4.25bpv)
Context: 200,000 tokens
GPU:     ngl=99 (all layers), ub=16, b=2048
Speed:   93% draft acceptance, no-OOM stable
VRAM:    22.9 GB
```

### VRAM Budget at 200K

| Component | Size |
|-----------|------|
| Model weights (slim GGUF, Q4_K) | 15,108 MiB |
| token_embd.weight (CPU_Mapped) | 682 MiB |
| KV cache (TBQ4, 16 attn layers) | 3,226 MiB |
| Recurrent state (Mamba, 48 layers) | 599 MiB |
| MTP head model (blk.64 Q6_K) | 1,014 MiB |
| MTP KV cache (1 layer) | 202 MiB |
| Compute buffers (CUDA) | ~1,100 MiB |
| **Total GPU** | **~22.9 GB** |

---

## Known Issues

### 1. CPU Leakage (~600% CPU at idle)
**Cause:** `token_embd.weight` (682 MB, Q4_K) is on CPU_Mapped — every forward pass accesses it via CPU. The `-t 20` flag creates 20 threads that spin doing CPU_Mapped access + recurrent state checkpoint management.
**Mitigation:** Reduce `-t` to 8. Also consider removing `--no-mmap` flag (allows GPU direct I/O for the mapped tensor).
**Long-term fix:** Squeeze VRAM to move token_embd to GPU. Requires another ~500 MB savings.

### 2. Prompt Processing Speed
**Cause:** Qwen3.6's hybrid architecture (Gated DeltaNet) requires full prompt reprocessing — can't use partial KV cache resumption. The `memory_seq_rm` call after every batch and checkpoint creation (149 MB each) add overhead.
**Mitigation:** Smaller batch size `-b 2048` reduces checkpoint overhead. The `--no-mmap` flag may also slow things down.

### 3. TBQ3_0 CUDA Kernel Bug
The 3-bit FWHT kernels produce `/////` garbage output — the pack/unpack bit layout doesn't round-trip correctly between CPU quantize and CUDA dequant. Needs debugging.

### 4. 262K Context OOM
Even with all optimizations, 262K doesn't fit with MTP at ngl=99 on 24GB. The compute buffer scales with context and hits the limit. Options: ngl=97 (saves ~240 MB), smaller ubatch (16→8), or wait for upstream PR merges.

---

## Wrapper Modes

```bash
# Default: MTP + TBQ4_0 + slim GGUF + shared tok_embd (200K, 93% accept)
qwen3.6-dense-mtp-quetza

# MTP + TBQ3_0 (200K, ~5.2x comp, BROKEN — kernel bug)
qwen3.6-dense-mtp-quetza --QKV=tbq3

# MTP + Q4_0 KV (200K, simpler KV compression)
qwen3.6-dense-mtp-quetza --QKV=q4

# dflash + turbo4 (300K, no MTP, 41 tok/s)
qwen3.6-dense-mtp-quetza --QKV=turbo4
```

---

## Build

```bash
cd /home/mal/AI/llama.cpp-mtp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc) --target llama-server
```

## GGUFs

| File | Size | Description |
|------|------|-------------|
| `Q4_K_P-MTP.gguf` | 17 GB | Original Q4_K_P + Q8_0 MTP heads (untouched) |
| `Q4_K_P-MTP-slim.gguf` | 15.8 GB | Q4_K model + IQ4_NL output + Q6_K MTP heads (default) |

## Blog

https://indrasmirror.au/blog-mtp-shared-tensors-200k.html

## Fork

https://github.com/Indras-Mirror/llama.cpp-mtp
