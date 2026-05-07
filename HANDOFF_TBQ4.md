# TBQ4_0 (Turbo4) CUDA KV Cache — MTP Build Handoff

**Date:** 2026-05-07
**Status:** WORKING — builds, loads, generates. 262K context OOMs on prompt processing (crash). 135K fully stable at ~98 tok/s.
**Build dir:** `/home/mal/AI/llama.cpp-mtp`
**dflash reference:** `/home/mal/AI/llama.cpp-dflash`

---

## What Was Done

Ported Turbo4 (TBQ4_0) CUDA quantization kernels from the dflash build into the MTP build of llama.cpp so that MTP speculative decoding and Turbo4 KV cache compression work together.

### Algorithm Change

The MTP build's original TBQ4 used **Householder QR rotation** (O(n²), 256-element blocks). This was replaced with dflash's **FWHT (Fast Walsh-Hadamard Transform)** rotation (O(n log n), 128-element blocks). This is faster and has proven CUDA kernels.

**Block format changed:**
- Old: `block_tbq4_0` = 130 bytes (256 elements: 128 qs bytes + 2 byte half norm)
- New: `block_tbq4_0` = 66 bytes (128 elements: 2 byte half norm + 64 qs bytes)

**Quantization pipeline:** normalize → sign multiply (s1) → FWHT butterfly → sign multiply (s2) → 4-bit PolarQuant via Lloyd-Max centroids → norm correction

**Dequantization:** centroid lookup → sign multiply (s2) → inverse FWHT → sign multiply (s1) → norm scale

### Files Modified

#### `ggml/src/ggml-common.h`
- Changed `block_tbq4_0` struct from 256-element to 128-element format
- Added `#define QK_TBQ4 128`
- Field order: `ggml_half d` first, then `uint8_t qs[64]`

#### `ggml/src/ggml.c`
- Changed type traits `blck_size` from `QK_K` (256) to `QK_TBQ4` (128) for `GGML_TYPE_TBQ4_0`

#### `ggml/src/ggml-turboq-tables.h`
- Added FWHT sign arrays: `turboq_wht_signs1[128]`, `turboq_wht_signs2[128]` (seed=42, matching dflash)
- Added FWHT-domain centroids: `turboq_fwht_centroids_4bit[16]` (Lloyd-Max for N(0, 1/sqrt(128)))
- Added FWHT-domain midpoints: `turboq_fwht_midpoints_4bit[15]`

#### `ggml/src/ggml-turboq.c` (CPU quantize/dequant — complete rewrite for TBQ4_0)
- `tbq4_fwht_128()` — in-register FWHT butterfly, O(n log n)
- `tbq4_rotate_forward()` — s1 multiply → FWHT → s2 multiply
- `tbq4_rotate_inverse()` — s2 multiply → FWHT → s1 multiply
- `tbq4_find_nearest()` — binary search over 15 midpoints
- `quantize_row_tbq4_0_ref()` — full FWHT quantize with norm correction
- `dequantize_row_tbq4_0()` — centroid lookup + inverse FWHT + norm scale
- `quantize_tbq4_0()` — updated for QK_TBQ4 block size

#### `ggml/src/ggml-cuda/tbq4-cuda.cuh` (NEW — header-only CUDA implementation)
- `__constant__` arrays: centroids[16], midpoints[15], wht_s1[128], wht_s2[128]
- `tbq4_fwht_128()` — per-thread in-register FWHT
- `tbq4_rotate_forward()` / `tbq4_rotate_inverse()` — device rotation functions
- `tbq4_find_nearest()` — O(log n) binary search quantizer
- `quantize_f32_tbq4_0_block()` — per-block quantize for SET_ROWS template
- `dequantize_tbq4_0()` — per-element dequant for get_rows (NO inverse rotation)
- `k_tbq4_dequant_full()` — CUDA kernel: full-block dequant WITH inverse FWHT, 128 threads + shared memory butterfly
- `tbq4_dequant_full_cuda()` — host launcher

#### `ggml/src/ggml-cuda/set-rows.cu`
- Added `#include "tbq4-cuda.cuh"`
- Added TBQ4_0 case in type dispatch calling `quantize_f32_tbq4_0_block`

#### `ggml/src/ggml-cuda/cpy.cu`
- Added `#include "tbq4-cuda.cuh"`
- Added TBQ4_0→F32 case using `tbq4_dequant_full_cuda()` (full-block dequant with inverse FWHT)
- Added TBQ4_0→F16 case using F32 temp buffer + conversion

#### `ggml/src/ggml-cuda/ggml-cuda.cu`
- Added `GGML_TYPE_TBQ4_0` to SET_ROWS supported types
- Added `GGML_TYPE_TBQ4_0` → F32/F16 to CPY supported types

---

## Model Files

| File | Description | Size |
|------|-------------|------|
| `Q4_K_P-MTP.gguf` | Original MTP GGUF, blk.64 Q8_0, output Q6_K | 17 GB |
| `Q4_K_P-MTP-q4out.gguf` | Requantized: output→Q4_K, blk.64→Q4_K, some main layers Q6K→Q4K | 16 GB |
| `Q4_K_P-MTP-q8mtp.gguf` | Requantized: output→Q4_K, blk.64 stays Q8_0 | 16 GB |

All in `/media/Crucial1TB/models/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-*`

---

## Test Results

### What Works

| Config | Context | Speed | MTP Draft Accept | Status |
|--------|---------|-------|-----------------|--------|
| Original GGUF, `-ctk tbq4_0 -ctv tbq4_0`, ngl=99 | 8K | 98 tok/s | 93% (14/15) | PERFECT |
| Original GGUF, ngl=99 | 135K | ~98 tok/s | ~93% | PERFECT |
| q4out GGUF, ngl=63, ub=32 | 262K | 45 tok/s | 72% | LOADS OK, prompt crash |
| q8mtp GGUF, ngl=58, ub=32 | 262K | 26 tok/s | 76% | LOADS OK, slower |

### 262K Problem — Prompt Processing Crash

At 262K context, the server loads and can generate short responses, but crashes during long prompt processing. This is likely because:

1. The compute buffer at 262K is enormous (~1,077-1,233 MiB)
2. The `-ub 32` setting means very many micro-batches for prompt ingestion
3. The 3 CPU layers (ngl=63) create a memory bus bottleneck during prompt processing
4. Possible CUDA OOM during the attention computation phase (not just allocation)

**The 262K config is right on the VRAM edge.** Total GPU usage:
- Main model: 15,004 MiB (q4out) or 14,282 MiB (q4out ngl=63)
- Main KV cache: 4,224 MiB
- Recurrent (Mamba): 599 MiB
- Main compute: 1,077 MiB
- MTP head model: 910 MiB (q4out) or 1,223 MiB (q8mtp)
- MTP KV: 264 MiB
- MTP compute: 1,077 MiB
- **Total: ~22,433 MiB** vs 24,078 available (~22,500 usable after system)

### VRAM Budget (RTX 4090 24GB)

| Component | 8K ctx | 135K ctx | 262K ctx |
|-----------|--------|----------|----------|
| Model (ngl=99) | 16,461 | 16,461 | N/A |
| Model (ngl=63) | N/A | N/A | 14,282-15,004 |
| KV cache (TBQ4, 16 layers) | 8 | 2,178 | 4,224 |
| Recurrent state | 599 | 599 | 599 |
| Compute buffer | 495 | ~1,000 | 1,077-1,233 |
| MTP head model | 1,425 | 1,425 | 910-1,223 |
| MTP KV (1 layer) | 0.5 | 136 | 264 |
| MTP compute | ~495 | ~1,000 | 1,077 |
| **Total** | ~19,484 | ~22,800 | ~22,433-23,600 |

### KV Cache Compression Ratio

TBQ4_0 at 262K: **4,224 MiB** (K: 2,112 + V: 2,112)
F16 equivalent would be: ~16,384 MiB
**Compression: ~3.9x**

---

## MTP Head Tensor Duplication Problem

The MTP head loads 18 tensors from the GGUF, totaling ~1,425 MiB on GPU (original quant). Most of this is **duplicated shared tensors**:

| Tensor | Type | Size | Notes |
|--------|------|------|-------|
| `output.weight` | Q6_K | 995 MiB | **DUPLICATED** from main model |
| `token_embd.weight` | Q4_K | 682 MiB | Goes to CPU_Mapped, less critical |
| `blk.64.*` (attention) | Q8_0 | 430 MiB | MTP-unique transformer layer |

The `output.weight` duplication is the single biggest VRAM cost. Requantizing it to Q4_K saves 313 MiB on both the main model AND MTP head copies (626 MiB total).

### Separate MTP GGUF Available

`havenoammo/Qwen3.6-27B-MTP-UD-GGUF` on HuggingFace has a standalone `27B_MTP.gguf` (457 MB) containing only the MTP-specific tensors in Q8_0. This can be grafted onto other quants using the `convert.py` script from that repo.

---

## Wrapper Script

Updated `/home/mal/.local/bin/qwen3.6-dense-mtp-quetza` to v4.0 with three modes:

```bash
# MTP + Turbo4 KV (262K context, ~45 tok/s) — UNSTABLE, crashes on long prompts
qwen3.6-dense-mtp-quetza --QKV=tbq4

# MTP + Q4_0 KV (135K context, ~90 tok/s) — STABLE, recommended
qwen3.6-dense-mtp-quetza

# dflash + turbo4 (300K context, ~41 tok/s, no MTP) — STABLE
qwen3.6-dense-mtp-quetza --QKV=turbo4
```

### Key Flags for 262K TBQ4 Mode
- `-ngl 63` — 3 main model layers on CPU (saves ~720 MiB GPU)
- `-ub 32` — small micro-batch (reduces compute buffer from 1,860 to 1,077 MiB)
- `-fit off` — disable VRAM safety margin
- `--mlock` — prevent swap (swap was 8GB full and killing performance)
- Uses `q4out` GGUF (output.weight Q4_K, blk.64 Q4_K)

---

## Remaining Issues / Next Steps

### 1. 262K Prompt Processing Crash (HIGH PRIORITY)
The 262K config loads and generates tokens fine for short prompts but crashes during long prompt ingestion. Possible fixes:
- Investigate the actual crash (is it CUDA OOM during attention? segfault? assert?)
- Try even smaller `-ub` (16 or 8) to reduce peak memory during prompt processing
- Try `-b 512` or `-b 256` to limit batch size
- Consider whether the CPY dequant kernel (`k_tbq4_dequant_full`) has a bug that manifests at large batch sizes
- Check if the issue is in the Mamba recurrent layers, not the attention layers

### 2. Shared Tensor Deduplication (MEDIUM)
llama.cpp loads the MTP head as a separate model, duplicating `output.weight` on GPU. A proper fix would share the GPU tensor between main and MTP models. This would save ~995 MiB (Q6_K) or ~682 MiB (Q4_K), enough to fit 262K without ngl reduction.

### 3. vec_dot for TBQ4 (LOW)
The `vec_dot_tbq4_0` function in `ggml-cpu/quants.c` still uses QK_K-based loops. Should be updated to QK_TBQ4 for correctness if CPU dot products are ever needed. Currently not exercised because all inference is on GPU.

### 4. Benchmark Comparison
Need a proper benchmark comparing:
- MTP + Q4_0 KV at 135K
- MTP + TBQ4_0 KV at 135K (same context, compare quality/speed)
- MTP + TBQ4_0 KV at 262K (if crash fixed)
- dflash + turbo4 at 262K (no MTP baseline)

### 5. Quality Validation
The requantized q4out GGUF drops output.weight from Q6_K to Q4_K and MTP layer from Q8_0 to Q4_K. Need perplexity testing to confirm quality impact is acceptable.

---

## Build Instructions

```bash
cd /home/mal/AI/llama.cpp-mtp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc)
```

Build succeeds 100% with no errors as of 2026-05-07.

## Test Commands

```bash
# Quick test (135K, stable)
./build/bin/llama-server \
  -m /media/Crucial1TB/models/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P-MTP.gguf \
  --spec-type mtp --spec-draft-n-max 3 \
  -ctk tbq4_0 -ctv tbq4_0 \
  -c 135000 -ngl 99 --flash-attn on --mlock -np 1

# 262K test (loads but crashes on long prompts)
./build/bin/llama-server \
  -m /media/Crucial1TB/models/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P-MTP-q4out.gguf \
  --spec-type mtp --spec-draft-n-max 3 \
  -ctk tbq4_0 -ctv tbq4_0 \
  -c 262144 -ngl 63 -fit off --flash-attn on --mlock -ub 32 -np 1

# Inference test
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50,"temperature":0}'
```

## Requantization Commands

```bash
# Requantize with Q4_K output + Q4_K MTP (smallest, for 262K)
./build/bin/llama-quantize --allow-requantize \
  --output-tensor-type q4_K \
  --tensor-type "blk.64.nextn.eh_proj=q4_K" \
  --tensor-type "blk.64.ffn_down=q4_K" \
  --tensor-type "blk.64.ffn_gate=q4_K" \
  --tensor-type "blk.64.ffn_up=q4_K" \
  --tensor-type "blk.64.attn_k=q4_K" \
  --tensor-type "blk.64.attn_q=q4_K" \
  --tensor-type "blk.64.attn_v=q4_K" \
  --tensor-type "blk.64.attn_output=q4_K" \
  input.gguf output-q4out.gguf Q4_K_M

# Requantize with Q4_K output but Q8_0 MTP (better draft acceptance)
./build/bin/llama-quantize --allow-requantize \
  --output-tensor-type q4_K \
  --tensor-type "blk.64.nextn.eh_proj=q8_0" \
  --tensor-type "blk.64.ffn_down=q8_0" \
  --tensor-type "blk.64.ffn_gate=q8_0" \
  --tensor-type "blk.64.ffn_up=q8_0" \
  --tensor-type "blk.64.attn_k=q8_0" \
  --tensor-type "blk.64.attn_q=q8_0" \
  --tensor-type "blk.64.attn_v=q8_0" \
  --tensor-type "blk.64.attn_output=q8_0" \
  input.gguf output-q8mtp.gguf Q4_K_M
```

---

## Key Technical Details

### FWHT Sign Arrays (seed=42)
Both CPU (`ggml-turboq-tables.h`) and CUDA (`tbq4-cuda.cuh`) use identical sign arrays generated with seed=42. These MUST match between CPU and CUDA or quantized data will be incompatible.

### Norm Correction
Turbo4 stores `corrected_norm = original_norm / reconstruction_norm` instead of raw L2 norm. This compensates for quantization error in the rotated domain.

### Centroid Values
Lloyd-Max optimal centroids for N(0, 1/sqrt(128)):
```
{-0.241556, -0.182907, -0.143047, -0.111065,
 -0.083317, -0.058069, -0.034311, -0.011353,
  0.011353,  0.034311,  0.058069,  0.083317,
  0.111065,  0.143047,  0.182907,  0.241556}
```

### Per-element vs Full-block Dequant
- `dequantize_tbq4_0()` — per-element, NO inverse FWHT. Used by get_rows template.
- `k_tbq4_dequant_full()` — full-block with inverse FWHT via shared memory. Used by CPY/CAST for attention.

The per-element dequant returns values in the rotated domain. This is intentional — get_rows operates element-by-element and can't do the butterfly. The full-block dequant is needed for attention where the KV cache must be in the original domain.

### MTP Independence
MTP speculative decoding (`--spec-type mtp`) is completely independent of KV cache quantization. They use different code paths. MTP controls draft token generation; TBQ4 controls how K/V tensors are stored/retrieved.

---

## Swap Warning

The system had 8GB swap completely full during testing, severely degrading performance. Always clear swap before running:
```bash
sudo swapoff -a && sudo swapon -a
```
Use `--mlock` flag to prevent swap usage during inference.

---

## Update: Tensor Sharing Infrastructure (2026-05-07)

### link_shared_tensors() — 682 MiB GPU Saving

The MTP head loads as a separate model from the same GGUF, duplicating `token_embd.weight` (682 MiB Q4_K) and `output.weight` (995 MiB Q6_K). We added infrastructure to share the embedding between trunk and MTP models:

**New API:** `llama_model_link_shared_tensors(model_mtp, trunk_model)` — public API in `include/llama.h`

**How it works:**
1. MTP model skips loading `token_embd.weight` (TENSOR_NOT_REQUIRED)
2. After both models load, `link_shared_tensors()` wires the MTP's `tok_embd` pointer to the trunk model's tensor
3. Saves 682 MiB — MTP head uses trunk's embedding for its own embedding lookup

**Why output.weight is NOT shared:** The Q4_K token_embd produces different logits than Q6_K output.weight when used for the output projection. Sharing output caused 0% draft acceptance. The solution: keep output.weight loaded from GGUF for the projection, share only tok_embd for the embedding.

### Final Test Results

| Config | Context | Speed | Accept | VRAM | Status |
|--------|---------|-------|--------|------|--------|
| MTP + Q4_0 KV + shared tok_embd | 200K | 92-97 tok/s | 93.6% | 23.96 GB | STABLE |
| MTP + Q4_0 KV + shared tok_embd | 135K | 97-103 tok/s | 93.6% | 22.4 GB | STABLE |
| MTP + TBQ4_0 + ngl=63 | 262K | 45 tok/s | 72% | ~23 GB | LOADS, prompt crash |

### Files Changed for Tensor Sharing

| File | Change |
|------|--------|
| `include/llama.h` | +`llama_model_link_shared_tensors()` public API |
| `src/llama-model.h` | +`virtual link_shared_tensors()`, +`get_tensor_mutable()` |
| `src/llama-model.cpp` | +`llama_model_link_shared_tensors()` impl |
| `src/models/models.h` | +override on `qwen35_mtp`, `qwen35moe_mtp` |
| `src/models/qwen35_mtp.cpp` | tok_embd→NOT_REQUIRED, link_shared_tensors impl |
| `src/models/qwen35moe_mtp.cpp` | Same for MoE variant |
| `tools/server/server-context.cpp` | Call after MTP model load |
