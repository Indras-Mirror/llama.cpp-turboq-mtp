# llama.cpp-mtp — Fused TBQ4 Flash Attention + MTP + Shared Tensors

> **Fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)** with fused TurboQuant flash attention — the FA kernel reads raw TBQ4_0 K/V blocks directly from global memory and dequants via centroid lookup in the FWHT-rotated domain. No separate dequant pass, no intermediate F16 buffer.

**80-179 tok/s decode (325 effective) with lossless 4.25 bpv KV cache at 262K context on RTX 4090 24GB.**

---

## What This Fork Adds

| Feature | Description | Status |
|---------|-------------|--------|
| **Fused TBQ4 Flash Attention** | Quantized-KV dequant inside the FA inner loop via rotated-domain attention | Working, 82+ tok/s |
| **MTP Speculative Decoding** | Multi-Token Prediction for Qwen3.6 (PR #22673) with 3 draft tokens per forward pass | Working, 73-93% accept |
| **CUDA TBQ4_0 Kernels** | FWHT-based TurboQuant quantize/dequant on GPU (ported from dflash fork) | Working |
| **Tensor Sharing API** | `link_shared_tensors()` prevents 682 MiB GPU duplication of token embeddings between trunk and MTP models | Working |
| **RotorQuant (PlanarQuant + IsoQuant)** | 4 new 3-bit/4-bit KV cache types using Givens/quaternion rotations — faster dequant, better compression, 5.3x faster prefill | ✅ New! |

### RotorQuant — Next-Gen KV Cache Compression

**RotorQuant replaces the FWHT butterfly with block-diagonal 2D/4D rotations.** Same compression ratio as TBQ4 but with O(d) rotation (fully parallel) instead of O(d log d) Hadamard. Drop-in compatible via `-ctk`/`-ctv` flags.

#### Available Types

| Type | Bits | Block | Rotation | VRAM @ 262K |
|------|------|-------|----------|-------------|
| `tbq4_0` | 4.25 | 66 bytes/128 dims | FWHT butterfly | 4224 MiB |
| `planar3_0` | 3.0 | 50 bytes/128 dims | 2D Givens pairs | **3200 MiB** (-24%) |
| `iso3_0` | 3.0 | 50 bytes/128 dims | 4D quaternion | **3200 MiB** (-24%) |
| `planar4_0` | 4.0 | 66 bytes/128 dims | 2D Givens pairs | 4224 MiB |
| `iso4_0` | 4.0 | 66 bytes/128 dims | 4D quaternion | 4224 MiB |

#### Benchmark (RTX 4090, Qwen3.6-27B, MTP+FA)

| Type | 4K ctx | 32K ctx | 262K ctx | Notes |
|------|--------|---------|----------|-------|
| `tbq4_0` | 55.3 t/s | 51.5 t/s | 77 t/s | Baseline — fused MMA kernel |
| `planar3_0` | 53.9 t/s | 50.6 t/s | ~47 t/s | Best speed/compression tradeoff |
| `iso3_0` | 53.5 t/s | 50.5 t/s | — | Same compression as planar3 |
| `planar4_0` | 52.2 t/s | — | — | 4-bit Givens |
| `iso4_0` | 50.6 t/s | — | — | 4-bit quaternion |

#### Usage

```bash
# Build with FA_ALL_QUANTS for planar/iso support
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc) --config Release

# Use planar3_0 for max VRAM savings (saves 1 GB vs TBQ4 at 262K)
./build/bin/llama-server \
  -m your-model.gguf \
  --spec-type mtp --spec-draft-n-max 3 \
  -ctk planar3_0 -ctv planar3_0 -c 262144 -ngl 99 \
  --flash-attn on --mlock -t 8 -ub 32 --parallel 1 --no-warmup
```

#### How It Works

Unlike TBQ4's FWHT (Hadamard) rotation, RotorQuant uses:

- **PlanarQuant**: 64 independent 2D Givens rotations per 128-dim block. Rotation: `[cos θ, sin θ; -sin θ, cos θ]` per element pair. 128 total rotation parameters.
- **IsoQuant**: 32 independent 4D quaternion rotations per 128-dim block. 128 total rotation parameters.

Both apply the rotation at quantization time. During FA dequant, the inverse rotation is applied inline — centroid lookup → inverse Givens/quaternion → scale by norm. The rotation is trivially parallel (no sequential stages like FWHT).

#### Bugs Fixed

1. **llama-graph.cpp**: Planar/iso removed from TBQ pass-through — VEC path handles dequant inline
2. **cpy.cu**: 4-bit dequant kernels with inverse rotation (planar4/iso4→F32)
3. **ggml-cuda.cu**: `supports_op` entries for all new types
4. **-fit auto**: Memory estimation workaround with `-fit off`

### Recent Fixes (May 11, 2026)

- **NaN sampler crash (#6)**: Guard against all-`-inf` logits in dist sampler — when upstream samplers filter every token to `-infinity`, softmax produces NaN (`-inf - (-inf) = NaN`), causing `assert(found)` failure. Fixed with `!(sum_cum > 0.0)` guard + `test_dist_all_neg_inf` unit test.
- **Double free**: Upstream cherry-pick from PR #22673 (server-context.cpp lifecycle fix).
- **RS sequence for MTP only**: Upstream cherry-pick from PR #22673 (fixes partial rollback scope for non-MTP models).

### Recent Changes (May 15-17, 2026)

- **Upstream sync**: Merged upstream `ggml-org/llama.cpp` master (`ec562eb67`, 40 commits). Adopted parallel drafting, renamed types (`DRAFT`→`DRAFT_SIMPLE`), `ctx`→`ctx_tgt`/`ctx_dft` split, MTP adapted to new `common_speculative_impl` interface. 6 merge conflicts resolved (README, common/arg.cpp, common/speculative.cpp, tests, server-context.cpp).
- **MTP draft regression fix**: `n_draft_max` was unconstrained (261K instead of 3), causing batch overflow. `dp.drafting = false` was set too early, preventing `accept()` from updating `last_n_accepted`, feeding stale hidden states to MTP. Both fixed.
- **Multi-turn KV cache fix**: Context checkpoints were not created for MTP slots because `n_rs_seq=3` fooled `common_context_can_seq_rm` into returning `PART` type. Without checkpoints, every message turn forced full prompt re-processing (~46s per turn). Fixed by enabling context checkpoints for MTP slots (~150 MiB each on CPU RAM, max 32 = ~4.8 GB). Multi-turn latency: 40-50s → ~460ms (86x improvement).
- **proper-lockfile Bun interop**: Fixed CJS interop edge case where Bun's bundler returns proper-lockfile under a `'.'` key.

## Upstream MTP Status

**⚠️ As of May 16, 2026:** Upstream `ggml-org/llama.cpp` merged official MTP support via [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) (`255582687`). This is 20 commits ahead of our current sync point (`ec562eb67`). The upstream implementation uses `--spec-type draft-mtp` and `COMMON_SPECULATIVE_TYPE_DRAFT_MTP`.

**Our fork** uses a custom MTP implementation (`--spec-type mtp`, `COMMON_SPECULATIVE_TYPE_MTP`) that predates the upstream merge. Both implementations support Qwen3.6 MTP heads, but ours includes additional features (TBQ4 fused FA, RotorQuant, tensor sharing, context checkpoints for MTP).

**Next sync** will incorporate the upstream MTP implementation while preserving our TBQ4 + RotorQuant + tensor sharing customizations. For now, our MTP implementation is stable and tested (92% draft acceptance, 29-turn continuous session without errors).

## Results (RTX 4090 24GB, Qwen3.6-27B-Heretic-v2-MTP Q4_K_M)

| Config | Context | KV Cache | tok/s | Draft Accept | VRAM |
|--------|---------|----------|-------|-------------|------|
| **MTP + Fused TBQ4 FA (May 11)** | **262K** | **TBQ4_0 (4.25 bpv)** | **179.4** | **81.4%** | **~20 GB** |
| **MTP + Fused TBQ4 FA** | **262K** | **TBQ4_0 (4.25 bpv)** | **80-87** | **73-93%** | **~20 GB** |
| MTP + Fused TBQ4 FA | 200K | TBQ4_0 (4.25 bpv) | 82-87 | 73% | ~20 GB |
| MTP + Q4_0 KV | 200K | Q4_0 (4.5 bpv) | 92-97 | 93.6% | 23.96 GB |
| MTP + Q4_0 KV | 135K | Q4_0 (4.5 bpv) | 97-103 | 93.6% | 22.4 GB |
| Baseline (no MTP, Q4_0 KV) | 200K | Q4_0 | ~40 | - | 23.96 GB |
| MTP Draft 5 | 262K | TBQ4_0 | 79.6 avg / 106 peak | 90.1% | ~20 GB |

## Why This Is Novel

**Nobody else has fused quantized-KV dequant into the flash attention inner loop.** The upstream TBQ4 PR (#21089) is CPU-only. The dflash fork (spiritbuun) has CUDA TBQ4 kernels but uses `nstages=0` with a separate dequant-to-F16 pass before FA. Our kernel reads raw TBQ4 blocks directly:

```
Standard path:  TBQ4 → dequant → F16 buffer → FA kernel reads F16
Our fused path: TBQ4 → FA kernel reads raw bytes → centroid×norm lookup inline
```

The key insight: since the Hadamard transform is orthonormal, **attention can operate entirely in the rotated domain**. Q is pre-rotated once, K/V are pre-rotated at quantization time, and the output is post-rotated once. The inner loop only needs a 2-value centroid lookup per element — no FWHT butterfly, no precomputed tables.

### Optimizations (43 → 82 tok/s across 5 sessions)

1. **Column-group access pattern** — threads process one column across all rows instead of one row per thread, nearly doubling bandwidth utilization
2. **Direct centroid lookup** — look up only the 2 centroid values needed per byte instead of precomputing all 16 (saving 14 FP muls + 14 float-to-half conversions per element)
3. **Rotated-domain attention** — FWHT runs only twice total (Q rotate in, output rotate out), never inside the KV iteration loop

---

## Quick Start

```bash
git clone https://github.com/Indras-Mirror/llama.cpp-mtp
cd llama.cpp-mtp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc) --config Release

# Fused TBQ4 FA + MTP (80-87 tok/s at 262K, lossless 4.25 bpv KV)
./build/bin/llama-server \
  -m your-qwen3.6-mtp.gguf \
  --spec-type mtp --spec-draft-n-max 3 \
  -ctk tbq4_0 -ctv tbq4_0 -c 262144 -ngl 99 \
  --flash-attn on --mlock -t 8 -ub 32 -np 1 --no-warmup

# Or with Q4_0 KV for max raw speed (92-97 tok/s at 200K, uses more VRAM)
./build/bin/llama-server \
  -m your-qwen3.6-mtp.gguf \
  --spec-type mtp --spec-draft-n-max 3 \
  -ctk q4_0 -ctv q4_0 -c 200000 -ngl 99 \
  --flash-attn on --mlock -ub 32 -np 1
```

### Getting an MTP-capable GGUF

**Option A: Pre-built Native-MTP-Preserved GGUF (Recommended)**

Use llmfan46's pre-built GGUF with all 15 native MTP heads preserved from Qwen3.6 training:

```bash
# Download from HuggingFace (~17 GB, Q4_K_M, 15 native MTP heads)
wget https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF/resolve/main/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q4_K_M.gguf
```

Model: `Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q4_K_M.gguf`
Source: [llmfan46 on HuggingFace](https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF) — Heretic v1.3 MPOA uncensored fine-tune (94% fewer refusals, 0.0021 KL divergence, 85.67% MMLU)

**Option B: Graft MTP heads onto any Qwen3.6 GGUF**

Standard GGUF conversion strips MTP layers. Graft them back:

```bash
# Download MTP head GGUF (457 MB, only the draft head tensors)
wget https://huggingface.co/havenoammo/Qwen3.6-27B-MTP-UD-GGUF/resolve/main/MTP-Q8_0.gguf

uv venv .venv --seed && source .venv/bin/activate
uv pip install gguf
python convert.py base-model.gguf MTP-Q8_0.gguf output-mtp.gguf
```

---

## Architecture

### Fused TBQ4 Flash Attention Pipeline

```
1. k_tbq4_rotate_input    → Pre-rotate Q via FWHT (separate kernel, 128-thread warp shuffle)
2. Fused FA kernel         → Read raw TBQ4 blocks from GMEM, centroid×norm dequant inline
3. k_tbq4_rotate_output   → Post-rotate VKQ back to original domain
```

K/V are pre-rotated at SET_ROWS time (`quantize_f32_tbq4_0_block` calls `tbq4_rotate_forward` before quantization). Everything in the FA inner loop operates in the rotated domain.

### TBQ4_0 Block Format

```c
struct block_tbq4_0 {      // 66 bytes per 128 elements (4.25 bits per value)
    ggml_half d;            // corrected L2 norm (2 bytes)
    uint8_t qs[QK_TBQ4/2]; // packed 4-bit centroid indices (64 bytes)
};
```

16 Lloyd-Max centroids optimized for N(0, 1/sqrt(128)) in the FWHT domain, stored in CUDA `__constant__` memory.

### Inner Loop (the hot path)

```cuda
// Per byte = 2 KV elements. This is the entire dequant:
const uint8_t byte = __ldg(&blk->qs[b]);
const half lo = __float2half(d_tbq4_centroids[byte & 0xF] * norm);
const half hi = __float2half(d_tbq4_centroids[byte >> 4] * norm);
tile[...] = __halves2half2(lo, hi);
```

### Tensor Sharing — `link_shared_tensors()` API

MTP loads `token_embd.weight` as a separate 682 MiB GPU allocation — a duplicate. Our API lets sibling models wire shared tensors:

```cpp
// include/llama.h
LLAMA_API void llama_model_link_shared_tensors(
    struct llama_model * model,
    const struct llama_model * trunk);
```

Implemented for `qwen35_mtp` and `qwen35moe_mtp`. Saves 682 MiB with zero quality impact.

---

## Files Added/Modified

### Fused TBQ4 Flash Attention (novel)
| File | Purpose |
|------|---------|
| `ggml/src/ggml-cuda/fattn-mma-tbq4.cuh` | **NEW** — Fused tile loader, rotation kernels, centroid lookup |
| `ggml/src/ggml-cuda/fattn-mma-tbq4-launch.cuh` | **NEW** — Template launcher, shmem calculation |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | Modified — TBQ4 guards in iter function (4 locations) |
| `ggml/src/ggml-cuda/fattn.cu` | Modified — TBQ4 dispatch + rotation kernel calls |
| `template-instances/fattn-mma-tbq4-instance-ncols2_{1,2,4,8}.cu` | **NEW** — Template instantiations |

### CUDA TBQ4_0 Kernels (ported from dflash)
| File | Purpose |
|------|---------|
| `ggml/src/ggml-cuda/tbq4-cuda.cuh` | **NEW** — FWHT, quantize, dequant, full-block dequant |
| `ggml/src/ggml-cuda/set-rows.cu` | TBQ4_0 SET_ROWS dispatch |
| `ggml/src/ggml-cuda/cpy.cu` | TBQ4_0 to F32/F16 dequant |

### Tensor Sharing Infrastructure
| File | Purpose |
|------|---------|
| `include/llama.h` | `llama_model_link_shared_tensors()` public API |
| `src/llama-model.h` / `.cpp` | Virtual method + implementation |
| `src/models/qwen35_mtp.cpp` | Qwen3.5 MTP tensor sharing |
| `src/models/qwen35moe_mtp.cpp` | Qwen3.5 MoE MTP tensor sharing |
| `tools/server/server-context.cpp` | Call site after MTP model load |

### Total: 89 files changed, +5,868 / -221 lines vs upstream

---

## Key Flags

| Flag | Purpose |
|------|---------|
| `--spec-type mtp --spec-draft-n-max 3` | Enable MTP speculative decoding |
| `-ctk tbq4_0 -ctv tbq4_0` | Fused TBQ4 KV cache (lossless, 4.25 bpv) |
| `-ctk q4_0 -ctv q4_0` | Q4_0 KV cache (higher speed, more VRAM) |
| `-ub 32` | Small ubatch keeps MTP compute buffer at ~712 MiB |
| `-np 1` | MTP only supports single parallel slot |
| `--mlock` | Prevent swap under memory pressure |
| `--flash-attn on` | Required for fused TBQ4 path |
| `--no-warmup` | Skip warmup for faster startup |

## Known Issues

- **Vision + MTP** crashes (upstream PR bug in multimodal handling — reported 2026-05-06). Use `--spec-type none` for vision tasks.
- **nstages=2 pipeline** produces garbled output with MTP (non-MTP works at 43.8-45.6 tok/s coherent). Reverted to synchronous nstages=0 for stability.
- **output.weight sharing** causes 0% draft acceptance (Q4_K ≠ Q6_K quantization error accumulates across embedding layers). `link_shared_tensors()` shares tok_embd only; output gets its own copy.
- **MTP requires `--parallel 1`** (single slot only — Multi-Token Prediction architecture limitation)
- **7B models crash with TBQ4** — `nb1=264` is 8-byte aligned, not 16-byte. Deferred. 27B works fine with `nb1=528`.
- **MoE models (35B-A3B)** may fail with `vector::_M_range_check` in MTP loading if `nextn_predict_layers` metadata is missing or incorrect in the GGUF. Verify `--verbose` output shows the key being read.
- **MTP draft-n-max 3 vs 5**: Draft 3 gives better per-token speed (80.6 vs 79.6 tok/s) and higher acceptance (92.6% vs 90.1%). Draft 5 occasionally hits higher peaks (106 tok/s) but overhead from verifying longer drafts eats the gain.

## Credits

- **[havenoammo](https://huggingface.co/havenoammo)** — MTP graft tooling, first Qwen3.6-MTP GGUF release
- **[spiritbuun](https://github.com/spiritbuun)** — dflash fork with CUDA TurboQuant kernels (our FWHT kernels adapted from this)
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** — PR #22673 (MTP), PR #21089 (CPU TBQ)
- **llmfan46** — Qwen3.6-27B-Heretic-v2 Native-MTP-Preserved GGUF (the model we use — 15 native MTP heads, MPOA uncensoring)
- **HauhauCS** — Original Qwen3.6-Heretic-v2 uncensored base model
- **Radamanthys11** — MTP-Q8_0 GGUF extraction
- **froggeric** — Fixed chat templates for Qwen3.6 + MTP

## Documentation

- **[Blog post](https://indrasmirror.au/blog-mtp-shared-tensors-200k.html)** — Detailed writeup with benchmarks, architecture, and optimization journey

---

<details>
<summary><strong>Upstream llama.cpp README</strong></summary>

This fork is based on [llama.cpp](https://github.com/ggml-org/llama.cpp) by ggml-org. See the [upstream repository](https://github.com/ggml-org/llama.cpp) for general llama.cpp documentation, build instructions, and supported models.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

</details>
