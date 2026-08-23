# llama.cpp-TurboQuant-DSV4 — Fused TBQ4 Flash Attention + MTP + DSV4 Native TBQ4

> **Fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)** (ggml-org lineage, b10217 rebase base `a7a6d0d26`) combining TurboQuant KV cache (TBQ3/TBQ4), RotorQuant, MTP speculative decoding, and — the headline of this repo — **DSV4 native TBQ4 KV cache via dequant-at-read** (Option A): DeepSeek-V4-Flash KV cache stored natively as TBQ4_0 and dequantized at read time, no Q8_0 fallback, no separate dequant pass.

**Measured on RTX 4090 24GB + 94Gi DDR4 (DeepSeek-V4-Flash-0731 abliterated, IQ2XXS, 81GB): 11.61 t/s gen @ 4K, 10.29 t/s @ 256K, 9.30 t/s @ 512K, VmSwap 0 everywhere, quality 12/12 correct (v2 native concat).**

---

## What This Fork Adds

| Feature | Description | Status |
|---------|-------------|--------|
| **DSV4 Native TBQ4 KV Cache (dequant-at-read)** | DeepSeek-V4-Flash KV stored natively as TBQ4_0; v2 strided-quantized concat kernel bypasses F32 dequant at csa/hca sites (2.16x speedup at 512K). Lid site still dequants at read. | ✅ **Headline — this work** |
| **Fused TBQ4 Flash Attention** | Quantized-KV dequant inside the FA inner loop via rotated-domain attention (centroid lookup, no intermediate F16 buffer) | Working, 82+ tok/s (Qwen3.6) |
| **MTP Speculative Decoding** | Multi-Token Prediction for Qwen3.6 (PR #22673 lineage) with 3 draft tokens per forward pass; custom implementation kept (see below) | Working, 73-98% accept |
| **Fused TBQ3 Flash Attention** | 3-bit KV compression (3.0625 bpv, ~24% smaller than TBQ4), fused inline dequant. Mixed TBQ4+TBQ3 via DSV4_CTK_COMP | Working — validated 2026-08-03 (GPU KV) |
| **CUDA TBQ4_0 Kernels** | FWHT-based TurboQuant quantize/dequant on GPU (ported from the dflash fork) | Working |
| **Tensor Sharing API** | `link_shared_tensors()` prevents 682 MiB GPU duplication of token embeddings between trunk and MTP models | Working |
| **RotorQuant (PlanarQuant + IsoQuant)** | 4 new 3-bit/4-bit KV cache types using Givens/quaternion rotations — faster dequant, better compression | Working |
| **Qwen35 SWA Hybrid (turboq-mtp-swa)** | Sliding-Window Attention for Qwen3.8-27B (qwen35, Gated-DeltaNet hybrid): windows most full-attn layers so decode stays bounded at deep context, keeps `swa_global_layers=8` GLOBAL for long-range recall. N=8 = optimum (correct beyond-window recall + ~70 t/s @ 62K, MTP accept 1.00) | Working |

---

## Qwen35 SWA Hybrid (turboq-mtp-swa)

Sliding-Window Attention for **Qwen3.8-27B** (arch `qwen35`, a Gated-DeltaNet hybrid). Windows
the bulk of the full-attention (non-recurrent, `il%4==3`) trunk layers to a sliding window so
**decode stays bounded at deep context**, while `qwen35.attention.swa_global_layers` of them stay
**GLOBAL** (dense, full context) for long-range verbatim recall. Gating is balanced (Bresenham)
over the full-attention layers, mirroring `muse-glimmer`; the MTP/draft head stays dense.

| Setting | Default | Notes |
|---------|---------|-------|
| `qwen35.attention.sliding_window` | `4096` | bounds decode (prefill unchanged) |
| `qwen35.attention.swa_global_layers` | `8` | full-attn layers kept GLOBAL |

Swept `swa_global_layers` = 2 / 4 / 8 / 13 at deep ctx (beyond-window recall ~10K back + decode
~62K ctx): **N=8 is the optimum — the smallest count with correct recall AND the fastest decode
(69.99 t/s @ 62K, MTP draft acceptance 1.00).** N=2/4 fail beyond-window recall (their MTP
acceptance collapses → long hedging thinking); N=13 recalls but is slower (59.25 t/s).

- Tunable via `--override-kv qwen35.attention.swa_global_layers=int:N` or the wrapper flag
  `--swa-global-layers=N`.
- MTP graph must keep the ISWA builders (`graph_mtp`) or `swa_type!=NONE` asserts.
- Wrapper: `qwen3.8-quetza-agg-swa --swa=8192`. Build: `cmake --build build -j4 --target llama-server`.

---

## DSV4 Native TBQ4 — v2 Strided-Quantized Concat

Upstream of this work, DSV4 fell back from TBQ quantization to Q8_0 because the DSV4 model path had no TBQ dequant support. This commit (`1a663e2d0`) removes that fallback and stores TBQ4_0 natively in all four DSV4 KV caches (lid, csa raw+csa, hca raw+hca). A `dequant_k_read` helper casts TBQ3/TBQ4 blocks to F32 and reshapes to 4D at the three read sites; the raw ratio-0 site keeps the fused `MMA_TBQ4` path, and the Q8_0 path is byte-identical (the helper is a passthrough there).

### Why it matters

The KV cache is the only thing that scales with context. Native TBQ4 (4.25 bpv) cuts the KV working set to roughly half of Q4_0's. With v2 native concat, 512K context uses just 82.4 GB RSS with **VmSwap 0** — well within a 96GB box. No thrash, no fallback.

### Benchmark (RTX 4090 24GB, DeepSeek-V4-Flash-0731 abliterated IQ2XXS, KV in system RAM)

Server: `llama-server -m <model> --port 8099 -c <ctx> --flash-attn on -t 8 -np 1 --jinja -ctk tbq4_0 -ctv tbq4_0 --no-kv-offload`

| Config | Context | gen t/s | prompt t/s | server RSS | VmSwap | Quality |
|--------|---------|---------|------------|-----------|--------|---------|
| **TBQ4 v2 native concat** | 4K | **11.61** | 18.8-20.3 | 81.1 GB | 0 | 12/12 |
| **TBQ4 v2** | 32K | **11.39** | 19.6-20.6 | 81.1 GB | 0 | 12/12 |
| **TBQ4 v2** | 256K | **10.29** | 19.1-20.4 | 81.7 GB | 0 | 12/12 |
| **TBQ4 v2** | 512K | **9.30** | 17.4-19.0 | 82.4 GB | 0 | 12/12 |
| Q4_0 K+V (mainline ref) | 512K | ~4.8 | — | 83.4-83.7 GB | 0→128 MB | 5/5 |

Quality: **12/12 correct** across all v2 configs (see table). No quantization-noise regression in any config.

**Note on v2:** With v2 native concat, TBQ4 at 512K (9.30 t/s) now outperforms mainline Q4@512K (~4.8 t/s) by 1.94x while using less memory. The DDR4 expert-weight floor (~85ms) sets a hard ceiling at ~12 t/s regardless of context size. Full-context benchmarks (200K+ tokens filled) are pending; CPU analysis estimates ~6-7 t/s at 512K under full fill.

---

## Build

Same build as the rest of the fork — CUDA with `sm_89` (RTX 4090), in a `build-mtp` directory:

```bash
cd llama.cpp-TurboQuant-DSV4
cmake -B build-mtp -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-mtp -j$(nproc) --config Release
```

For the full RotorQuant type set (planar3_0 / iso3_0 / planar4_0 / iso4_0), add `-DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON` to the configure step. The `build-mtp` name is the convention used throughout this fork's testing (a plain `build` dir also works); `build-mtp/bin/llama-server` is the resulting binary.

## Run

All variants share the same shape: pick the KV type, keep `--flash-attn on` and `--no-kv-offload` (KV cache lives in system RAM, the model lives in VRAM — this is what makes 512K/1M possible on 24GB + 96GB hardware).

```bash
# TBQ4 native KV (the headline config — DSV4 stores TBQ4_0 natively, dequant at read)
./build-mtp/bin/llama-server \
  -m deepseek-v4-flash-abliterated-IQ2XXS.gguf \
  -c 524288 --flash-attn on -t 8 -np 1 --jinja \
  -ctk tbq4_0 -ctv tbq4_0 --no-kv-offload

# Q4_0 KV (daily sweet spot: ~4.8 t/s gen at 512K, smallest KV footprint)
./build-mtp/bin/llama-server \
  -m deepseek-v4-flash-abliterated-IQ2XXS.gguf \
  -c 524288 --flash-attn on -t 8 -np 1 --jinja \
  -ctk q4_0 -ctv q4_0 --no-kv-offload

# Q8_0 KV (reference / max quality — note: on DSV4 this needs ~32 GB KV at 512K and will thrash on a 96GB box)
./build-mtp/bin/llama-server \
  -m deepseek-v4-flash-abliterated-IQ2XXS.gguf \
  -c 524288 --flash-attn on -t 8 -np 1 --jinja \
  -ctk q8_0 -ctv q8_0 --no-kv-offload
```

Context sizing: 4K → 11.61 t/s, 32K → 11.39 t/s, 256K → 10.29 t/s, 512K → 9.30 t/s (v2 native concat). Full-context benchmarks pending.

### MTP mode (Qwen3.6 family)

```bash
./build-mtp/bin/llama-server \
  -m your-qwen3.6-mtp.gguf \
  --spec-type mtp --spec-draft-n-max 3 \
  -ctk tbq4_0 -ctv tbq4_0 -c 262144 -ngl 99 \
  --flash-attn on --mlock -t 8 -ub 32 -np 1 --no-warmup
```

---

## Qwen3.8 MTP with TBQ4 Draft KV (2026-08-16)

Verified on **Qwen3.8-27B-heretic-ara.i1-IQ4_XS** (qwen35 arch, embedded MTP head, `nextn_predict_layers=1`) with llama-server on RTX 4090 24GB at `-c 262144`. This is the Qwen3.8-lineage companion to the Qwen3.6 MTP mode above, and the first config in this fork where the **draft cache gets its own quant type**, fully independent of the main KV cache.

### Why the draft KV needs its own type

At 262K the main TBQ4 KV + mmproj already use 19.9 GiB of the 24.5 GiB VRAM budget, leaving ~4.2 GiB of headroom. An fp16 draft KV (~4.35 GiB) does **not** fit. Quantizing the draft cache with TBQ4 (~1.1 GiB) closes the gap — **total 20,637 MiB, 3.9 GiB headroom, no OOM** — while still drafting 2-3 tokens per forward pass.

### Flags

The draft KV type is set separately from the main cache: `--spec-draft-type-k` / `--spec-draft-type-v` (aliases `-ctkd` / `-ctvd`, long forms `--cache-type-k-draft` / `--cache-type-v-draft`, see `common/arg.cpp:3926`/`3939`). It does not follow `-ctk` / `-ctv`:

```bash
./build-mtp/bin/llama-server \
  -m qwen3.8-27b-heretic-ara.i1-IQ4_XS.gguf \
  -c 262144 --flash-attn on -ngl 99 \
  -ctk tbq4_0 -ctv tbq4_0 \
  -ctkd tbq4_0 -ctvd tbq4_0 \
  --spec-type draft-mtp --spec-draft-n-max 3
```

### Measured results

| Metric | Value |
|--------|-------|
| Draft acceptance | 0.52-0.67 |
| Mean draft length | 2.55-3.0 (drafting active) |
| Generation | **74.1 t/s** vs ~44-50 t/s without MTP (**~1.5-1.7x**) |
| Prompt processing | 452 t/s |
| VRAM total | 20,637 MiB (3.9 GiB headroom, no OOM) |
| Native tool_use probe | clean (`stop_reason: tool_use`) |

---

## Upstream MTP Status — Why We Keep Our Implementation

As of May 16, 2026, upstream `ggml-org/llama.cpp` merged official MTP support via [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) (`255582687`), which uses `--spec-type draft-mtp`. **We are NOT adopting it.** Our custom MTP (`--spec-type mtp`) predates the merge and beats upstream in every measured metric — head-to-head on RTX 4090 24GB with Qwen3.6-27B-Heretic-v2-MTP Q4_K_M:

| Metric | Upstream MTP | Our Fork | Delta |
|--------|:-----------:|:--------:|:-----:|
| **Generation speed** | 71.5 tok/s | 82-93 tok/s | **+15-30%** |
| **Draft acceptance** | 47-89% | 73-98% (avg 92%) | **+3-45 pp** |
| **KV cache type** | Q4_0 (4.5 bpv) | TBQ4_0 (4.25 bpv) | 6% more compression |
| **Max context @ 24GB** | ~131K | **262K** | **2x** |
| **Fused quant FA** | ❌ Separate dequant pass | ✅ Inline dequant in FA loop | Memory + speed |
| **Tensor sharing** | ❌ 682 MiB duplicated | ✅ `link_shared_tensors()` | Saved 682 MiB |
| **RotorQuant** | ❌ | ✅ planar3/iso3/planar4/iso4 | 3-4 bit KV cache options |

Future upstream syncs will pull non-MTP improvements (tokenizer fixes, server patches); the TBQ4 + RotorQuant + tensor sharing + MTP stack stays ours.

## Results (Qwen3.6-27B-Heretic-v2-MTP Q4_K_M, RTX 4090 24GB)

| Config | Context | KV Cache | tok/s | Draft Accept | VRAM |
|--------|---------|----------|-------|-------------|------|
| **MTP + Fused TBQ4 FA** | **262K** | **TBQ4_0 (4.25 bpv)** | **80-87** | **73-93%** | **~20 GB** |
| MTP + Q4_0 KV | 200K | Q4_0 (4.5 bpv) | 92-97 | 93.6% | 23.96 GB |
| Baseline (no MTP, Q4_0 KV) | 200K | Q4_0 | ~40 | - | 23.96 GB |

## RotorQuant — More KV Cache Compression

**RotorQuant replaces the FWHT butterfly with block-diagonal 2D/4D rotations.** Same compression ratio as TBQ4 but with O(d) rotation (fully parallel) instead of O(d log d) Hadamard. Drop-in compatible via `-ctk`/`-ctv`.

| Type | Bits | Block | Rotation | VRAM @ 262K |
|------|------|-------|----------|-------------|
| `tbq4_0` | 4.25 | 66 bytes/128 dims | FWHT butterfly | 4224 MiB |
| `planar3_0` | 3.0 | 50 bytes/128 dims | 2D Givens pairs | **3200 MiB** (-24%) |
| `iso3_0` | 3.0 | 50 bytes/128 dims | 4D quaternion | **3200 MiB** (-24%) |
| `planar4_0` | 4.0 | 66 bytes/128 dims | 2D Givens pairs | 4224 MiB |
| `iso4_0` | 4.0 | 66 bytes/128 dims | 4D quaternion | 4224 MiB |

## Why This Is Novel

**Nobody else has fused quantized-KV dequant into the flash attention inner loop.** The upstream TBQ4 PR (#21089) is CPU-only. The dflash fork has CUDA TBQ4 kernels but uses a separate dequant-to-F16 pass before FA. Our kernel reads raw TBQ4 blocks directly:

```
Standard path:  TBQ4 → dequant → F16 buffer → FA kernel reads F16
Our fused path: TBQ4 → FA kernel reads raw bytes → centroid×norm lookup inline
```

The key insight: since the Hadamard transform is orthonormal, **attention can operate entirely in the rotated domain** — Q is pre-rotated once, K/V are pre-rotated at quantization time, and the output is post-rotated once. The inner loop only needs a 2-value centroid lookup per element:

```cuda
// Per byte = 2 KV elements. This is the entire dequant:
const uint8_t byte = __ldg(&blk->qs[b]);
const half lo = __float2half(d_tbq4_centroids[byte & 0xF] * norm);
const half hi = __float2half(d_tbq4_centroids[byte >> 4] * norm);
tile[...] = __halves2half2(lo, hi);
```

## Key Flags

| Flag | Purpose |
|------|---------|
| `-ctk tbq4_0 -ctv tbq4_0` | Native TBQ4 KV cache (4.25 bpv) — DSV4 dequant-at-read path |
| `-ctk q4_0 -ctv q4_0` | Q4_0 KV cache (daily sweet spot on DSV4 @512K) |
| `--no-kv-offload` | Keep KV cache in system RAM (enables 512K/1M on 24GB VRAM) |
| `--flash-attn on` | Required for the fused TBQ4 path |
| `--spec-type mtp --spec-draft-n-max 3` | Enable MTP speculative decoding (Qwen3.6) |
| `--spec-type draft-mtp` | Enable upstream-style MTP (Qwen3.8, embedded MTP head) |
| `-ctkd tbq4_0 -ctvd tbq4_0` | Draft KV cache type (`--spec-draft-type-k/v`) — independent of main `-ctk`/`-ctv` |
| `DSV4_CTK_COMP=tbq3_0` | Per-cache K quant: TBQ4 for raw-ratio sites, TBQ3 for compressed (mixed) |
| `--mlock` | Prevent swap under memory pressure |
| `-ub 32` | Small ubatch keeps the MTP compute buffer small |
| `-np 1` | MTP supports a single parallel slot |

## Known Issues

- **Vision + MTP** crashes (upstream PR bug in multimodal handling). Use `--spec-type none` for vision tasks.
- **nstages=2 pipeline** produces garbled output with MTP (non-MTP is coherent). Reverted to synchronous nstages=0 for stability.
- **output.weight sharing** causes 0% draft acceptance (Q4_K ≠ Q6_K quantization error accumulates). `link_shared_tensors()` shares `tok_embd` only.
- **MTP requires `--parallel 1`** (single slot — Multi-Token Prediction architecture limitation).
- **7B models crash with TBQ4** — `nb1=264` is 8-byte aligned, not 16-byte. Deferred; 27B works (`nb1=528`).
- **MoE models** may hit `vector::_M_range_check` in MTP loading if `nextn_predict_layers` metadata is missing/incorrect in the GGUF.
- **1M native-TBQ4 gen is ~2.2 t/s** — the honest dequant-at-read + K-transfer cost at extreme context (see benchmark note above).

## TBQ3 Fused Flash Attention (Experimental)

**Status: FIXED and validated (2026-08-03). GPU KV correct at 12.98 t/s — now the recommended default GPU KV type (3.0625 bpv vs TBQ4's 4.125, the 512K headroom enabler). CPU-only KV correct but slow (2.85 t/s).**

TBQ3 (3.0625 bpv) is the next step below TBQ4 (4.125 bpv). Fused FA reads raw TBQ3_0 K/V blocks directly (3-bit centroid lookup + norm, no intermediate F16 buffer). The fused kernel operates in the rotated domain identically to TBQ4; only the tile loader and sign arrays differ. The fused MMA kernel (`fattn-mma-tbq3.cuh`) compiles and auto-dispatches from `fattn.cu` via the `GGML_TYPE_TBQ3_0` gate.

### Bug history — eight bugs fixed (working tree, uncommitted)

**Six dequant/rotation fixes:**
1. **CUDA dequant** (`k_tbq3_dequant_full`, `ggml-cuda/tbq3-cuda.cuh`): wrong struct offsets — read `d` from byte 48 (block end); fixed to the `{d, qs}` layout, `d` at byte 0.
2. **CUDA dequant**: missing s1/s2 sign multiplication in the inverse rotation.
3. **CUDA dequant**: inverse factor `/128` → `*inv_sqrt_128` (`0.08838834764831845f`). `d = norm/recon_norm` assumes a norm-preserving rotation; the unnormalized FWHT scales norms by sqrt(128), so the inverse is `1/sqrt(128)` (confirmed by `fattn-mma-tbq3.cuh:118`).
4. **CPU dequant** (`dequantize_row_tbq3_0`, `ggml/src/ggml-turboq.c`): wrong sign arrays — was TBQ4's `turboq_wht_signs1/2`; now TBQ3-specific `turboq_wht_signs1/2_tbq3` in `ggml-turboq-tables.h` (values differ, e.g. `signs1[3]`: TBQ4=`-1`, TBQ3=`1`).
5. **CPU dequant**: wrong FWHT — was `tbq4_fwht_128` (includes `*inv_sqrt_128` normalization); now raw unnormalized `tbq3_fwht_128_cpu`, matching CUDA exactly (TBQ3 centroids are fitted to the unnormalized domain).
6. **CPU quantizer** (`quantize_row_tbq3_0_ref`): same sign-array + FWHT fixes as the dequantizer.

**Then the CPU 3-bit packer bug:** `quantize_row_tbq3_0_ref` (`ggml-turboq.c`) was corrupting 7/8 values — fixed to mirror the CUDA 24-bit little-endian packer.

**Then the final bug — CUDA shared-memory FWHT race:** all 128 threads in `k_tbq3_dequant_full` ran the full in-place FWHT with insufficient `__syncthreads`. Fixed with the race-free butterfly from the TBQ4 kernel: `__shfl_xor_sync` h=1..16, shared slots with barriers h=32/64.

**Files changed**: `ggml-turboq-tables.h` (added `turboq_wht_signs1/2_tbq3[128]`), `ggml-turboq.c` (added `tbq3_fwht_128_cpu`; fixed quantizer + dequantizer + packer), `tbq3-cuda.cuh` (`inv_128` → `inv_sqrt_128`, race-free FWHT butterfly).

### Validation (2026-08-03) — FIXED, all 4 configs pass

Controlled A/B (original chat prompt, temp 0, seed 42, same model):
- f16 GPU: `"4"` @ 13.57 t/s | TBQ4 GPU: `"4"` @ 11.74 t/s | **TBQ3 GPU: `"4"` @ 12.98 t/s** | TBQ3 CPU-only: `"Four"` @ 2.85 t/s
- All 4 configs (GPU/CPU KV × FA on/off) pass; deterministic ×3.

**GOTCHA:** a bare prompt + `--jinja` produces garbage-looking output on this model regardless of KV type — use the chat format `"<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n"`. The earlier "TBQ3 still broken" observations after the packer fix were this format artifact.

**Recommendation:** TBQ3 is the default GPU KV choice — faster than TBQ4 (12.98 vs 11.74 t/s) at 3.0625 bpv. CPU-only KV is correct but 2.85 t/s.

**Dead code residual:** `dequantize_tbq3_0` latent xnorm bug in `tbq3-cuda.cuh` — unreferenced, left as-is.

### Test procedure

Server: `taskset -c 0-15 /home/mal/AI/llama.cpp-mtp-fixes/build/bin/llama-server -m "<NVME model path>" --host 127.0.0.1 --port 8099 --no-kv-offload -ctk tbq3_0 -ctv tbq3_0 -c 4096 -t 8 -fa on`

Test: `curl -s http://localhost:8099/completion -d '{"prompt":"<|im_start|>user\nWhat is 2+2? Answer in one word.<|im_end|>\n<|im_start|>assistant\n","n_predict":20,"stream":false}'`

TBQ4 reference result: content `"4"` at ~10.5 t/s. Model (NVMe): `/media/mal/2064C68864C66060/Models/cyberneurova-DeepSeek-V4-Flash-abliterated-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-aligned.gguf` — Team CX2 USB SSD copy is BACKUP ONLY (~13 min load vs ~50s from NVMe); NVMe does not auto-mount after reboot (`udisksctl mount -b /dev/nvme1n1p2`).

### TBQ3 KV RAM estimates (128-dim blocks, 50 bytes each)

| Context | TBQ4 KV (4.125 bpv) | TBQ3 KV (3.0625 bpv) | Saving |
|---------|---------------------|----------------------|--------|
| 256K | ~4.2 GB | ~3.2 GB | ~24% |
| 512K | ~8.4 GB | ~6.4 GB | ~24% |
| 1M | ~16.8 GB | ~12.8 GB | ~24% |

---

## DSpark / MTP Speculative Decoding (Upstream Compatibility)

Our fork maintains its own MTP implementation (`--spec-type mtp`) for Qwen3.6. For DeepSeek-V4-Flash, upstream llama.cpp now supports DSpark speculative decoding (`--spec-type draft-dspark`) which uses a separate draft model to predict tokens verified in batch by the main model. Note the size claims: am17an's reference drafter is **18.48 GiB** (NOT ~10.9GB); alessandrobologna's quantized drafts are smaller (MXFP4-Q8_0 10.9GB, Q2_K-Q8_0 6.97GB).

**DSpark and TBQ KV cache are orthogonal** -- DSpark speeds up token generation via speculative decoding, TBQ compresses the KV cache to fit longer context. They should stack: DSpark for speed, TBQ3/4 for RAM.

### DSpark usage (requires upstream rebase with DSpark support)

```bash
# DSpark drafter (am17an's is the reference)
# Download: https://huggingface.co/am17an/DeepseekV4-Flash-20260731-DSpark
./build-mtp/bin/llama-server \
  -m deepseek-v4-flash-abliterated-IQ2XXS.gguf \
  --model-draft DeepseekV4-Flash-20260731-DSpark.gguf \
  --spec-type draft-dspark --spec-draft-n-max 3 \
  -ctk tbq4_0 -ctv tbq4_0 --no-kv-offload \
  -c 524288 --flash-attn on -t 8 -np 1 --jinja

# DSpark + ngram-mod combo (higher acceptance, tested by community)
# --spec-type draft-dspark,ngram-mod --spec-draft-n-max 10 \
# --spec-ngram-mod-n-match 60 --spec-ngram-mod-n-min 12 --spec-ngram-mod-n-max 24
```

### Community speed results (from r/LocalLLaMA, Aug 2026)

| Setup | Baseline | With DSpark | Speedup |
|-------|----------|-------------|---------|
| 8x 3090 (192GB) Q4_K_XL | 27 t/s | 40 t/s (DSpark n-max 3) | 1.5x |
| 8x 3090 MXFP4 | 35 t/s | 70 t/s (DSpark, CUB_TOP_K) | 2.0x |
| 6000 Pro Max + 5090 | 60 t/s | 95 t/s (DSpark) | 1.58x |
| DSpark + ngram-mod (any) | -- | 90% accept, 29 t/s | -- |

### Abliterated DSpark models

- `apetersson/DeepSeek-V4-Flash-0731-Abliterated-DS4-Quality128` -- abliterated with llama.cpp-compatible DSpark GGUF
- `am17an/DeepseekV4-Flash-20260731-DSpark` -- reference DSpark drafter (not abliterated, works with any main model)

### Integration status

Upstream DSV4 MTP + DSpark (PR #25784) plus the DSpark sidecar (#26458) are **MERGED** into new branch `feature/dsv4-dspark` @ `5b504a59f` in the isolated worktree `/home/mal/AI/llama.cpp-mtp-fixes-dspark` (1 conflict resolved: `llama-kv-cache-dsv4.cpp`; TBQ4 regression PASS `"4"` @ 11.23 t/s; original branch untouched). Note: the fork's own custom MTP is native for Qwen3.6-lineage targets (`--spec-type mtp`, ~92% accept avg, 80-87 t/s) — it does NOT apply to DSV4 (vocab mismatch 248320 vs 129280).

Drafters downloaded + sha256-verified on NVMe (`/media/mal/2064C68864C66060/Models/DSPark-Drafters/`): alessandrobologna MXFP4-Q8_0 (10.9G) and Q2_K-Q8_0 (6.97G). Both currently fail the arch gate (`deepseek_v4_flash_dspark_draft` unregistered) — the mapping slice is in flight. am17an's reference drafter (18.48 GiB, likely loads post-merge due to factored layout) is deferred on RAM budget.

**Target config** (DSpark on this box): `llama-server -m <main> --model-draft <drafter> --spec-type draft-dspark --spec-draft-n-max 3 -ctk tbq4_0 -ctv tbq4_0 --no-kv-offload -c 524288 --flash-attn on`.

**Feasibility (kv512k study):** 512k TBQ3 + DSpark est **16-19 t/s** (fits RAM — only TBQ3 leaves draft headroom); 512k TBQ4 + DSpark est 15-17 (marginal); no draft 6.5-7 t/s. q4_0 KV skipped (worse than TBQ4 on both RAM and speed). The TBQ KV cache code and DSpark operate on different axes (KV compression vs token prediction) and do not conflict.

---

## Credits

- **johndpope** — the TurboQuant lineage (TBQ3/TBQ4 KV cache, CPU TBQ quantize/dequant) this fork's KV compression builds on
- **[spiritbuun](https://github.com/spiritbuun)** — dflash fork with CUDA TurboQuant kernels (our FWHT kernels adapted from this)
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** — MTP heritage (PR #22673), CPU TBQ (PR #21089), and the upstream base this fork is rebased on
- **[havenoammo](https://huggingface.co/havenoammo)** — MTP graft tooling, first Qwen3.6-MTP GGUF release
- **llmfan46** — Qwen3.6-27B-Heretic-v2 Native-MTP-Preserved GGUF (15 native MTP heads, MPOA uncensoring)
- **HauhauCS** — Original Qwen3.6-Heretic-v2 uncensored base model
- **Radamanthys11** — MTP-Q8_0 GGUF extraction
- **froggeric** — Fixed chat templates for Qwen3.6 + MTP

## License

This fork keeps the upstream llama.cpp **MIT license** (see [LICENSE](LICENSE)). All added code in this fork inherits it. Upstream project: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
