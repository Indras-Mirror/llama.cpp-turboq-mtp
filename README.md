# llama.cpp-TurboQuant-DSV4 — Fused TBQ4 Flash Attention + MTP + DSV4 Native TBQ4

> **Fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)** (ggml-org lineage, b10217 rebase base `a7a6d0d26`) combining TurboQuant KV cache (TBQ3/TBQ4), RotorQuant, MTP speculative decoding, and — the headline of this repo — **DSV4 native TBQ4 KV cache via dequant-at-read** (Option A): DeepSeek-V4-Flash KV cache stored natively as TBQ4_0 and dequantized at read time, no Q8_0 fallback, no separate dequant pass.

**Measured on RTX 4090 24GB + 94Gi DDR4 (DeepSeek-V4-Flash-0731 abliterated, IQ2XXS, 81GB): 12.15 t/s gen @ 4K, ~4.3 t/s @ 512K, ~2.2 t/s @ 1M, VmSwap 0 at both 512K and 1M, quality 5/5.**

---

## What This Fork Adds

| Feature | Description | Status |
|---------|-------------|--------|
| **DSV4 Native TBQ4 KV Cache (dequant-at-read)** | DeepSeek-V4-Flash KV stored natively as TBQ4_0 in all four caches; dequant to F32 at read time via `dequant_k_read` at the lid/csa/hca sites. No Q8_0 fallback. 1M context on a 96GB box with zero swap. | ✅ **Headline — this work** |
| **Fused TBQ4 Flash Attention** | Quantized-KV dequant inside the FA inner loop via rotated-domain attention (centroid lookup, no intermediate F16 buffer) | Working, 82+ tok/s (Qwen3.6) |
| **MTP Speculative Decoding** | Multi-Token Prediction for Qwen3.6 (PR #22673 lineage) with 3 draft tokens per forward pass; custom implementation kept (see below) | Working, 73-98% accept |
| **CUDA TBQ4_0 Kernels** | FWHT-based TurboQuant quantize/dequant on GPU (ported from the dflash fork) | Working |
| **Tensor Sharing API** | `link_shared_tensors()` prevents 682 MiB GPU duplication of token embeddings between trunk and MTP models | Working |
| **RotorQuant (PlanarQuant + IsoQuant)** | 4 new 3-bit/4-bit KV cache types using Givens/quaternion rotations — faster dequant, better compression | Working |

---

## DSV4 Native TBQ4 via Dequant-at-Read (Option A)

Upstream of this work, DSV4 fell back from TBQ quantization to Q8_0 because the DSV4 model path had no TBQ dequant support. This commit (`1a663e2d0`) removes that fallback and stores TBQ4_0 natively in all four DSV4 KV caches (lid, csa raw+csa, hca raw+hca). A `dequant_k_read` helper casts TBQ3/TBQ4 blocks to F32 and reshapes to 4D at the three read sites; the raw ratio-0 site keeps the fused `MMA_TBQ4` path, and the Q8_0 path is byte-identical (the helper is a passthrough there).

### Why it matters

The KV cache is the only thing that scales with context. Native TBQ4 (4.25 bpv) cuts the KV working set to roughly half of Q4_0's — which is exactly what lets a 1M-context DeepSeek-V4-Flash session fit in a 96GB box without touching swap. At 512K context the server RSS is 86.3 GB with **VmSwap 0**; at 1M it is 87.5 GB, also **VmSwap 0**. No thrash, no fallback.

### Benchmark (RTX 4090 24GB, DeepSeek-V4-Flash-0731 abliterated IQ2XXS, KV in system RAM)

Server: `llama-server -m <model> --port 8099 -c <ctx> --flash-attn on -t 8 -np 1 --jinja -ctk tbq4_0 -ctv tbq4_0 --no-kv-offload`

| Config | Context | gen t/s | server RSS | VmSwap | Quality (5-prompt suite) |
|--------|---------|---------|-----------|--------|--------------------------|
| **TBQ4 native (this work)** | 4K | **12.15** | — | — | — |
| **TBQ4 native** | 32K | **11.67 / 11.56** | — | — | Q8-vs-TBQ4 A/B identical |
| **TBQ4 native** | 512K | **~4.3** | **86.3 GB** | **0** | **5/5 correct** |
| **TBQ4 native** | 1M | **~2.2** | **87.5 GB** | **0** | **5/5 correct** |
| Q4_0 K+V (mainline ref) | 512K | ~4.8 | 83.4–83.7 GB | 0→128 MB | 5/5 correct |
| Q4_0 K+V (mainline ref) | 1M | ~7.4 | 86.1–87.0 GB | 0 MB | 5/5 correct |

Quality was validated with a 5-prompt suite (math 122.7, LIS, reasoning 690, TCP/UDP, capitals) at both 512K and 1M: **5/5 correct**, and a 32K Q8-vs-TBQ4 A/B run produced identical answers. No quantization-noise regression in any config.

**Honest note on the 1M cost:** at 1M context the native-TBQ4 path generates at ~2.2 t/s, slower than mainline Q4@1M (~7.4 t/s). The gap is the dequant-at-read CPU cost plus the K-transfer cost at extreme context — that is the price of the design's premise (full 1M on a 96GB box, which Q8 cannot do at all: Q8@512K thrashes to ~1–3 t/s). For daily use, **Q4 K+V @ 512K (~4.8 t/s) is the sweet spot**; reach for TBQ4@512K when you want the extra ~3 GB of headroom, and TBQ4@1M only when you actually need the full megatoken context.

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

Context sizing: 4K → 12.15 t/s, 32K → ~11.6 t/s, 512K → ~4.3 t/s, 1M → ~2.2 t/s (see table above). For a 1M run use `-c 1048576` and expect roughly 87 GB RSS.

### MTP mode (Qwen3.6 family)

```bash
./build-mtp/bin/llama-server \
  -m your-qwen3.6-mtp.gguf \
  --spec-type mtp --spec-draft-n-max 3 \
  -ctk tbq4_0 -ctv tbq4_0 -c 262144 -ngl 99 \
  --flash-attn on --mlock -t 8 -ub 32 -np 1 --no-warmup
```

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
