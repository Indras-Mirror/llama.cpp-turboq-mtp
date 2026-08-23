# llama.cpp SWA-TurboQuant — Qwen35 Sliding-Window Attention + Fused TBQ4 Flash Attention + MTP

> **Fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)** (ggml-org lineage, rebase base `a7a6d0d26`) adding **Qwen35 SWA Hybrid** (sliding-window attention for Qwen3.8-27B), a **fused quantized-KV flash attention** path (TBQ4/TBQ3 — no separate dequant pass), **MTP speculative decoding**, and **RotorQuant** KV compression.

---

## Headline: Qwen35 SWA Hybrid

Sliding-Window Attention for **Qwen3.8-27B** (arch `qwen35`, a Gated-DeltaNet hybrid). It windows most of the full-attention trunk layers so **decode stays bounded at deep context**, while `qwen35.attention.swa_global_layers` of them stay **GLOBAL** (dense, full context) to preserve long-range verbatim recall. Gating is balanced (Bresenham), mirroring `muse-glimmer`; the MTP draft head stays dense.

**Swept `swa_global_layers` = 2 / 4 / 8 / 13 → N = 8 is the optimum**: the smallest count that both recalls correctly *and* decodes fastest (**69.99 t/s @ 62K ctx, MTP draft acceptance 1.00**). N = 2/4 fail beyond-window recall; N = 13 recalls but is slower (59.25 t/s).

| Setting | Default | Notes |
|---------|---------|-------|
| `qwen35.attention.sliding_window` | `4096` | bounds *decode* (prefill unchanged) |
| `qwen35.attention.swa_global_layers` | `8` | full-attention layers kept GLOBAL |

**Tunable** via `--override-kv qwen35.attention.swa_global_layers=int:N` (and `--override-kv qwen35.attention.sliding_window=int:N`), or the wrapper flag `--swa-global-layers=N`. Wrapper: `qwen3.8-quetza-agg-swa`.

> The wrapper's default is `swa_global_layers=8`. If you load a Qwen3.8 model **bare** (no override and no `swa_global_layers` GGUF metadata), the in-tree fallback is `13` (the `muse-glimmer` ratio) — pass the override explicitly to pin your config.

### How it works

`qwen35` is a Gated-DeltaNet hybrid: ~48 DeltaNet (recurrent / linear-attention) layers plus ~16 full-attention (dense) layers sitting at `il % 4 == 3`. Decode cost grows with context because the full-attention layers read the whole KV cache. SWA windows those full-attention layers to a fixed window so **decode cost stays ~constant per token at deep context**, then keeps a small number of them GLOBAL so the model can still attend across the entire context for recall.

The **balanced (Bresenham) gating** spreads the `swa_global_layers` GLOBAL layers evenly across the stack (not a block at one end), so there is a full-context path at regular intervals — mirroring the `muse-glimmer` ratio. The MTP draft head stays dense.

```
Recurrent (DeltaNet) layers  → linear attention (not windowed)
Full-attention layers         → GLOBAL at il%4==3 where the Bresenham counter hits,
                                WINDOWED elsewhere
MTP draft head                → dense (unbounded)
```

A single global/dense path per interval is exactly what keeps decode cheap *while* retaining the long-range recall a pure window erases.

### Benchmarks (RTX 4090 24GB, Qwen3.8-27B Q4_K_P, TBQ4 KV, MTP draft)

Beyond-window recall is a decisive exact-match gate: a secret code planted ~10K tokens back must be reproduced verbatim. Decode is measured at ~62K-token context.

**`swa_global_layers` sweep:**

| N (`swa_global_layers`) | decode t/s @ ~62K | beyond-window recall | MTP accept |
|:---:|:---:|:---:|:---:|
| 2 | 51.9 | ❌ hallucinated (`ZXQ-7841`) | — |
| 4 | 54.4 | ❌ hallucinated | — |
| **8** | **69.99** | **✅ `ZXQ-9471-BRAVO`** | **1.00** |
| 13 | 59.25 | ✅ correct | — |

**Hybrid vs the extremes:**

| Config | decode t/s @ ~62K | beyond-window recall |
|--------|:---:|:---:|
| Pure SWA (all full-attn windowed) | ~97 | ❌ hallucinated |
| **SWA hybrid (`swa_global_layers=8`)** | **~70** | **✅ correct** |
| Dense (no SWA) | ~18–20 | ✅ correct (slow) |

A live wrapper run did a **43K-token prefill at ~635 t/s** and decoded **~2K tokens at ~52 t/s**.

**Summary:** SWA bounds *decode* only — prefill is unchanged and still scales with context. The hybrid keeps decode ~4× faster than dense while preserving the beyond-window recall that a pure window loses.

### Reproduce

```bash
# Smoke test: load + one short generation with SWA override + MTP draft
./qwen-swa-smoke.sh

# Deep-context stress: ~120K-token prefill, decode t/s + VRAM measurement
./qwen-swa-stress.sh
```

Decisive recall gate (the test that separates N=8 from N=2/4):

```bash
# prompt = filler*1000 + "The secret vault code is ZXQ-9471-BRAVO. " + filler*200
#          + "\n\nRecall that secret vault code. Answer with only the code."
./build/bin/llama-server -m Qwen3.8-27B-...-Q4_K_P.gguf \
  --override-kv qwen35.attention.sliding_window=int:4096 \
  --override-kv qwen35.attention.swa_global_layers=int:8 \
  --spec-type draft-mtp --spec-draft-n-max 3 --flash-attn on \
  -t 8 -ngl 99 --jinja --no-warmup --seed 3407
# expect: ZXQ-9471-BRAVO
```

**Gotchas:** the MTP graph must keep the ISWA builders (`graph_mtp`) or `swa_type != NONE` asserts. Keep `--parallel 1`.

---

## What This Fork Adds

| Feature | Description | Status |
|---------|-------------|--------|
| **Qwen35 SWA Hybrid** | Sliding-window attention for Qwen3.8-27B; keeps `swa_global_layers` GLOBAL for long-range recall. N=8 optimum (correct recall + ~70 t/s @ 62K) | ✅ **This work** |
| **Fused TBQ4 Flash Attention** | Quantized-KV dequant *inside* the FA inner loop (rotated-domain centroid lookup, no intermediate F16) | Working, 82+ t/s (Qwen3.6) |
| **MTP Speculative Decoding** | Qwen3.6 `--spec-type mtp` + Qwen3.8 `--spec-type draft-mtp` (embedded head, own TBQ4 draft KV) | Working |
| **Fused TBQ3 Flash Attention** | 3-bit KV (3.0625 bpv, ~24% smaller than TBQ4), fused inline dequant | Working |
| **DSV4 Native TBQ4 KV Cache** | DeepSeek-V4-Flash KV stored natively as TBQ4_0, dequant-at-read (Option A) | ✅ Headline |
| **RotorQuant** | 4 new 3/4-bit KV types via Givens/quaternion rotations — faster dequant | Working |
| **CUDA TBQ4/TBQ3 Kernels** | FWHT-based quantize/dequant on GPU (adapted from the dflash fork) | Working |
| **Tensor Sharing API** | `link_shared_tensors()` prevents a 682 MiB GPU token-embedding duplication between trunk and MTP | Working |

---

## The Rest of the Fork

### Fused TBQ4 / TBQ3 Flash Attention

**Nobody else fuses quantized-KV dequant into the flash attention inner loop.** The upstream TBQ4 PR (#21089) is CPU-only; the dflash fork has CUDA TBQ4 kernels but uses a separate dequant-to-F16 pass. Here the FA kernel reads raw TBQ blocks directly:

```
Standard path:  TBQ4 → dequant → F16 buffer → FA kernel reads F16
Our fused path: TBQ4 → FA kernel reads raw bytes → centroid × norm lookup inline
```

Because the Hadamard transform is orthonormal, attention runs entirely in the rotated domain — Q is pre-rotated once, K/V at quantization time, output post-rotated once. The inner loop needs only a 2-value centroid lookup per element:

```cuda
// Per byte = 2 KV elements. This is the entire dequant:
const uint8_t byte = __ldg(&blk->qs[b]);
const half lo = __float2half(d_tbq4_centroids[byte & 0xF] * norm);
const half hi = __float2half(d_tbq4_centroids[byte >> 4] * norm);
tile[...] = __halves2half2(lo, hi);
```

**TBQ3 (3.0625 bpv)** is validated (2026-08-03): GPU KV correct at **12.98 t/s** vs TBQ4's 11.74 t/s at 3.0625 vs 4.125 bpv — the recommended default GPU KV type (validated ×3, deterministic). CPU-only KV is correct but slow (2.85 t/s). Fused MMA kernel (`fattn-mma-tbq3.cuh`) auto-dispatches from `fattn.cu` via the `GGML_TYPE_TBQ3_0` gate.

### MTP Speculative Decoding

Two targets, two modes:

- **Qwen3.6** (`--spec-type mtp`): custom 3-draft-token implementation that predates upstream and beats it in every measured metric (82–93 t/s, 73–98% accept, 262K max ctx @ 24GB).
- **Qwen3.8** (`--spec-type draft-mtp`): embedded MTP head (`nextn_predict_layers=1`) with its **own TBQ4 draft KV cache**, independent of the main KV (`-ctkd tbq4_0 -ctvd tbq4_0`). At 262K the main TBQ4 KV + mmproj use 19.9 GiB of the 24.5 GiB VRAM budget; an fp16 draft KV (~4.35 GiB) doesn't fit, so the TBQ4 draft KV (~1.1 GiB) is what closes the gap (20,637 MiB total, 3.9 GiB headroom, no OOM).

**Upstream MTP status:** upstream merged official MTP via [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) (`--spec-type draft-mtp`). **We are not adopting it** — our custom MTP predates the merge and leads on every measured metric (head-to-head on RTX 4090 24GB, Qwen3.6-27B-Heretic-v2-MTP Q4_K_M):

| Metric | Upstream MTP | Our Fork | Delta |
|--------|:-----------:|:--------:|:-----:|
| Generation speed | 71.5 tok/s | 82–93 tok/s | **+15–30%** |
| Draft acceptance | 47–89% | 73–98% (avg 92%) | **+3–45 pp** |
| KV cache type | Q4_0 (4.5 bpv) | TBQ4_0 (4.25 bpv) | 6% more compression |
| Max context @ 24GB | ~131K | **262K** | **2×** |
| Fused quant FA | ❌ separate dequant pass | ✅ inline dequant in FA loop | memory + speed |
| Tensor sharing | ❌ 682 MiB duplicated | ✅ `link_shared_tensors()` | saved 682 MiB |
| RotorQuant | ❌ | ✅ planar3/iso3/planar4/iso4 | 3–4 bit KV options |

Future upstream syncs still pull non-MTP improvements (tokenizer fixes, server patches); the TBQ4 + RotorQuant + tensor sharing + MTP stack stays ours.

**Qwen3.8 MTP measured (Qwen3.8-27B-heretic-ara.i1-IQ4_XS, `-c 262144`):**

| Metric | Value |
|--------|-------|
| Draft acceptance | 0.52–0.67 |
| Mean draft length | 2.55–3.0 (drafting active) |
| Generation | **74.1 t/s** vs ~44–50 t/s without MTP (~1.5–1.7×) |
| Prompt processing | 452 t/s |
| VRAM total | 20,637 MiB (3.9 GiB headroom, no OOM) |
| Native tool_use probe | clean (`stop_reason: tool_use`) |

### DSV4 Native TBQ4 KV Cache

DeepSeek-V4-Flash KV is stored **natively as TBQ4_0** and dequantized at read time — no Q8_0 fallback, no separate dequant pass. The v2 strided-quantized concat kernel bypasses the F32 dequant at the csa/hca sites (2.16× speedup at 512K). Native TBQ4 (4.25 bpv) cuts the KV working set to roughly half of Q4_0's, so 512K fits in a 96GB box with **VmSwap 0**.

**Benchmark (RTX 4090 24GB, DeepSeek-V4-Flash-0731 abliterated IQ2XXS, KV in system RAM):**

Server: `llama-server -m <model> --port 8099 -c <ctx> --flash-attn on -t 8 -np 1 --jinja -ctk tbq4_0 -ctv tbq4_0 --no-kv-offload`

| Config | Context | gen t/s | prompt t/s | server RSS | VmSwap | Quality |
|--------|---------|---------|------------|-----------|--------|---------|
| **TBQ4 v2 native concat** | 4K | **11.61** | 18.8–20.3 | 81.1 GB | 0 | 12/12 |
| **TBQ4 v2** | 32K | **11.39** | 19.6–20.6 | 81.1 GB | 0 | 12/12 |
| **TBQ4 v2** | 256K | **10.29** | 19.1–20.4 | 81.7 GB | 0 | 12/12 |
| **TBQ4 v2** | 512K | **9.30** | 17.4–19.0 | 82.4 GB | 0 | 12/12 |
| Q4_0 K+V (mainline ref) | 512K | ~4.8 | — | 83.4–83.7 GB | 0→128 MB | 5/5 |

Quality: **12/12 correct** across all v2 configs — no quantization-noise regression. At 512K, TBQ4 (9.30 t/s) outperforms mainline Q4@512K (~4.8 t/s) by **1.94×** while using less memory. The DDR4 expert-weight floor (~85 ms) sets a hard ceiling at ~12 t/s regardless of context.

### RotorQuant

RotorQuant replaces the FWHT butterfly with block-diagonal 2D/4D rotations — the same compression ratio as TBQ4 with an O(d) rotation (fully parallel) instead of O(d log d). Drop-in compatible via `-ctk`/`-ctv`.

| Type | Bits | Block | Rotation | VRAM @ 262K |
|------|------|-------|----------|-------------|
| `tbq4_0` | 4.25 | 66 bytes/128 dims | FWHT butterfly | 4224 MiB |
| `planar3_0` | 3.0 | 50 bytes/128 dims | 2D Givens pairs | **3200 MiB** (−24%) |
| `iso3_0` | 3.0 | 50 bytes/128 dims | 4D quaternion | **3200 MiB** (−24%) |
| `planar4_0` | 4.0 | 66 bytes/128 dims | 2D Givens pairs | 4224 MiB |
| `iso4_0` | 4.0 | 66 bytes/128 dims | 4D quaternion | 4224 MiB |

### DSpark Speculative Decoding (upstream compatibility)

For DeepSeek-V4-Flash, upstream supports DSpark (`--spec-type draft-dspark`) with a separate draft model. DSpark (speed) and TBQ KV compression (RAM) are orthogonal and **stack**. This branch tracks upstream DSpark support; the fork's custom MTP is native for Qwen3.6 targets and does **not** apply to DSV4 (vocab mismatch 248320 vs 129280).

```bash
./build/bin/llama-server \
  -m deepseek-v4-flash-abliterated-IQ2XXS.gguf \
  --model-draft DeepseekV4-Flash-20260731-DSpark.gguf \
  --spec-type draft-dspark --spec-draft-n-max 3 \
  -ctk tbq4_0 -ctv tbq4_0 --no-kv-offload \
  -c 524288 --flash-attn on -t 8 -np 1 --jinja
```

---

## Key Flags

| Flag | Purpose |
|------|---------|
| `--override-kv qwen35.attention.sliding_window=int:N` | SWA window size for qwen35 (default 4096) |
| `--override-kv qwen35.attention.swa_global_layers=int:N` | qwen35 full-attn layers kept GLOBAL (default 8) |
| `-ctk tbq4_0 -ctv tbq4_0` | Native TBQ4 KV cache (4.25 bpv) — DSV4 dequant-at-read path |
| `-ctk tbq3_0 -ctv tbq3_0` | TBQ3 KV cache (3.0625 bpv, ~24% smaller) |
| `-ctk q4_0 -ctv q4_0` | Q4_0 KV (smallest footprint on DSV4 @ 512K) |
| `-ctkd tbq4_0 -ctvd tbq4_0` | Draft KV cache type (`--spec-draft-type-k/v`) — independent of main `-ctk`/`-ctv` |
| `--no-kv-offload` | Keep KV cache in system RAM (enables 512K/1M on 24GB VRAM) |
| `--flash-attn on` | Required for the fused TBQ4/TBQ3 path |
| `--spec-type mtp --spec-draft-n-max 3` | Custom MTP (Qwen3.6) |
| `--spec-type draft-mtp` | Embedded-head MTP (Qwen3.8) |
| `DSV4_CTK_COMP=tbq3_0` | Per-cache K quant: TBQ4 for raw-ratio sites, TBQ3 for compressed (mixed) |
| `--mlock` | Prevent swap under memory pressure |
| `-ub 32` | Small ubatch keeps the MTP compute buffer small |
| `-np 1` | MTP supports a single parallel slot |

## Build

CUDA with `sm_89` (RTX 4090):

```bash
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc) --config Release
```

For the full RotorQuant type set (planar3_0 / iso3_0 / planar4_0 / iso4_0), add `-DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON` to the configure step. `build/bin/llama-server` is the resulting binary.

## Known Issues

- **Vision + MTP** crashes (upstream multimodal-handling bug). Use `--spec-type none` for vision tasks.
- **`nstages=2` pipeline** produces garbled output with MTP (non-MTP is coherent). Reverted to synchronous `nstages=0` for stability.
- **`output.weight` sharing** causes 0% draft acceptance (Q4_K ≠ Q6_K quantization error accumulates). `link_shared_tensors()` shares `tok_embd` only.
- **MTP requires `--parallel 1`** (single slot — Multi-Token Prediction architecture limitation).
- **7B models crash with TBQ4** — `nb1=264` is 8-byte aligned, not 16-byte. Deferred; 27B works (`nb1=528`).
- **MoE models** may hit `vector::_M_range_check` in MTP loading if `nextn_predict_layers` metadata is missing/incorrect in the GGUF.
- **1M native-TBQ4 gen is ~2.2 t/s** — the honest dequant-at-read + K-transfer cost at extreme context.

---

## Credits

- **johndpope** — the TurboQuant lineage (TBQ3/TBQ4 KV cache, CPU quantize/dequant) this fork's KV compression builds on
- **[spiritbuun](https://github.com/spiritbuun)** — dflash fork with CUDA TurboQuant kernels (our FWHT kernels adapted from this)
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** — MTP heritage (PR #22673), CPU TBQ (PR #21089), and the upstream base this fork is rebased on
- **havenoammo** — MTP graft tooling, first Qwen3.6-MTP GGUF release
- **llmfan46** — Qwen3.6-27B-Heretic-v2 Native-MTP-Preserved GGUF (15 native MTP heads, MPOA uncensoring)
- **HauhauCS** — Original Qwen3.6-Heretic-v2 uncensored base model
- **Radamanthys11** — MTP-Q8_0 GGUF extraction
- **froggeric** — Fixed chat templates for Qwen3.6 + MTP

## License

This fork keeps the upstream llama.cpp **MIT license** (see [LICENSE](LICENSE)). All added code in this fork inherits it. Upstream project: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
