# Qwen4-Exp (qwen4exp) — Sliding-Window Attention + MTP + PLE-on-iswa on the TurboQuant/TBQ4 fork

A llama.cpp fork that ports **Qwen3.8-Flash-Next (qwen4exp)** — a hybrid hyper-connection MoE —
and adds **sliding-window attention (SWA)** to **stop decode from collapsing at deep context**,
plus **multi-token-prediction (MTP)**, **PLE n-gram support on the SWA path**, and the **TBQ4**
4-bit KV cache.

> **Headline:** SWA keeps decode fast at 100k+ context **without losing long-range recall**.
> Measured here (TBQ4, 4096 window, 8 global layers): decode **22.8 t/s @ 12k → 15.7 t/s @ 100k**,
> and a code planted at position 0 is **recalled exactly at 100k** — the windowed layers bound
> the cost while the 8 global layers + DeltaNet recurrent path retain the context.
>
> **Which branch is this?** `qwen4-swa` = the **Qwen4-Exp (qwen4exp)** extension, and this repo's
> **default** branch. `master` = the **Qwen35 hybrid** build (Qwen3.8-27B SWA, the ~94-star
> TurboQuant/TBQ4 fork, a.k.a. the "mtp-fixes" master) on upstream base `a7a6d0d26`. `qwen4-swa` is
> **rebased on newer upstream (`d7bd3bfca`)**, so the two branches share the same *recipe* but not the
> same history — they do **not** merge cleanly. `master` is the Qwen35 build; this branch is the
> Qwen4-Exp build. Both live in `Indras-Mirror/llama.cpp-turboq-mtp`.

---

## The problem

Qwen3.8-Flash-Next is a **hybrid**: some layers dense self-attention, some **DeltaNet (recurrent)**,
joined by **hyper-connections** with a **QSA indexer**. Its dense full-attention layers attend the
**entire window every step**, so decode grows linearly with context — the classic collapse
(e.g. 60 → 18 t/s as ctx goes 6K → 84K on Qwen35-class hybrids). That is exactly what **SWA**
fixes.

## The fix — Sliding-Window Attention (qwen4exp)

SWA **windows the dense full-attention layers** so each step only attends the last `N` tokens,
bounding decode cost at deep context. It is **opt-in**: a plain GGUF (no
`qwen4exp.attention.sliding_window`) loads the clean dense path unchanged.

Layer treatment (mirrors the proven Qwen35 hybrid recipe):

| Layer type | SWA treatment |
|---|---|
| Dense full-attention (non-recurrent, non-QSA) | **Windowed** (the cheap layers) |
| `swa_global_layers` of them (default 8) | **Kept GLOBAL** — full attention, for long-range recall |
| Recurrent (DeltaNet) | Dense — not an attention layer, never windowed |
| QSA-indexed | GLOBAL (QSA already bounds them) |
| MTP draft block | Dense |

The globals are **Bresenham-distributed** across the stack so a full-attention path survives at
regular intervals, and the recurrent lanes span the whole context. **Result: a cheap windowed
layer set for speed + a sparse set of global/global-capable layers for memory.**

### Features wired in

- **MTP draft-head** (`nextn`): a separate `blk.48` draft block run via `-md`
  (`--spec-type draft-mtp`) — attention + 512-expert MoE + hyper-connections (4B params).
- **PLE on the SWA path**: the n-gram "per-layer token embedding" table works with SWA — the
  predecessor-token gather reads the iswa **base** (full-retention) cache, so PLE layers are
  never windowed.
- **TBQ4** 4-bit KV cache (K and V) — the fork's cheap long-context KV kernel.

## How it performs on this setup

Measured on the fork's TBQ4 + `--n-cpu-moe` offload setup, 4096 SWA window, 8 global layers,
temperature 0. **Real numbers**:

| Metric | Result |
|---|---|
| Prefill throughput | **200–235 tok/s** |
| Decode @ 12k ctx (SWA + TBQ4) | **22.8 t/s** |
| Decode @ ~19.5k ctx (coding battle) | **20.5 t/s** |
| Decode @ 100k ctx (SWA + TBQ4) | **15.7 t/s** |
| Bare TBQ4, no SWA (short ctx baseline) | ~21.6–25 t/s |
| **Context retention @ 100k** | **YES** — code at position 0 recalled exactly |

### The SWA advantage

- **Stops decode degradation at large contexts.** The dense full-attention layers (the dominant
  per-step cost) stop scaling with the full window; only the 8 global layers + DeltaNet scale, so
  the decode curve flattens instead of collapsing.
- **Still retains memory.** Long-range recall survives because the global layers and the recurrent
  (DeltaNet) lanes span the whole context. Verified: an exact string planted ~100k back was recalled,
  and a function defined ~14k back (outside the window) was correctly recalled in a coding task.
- **Opt-in + non-regressing.** Without the override the model runs the identical dense path. Other
  arches (Qwen35, Qwen3-Next) are untouched — the SWA layer logic lives entirely in `qwen4exp.cpp`.

## Run it

```bash
# SWA on (4096 window, 8 global layers) + TBQ4, 200k context
LLAMA_ATTN_ROT_DISABLE=1 llama-server \
  -m Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf \
  --override-kv "qwen4exp.attention.sliding_window=int:4096,qwen4exp.attention.swa_global_layers=int:8" \
  -ngl 99 --n-cpu-moe 38 -fa on -ctk tbq4_0 -ctv tbq4_0 -c 204800
```

Or with a wrapper that does the same (SWA on by default, window/globals tunable, MTP via
`NEXT_QUETZA_MTP=on`): `next-quetza-swa`.

> qwen4exp needs `LLAMA_ATTN_ROT_DISABLE=1` (upstream #21038 rotation is unsupported by this arch).
> For deep-reasoning agentic prompts use `--reasoning off` (or a matching chat template) to avoid
> the server's strict reasoning-output parser (a reasoning `<think>` block can 500 the default parse).

---

## The Rest of the Fork

### Qwen35 SWA Hybrid

Sliding-Window Attention for **Qwen3.8-27B** (arch `qwen35`, a Gated-DeltaNet hybrid). It windows
most full-attention trunk layers so **decode stays bounded at deep context**, while
`qwen35.attention.swa_global_layers` of them stay **GLOBAL** (dense) to preserve long-range recall.
Gating is balanced (Bresenham), mirroring `muse-glimmer`; the MTP draft head stays dense.

**Swept `swa_global_layers` = 2 / 4 / 8 / 13 → N = 8 is the optimum** (recalls correctly *and*
decodes fastest: **69.99 t/s @ 62K ctx, MTP acceptance 1.00**). N=2/4 fail beyond-window recall;
N=13 recalls but is slower (59.25 t/s). Tunable via `--override-kv qwen35.attention.swa_global_layers`.

| Config | decode t/s @ ~62K | beyond-window recall |
|--------|:---:|:---:|
| Pure SWA (all windowed) | ~97 | ❌ hallucinated |
| **SWA hybrid (N=8)** | **~70** | ✅ correct |
| Dense (no SWA) | ~18–20 | ✅ correct (slow) |

### Fused TBQ4 / TBQ3 Flash Attention

The FA kernel reads raw TBQ blocks directly (no separate F16 dequant):

```
Standard path:  TBQ4 → dequant → F16 buffer → FA kernel reads F16
Our fused path: TBQ4 → FA kernel reads raw bytes → centroid × norm lookup inline
```

Because the Hadamard transform is orthonormal, attention runs entirely in the rotated domain.
**TBQ3 (3.0625 bpv)** validated at **12.98 t/s** vs TBQ4's 11.74 t/s at 4.125 bpv — the recommended
default GPU KV type (deterministic ×3). Fused MMA kernel auto-dispatches via the `GGML_TYPE_TBQ3_0` gate.

### MTP Speculative Decoding

- **Qwen3.6** (`--spec-type mtp`): custom 3-draft-token MTP that predates upstream and beats it on
  every measured metric (82–93 t/s, 73–98% accept, 262K max ctx @ 24GB).
- **Qwen3.8** (`--spec-type draft-mtp`): embedded MTP head with its **own TBQ4 draft KV cache**.
  At 262K, TBQ4 draft KV (~1.1 GiB) is what fits the 24.5 GiB VRAM budget (20,637 MiB, 3.9 GiB headroom).
- Qwen3.8 MTP measured: **74.1 t/s** vs ~44–50 without, acceptance 0.52–0.67 (draft length 2.55–3.0).

**Upstream status:** upstream merged MTP (PR #22673). We keep our custom MTP — it leads on every
measured metric (head-to-head, RTX 4090 24GB, Qwen3.6-27B Q4_K_M):

| Metric | Upstream MTP | Our Fork |
|--------|:-----------:|:--------:|
| Generation speed | 71.5 tok/s | 82–93 tok/s |
| Draft acceptance | 47–89% | 73–98% (avg 92%) |
| Max context @ 24GB | ~131K | **262K** |
| Fused quant FA | ❌ separate dequant | ✅ inline in FA loop |

### DSV4 Native TBQ4 KV Cache

DeepSeek-V4-Flash KV stored natively as TBQ4_0, dequantized at read (no Q8_0 fallback, no separate
pass). At 512K, TBQ4 (9.30 t/s) beats mainline Q4@512K (~4.8 t/s) by **1.94×** using less memory;
quality **12/12** correct (no quantization-noise regression).

### RotorQuant

Replaces the FWHT butterfly with block-diagonal 2D/4D rotations — same compression as TBQ4 with an
O(d) (fully parallel) rotation instead of O(d log d). Drop-in via `-ctk`/`-ctv`. Types: `planar3_0`,
`iso3_0`, `planar4_0`, `iso4_0` (3–4 bit RV options).

---

## Key Flags

| Flag | Purpose |
|------|---------|
| `--override-kv qwen4exp.attention.sliding_window=int:N` | qwen4exp SWA window (default 4096) |
| `--override-kv qwen4exp.attention.swa_global_layers=int:N` | qwen4exp global layers (default 8) |
| `--override-kv qwen35.attention.sliding_window=int:N` | qwen35 SWA window (default 4096) |
| `--override-kv qwen35.attention.swa_global_layers=int:N` | qwen35 global layers (default 8) |
| `-ctk tbq4_0 -ctv tbq4_0` | Native TBQ4 KV cache (4.25 bpv) |
| `-ctk tbq3_0 -ctv tbq3_0` | TBQ3 KV cache (3.0625 bpv, ~24% smaller) |
| `-ctkd tbq4_0 -ctvd tbq4_0` | Draft KV cache type (independent of main) |
| `--flash-attn on` | Required for the fused TBQ4/TBQ3 path |
| `--spec-type draft-mtp` | Embedded-head MTP (Qwen3.8) |
| `--no-kv-offload` | Keep KV in system RAM (512K/1M on 24GB) |

## Build

CUDA with `sm_89` (RTX 4090):

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc) --config Release
```

For the full RotorQuant set add `-DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON`.

## Known Issues

- **Vision + MTP** crashes (upstream multimodal-handling bug) — use `--spec-type none` for vision.
- **MTP requires `--parallel 1`** (single slot).
- **7B models crash with TBQ4** (`nb1=264` is 8-byte aligned, not 16). 27B works (`nb1=528`).
- **1M native-TBQ4 gen ~2.2 t/s** — honest dequant-at-read + K-transfer cost at extreme context.

## Credits / License

- qwen4exp arch support: Daniel Han (PR 27742 upstream). MTP draft graph: JJJYmmm (PR 27739),
  reconciled by LaurentZuijdwijk. SWA hybrid recipe adapted from this fork's Qwen35 work.
- johndpope (TurboQuant lineage), spiritbuun (dflash CUDA kernels), ggml-org llama.cpp (upstream base).
- Base model: the Qwen team. Fork keeps the upstream **MIT license** ([LICENSE](LICENSE)).
