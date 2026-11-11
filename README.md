# llama.cpp-turboq-mtp

Maintained by Jack Trefon — [trefon.com](https://trefon.com)

This repository is Jack Trefon's fork of [Indras-Mirror/llama.cpp-mtp](https://github.com/Indras-Mirror/llama.cpp-mtp), which is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). `master` follows the Indras-Mirror history through 17 May 2026 (`0bc0a270d`). The license is unchanged: MIT, copyright The ggml authors ([LICENSE](LICENSE)).

TurboQuant, the fused KV flash-attention kernel, RotorQuant, multi-token prediction, and every throughput or context-length figure in the [upstream writeup](#upstream-writeup) below come from that upstream line. See [Upstream work](#upstream-work). The list that follows is the work added on this fork, checked against the commits and code on `master`.

## What this fork adds

### Consumer NVIDIA GPU performance

CUDA work for consumer NVIDIA GPUs. This is separate from TurboQuant and from the fused KV flash-attention kernel, which are upstream.

- Gated DeltaNet prefill uses `GGML_OP_GATED_DELTA_NET_PIPE` (`ggml/src/ggml-cuda/gated_delta_net_pipe.cu`). One launch per layer holds the recurrent state in registers and runs the chunk loop there, in place of the per-chunk matmul chain in `src/models/delta-net-base.cpp`. The kernel still does those matmuls as plain loops.
- Prompt checkpoints can be copied device-to-host on `cudaStreamPerThread` into a two-slot pinned staging ring. A `cudaLaunchHostFunc` callback then copies them into pageable memory, so prefill is not blocked on the readback. Builds without CUDA keep the synchronous path.
- `LLAMA_KV_PAD_MIN` (default 4096) is the floor for the KV length used when the CUDA graph is captured. The pad in this code before that change was 256, so the graph was recaptured on a shorter stride. This is a pad change. There is no speed table for it in this repository.
- `--spec-adapt` watches an exponential moving average of MTP draft acceptance and, below 0.35, drops the draft depth to `--spec-draft-n-min`. Above 0.50 it returns to `--spec-draft-n-max`.

### Sliding-window attention in gated DeltaNet (Qwen)

Qwen3.5 in this tree is a hybrid: gated DeltaNet layers plus sliding-window attention. The MTP shadow context was created with `n_rs_seq = 0`. On a rejected draft the KV rows were removed and the DeltaNet recurrent state was left in place, so the draft state drifted. At long context the draft head repeated one token. The shadow context now uses the same recurrent rollback depth as the target. Qwen3.5 supports partial rollback, so that depth is not clamped. Rejected draft rows are also removed on the target context, which covers the M-RoPE position check on Qwen3.8. When target memory is removed, copied, or shifted, the MTP pending stash is cleared so the draft head is re-seeded.

Context checkpoints on a hybrid model with sliding-window attention called a state-copy path that cannot serialize the SWA cache (the tensor backend is null). That aborted prompts which span more than one prefill batch. Checkpoints are now created only when speculative decoding, MTP, or cache reuse will roll the cache back.

The sliding-window width, and which Qwen layers stay full attention, are set on the Indras-Mirror line (Excidos and Indras-Mirror). Those commits are not in this branch.

### Large-context stability

On top of the DeltaNet rollback above, long contexts were aborting or spinning:

- An MTP draft that indexed past the allocated KV cache asserted inside `ggml_backend_tensor_get`. The draft path returns early instead.
- With the context full, the server looped on "decode: failed to find a memory slot". If the accumulated position is already at `n_ctx`, the slot returns `exceed_context_size` ("accumulated context is full"). A generation that would run past the context with context-shift disabled returns that same error type.
- After rejection rollback started trimming the target with `llama_memory_seq_rm` directly, the MTP pending stash still pointed past the trimmed rows. The draft head then stayed desynchronized. The commit note records that loop at about 227K tokens. The stash is reset from `llama_context_seq_rm`, `seq_cp`, and `seq_add`, and after a rejection trim in `accept()`.
- After 60 MTP rounds that accept every drafted token, the next round samples one token from the target model only. That gives end-of-sequence a chance to win during a tool-call run.

The recurrent-state commit records a 1500-token generation around 95K context and a session through about 200K without the repeated-token lock-up. Those are stability checks from the commit messages. This fork does not add a throughput number for them.

### MLX stability

This repository has no MLX backend, no MLX sources, and no commit that mentions MLX. The item is listed because it is part of the maintainer's stated scope. There is no MLX change here to build or review.

### Hot model reload

The server loads an INI preset file (`--models-preset`; `models.ini` is an example) and can swap models without a restart.

- `POST /models/load` with `{"model": "<preset-key-or-alias>"}` runs the swap on the main loop and returns when the new model is loaded.
- The same swap runs when `model` on the request names a preset that is not the loaded one, on `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, and the Anthropic `/v1/messages` route.
- The swap waits while a slot is busy, drops tasks that were tokenized with the old vocabulary, drops the prompt cache, and tears down speculative state, the MTP context, and the multimodal context before the model. If the new load fails, the previous model is loaded again.
- Preset keys are applied on top of the CLI arguments captured at startup, so context size, KV cache type, mmproj, and spec type come from that preset.
- `GET /v1/models` lists each preset with `status` (`loaded` or `unloaded`) and `context_length` taken from the preset's context-size field.
- `swap-model.sh` is a client for the load endpoint.

### OpenAI-compatible reasoning and context length

[llama.cpp's server](https://github.com/ggml-org/llama.cpp) already exposes OpenAI-style routes. This fork changes how reasoning mode and context length are applied on those routes:

- `reasoning_effort` accepts `none` and `off` (thinking disabled), `minimal` (512 tokens), `low` (1024), `medium` (2048), `high` (4096), `xhigh` (8192), and `max`, `ultra`, or `unlimited` (no cap). `none` and `off` also set `enable_thinking` off.
- A numeric `thinking_budget_tokens` or `reasoning_budget_tokens` on the request overrides the word. Either one overrides the server default from `--reasoning-budget` or the preset. With none of these fields set, the configured budget is used.
- Before the chat template is rendered, messages that do not fit in `n_ctx` minus a margin are dropped from the oldest non-system turns. The margin is the reasoning budget plus 8192, and it is capped at one quarter of `n_ctx`, so an 8K context is not trimmed to an empty budget. The system prompt stays. The last message stays even when that single message is larger than the estimate, so the template still receives the user query. The estimate includes message text and tool-call name, arguments, and id.
- The Qwen templates under `models/templates` emit `reasoning_content` whenever that field is present, including thinking that sits on a tool turn before the last user message.
- Tool-call arguments that are not valid JSON are replaced with `{}` and logged, instead of failing the request.

That is the compatibility work on `master`. It is the reasoning-mode and context-length behavior above, on the existing OpenAI-compatible routes.

## Upstream work

- **TurboQuant and the fused KV kernel.** On the Indras-Mirror tree, Excidos added the fused TBQ4 flash-attention path: the kernel reads raw TBQ4 K/V blocks in the attention loop (`ggml/src/ggml-cuda/fattn-mma-tbq4.cuh`, first committed May 2026). The same history's README says the CUDA TurboQuant FWHT kernels were adapted from [spiritbuun](https://github.com/spiritbuun)'s dflash fork. CPU TurboQuant in upstream llama.cpp is [ggml-org/llama.cpp#21089](https://github.com/ggml-org/llama.cpp/pull/21089).
- **RotorQuant** (planar and iso KV types) was integrated on that fork by Excidos.
- **Multi-token prediction.** Upstream llama.cpp merged draft-MTP in [ggml-org/llama.cpp#22673](https://github.com/ggml-org/llama.cpp/pull/22673). The Indras-Mirror fork kept its own MTP implementation and published the comparison against that upstream MTP.
- **A TBQ4 crash fix on this repo, same kernel.** Jean-Maximilien Cadic fixed a misaligned-address crash in the TBQ4 `nstages=2` staging loader ([jtrefon/llama.cpp-turboq-mtp#10](https://github.com/jtrefon/llama.cpp-turboq-mtp/pull/10)). That patch is a fix to the upstream kernel.

Reported speeds in the writeup below, including 80–179 tok/s and the 262K-context RTX 4090 tables, were published by Indras-Mirror and Excidos. They were not re-measured for this repository. [indrasmirror.au](https://indrasmirror.au) is that upstream author's site. The long writeup is [blog-mtp-shared-tensors-200k](https://indrasmirror.au/blog-mtp-shared-tensors-200k.html).

Later Indras-Mirror work (Qwen sliding-window hybrids, Qwen4-Exp SWA, DSV4 KV) is on the `sync/indras-2026-09` branch and is not in `master`.

## Upstream writeup

From here to the credits is the Indras-Mirror / Excidos README as carried in this tree, with a few paragraphs added on this branch (the async-checkpoint, KV-pad, and `--spec-adapt` rows, the production-configuration block, and the 12 July 2026 checkpoint note). Those paragraphs match the CUDA items above. In this writeup, "this fork" and "our fork" mean the Indras-Mirror tree. Every tok/s figure and every context length here is an upstream-reported number, including where this branch repeated it. The quick start clones `Indras-Mirror/llama.cpp-mtp`. This repository is [jtrefon/llama.cpp-turboq-mtp](https://github.com/jtrefon/llama.cpp-turboq-mtp).

### llama.cpp-mtp — fused TBQ4 flash attention, MTP, shared tensors

> Fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) with fused TurboQuant flash attention. The FA kernel reads raw TBQ4_0 K/V blocks from global memory and dequantizes with a centroid lookup in the FWHT-rotated domain. There is no separate dequant pass and no intermediate F16 buffer. This paragraph is the upstream description.

**Upstream-reported figure, not measured on this fork:** 80-179 tok/s decode (325 effective) with lossless 4.125 bpv KV cache at 262K context on RTX 4090 24GB.

## Upstream feature list

| Feature | Description | Status |
|---------|-------------|--------|
| **Fused TBQ4 Flash Attention** | Quantized-KV dequant inside the FA inner loop via rotated-domain attention | Working, 82+ tok/s |
| **MTP Speculative Decoding** | Multi-Token Prediction for Qwen3.6 (PR #22673) with 3 draft tokens per forward pass | Working, 50–68% accept (head-quality limited; see [PROPOSAL](PROPOSAL-mtp-acceptance.md)) |
| **CUDA TBQ4_0 Kernels** | FWHT-based TurboQuant quantize/dequant on GPU (ported from dflash fork) | Working |
| **Tensor Sharing API** | `link_shared_tensors()` prevents 682 MiB GPU duplication of token embeddings between trunk and MTP models | Working |
| **RotorQuant (PlanarQuant + IsoQuant)** | 4 new 3-bit/4-bit KV cache types using Givens/quaternion rotations — faster dequant, better compression, 5.3x faster prefill | ✅ New! |
| **Async Checkpoint Capture** | Prompt checkpoints captured via async D2H into a pinned staging ring (≤2 slots, ~300 MiB pinned) so the 32× ~149 MiB state readbacks overlap with prefill compute instead of stalling it | ✅ New! |
| **KV Pad Floor** | Configurable via `LLAMA_KV_PAD_MIN` env (default 4096); 16× fewer graph re-captures than upstream hardcoded 256 | Working |
| **Adaptive Draft Depth** | `--spec-adapt` throttles MTP draft depth on pathologically low acceptance (<35%); guards against wasted compute during dips | Working |
| **P-Core Binding** | Architecture proposal for pinning hot main thread to dedicated P-core on i9-12900K; see [ARCHITECTURE](ARCHITECTURE-pcore-binding.md) | Proposed |

### Documentation / Proposals

| Document | Purpose |
|----------|---------|
| [PROPOSAL-mtp-acceptance.md](PROPOSAL-mtp-acceptance.md) | Root-cause analysis of 50% acceptance gap + four ranked levers (temp, adaptive depth, EAGLE3, FastMTP) |
| [FASTMTP-RETRAIN.md](FASTMTP-RETRAIN.md) | Handoff for model owner: retrain MTP head with FastMTP objective (+82% accept) |
| [ARCHITECTURE-pcore-binding.md](ARCHITECTURE-pcore-binding.md) | P-core affinity + OpenMP isolation for single-threaded ggml build path |
| [CHANGES-UNCOMMITTED.md](CHANGES-UNCOMMITTED.md) | Revert guide for all local changes |

### Target Audience & Production Configuration

This fork is tuned for **Qwopus3.6-27B-v2-MTP** (or compatible Qwen3.5 27B) on a **single RTX 4090 (24 GiB)** at **262K context** with **lossless TBQ4 KV cache**.

The production server (`llama-turboq.service`) runs:

```
--temp 0.6 --top-p 0.95 --min-p 0.05     # fixes agent tool-call loops, raises decode 62→82 tok/s (+31%)
--reasoning on --reasoning-budget 8192    # thinking tokens inside <think> tags
--jinja                                    # model's native Qwen3.5 chat template
--spec-type mtp --spec-draft-n-max 3      # MTP speculative decoding
--spec-adapt                              # adaptive draft-depth guard
-ctk tbq4_0 -ctv tbq4_0                  # TBQ4 lossless KV (4.125 bpv, ~4 GiB @ 262K)
--flash-attn on                           # fused dequant + FA kernel
-c 262144 -ngl 99                         # full context, all layers on GPU
```

Upstream-reported decode throughput at 262K context: **~82 tok/s** (temp 0.6, MTP 3 drafts).
The remaining throughput headroom (target: 70–95% MTP acceptance) requires model-side retraining — see [FASTMTP-RETRAIN.md](FASTMTP-RETRAIN.md).

### RotorQuant — Next-Gen KV Cache Compression

**RotorQuant replaces the FWHT butterfly with block-diagonal 2D/4D rotations.** Same compression ratio as TBQ4 but with O(d) rotation (fully parallel) instead of O(d log d) Hadamard. Drop-in compatible via `-ctk`/`-ctv` flags.

#### Available Types

| Type | Bits | Block | Rotation | VRAM @ 262K |
|------|------|-------|----------|-------------|
| `tbq4_0` | 4.125 | 66 bytes/128 dims | FWHT butterfly | 4224 MiB |
| `planar3_0` | 3.125 | 50 bytes/128 dims | 2D Givens pairs | **3200 MiB** (-24%) |
| `iso3_0` | 3.125 | 50 bytes/128 dims | 4D quaternion | **3200 MiB** (-24%) |
| `planar4_0` | 4.125 | 66 bytes/128 dims | 2D Givens pairs | 4224 MiB |
| `iso4_0` | 4.125 | 66 bytes/128 dims | 4D quaternion | 4224 MiB |

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

### Recent Changes (Jul 12, 2026)

- **Async checkpoint capture (RTX-4090 / NVIDIA-dedicated)**: Prompt checkpoints (32 × ~149 MiB state readbacks during a 262K prefill) previously stalled the prefill compute stream — `llama_state_seq_get_data_ext` did a full `ctx->synchronize()` plus one blocking `ggml_backend_tensor_get` per state tensor. Now capture is **asynchronous**: the state is copied device→host into a **pinned staging ring** (2 slots, ~300 MiB pinned RAM max) on `cudaStreamPerThread`, overlapping with subsequent prefill compute. A CUDA host callback (`cudaLaunchHostFunc`) copies staging→pageable store and frees the slot within milliseconds, so pinned RAM never exceeds ~300 MiB regardless of checkpoint count. The data is only synchronized (event wait) when a checkpoint is actually *consumed* (cache reuse), where the GPU is idle. Non-CUDA builds fall back to the synchronous path. Verified: checkpoint create + restore work correctly end-to-end.

### Async Checkpoint Capture — Design

```
create_checkpoint (prefill loop, no stall)
   └─ update_tgt/update_dft
        ├─ llama_state_seq_get_data_ext_async  → async D2H into pinned staging (cudaStreamPerThread)
        └─ cudaLaunchHostFunc(cb): staging → pageable data_tgt/data_dft, release staging slot

follow-up request (GPU idle)
   └─ load_tgt/load_dft
        └─ if async still in flight: cudaStreamSynchronize(cudaStreamPerThread)  // cheap, GPU idle
        └─ llama_state_seq_set_data_ext  (restore)
```

Pinned RAM is bounded to a 2-slot ring (`ckpt_staging_pool` in `common/common.cpp`) because the RTX 4090 host has little headroom — we deliberately avoid pinning all 32 checkpoints (~4.7 GB).

## Upstream MTP Status

**⚠️ As of May 16, 2026:** Upstream `ggml-org/llama.cpp` merged official MTP support via [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) (`255582687`). This is 20 commits ahead of our current sync point (`ec562eb67`). The upstream implementation uses `--spec-type draft-mtp` and `COMMON_SPECULATIVE_TYPE_DRAFT_MTP`.

**Our fork** uses a custom MTP implementation (`--spec-type mtp`, `COMMON_SPECULATIVE_TYPE_MTP`) that predates the upstream merge. Both implementations support Qwen3.6 MTP heads, but ours includes additional features (TBQ4 fused FA, RotorQuant, tensor sharing, context checkpoints for MTP).

**We are keeping our custom MTP implementation.** Head-to-head testing (see below) shows our fork exceeds upstream in every performance metric. Future upstream syncs will pull non-MTP improvements (tokenizer fixes, server patches, etc.) but our TBQ4 + RotorQuant + tensor sharing + MTP stack will remain the core. This fork is stable and production-tested (92% draft acceptance, 29-turn continuous session without errors, 262K context on 24GB VRAM).

### Upstream vs Fork — Head-to-Head (May 17, 2026)

We tested upstream `draft-mtp` (PR #22673, merged May 16) against our custom MTP on identical hardware (RTX 4090 24GB, Qwen3.6-27B-Heretic-v2-MTP Q4_K_M):

| Metric | Upstream MTP | Our Fork | Delta |
|--------|:-----------:|:--------:|:-----:|
| **Generation speed** | 71.5 tok/s | 82-93 tok/s | **+15-30%** |
| **Draft acceptance** | 47-89% | 73-98% (avg 92%) | **+3-45 pp** |
| **KV cache type** | Q4_0 (4.5 bpv) | TBQ4_0 (4.125 bpv) | 8% smaller |
| **Max context @ 24GB** | ~131K | **262K** | **2x** |
| **262K context VRAM** | ❌ Won't fit (needs 32 GB) | ✅ ~20 GB | — |
| **Fused quant FA** | ❌ Separate dequant pass | ✅ Inline dequant in FA loop | Memory + speed |
| **Tensor sharing** | ❌ 682 MiB duplicated | ✅ `link_shared_tensors()` | Saved 682 MiB |
| **RotorQuant** | ❌ | ✅ planar3/iso3/planar4/iso4 | 3-4 bit KV cache options |
| **Multi-turn cache** | ✅ Checkpoints (native) | ✅ Checkpoints (our fix) | Same mechanism |

**Why we are not adopting upstream MTP:** Upstream's implementation is a clean starting point, but our fork's TBQ4 fused flash attention + RotorQuant + tensor sharing stack delivers significantly higher performance, 2x the context capacity, and better draft acceptance. Upstream MTP lacks the KV cache compression needed for 262K context on consumer GPUs — the TBQ4_0 format (4.125 bpv) is the critical differentiator, using only 4.2 GB for KV cache vs 16.4 GB for upstream's Q4_0 at 262K.

**Upstream commits assessed (20 total, May 14-17):** Of 40+ commits since our sync point, the vast majority are AMD/Vulkan/WebGPU/Hexagon/SYCL backend changes, CI fixes, web UI updates, and Docker configs — none relevant to our CUDA single-GPU setup. The few potentially useful commits (Qwen3.5 tokenizer improvements, reasoning-budget deep-copy fix, server log reduction) are minor quality-of-life improvements that do not warrant destabilizing our stable build. They will be picked up in the next scheduled sync.

## Results (RTX 4090 24GB, Qwen3.6-27B-Heretic-v2-MTP Q4_K_M)

| Config | Context | KV Cache | tok/s | Draft Accept | VRAM |
|--------|---------|----------|-------|-------------|------|
| **MTP + Fused TBQ4 FA (May 11)** | **262K** | **TBQ4_0 (4.125 bpv)** | **179.4** | **81.4%** | **~20 GB** |
| **MTP + Fused TBQ4 FA** | **262K** | **TBQ4_0 (4.125 bpv)** | **80-87** | **73-93%** | **~20 GB** |
| MTP + Fused TBQ4 FA | 200K | TBQ4_0 (4.125 bpv) | 82-87 | 73% | ~20 GB |
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

# Fused TBQ4 FA + MTP (80-87 tok/s at 262K, lossless 4.125 bpv KV)
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
struct block_tbq4_0 {      // 66 bytes per 128 elements (4.125 bits per value)
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
| `-ctk tbq4_0 -ctv tbq4_0` | Fused TBQ4 KV cache (lossless, 4.125 bpv) |
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
