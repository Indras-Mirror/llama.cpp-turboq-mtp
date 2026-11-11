# Upstream Sync Tracker

Living document for tracking merges from upstream llama.cpp and the parent fork
(Indras-Mirror) into this fork. Update it on every batch.

## Current baseline

| Item | Value |
|---|---|
| Branch | `sync/indras-2026-09` |
| Upstream compared against | `ggml/master` @ `6716df694` (fetched 2026-10-04) |
| Commits behind upstream | ~695 (selective sync; see below) |
| Parent fork (`origin`) | `Indras-Mirror/llama.cpp-turboq-mtp` — **fully contained** (0 missing) |
| Our own remote (`myfork`) | `jtrefon/llama.cpp-turboq-mtp` — push target |

## Synced

### Batch 1 — 2026-09-25 — absorb Indras-Mirror master

`0348d3dd9 merge: Indras-Mirror TurboQ/MTP base into our fork` — contains the whole
`origin/master` history (qwen4exp SWA + TBQ/RotorQuant + DSV4/DSpark lineage).

Follow-up commits complete the merge (it did not compile / had dropped features):

- `3192e5795` core: restore build + fork features (GGML_OP_GATED_DELTA_NET_PIPE,
  draft-mtp adoption, old-MTP excision, MTP guards, helpers)
- `ca10d1ac7` server: restore hot-swap + model advertising; tokenizer-accurate
  prompt trimming (fixes the ~131k quality/perf cliff); reasoning_effort ladder

### Batch 2 — 2026-10-04 — selective upstream forward-port

| Upstream | Subject | Result |
|---|---|---|
| `2f539596c` | ggml-cpu: fix heap corruption (CACHE_LINE_SIZE) | picked |
| `2149c00f4` | ggml/gguf: fix integer overflow | picked |
| `134b2bb75` | ggml-cuda: fix cpy transposed path corrupting non-contiguous dst | picked |
| `b74f590ea` | ggml-cuda: fix divergent barrier in f16 flash attention | picked |
| `b23701f77` | cuda: fix CUB argsort corruption (in-place keys) | picked |
| `73a43d1f6` | cuda: fix races in mmid and mmf | picked |
| `187664b53` | llama-bench: fix OOB access of hf_file | picked |
| `42916d83f` | server: fix token counting API crash on sleep | picked |
| `160bd031b` | server: fix LRU hang on multiple requests same model | picked |
| `b0dcb8192` | server: fix speculation after an image | picked |
| `bf9a0ccce` | server: fix dead LLAMA_ARG_HF_REPO_FILE key in preset allow-list | picked |

### Fork-side

| Commit | Subject |
|---|---|
| `4f5648fcf` | ggml-cuda: fix misaligned-address crash in TBQ4 nstages=2 staging loader (PR #10, myfork) |

## Pending / candidates

| Upstream | Subject | Why pending |
|---|---|---|
| `6d1479c14` | ggml: fix ggml_backend_buft_get_alloc_size() guard | N/A — fix targets the newer exempting assert; our `assert(size >= ggml_nbytes(tensor))` never fires |
| `526c43b8f` | mtmd: fix GCC 15 stringop-overflow in decode_embd_batch | N/A — our tree already has `logits.resize(n_tokens)` (no `, 0`) |
| `991991118` | server: fix router eviction races | not on our path — single model + in-process swap, no router |
| rest of ~695 | Metal / Vulkan / SYCL / OpenCL / CANN / WebGPU / CI / UI / convert / gguf-py-only / docs | not relevant to this fork's path (RTX 4090, CUDA, TBQ) |

## Fork-local changes to re-verify after any sync

- `common/speculative.cpp` — draft-mtp loop breaker (`FULL_ACCEPT_LIMIT`) + spec-adapt
- `common/arg.cpp` — `--spec-type none` resets the type list (preset override)
- `tools/server/server-context.cpp` — in-process swap, preset registry, model
  advertising (`get_res_models_ext`), chat_params.vocab for trimming
- `tools/server/server-common.cpp` — tokenizer-accurate prompt trimming,
  `reasoning_effort` -> budget ladder + precedence
- `ggml/src/ggml-cuda/` — TBQ4/TBQ3 fused FA, planar/iso, concat, gated_delta_net_pipe
- `common/common.cpp` — async checkpoint capture currently gated off (sync fallback)
- `models.ini` — per-model ctk/ctv, spec-type, ctx-checkpoints

## Sync history

| Date | From | Commits | Result |
|---|---|---|---|
| 2026-08-02 | upstream (absorb, `sync/absorb-dsv4`) | 13 | superseded by Batch 1 |
| 2026-08-09 | upstream | 15 | superseded by Batch 1 |
| 2026-09-25 | parent fork (Indras-Mirror) | 1596 | merged + fixed (`0348d3dd9` + follow-ups) |
| 2026-10-04 | upstream (selective) | 11 of 14 | build + tests green |
