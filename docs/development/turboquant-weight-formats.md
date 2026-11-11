# TurboQuant+ weight formats (`TQ*_1S`) — port scoping

**Goal.** Near-lossless *weight* compression: WHT-rotated polar-quantized weight
block types, so a 27B can shrink **below Q4_K_M size while keeping PPL near
Q8/f16** — freeing VRAM for longer KV / larger context. This is *not* a KV-cache
format (our TBQ/RotorQuant already cover KV).

**Source.** `TheTom/llama-cpp-turboquant` ("TurboQuant+"; also mirrored at
`david568303/llama-cpp-turboquant`). Reference: DeepWiki *Quantization and Model
Optimization* + *Codec Internals*.

| Item | Detail |
|---|---|
| New types | `GGML_TYPE_TURBO2_0` / `TURBO3_0` / `TURBO4_0` (2/3/4-bit) |
| Quantize formats | `TQ3_1S` (4.00 bpw), `TQ4_1S` (5.00 bpw) |
| Rotation | FWHT in-place (group = 128) + deterministic Gaussian rotation (LCG seed 42, Gram-Schmidt orthogonalized) |
| Codebook | polar codebooks, Lloyd-Max centroids for 2/3/4-bit |
| Block | scale `d` + packed indices |
| CPU | `ggml/src/ggml-turbo-quant.c` (FWHT, centroids, quant/dequant), `ggml_vec_dot_turbo{2,3,4}_0_f32`, `type_traits_cpu` |
| Loader/quantize | `llama-model-loader.cpp` applies the WHT during quantization; `llama-quantize` emits the formats |
| Validation | `test-turbo-quant.c` (FWHT invariance, centroid accuracy, LCG determinism), `turbo-quality-gate.sh` (PPL/NMSE) |
| Claim | up to 4.6x compression with minimal PPL loss |

## Why it is not free: the rotation must be folded

WHT/Gaussian rotation is orthogonal, so `W·x = Rᵀ (R·W) x` — quantizing `R·W`
requires the runtime to rotate `x` (or fold `R` into the *producing* op, as
QuaRot does into RMSNorm / the previous matmul). That folding is
**architecture-specific** and is the real integration cost (attention QKV/O,
FFN gate/up/down, MoE experts, embeddings, output head, MTP/nextn blocks). A
loader-local rotation without folding only works if activations are rotated at
runtime, which costs an extra op per matmul.

## Phased plan

**Phase 1 — CPU types + quantize + PPL (smallest useful step).**
- add `block_turbo{2,3,4}_0` (ggml-common.h) + `GGML_TYPE_*` + traits
- port `ggml-turbo-quant.c` (FWHT, centroids, `quantize_row_*_ref`, `dequantize_row_*`)
- CPU `vec_dot` kernels + `type_traits_cpu` registration
- expose `TQ3_1S` / `TQ4_1S` in `llama-quantize`
- validate with `test-turbo-quant.c` and PPL (vs Q4_K_M / Q6_K / Q8_0)

**Phase 2 — CUDA kernels (the bulk).** dequant + `vec_dot`/MMQ integration so
inference is fast on the 4090.

**Phase 3 — conversion/imatrix.** `gguf-py` constants + quants.py, imatrix-aware
quantization (`--imatrix`) for the TQ formats.

**Phase 4 — fork integration.** rotation folding per architecture (start with
qwen35/qwen35moe), `models.ini` presets, docs, benchmark gates.

## Risks / open questions

- rotation folding per architecture (correctness); must be validated with PPL,
  not just MSE
- MoE (qwen35moe) and fused ops (our TBQ FA / gated-delta-net) interactions
- token embeddings / output head are usually kept higher precision
  (`_1S` = only the "1" sensitive group rotated?) — confirm the exact `_1S`
  semantics from the source before porting
- interaction with our existing rotation infra (`ggml-turboq.c` FWHT +
  planar/iso) — reuse where possible

## Immediate next step

Fetch `ggml/src/ggml-turbo-quant.c` + `ggml-common.h` + `llama-quant.cpp`
hunks from TheTom/llama-cpp-turboquant, diff against our tree, and do a
CPU-only spike (Phase 1) to measure PPL at equal bpw vs Q4_K_M/Q5_K_M.
