# llama.cpp-mtp — MTP + Shared Tensors + CUDA TBQ4

Fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) with three enhancements:

1. **MTP (PR #22673)** — Multi-Token Prediction for Qwen3.6, 2.5× generation speed
2. **Tensor sharing** — Prevents 682 MiB GPU duplication of token embeddings
3. **CUDA TBQ4_0** — TurboQuant 4-bit KV cache with FWHT kernels

## Quick Start

```bash
git clone https://github.com/excidos/llama.cpp-mtp
cd llama.cpp-mtp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc) --target llama-server

# Run with MTP + Q4_0 KV (200K context, 92-97 tok/s)
./build/bin/llama-server \
  -m model.gguf --spec-type mtp --spec-draft-n-max 3 \
  -ctk q4_0 -ctv q4_0 -c 200000 -ngl 99 \
  --flash-attn on --mlock -ub 32 -np 1
```

## Results (RTX 4090 24GB, Qwen3.6-27B Q4_K_P)

| Config | Context | tok/s | Draft Accept | VRAM |
|--------|---------|-------|-------------|------|
| MTP + Q4_0 KV | 200K | 92-97 | 93.6% | 23.96 GB |
| MTP + Q4_0 KV | 135K | 97-103 | 93.6% | 22.4 GB |
| MTP + TBQ4_0 (ngl=63) | 262K | 45 | 72% | ~23 GB |

## What This Fork Adds

### 1. `link_shared_tensors()` API (682 MiB saved)

The MTP head loads token embeddings from the GGUF as a separate GPU allocation — a 682 MiB duplicate of the trunk model's `token_embd.weight`. New virtual method on `llama_model` lets sibling models wire shared tensors:

```cpp
// include/llama.h
LLAMA_API void llama_model_link_shared_tensors(
    struct llama_model * model,
    const struct llama_model * trunk);
```

Models opt in via `link_shared_tensors()` override. Currently implemented for `qwen35_mtp` and `qwen35moe_mtp`.

### 2. CUDA TBQ4_0 Kernels (from dflash fork)

Ported FWHT-based TurboQuant CUDA kernels into the upstream TBQ PR's type system:

- `ggml/src/ggml-cuda/tbq4-cuda.cuh` — header-only: FWHT butterfly, quantize, dequant, full-block dequant with shared memory
- `ggml/src/ggml-cuda/set-rows.cu` — TBQ4_0 SET_ROWS dispatch
- `ggml/src/ggml-cuda/cpy.cu` — TBQ4_0→F32/F16 dequant
- `ggml/src/ggml-cuda/ggml-cuda.cu` — TBQ4_0 op support

Block format: 128 elements (QK_TBQ4), fp16 norm first, 64 bytes of packed 4-bit data. Uses Lloyd-Max centroids optimized for N(0, 1/sqrt(128)) in FWHT domain.

### 3. MTP PR #22673 (base)

The foundation — MTP architecture support, speculative decoding integration, `qwen35_mtp` and `qwen35moe_mtp` model classes.

## Grafting MTP Heads

```bash
wget https://huggingface.co/havenoammo/Qwen3.6-27B-MTP-UD-GGUF/resolve/main/MTP-Q8_0.gguf
uv venv .venv --seed && source .venv/bin/activate
uv pip install gguf
python convert.py base-model.gguf MTP-Q8_0.gguf output-mtp.gguf
```

## Key Flags

- `--spec-type mtp --spec-draft-n-max 3` — enable MTP speculative decoding
- `-ctk q4_0 -ctv q4_0` — Q4_0 KV cache (best VRAM/speed balance)
- `-ctk tbq4_0 -ctv tbq4_0` — TBQ4_0 KV (near-lossless, needs ngl=63 for 262K)
- `-ub 32` — small ubatch keeps MTP compute buffer at ~712 MiB
- `-np 1` — MTP only supports single parallel slot
- `--mlock` — prevent swap under memory pressure

## Known Issues

- **Vision crashes with MTP** — upstream PR bug (reported 2026-05-06)
- **262K prompt processing** — TBQ4_0 config loads but crashes on long prefill
- **output.weight sharing** — Q4_K tok_embd ≠ Q6_K output for projection; sharing causes 0% acceptance
- **No parallel slots** — MTP requires `--parallel 1`

## Full Handoff

See [HANDOFF_TBQ4.md](HANDOFF_TBQ4.md) for complete technical documentation including:
- FWHT algorithm details and sign arrays
- CUDA kernel architecture (per-element vs full-block dequant)
- VRAM budget breakdowns
- Requantization commands
- All file modifications with descriptions

## Credits

- **havenoammo** — MTP graft tooling, first GGUF release
- **spiritbuun** — dflash fork with CUDA TurboQuant kernels
- **ggml-org/llama.cpp** — PR #22673 (MTP), PR #21089 (CPU TBQ)
- **HauhauCS** — Uncensored Qwen3.6 K_P quants
