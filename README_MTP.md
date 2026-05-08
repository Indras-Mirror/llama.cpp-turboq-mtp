# llama.cpp-mtp — Fused TBQ4 Flash Attention + MTP + Shared Tensors

Fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) with **fused TurboQuant flash attention** — the FA kernel reads raw TBQ4_0 K/V blocks directly from global memory and dequants via centroid lookup in the FWHT-rotated domain. No separate dequant pass, no intermediate F16 buffer.

**82+ tok/s with lossless 4.25 bpv KV cache at 200K context on RTX 4090 24GB.**

## Results (RTX 4090 24GB, Qwen3.6-27B Q4_K_M)

| Config | Context | KV Cache | tok/s | Draft Accept | VRAM |
|--------|---------|----------|-------|-------------|------|
| **MTP + Fused TBQ4 FA** | **200K** | **TBQ4_0 (4.25 bpv)** | **82–87** | **73%** | **~20 GB** |
| MTP + Q4_0 KV | 200K | Q4_0 (4.5 bpv) | 92–97 | 93.6% | 23.96 GB |
| MTP + Q4_0 KV | 135K | Q4_0 (4.5 bpv) | 97–103 | 93.6% | 22.4 GB |
| Baseline (no MTP, Q4_0) | 200K | Q4_0 | ~40 | — | 23.96 GB |

The fused TBQ4 path trades some draft acceptance (73% vs 93.6%) for significantly lower VRAM — the 4.25 bpv lossless compression means ~4 GB less KV cache at 200K, leaving headroom for even longer contexts.

## What Makes This Novel

**Nobody else has fused quantized-KV dequant into the flash attention inner loop.** The upstream TBQ4 PR (#21089) is CPU-only. The dflash fork (spiritbuun) has CUDA TBQ4 kernels but uses `nstages=0` with a separate dequant-to-F16 pass before FA. Our kernel reads raw TBQ4 blocks directly:

```
Standard path:  TBQ4 → dequant → F16 buffer → FA kernel reads F16
Our fused path: TBQ4 → FA kernel reads raw bytes → centroid×norm lookup inline
```

The key insight: since FWHT is orthonormal, attention can operate entirely in the rotated domain. Q is pre-rotated once (separate kernel), K/V are pre-rotated at quantization time (SET_ROWS), and the output is post-rotated once. The inner loop only needs a 2-value centroid lookup per element — no FWHT, no precomputed tables.

### Optimizations Applied

1. **Column-group access pattern** — threads process one column across all rows instead of one row per thread, nearly doubling bandwidth utilization
2. **Direct centroid lookup** — instead of precomputing all 16 centroid×norm products per thread, we look up only the 2 values actually needed per byte (saving 14 FP muls + 14 float-to-half conversions)
3. **Rotated-domain attention** — FWHT runs only twice total (Q rotate, output un-rotate), not per-element in the inner loop

## Quick Start

```bash
git clone https://github.com/Indras-Mirror/llama.cpp-mtp
cd llama.cpp-mtp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc) --target llama-server

# Fused TBQ4 FA + MTP (82+ tok/s at 200K, lossless KV)
./build/bin/llama-server \
  -m model.gguf --spec-type mtp --spec-draft-n-max 3 \
  -ctk tbq4_0 -ctv tbq4_0 -c 200000 -ngl 99 \
  --flash-attn on --mlock -t 8 -ub 32 -np 1 --no-warmup

# Or with Q4_0 KV for max speed (92-97 tok/s, higher VRAM)
./build/bin/llama-server \
  -m model.gguf --spec-type mtp --spec-draft-n-max 3 \
  -ctk q4_0 -ctv q4_0 -c 200000 -ngl 99 \
  --flash-attn on --mlock -ub 32 -np 1
```

## Architecture

### Fused TBQ4 Flash Attention Pipeline

```
1. k_tbq4_rotate_input    → Pre-rotate Q via FWHT (separate kernel, 128-thread warp shuffle)
2. Fused FA kernel         → Read raw TBQ4 blocks from GMEM, centroid×norm dequant inline
3. k_tbq4_rotate_output   → Post-rotate VKQ back to original domain
```

K/V are pre-rotated at SET_ROWS time — `quantize_f32_tbq4_0_block` calls `tbq4_rotate_forward` before quantization. Everything in the FA inner loop operates in the rotated domain.

### TBQ4_0 Block Format

```
struct block_tbq4_0 {      // 66 bytes per 128 elements (4.25 bpv)
    ggml_half d;            // corrected L2 norm (2 bytes)
    uint8_t qs[QK_TBQ4/2]; // packed 4-bit centroids (64 bytes)
};
```

Centroids are Lloyd-Max optimal for N(0, 1/√128) in the FWHT-transformed domain. 16 centroids stored in `__constant__` memory, indexed by 4-bit nibbles.

### Tensor Sharing — `link_shared_tensors()` API

The MTP head loads `token_embd.weight` as a separate 682 MiB GPU allocation — a duplicate of the trunk model's copy. Our `link_shared_tensors()` virtual method lets sibling models wire shared tensors after loading:

```cpp
// include/llama.h
LLAMA_API void llama_model_link_shared_tensors(
    struct llama_model * model,
    const struct llama_model * trunk);
```

Implemented for `qwen35_mtp` and `qwen35moe_mtp`. Saves 682 MiB with no quality impact.

## Files Added/Modified

### Fused TBQ4 Flash Attention (the novel part)
| File | Purpose |
|------|---------|
| `ggml/src/ggml-cuda/fattn-mma-tbq4.cuh` | **NEW** — Fused tile loader, rotation kernels, centroid lookup |
| `ggml/src/ggml-cuda/fattn-mma-tbq4-launch.cuh` | **NEW** — Template launcher, shmem calculation |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | Modified — TBQ4 guards in iter function (4 locations) |
| `ggml/src/ggml-cuda/fattn.cu` | Modified — TBQ4 dispatch + rotation kernel calls |
| `template-instances/fattn-mma-tbq4-instance-ncols2_{1,2,4,8}.cu` | **NEW** — Template instantiations |

### CUDA TBQ4_0 Kernels
| File | Purpose |
|------|---------|
| `ggml/src/ggml-cuda/tbq4-cuda.cuh` | **NEW** — FWHT, quantize, dequant, full-block dequant |
| `ggml/src/ggml-cuda/set-rows.cu` | TBQ4_0 SET_ROWS dispatch |
| `ggml/src/ggml-cuda/cpy.cu` | TBQ4_0→F32/F16 dequant |

### Tensor Sharing Infrastructure
| File | Purpose |
|------|---------|
| `include/llama.h` | `llama_model_link_shared_tensors()` public API |
| `src/llama-model.h` | Virtual method + `get_tensor_mutable()` |
| `src/llama-model.cpp` | Implementation |
| `src/models/qwen35_mtp.cpp` | Qwen3.5 MTP sharing |
| `src/models/qwen35moe_mtp.cpp` | Qwen3.5 MoE MTP sharing |
| `tools/server/server-context.cpp` | Call site after MTP model load |

## Grafting MTP Heads

```bash
# Download MTP head GGUF (457 MB, only blk.64.* tensors)
wget https://huggingface.co/havenoammo/Qwen3.6-27B-MTP-UD-GGUF/resolve/main/MTP-Q8_0.gguf

uv venv .venv --seed && source .venv/bin/activate
uv pip install gguf
python convert.py base-model.gguf MTP-Q8_0.gguf output-mtp.gguf
```

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

## Known Issues

- **Vision crashes with MTP** — upstream PR bug (reported 2026-05-06)
- **nstages=2 pipeline** — pipelined staging produces garbled output; reverted to nstages=0
- **output.weight sharing** — Q4_K tok_embd ≠ Q6_K output; sharing causes 0% acceptance
- **No parallel slots** — MTP requires `--parallel 1`

## Credits

- **havenoammo** — MTP graft tooling, first GGUF release
- **spiritbuun** — dflash fork with CUDA TurboQuant kernels (our FWHT kernels adapted from this)
- **ggml-org/llama.cpp** — PR #22673 (MTP), PR #21089 (CPU TBQ)
- **HauhauCS** — Uncensored Qwen3.6 K_P quants
- **Radamanthys11** — MTP-Q8_0 GGUF extraction
- **froggeric** — Fixed chat templates for Qwen3.6 + MTP

## Full Technical Documentation

See [HANDOFF_TBQ4.md](HANDOFF_TBQ4.md) for complete internals including FWHT algorithm, guard structure, bug tracker (15 bugs fixed), and VRAM budget breakdowns.

## Blog Post

Detailed writeup with benchmarks and architecture diagrams: [Fused TBQ4 Flash Attention: 82 tok/s at 200K Context](https://indrasmirror.au/blog-mtp-shared-tensors-200k.html)
