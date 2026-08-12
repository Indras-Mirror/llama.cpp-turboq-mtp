# TurboQuant KV cache types — support matrix, quality, benchmarks

Reference for the TBQ3_0 / TBQ4_0 KV cache types in this fork.

## Block formats

Both types quantize 128-element blocks with a Walsh-Hadamard rotation:

| type | bits/value | block | layout | notes |
|---|---|---|---|---|
| `tbq3_0` | 3.0 | 50 B | `half d` + 48 B packed 3-bit indices | 8 Lloyd-Max centroids (FWHT domain) |
| `tbq4_0` | 4.25 | 66 B | `half d` + 64 B packed 4-bit indices | 16 Lloyd-Max centroids (FWHT domain) |

Values are stored in the **WHT-rotated domain**: quantize applies `s1 -> WHT128 -> s2`
(plus norm correction `d = ||x|| / ||centroids||`), dequantize inverts. Because WHT
is orthonormal, attention can run entirely in the rotated domain (Q pre-rotated,
K/V read as centroids) — that is the fused CUDA path. The Metal backend dequantizes
to the original domain before FA (no fused kernel yet).

## Backend support

| backend | quantize/write | attention | status |
|---|---|---|---|
| CPU | ref codec (`ggml-turboq.c`) | vec_dot (dequant inline) | works |
| CUDA | `tbq4-cuda.cuh` | fused MMA FA (`fattn-mma-tbq4.cuh`) | works, production (RTX 4090, 262K ctx) |
| Metal | cpy/set_rows kernels (`ggml-metal.metal`) | dequant fallback (cast F32, FA on f16) | works since 2026-08-11 (PR #4) |
| Vulkan/SYCL/etc | — | — | unsupported |

## Quality — 3-bit is the risk axis

Measured 2026-08-11 (Qwen3.5-4B, CPU, same prompt/params):

| KV config | result |
|---|---|
| q4_0 K + q4_0 V | clean |
| q4_0 K + tbq4_0 V | clean |
| q4_0 K + tbq3_0 V | garbage (`.class` repetition) — **was a packing bug**, see below |
| tbq3_0 K + tbq3_0 V | garbage (V-dominant) |

Two independent causes were conflated:

1. **Packing bug (fixed)**: the 3-bit packer wrote `idx >> (8 - bit)` (negative-shift UB)
   for `bit >= 8` and corrupted `qs[0]` for values 3..7 of each 8-group. Every backend
   that used the reference codec produced garbled tbq3_0 KV. Fixed with 24-bit
   little-endian lanes (PR #4). After the fix, tbq3_0 V produces clean text.
2. **Inherent 3-bit lossiness**: 3-bit V is quantized to 8 centroids; expect quality
   degradation vs 4-bit (consistent with other TurboQuant implementations: ~+5.8% PPL
   for 3.5-bit KV measured on Qwen3.5-0.8B in unixsysdev/llama-turboquant). 4-bit K/V
   is the supported default; treat tbq3_0 as experimental.

The context-creation guard (PR #2) refuses `tbq3_0` V unless `LLAMA_ALLOW_TBQ3_KV=1`
and warns for `tbq3_0` K — re-benchmark quality before relaxing it.

## Benchmarks

Apple M4, Qwen3.5-4B Q4_K_M, 512 ctx, `-fa on`, decode 16 tokens:

| config | t/s |
|---|---|
| Metal q4_0 K + q4_0 V | 26.9 |
| Metal q4_0 K + tbq4_0 V | 24.1 |
| Metal q4_0 K + tbq4_0 V + MTP (n=2) | 28.7 (accept 9/10) |
| CPU q4_0 K + tbq4_0 V | 23.3 |

tbq4_0 V on Metal runs at ~90% of q4_0 — the remaining gap is the dequant-to-f32
fallback (full inverse WHT per attention step). The rotated-domain fusion (dequant
to centroids only + Q pre-rotation) removes the butterfly entirely; that is the
planned next optimization.

## Notes for maintainers

- The Metal kernels require the non-`static` device functions — the MSL compiler
  silently drops template-instantiated kernels whose template arguments are
  `static` functions (found the hard way, 2026-08-11).
- `-fit auto` crashes on low-memory hosts (silent death during
  `common_params_fit_impl`); use `-fit off` on small machines.
- Quantized V requires `-fa on` (flash attention), enforced at context creation.
