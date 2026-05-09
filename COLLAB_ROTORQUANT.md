# COLLAB NOTES — Integrate RotorQuant (PlanarQuant + IsoQuant)

**Branch:** `feature/rotorquant`  
**GitHub Issue:** Indras-Mirror/llama.cpp-mtp#3  
**Date:** 2026-05-10  
**Status:** PLANNING — integration not started  

---

## What Is RotorQuant?

Drop-in KV cache compression that replaces the 128-element WHT butterfly (TurboQuant/TBQ4) with block-diagonal 2D/4D rotations. Same compression ratio, but faster because the rotation is O(d) instead of O(d log d) and fully parallelizable.

**Claimed benefits over TurboQuant (same 10.3x compression, Llama 3.1 8B, RTX 5090):**

| Metric | TurboQuant (turbo3) | PlanarQuant (planar3) | IsoQuant (iso3) |
|--------|---------------------|-----------------------|-----------------|
| Decode tok/s | 93 | 119 | 118 |
| Prefill tok/s | 722 | 3,822 | 3,397 |
| PPL (wiki-2) | 7.07 | 7.05 | 6.91 |
| Params | 16,384 | 128 | 128 |

- **28% faster decode**, **5.3x faster prefill**, **44x fewer params** vs TurboQuant
- Same 3-bit symmetric compression (10.3x)
- PlanarQuant uses 2D Givens rotations (256 rotations per 128-dim block)
- IsoQuant uses 4D quaternion rotations (32 quaternion rotations per 128-dim block)

## Architecture Comparison

| Feature | TBQ4_0 (our current) | PlanarQuant 3-bit | IsoQuant 3-bit |
|---------|---------------------|-------------------|----------------|
| Rotation | 128-dim WHT butterfly | 2D Givens pairs | 4D quaternion |
| Rotation complexity | O(d log d), 7 stages | O(d), fully parallel | O(d), fully parallel |
| Dequant in FA | centroid×norm lookup | 2D inverse rotation + quant lookup | 4D inverse rotation + quant lookup |
| Block format | 66 bytes / 128 elems | TBD | TBD |
| Bits per value | 4.25 | ~3.0 | ~3.0 |

**Key insight for our fork:** Our TBQ4 fused FA kernel already does centroid×norm lookup in the rotated domain. For PlanarQuant/IsoQuant, the dequant would be: inverse 2D/4D rotation → centroid lookup → scale by norm. The rotation is trivially parallel (64 independent 2D rotations for planar, 32 independent 4D rotations for iso), so it should be faster in the FA inner loop than our current TBQ4 approach.

## Source Material

**Reference fork:** `johndpope/llama-cpp-turboquant` branch `feature/planarquant-kv-cache`
- Cloned at: `/tmp/rotorquant-fork` (available locally)
- Last commit: 20efe75 (2026-04-01)

## Files To Port

### New files (create in our fork):

| Source File | Lines | Purpose |
|-------------|-------|---------|
| `ggml/src/ggml-cuda/set-rows-planar-iso.cuh` | 239 | GPU quantize kernels for planar3/iso3/planar4/iso4 |
| `ggml/src/ggml-cuda/cpy-planar-iso.cuh` | 9 | CPU F16→planar/iso conversion declarations |
| `ggml/src/ggml-cuda/cpy-planar-iso.cu` | 320 | CPU-side conversion + constant init |
| `ggml/src/ggml-cuda/planar-iso-constants.cuh` | 31 | Rotor rotation angle constants |
| 24 template instances in `template-instances/` | ~50 each | `fattn-vec-instance-*.cu` files |

### Modified files (merge into our fork):

| File | What Changed | Conflict Risk |
|------|-------------|---------------|
| `ggml/src/ggml-cuda/fattn.cu` | Add planar/iso dispatch cases | **HIGH** — we modified for TBQ4 dispatch |
| `ggml/src/ggml-cuda/fattn-common.cuh` | Add planar/iso type constants | **LOW** — just new enum values |
| `ggml/src/ggml-cuda/cpy.cu` | Add planar/iso dequant cases | **MED** — we modified for TBQ4 dequant |
| `ggml/src/ggml-cuda/set-rows.cu` | Add planar/iso quantize dispatch | **MED** — we modified for TBQ4 quantize |
| `ggml/src/ggml-cuda/dequantize.cuh` | Add planar/iso dequant functions | **LOW** — new functions |

### Files we might ALSO need to modify:

| File | Why |
|------|-----|
| `src/llama-graph.cpp` | New KV cache types need pass-through logic |
| `ggml/include/ggml.h` | New `ggml_type` enum values for cache types |
| `common/common.cpp` | CLI flags for `--cache-type-k planar3` etc |
| `tools/server/server-context.cpp` | Server-side cache type handling |

## Integration Plan

### Step 1 — Port core kernel files (low risk)
- Copy all 4 new `.cuh`/`.cu` files directly (no conflicts)
- Add the 24 template instances
- Verify: builds on sm_89

### Step 2 — Merge modified files (medium risk)
- For each of the 5 conflict files:
  - Diff the rotorquant fork's version vs the upstream base version
  - Apply only the planar/iso additions to our fork's version
  - Preserve ALL our TBQ4 code paths
- Key rule: planquant/iso additions go NEXT TO our TBQ4 additions, not replacing them

### Step 3 — Wire through the stack
- Add `GGML_TYPE_PLANAR3_0`, `GGML_TYPE_PLANAR4_0`, `GGML_TYPE_ISO3_0`, `GGML_TYPE_ISO4_0` to ggml types
- Add `--cache-type-k planar3` / `--cache-type-v planar3` CLI flags
- Add pass-through logic in `llama-graph.cpp` (similar to TBQ4 pass-through)
- Wire FA dispatch to route planar/iso KV types through the correct kernels

### Step 4 — Build & test
- Build for sm_89 (4090)
- Test with Qwen3.6-27B at 200K context
- Benchmark: planar3 vs tbq4_0 at same context
- PPL comparison

### Step 5 — Verify quality
- Run factual recall test at high context
- Compare output quality vs TBQ4_0
- If PPL is better (as claimed), planar3 becomes the recommended cache type

## VRAM Math (planar3 at 262K, Qwen3.6-27B)

24 heads × 4 KV heads × 128 dim × 262K tokens = 
- FP16: ~6.4 GB KV cache
- planar3 (3-bit): ~2.1 GB KV cache
- TBQ4_0 (4.25-bit): ~2.8 GB KV cache

So planar3 saves ~0.7 GB vs TBQ4_0 at 262K — enough for another ~40K context.

## Danger Zone

- **Do NOT** merge rotorquant's modified `fattn.cu` directly — it will overwrite our TBQ4 fused FA dispatch
- **Do NOT** change the master branch until integration is verified working
- **Do NOT** remove TBQ4 code paths — planar/iso are additive

## How To Help (for peers)

1. **Port the kernel files**: Copy the 4 new files + 24 template instances from the rotorquant fork. Build and verify no compile errors.

2. **Merge fattn.cu**: The hardest file. Need to combine our TBQ4 fused FA dispatch with rotorquant's planar/iso dispatch. Both add new switch cases in `ggml_cuda_flash_attn_ext_mma()`. 

3. **Add ggml types**: Define the 4 new `ggml_type` values and their properties (block size, type size, dequant function pointers).

4. **Wire CLI flags**: Add `planar3`, `planar4`, `iso3`, `iso4` as valid `--cache-type-k` / `--cache-type-v` values.

5. **Test on Ada (4090) first**: Don't need Ampere. If it works on Ada, test on Ampere to see if the same sm_86 garbage bug exists (if it does, it's a separate issue from the TBQ4 bug).

## Reference Commands

```bash
# Build
cd /home/mal/AI/llama.cpp-mtp-fixes
git checkout feature/rotorquant
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc) --target llama-server

# Test with planar3 (FUTURE — once wired)
./build/bin/llama-server \
  -m /media/Crucial1TB/models/Qwen3.6-Heretic-MTP/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q4_K_M.gguf \
  --port 8098 -c 262144 --flash-attn on --mlock -t 8 -ngl 99 \
  --parallel 1 --spec-type mtp --spec-draft-n-max 3 \
  -ctk planar3 -ctv planar3 --jinja --temp 0.6 --seed 3407
```

## Contact

- **GitHub:** @Indras-Mirror (issue #3)
- **Relay:** `quetza-codetl` (QuetzaCodetl session)
- **Thread:** `tbq4-coordination`
- **RotorQuant repo:** `scrya-com/rotorquant`
- **Reference fork:** `johndpope/llama-cpp-turboquant` (branch `feature/planarquant-kv-cache`)
