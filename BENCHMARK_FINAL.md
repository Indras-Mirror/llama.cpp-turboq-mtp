# DSV4 KV quant + draft benchmark — FINAL (2026-08-03)

Session handoff replacement (original HANDOFF_SESSION_2026_08_03.md was lost with
the conductor session). Numbers verified from worker reports with per-run log
evidence. Model: cyberneurova-DeepSeek-V4-Flash-abliterated-IQ2XXS (81GB),
RTX 4090 24GB, NVME.

## Branch state (feature/dsv4-dspark, worktree clean)
- `5b504a59f` merge upstream DSV4 MTP + DSpark (#25784, #26458)
- `4d7b7bbae` DSpark drafter GGUF loading (arch alias + dspark.* mapping)
- `a8b61c3d9` TBQ3 KV end-to-end fix (cherry-pick of `fa49662e1` from
  feature/dsv4-tbq4-native: CPU 3-bit packer, CUDA FWHT race, rotation fixes;
  12 files, +619/−75)

## TBQ3 fix — sanity gate (PASS)
`2+2` chat probe (temp 0, seed 42, 4096 ctx, TBQ3 KV): `'2+2 is 4.'`, 11.22 t/s,
no garbage. (GOTCHA: bare prompt + --jinja produces garbage on this model
regardless of KV type — always use chat format.)

## Quality @4096 (PPL, 133KB sample; task eval temp 0 seed 42)
| KV | PPL | task eval |
|---|---|---|
| q8_0 (ref) | 1.5970 | 6/6 |
| tbq4_0 | 1.6387 (+2.6%) | 5/6 |
| tbq3_0 (fixed) | 1.8314 (+14.7% vs q8_0) | 6/6 |

TBQ3 PPL measurably worse than TBQ4; task 6/6 vs 5/6 (small-sample inversion).

## Speed @256k (main GPU, draft CPU `-ngld 0`, 200-tok unless noted)
| Run | KV | draft | t/s | accept | notes |
|---|---|---|---|---|---|
| R1 | tbq3_0 | — | 7.87 | — | coherent water-cycle output |
| R2 | tbq3_0 | Q2_K | — | — | CUDA OOM 3/3 (draft-active decode crosses 24GB) |
| R3 | tbq4_0 | — | 9.42 | — | best working config |
| R4 | tbq4_0 | Q2_K | 7.92 | 0.60 | draft hurts sustained gen |
| MTP | tbq4_0 | MTP Q8_0 | 4.33 | 0.37 | draft CPU can't feed GPU |

Prefill (1016-tok, n_pred 1): TBQ3 117.98 t/s (corrected — broken build
overreported 153.58); TBQ4 110-115 t/s. DSpark short-burst 2+2: 10.33 t/s
(accept 0.76) — CPU draft helps short bursts only.

## Verdict
1. **Best 256k config: TBQ4 KV, no draft — 9.42 t/s, PPL 1.6387.**
2. TBQ3 is now correct + functional (was broken) but: −16% gen speed
   (7.87 vs 9.42), +11.8% PPL vs TBQ4, and its Q2_K draft config OOMs this rig.
3. Drafters (both lose on sustained gen): DSpark Q2_K 7.92, MTP Q8_0 4.33.
   DSpark Q2_K remains the best drafter (half the size of MXFP4, same speed).
4. 15-20 t/s @ 512k is a hardware wall (draft can't share GPU with 81GB main) —
   needs 2× 4090-class, or accept 256k TBQ4 ~9.4 t/s.

## Models
- Main: `/media/mal/2064C68864C66060/Models/cyberneurova-DeepSeek-V4-Flash-abliterated-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-aligned.gguf`
- DSpark drafters (sha256 verified): `DSPark-Drafters/DeepSeek-V4-Flash-0731-DSpark-Drafter-Q2_K-Q8_0.gguf` (6.97G), `-MXFP4-Q8_0.gguf` (10.9G)
- MTP draft: `DSPark-Drafters/DeepSeek-V4-Flash-MTP-Q8_0.gguf` (4.73G, size-verified vs HF API, loads with `--spec-type draft-mtp`)
- NVME mount: `udisksctl mount -b /dev/nvme1n1p2`
