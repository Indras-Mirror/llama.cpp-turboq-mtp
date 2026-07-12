# Architecture Proposal: P-Core-Bound Execution for `llama-turboq`

**Target host:** 1× RTX 4090 (sm_89) + i9-12900K (Intel hybrid: 8 P-cores, 8 E-cores)
**Goal:** Reduce decode latency spikes / choke points by binding the
latency-critical server thread(s) to dedicated performance (P) cores and
isolating the idle OpenMP pool, **without** lowering quants or changing the
model/GPU execution path.

---

## 1. Problem Statement (from measurement)

During decode the workload is **GPU-bound** (GPU ~94–100%), but each step
still incurs a fixed slice of **single-threaded CPU work** on the llama-server
main thread:

- ggml graph **construction** (`ggml_build_forward_impl` →
  `ggml_visit_parents_graph`) — a dependency-ordered topological visit,
  **inherently sequential** (0 OpenMP pragmas in the build path).
- MTP orchestration (`can_speculate` / `spec_draft`) and server timing.
- Per-step CUDA-graph reuse checks (`ggml_cuda_graph_update_required` —
  a full node-property scan over the ~thousands of decode-graph nodes,
  run **every step** even on reuse).

Observations (live, `/tmp/opencode` artifacts):

- Only **one** logical core (`cpu5`, a P-core) sits at ~100%; the other 23 are
  idle → the hot work is single-threaded and already on a P-core.
- The 24-thread OpenMP pool (`-t 24 -tb 24`) is **idle** during all-GPU decode
  yet still causes **scheduler jitter** by competing for the same P-cores the
  build thread needs.
- A **re-capture spike** hits every time `n_kv` crosses a padding boundary.
  The KV pad floor is hardcoded to **256** (`llama-kv-cache.cpp:1134`:
  `n_pad_cur = std::max(n_pad, 256u)`), so the decode graph re-captures
  **every 256 tokens** — a heavier step (full graph capture) = periodic spike.

**Conclusion:** there is **no "make the build multi-core" switch** — ggml builds
the graph single-threaded by design, and parallelizing it needs a risky ggml
rewrite. The symptom (spikes/choke) is best attacked by **core affinity +
isolation + less-frequent re-capture**.

---

## 2. Host CPU Topology (i9-12900K)

| Group        | Physical cores | Logical CPUs | Notes                         |
|--------------|----------------|--------------|-------------------------------|
| **P-cores**  | 8 (HT)         | `cpu0–15`    | 2 threads each; fast, low latency |
| **E-cores**  | 8 (no HT)      | `cpu16–23`   | efficient, slower per-thread  |

P-core ↔ SMT sibling map:

```
P0: cpu0,cpu1     P1: cpu2,cpu3     P2: cpu4,cpu5     P3: cpu6,cpu7
P4: cpu8,cpu9     P5: cpu10,cpu11   P6: cpu12,cpu13   P7: cpu14,cpu15
E0-7: cpu16-23 (1 thread each)
```

---

## 3. Thread Taxonomy in `llama-server`

| Thread role                         | CPU cost @ decode | Criticality |
|-------------------------------------|-------------------|-------------|
| Main / server loop (graph build + MTP + CUDA-graph reuse scan) | **high, single-threaded** | latency-critical |
| CUDA worker threads                 | low (GPU-bound)   | non-critical |
| OpenMP compute pool (`-t 24`)       | idle (GPU does the math) | jitter source |
| HTTP / I-O / sampler / log threads  | low, bursty       | non-critical |

---

## 4. Proposed Binding Architecture

```
 P-cores (cpu0-15)                 E-cores (cpu16-23)
 ┌─────────────────────────────┐   ┌─────────────────────────────┐
 │ P7 (cpu14)  ▶ MAIN/BUILD    │   │ OMP compute pool  (8 thr)   │
 │   (cpu15 left IDLE to avoid  │   │   OMP_NUM_THREADS=8         │
 │    SMT sibling contention)   │   │   OMP_PLACES={16-23}        │
 │                              │   │                             │
 │ P0-P6 (cpu0-13) ▶ CUDA       │   │ HTTP / I-O / sampler / log  │
 │   worker threads (low CPU)   │   │   (background, bursty)      │
 └─────────────────────────────┘   └─────────────────────────────┘
```

1. **Hot main thread → dedicated P-core `cpu14`** (P7). Its SMT sibling
   `cpu15` is deliberately left unbound so no co-resident thread steals
   P7's execution resources (no SMT contention on the critical path).
2. **OpenMP compute pool → E-cores `cpu16–23`** via
   `OMP_NUM_THREADS=8`, `OMP_PROC_BIND=spread`, `OMP_PLACES="{16-23}"`.
   This removes the 24 idle threads from the P-cores and ends the scheduler
   jitter on the build thread. (Pool still available for any CPU-side work
   during prefill.)
3. **CUDA worker threads → P-cores `cpu0–13`** (low CPU; harmless).
4. **HTTP / I-O / sampler / log → E-cores** (background, bursty; keep off
   the hot core).
5. **Re-capture smoothing (the real spike source):** raise the KV pad floor
   from 256 → **4096** in `llama-kv-cache.cpp:1134`. Re-captures become
   **16× less frequent**. Safe: the kq-mask already excludes padded (unused)
   positions, so attention compute is unchanged.

---

## 5. Implementation

### 5.1 systemd unit (production) — no code change required
Edit `/etc/systemd/system/llama-turboq.service`:

```ini
[Service]
# Keep OMP off the P-cores entirely
Environment=OMP_NUM_THREADS=8
Environment=OMP_PROC_BIND=spread
Environment=OMP_PLACES={16-23}
# Pin the whole process tree to P-cores; main thread pinned tighter in code
ExecStartPre=/usr/bin/taskset -c 0-15 /bin/true
ExecStart=/usr/bin/taskset -c 0-15 /home/jack/llm/llama-turboq/build/bin/llama-server ... (existing args)
```

### 5.2 Finer per-thread binding (optional, code-level)
For a hard guarantee that the build thread owns `cpu14` and nothing else
migrates onto it, add `pthread_setaffinity_np` at:
- server main loop entry (`tools/server/server.cpp`) → mask `{14}`,
- `ggml_threadpool` worker creation (`ggml/src/ggml.c`) → mask `{16-23}`.

Trade-off: code-level affinity is more robust than `taskset` but couples the
binary to this specific topology. Keep it behind a `#ifdef`/env guard
(`LLAMA_PIN_PCORES`) so the binary stays portable.

### 5.3 Manual / dev runs
Wrap the launch script:
```sh
OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES={16-23} \
  taskset -c 0-15 /home/jack/llm/llama-turboq/build/bin/llama-server ...
```

### 5.4 KV pad change
`src/llama-kv-cache.cpp`:
```cpp
const uint32_t n_pad_cur = std::max(n_pad, 4096u);   // was 256u
```

---

## 6. Verification / Acceptance Criteria

1. **Affinity correct:** `htop` / `perf` shows main thread pinned to `cpu14`
   (no migration), OMP threads confined to `cpu16–23`.
2. **Re-capture frequency:** count `CUDA graph warmup reset` in a verbose
   (`-v`) decode log over 4096 tokens — expect ~1 (was ~16).
3. **Latency:** measure inter-token delay percentiles (p50 / p95 / p99) at
   `temp 1.0`, 256-token generation, before vs after. Target: **p99 reduced**
   and fewer outliers, even if mean tok/s is unchanged (GPU remains the
   throughput limiter).
4. **Throughput unchanged:** `predicted_per_second` ~62 (temp 1.0) / ~92
   (temp 0) preserved — confirms we did not regress decode.

---

## 7. Risks & Rollback

- **Env/units-only path** (§5.1, §5.3) is fully reversible: remove the
  `Environment=`/`taskset` lines, `systemctl daemon-reload && restart`.
- **Code affinity** (§5.2) risks starving OMP during prefill if mis-scoped;
  mitigated by keeping the full E-core set for the pool. Gate behind
  `LLAMA_PIN_PCORES` so it defaults off.
- **KV pad 4096** slightly enlarges the decode tensor shape per step
  (masked, negligible). If any edge case appears, revert to 256.

---

## 8. Out of Scope

- **True multi-core parallelization of ggml graph construction** — not
  feasible via config; would require a speculative parallel graph builder
  with dependency tracking (poor scaling, high risk). Explicitly **not**
  proposed.
- **Lowering quants** — rejected by user.
- **CUDA-graph capture enablement** — already active/verified; nothing to do.
