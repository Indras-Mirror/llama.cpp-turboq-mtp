# Reasoning-Effort Implementation Proposal (llama-turboq fork)

**Status: DESIGN APPROVED IN PRINCIPLE — NOT YET IMPLEMENTED (Jack: "dont proceed yet").**
Owner: Jack + Frank · Model in scope: **Qwen3.8-27B (MTP)** on the llama-turboq fork, `:8081/v1`.
Date: 2026-09-23. **This file is the single source of truth and is written to survive context compression. If it is truncated, trust the §0 header + §10 checklist.**

---

## 0. COMPRESSION-SAFE SUMMARY (the 10 facts that must survive)

1. **Goal:** stop Hermes "reasoning limit reached" failures + implement the extended reasoning-effort flags the fork is missing, with **config = fallback, request = override**.
2. **Root cause (2 bugs, both confirmed file:line):**
   - **Bug 1 — the effort WORD is not parsed.** Hermes sends top-level `reasoning_effort` (`custom/__init__.py:83`); the fork has **zero** code reading it (only 2 commented fossils at `common/arg.cpp:4015,4034`). `xhigh`/`ultra` is silently dropped.
   - **Bug 2 — precedence inverted.** `server-common.cpp:1278` reads the request budget **only when config is unset (-1)**. `models.ini` sets `LLAMA_ARG_THINK_BUDGET=8192`, so the request can **never** override → the 8192 hard cap always wins → thinking truncated (`common/reasoning-budget.cpp:107` "budget exhausted, forcing end").
3. **The fork ALREADY HAS the hard-cap mechanism** (`common/reasoning-budget.{h,cpp}` state machine; wired `common/sampling.cpp:296`; `-1`=unlimited, `0`=immediate-end, `N`=cap) and the **soft instruction path** (`chat_template_kwargs` merge at `server-common.cpp:1060`, `enable_thinking` special-case `:1067`). → **We parse + route + flip precedence; we do NOT build the cap.**
4. **The endpoint IS served:** `server.cpp:183` → `POST /v1/chat/completions`. Good.
5. **Final ladder (Jack's decision, ×2, non-redundant, TOP = UNLIMITED):** `none=off · minimal=512 · low=1024 · medium=2048 · high=4096 · xhigh=8192 · max=UNLIMITED · ultra=UNLIMITED · unlimited=UNLIMITED`. Raw number always overrides the word; `-1`=unlimited. **`max` is the top because it's the word the most common clients (Hermes top, OpenAI, opencode) actually send — so "top = unlimited" AND "top = widely used" both land on `max`.**
6. **Both doors (Jack: "can we do both"):** accept the **word** (`reasoning_effort`) for Hermes/pi/opencode AND the **raw number** (`reasoning_budget_tokens`/`thinking_budget_tokens`) for vLLM/SGLang/llama.cpp clients. One budget, two entry points.
7. **Priority (highest wins):** raw number > word (mapped) > `models.ini` config > built-in default.
8. **TOP LEVEL = UNLIMITED (Jack: "our top has to be unlimited AND widely used").** Hermes **clamps its top setting down to `max` on the wire** (`clamp_effort(effort, OPENAI_COMPAT_WIRE_EFFORTS)`, wire tops at `max`; docstring "ultra never reaches a wire"). So the TUI's "ultimate" arrives as **`max`**. Solution: **map `max` → unlimited (`-1`)** — because `max` is precisely the word OpenAI/Hermes/opencode/Ollama/GLM all send as their top. One word gives us "top = unlimited" AND "top = widely used." `ultra`/`unlimited` are aliases → also unlimited. `xhigh`=8192 is the highest FINITE level; raw numbers cover anything in between. **DECISION RESOLVED — no longer pending.**
9. **Do NOT implement yet.** Jack wants to compress context first.
10. **Verification:** unit (curl each word + raw number, assert `reasoning-budget: tokens=N` in log) + E2E (a >8192-thinking prompt completes at `xhigh`/`max`) + precedence (raw beats word; nothing-sent → config fallback) + back-compat + re-run the real Hermes session.

---

## 1. The two confirmed bugs (root cause, file:line)

### Bug 1 — `reasoning_effort` word is dropped
- Hermes sends top-level `reasoning_effort` (`custom/__init__.py:83` → `top_level["reasoning_effort"] = clamp_effort(effort, OPENAI_COMPAT_WIRE_EFFORTS)`).
- The fork has **zero** code reading it. Confirmed: `grep` of `tools/server` + `common` for the word → only the two commented fossils `common/arg.cpp:4015` and `:4034` (`//params.default_template_kwargs["reasoning_effort"] = "\"high\"";`).
- Result: Hermes's `xhigh`/`ultra` is silently ignored; the model falls back to the config budget.

### Bug 2 — precedence is inverted (config wins over request)
`tools/server/server-common.cpp:1276-1290` (the OpenAI-chat path):
```cpp
// Reasoning budget: pass parameters through to sampling layer
{
    int reasoning_budget = opt.reasoning_budget;                 // config (models.ini 8192) — ALWAYS used
    if (reasoning_budget == -1 && body.contains("thinking_budget_tokens")) {
        reasoning_budget = json_value(body, "thinking_budget_tokens", -1);   // request ONLY if config is -1
    }
    if (!chat_params.thinking_end_tag.empty()) {
        llama_params["reasoning_budget_tokens"] = reasoning_budget;
        llama_params["reasoning_budget_start_tag"] = chat_params.thinking_start_tag;
        llama_params["reasoning_budget_end_tag"]   = chat_params.thinking_end_tag;
        llama_params["reasoning_budget_message"]   = opt.reasoning_budget_message;
    }
}
```
The per-request budget is consulted **only when config is unset (-1)**. `models.ini` sets 8192, so the request is never read → the 8192 cap is absolute. Opposite of the industry standard (request overrides config).

### Symptom (what Jack saw)
Model needs >8192 thinking tokens → `common/reasoning-budget.cpp:107` fires `"reasoning-budget: budget exhausted, forcing end sequence"` → truncated thinking, empty/short `content`, `finish_reason: length` → Hermes reports "reasoning limit reached."

---

## 2. What the fork ALREADY has (the hard part is done — we route, not build)

1. **Hard token cap:** `common/reasoning-budget.{h,cpp}` state machine (`IDLE→COUNTING→FORCING→DONE`), wired at `common/sampling.cpp:296` (`params.reasoning_budget_tokens >= 0` → attach; `< 0 ? INT_MAX : N`). `-1`=unlimited, `0`=immediate end, `N`=cap. Config flag `--reasoning-budget` (env `LLAMA_ARG_THINK_BUDGET`, `common/arg.cpp:3110-3117`). `models.ini` sets `=8192`. **Same design as vLLM `thinking_token_budget` / SGLang `max_thinking_tokens`.**
2. **Per-request-capable (in principle):** the sampling layer reads `reasoning_budget_tokens` directly from the request body (`server-task.cpp:489` → `params.sampling.reasoning_budget_tokens = budget`). So the cap **can** vary per request — the OpenAI-chat path just gates it on config (Bug 2).
3. **Soft instruction path:** `server-common.cpp:1060-1064` merges the request's `chat_template_kwargs` into the Jinja template; `:1067` already special-cases `enable_thinking`. Qwen3.8-27B's template turns a word into a system instruction ("keep thinking brief" / baseline / "think carefully").

→ **The implementation is: parse `reasoning_effort`, map word→budget, feed BOTH mechanisms, and flip the precedence gate.** No new sampler, no new architecture.

---

## 3. The industry standard (research, with sources)

Consistent across OpenAI/codex, pi, opencode, vLLM, SGLang, llama.cpp upstream, and Qwen's own docs:

- **Wire field:** top-level `reasoning_effort` (string) on `/v1/chat/completions` — what Hermes already sends.
- **Ladder (low→high):** OpenAI `none, minimal, low, medium, high, xhigh, max`; pi `off, minimal, low, medium, high, xhigh`; opencode `default, low, medium, high, xhigh, max`; Hermes internal `EFFORT_LADDER = none, minimal, low, medium, high, xhigh, max, ultra` (`agent/reasoning_effort.py:25`) and clamps to `OPENAI_COMPAT_WIRE_EFFORTS = none, minimal, low, medium, high, xhigh, max` (`:28`).
- **Implementation = word → token budget → cap** (NOT on/off):
  - llama.cpp discussion #20408: "accept the usual reasoning levels and map them to percentages with the max-tokens left, then set that as the reasoning budget."
  - opencode per-model: `low→512, medium→2048, xhigh→8192, none→enable_thinking:false`.
  - Qwen3.8-27B production config (komikndr gist): `low→512, medium→2048, default→4096, xhigh→8192`.
  - QwenCloud: `thinking_budget` default 4000; qwen3.8 `low/medium/xhigh`.
  - vLLM: `reasoning_effort`→`enable_thinking` + `thinking_token_budget` (per-request cap). SGLang: `max_thinking_tokens`/`thinking_budget`.
- **Priority:** per-request value overrides server default; server default (config/flag) is the fallback. (vLLM: explicit `chat_template_kwargs` "takes priority over the automatic injection".)
- **"Unlimited" = no budget (`-1`), not a named level** (llama.cpp `--reasoning-budget -1`; vLLM "no budget"; SGLang).

**Qwen3.8-27B specifics (our exact model):**
- Native levels: `low`, `medium`, `xhigh` (`high` aliased→`xhigh`); default `xhigh`.
- Effort is an **instruction**, not a hard dial (graphometer.ai): `xhigh` adds a "think carefully" block; `low` a "keep brief" block; `medium` baseline.
- GDN/Qwen self-terminate reasoning well by task complexity → an "unlimited" top is reasonable and safe (Jack's own reasoning). `xhigh`=8192 remains the highest finite level; raw numbers cover anything below the top.

---

## 4. Final design

### 4.1 Word → budget table (Qwen3.8-27B; per-model, configurable)
| word | budget (tokens) | note |
|---|---|---|
| none | off | `enable_thinking=false` |
| minimal | 512 | kept (Jack) |
| low | 1024 | moved up from 512 (was redundant with minimal) |
| medium | 2048 | wire-vocab fill (Hermes can send it) |
| high | 4096 | dropped from 8192 to make room (Jack) |
| xhigh | 8192 | native Qwen3.8 top, kept |
| max | -1 (unlimited) | **TOP = unlimited** (Jack); also the word Hermes-top/OpenAI/opencode send |
| ultra | -1 (unlimited) | alias → unlimited (Hermes-internal top, future-proof) |
| unlimited | -1 (unlimited) | alias → unlimited (raw escape) |
| raw number | as-sent | `-1` = unlimited; **always overrides the word** |

Rationale: ×2 geometric, non-redundant (512,1024,2048,4096,8192,∞). 512/2048/8192 are research-grounded (Qwen3.8-27B production configs + QwenCloud); 1024/4096 are the ×2 fills. `none`=off. **`max`/`ultra`/`unlimited`=`-1` (no cap). `max` is the top because it's the most widely-sent top word — so the top is unlimited AND widely used.**

### 4.2 Two mechanisms, one budget
- **Soft (instruction):** route the word into the Jinja template via the existing `chat_template_kwargs` path (`server-common.cpp:1060`) so the model's own "think carefully / keep brief" instruction applies.
- **Hard (cap):** map the word to a token budget and feed the existing `reasoning-budget` sampler (`reasoning_budget_tokens`), which force-ends thinking at the cap.
- Both from the **same** resolved budget. A raw number (`reasoning_budget_tokens`/`thinking_budget_tokens`) overrides the word and is the unlimited escape hatch (`-1`).

### 4.3 Precedence (flip Bug 2)
`raw number  >  reasoning_effort word (mapped)  >  models.ini config  >  built-in default`
- No field sent → `models.ini` / built-in default applies (Jack's "config = fallback" requirement). ✓
- Field sent → it overrides. ✓

### 4.4 Both doors (Jack: "can we do both, to cover more clients")
Raw numeric budget = the precise escape hatch + unlimited. Named level = the convenience protocol (what Hermes/pi/opencode send). Accepting **both** covers word-based AND number-based clients. One mechanism, two entry points.

---

## 5. Implementation plan (phased; NOT started)

### Phase 1 — Parse `reasoning_effort` + flip precedence (fixes the current failure)
Files: `tools/server/server-common.cpp` (OpenAI-chat path), `tools/server/server-chat.cpp` (completions path), `common/` (word→budget table).
- Read top-level `reasoning_effort` (and `chat_template_kwargs.reasoning_effort`) from the request body.
- Map word→budget via the §4.1 table (per-model from `models.ini` if present, else built-in).
- Route to BOTH mechanisms: set the template kwarg (soft) AND `reasoning_budget_tokens` (hard).
- Flip `server-common.cpp:1276-1290` so a per-request budget (word-mapped or raw) **overrides** the config default; config is the fallback when no field is present.
- Accept raw `reasoning_budget_tokens`/`thinking_budget_tokens` as the top-priority override.

### Phase 2 — Per-model config in `models.ini`
Optional per-model effort table under each `[model]` section, e.g. `[qwen38-27b-mtp]`:
```
EFFORT_MINIMAL=512
EFFORT_LOW=1024
EFFORT_MEDIUM=2048
EFFORT_HIGH=4096
EFFORT_XHIGH=8192
EFFORT_MAX=-1        ; TOP = unlimited (Jack)
; EFFORT_ULTRA=-1    ; alias → unlimited (Hermes-internal top)
```
Server falls back to the built-in §4.1 table if a model doesn't define one. Keep `LLAMA_ARG_THINK_BUDGET=8192` as the documented **fallback** (no longer a hard cap).

### Phase 3 — (Optional) expose in `/v1/models` + docs
Echo supported effort levels per model so clients (pi/opencode `supported_reasoning_effort`) can discover them.

### Back-compat guarantees
- Existing `thinking_budget_tokens`/`reasoning_budget_tokens` requests still honored.
- `models.ini` fallback still applies when no field is sent.
- `--reasoning-budget` flag still works.

---

## 6. Verification plan (after implementation)
1. **Unit:** curl each word (`none…max`, `ultra`, `unlimited`) + raw numbers; assert server log shows the expected `reasoning-budget: tokens=N`.
2. **E2E:** a prompt needing >8192 thinking tokens — confirm `xhigh`/`max` complete (no forced end-tag); `none`/`low` stay short.
3. **Precedence:** send a raw budget with a different word → raw wins. Send nothing → config fallback applies.
4. **Back-compat:** existing `thinking_budget_tokens` requests still honored.
5. **Hermes:** re-run the actual Hermes `xhigh`/`ultra` session that hit the limit — confirm it completes. (Capture a request dump to confirm exactly what word Hermes sends for its top setting — see §7.)

---

## 7. TOP LEVEL = UNLIMITED (RESOLVED — Jack: "our top has to be unlimited AND widely used")

Hermes's custom provider clamps its top setting down to **`max`** on the wire (`clamp_effort(effort, OPENAI_COMPAT_WIRE_EFFORTS)`, `custom/__init__.py:83`; wire tops at `max`, `reasoning_effort.py:28`). So the TUI's "ultimate" arrives as **`max`**.

**Resolution:** map **`max` → unlimited (`-1`)** in the fork. This is the right word for the top because:
- It is the word **OpenAI, Hermes (clamped top), opencode, Ollama-Cloud, GLM-5.3, DeepSeek-V4** all send as their highest level.
- One word therefore satisfies both Jack requirements: **top = unlimited** AND **top = widely used** (the most common clients' top).
- `xhigh`=8192 stays the highest FINITE level; raw numbers (e.g. `4096`, `8192`, `16384`) cover anything in between; `-1`/`unlimited`/`ultra` are aliases for the top.

**No Hermes core change needed** — the fork just interprets `max` as unlimited. (The earlier "Option B: advertise `ultra` as a wire level" is no longer required.)

---

## 8. Open decisions / risks
1. **Top level: unlimited** → §7. **RESOLVED (Jack: max=unlimited, widely-used top).**
2. **Tuning the numbers:** 512/2048/8192 research-grounded; 1024/4096 are ×2 fills. Re-tune after it's working if the 27B behaves differently.
3. **Per-model vs global table:** per-model (`models.ini`) with a built-in global fallback (design).
4. **Safety of an unlimited top:** GDN/Qwen self-terminates reasoning by task complexity (Jack's reasoning), so unlimited is acceptable; `xhigh`=8192 remains the finite top if a cap is ever wanted.

---

## 9. Key references

**Fork code (llama-turboq, `/home/jack/llm/llama-turboq`):**
- `tools/server/server.cpp:183` — `POST /v1/chat/completions` IS served.
- `tools/server/server-common.cpp:1060-1067` — `chat_template_kwargs` merge + `enable_thinking` special-case.
- `tools/server/server-common.cpp:1276-1290` — the inverted precedence gate (Bug 2).
- `tools/server/server-task.cpp:489-506` — `reasoning_budget_tokens` read from request (per-request-capable).
- `common/reasoning-budget.{h,cpp}` — the token-cap state machine; `:107` "budget exhausted" log.
- `common/sampling.cpp:296` — sampler wiring (`< 0 ? INT_MAX : N`).
- `common/arg.cpp:3110-3117` — `--reasoning-budget` flag (`LLAMA_ARG_THINK_BUDGET`).
- `common/arg.cpp:4015,4034` — commented-out `reasoning_effort` fossils (Bug 1).
- `models.ini [qwen38-27b-mtp]` — `LLAMA_ARG_THINK_BUDGET=8192` (the current hard cap).

**Hermes (client, `~/.hermes/hermes-agent`):**
- `plugins/model-providers/custom/__init__.py:83` — sends `reasoning_effort` (clamped).
- `agent/reasoning_effort.py:25,28` — `EFFORT_LADDER`, `OPENAI_COMPAT_WIRE_EFFORTS`; `clamp_effort` (ultra→max).
- `~/.hermes/config.yaml:38` — `reasoning_effort: xhigh` (current).

**Model:** `/data/models/Qwen3.8-27B-MTP/Qwen3.8-27B-MTP-Q4_K_M.gguf` (Qwen3.8-27B, native levels low/medium/xhigh).

**Research sources:** OpenAI chat-completions `reasoning_effort` spec; pi `thinkingLevelMap`/`reasoningEffortMap`; opencode variant→`reasoning_effort`; vLLM `thinking_token_budget`/`ReasoningBudgetLogitsProcessor`; SGLang `max_thinking_tokens`; llama.cpp `--reasoning-budget` + discussion #20408; QwenCloud thinking docs (`thinking_budget` default 4000, qwen3.8 `low/medium/xhigh`); komikndr Qwen3.8-27B production config (low 512 / medium 2048 / xhigh 8192); graphometer.ai effort-as-instruction measurement.

---

## 10. Status checklist
- [x] Research complete (industry standard + Qwen3.8-27B specifics + fork code audit).
- [x] Root cause confirmed (2 bugs, file:line).
- [x] Ladder decided (Jack: minimal 512 / low 1024 / medium 2048 / high 4096 / xhigh 8192 / **max=unlimited** / ultra=unlimited).
- [x] Both-doors decision (word + raw number) confirmed.
- [x] **§7 top-level decision RESOLVED: `max` = unlimited (the widely-used top word).**
- [ ] Phase 1 implemented.
- [ ] Phase 2 (models.ini table) implemented.
- [ ] Verification (unit + E2E + Hermes) passed.
