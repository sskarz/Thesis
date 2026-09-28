# Experiment Plan

Attack plan for supporting the thesis claim that SLMs "can be optimized" for multi-tool selection and sequential tool calling, with measurable accuracy/latency/memory trade-offs.

## Current Gaps

| Problem | Evidence | Severity |
|---------|----------|----------|
| Parallel calling = 0% for all models | Structural failure across all 4 models | Critical — biggest SLM-vs-cloud gap |
| Irrelevance inconsistent | Qwen2.5 at 58.75%, Llama at 0.83% | High — unsafe for agentic deployment |
| Multi-turn not yet reported | `generate_multi_turn.py` exists but no results in THESIS.md | Gap in thesis |
| No optimization tested yet | Only type coercion + system prompt so far | Core thesis claim is unsupported |

## Tier 1 — Must Do

These directly support the thesis claim of "optimization" and are required for the thesis to hold.

### 1. Guided-Structured Templates (METHODS §1)

- **What:** Replace `TOOL_CALLING_SYSTEM_PROMPT` with the 7-step reasoning template
- **Why:** Cheapest "optimization" data point — pure prompt change, no training
- **Target categories:** `multiple`, `irrelevance`, `parallel`
- **Expected impact:** +3–12% on BFCL
- **Effort:** Low
- **Source:** He et al., "Improving LLMs Function Calling via Guided-Structured Templates," arXiv:2509.18076
- **Note on reasoning:** The baseline evaluation uses "no reasoning mode" to isolate raw tool-calling capability. This template is a form of structured chain-of-thought — it explicitly asks the model to reason through 7 steps before producing a tool call. Frame this as an experimental variable, not a baseline condition: "baseline without reasoning = X%, with structured prompting = Y%." The delta measures the value of structured reasoning for SLM tool calling.

### 2. Best-of-N Majority Voting (METHODS §4)

- **What:** Generate N=5 candidates at temp=0.7, majority-vote on `(function_name, args)` tuples
- **Why:** Directly measures the accuracy vs. latency trade-off the thesis promises
- **Target categories:** All
- **Measured (Gemma `multi_turn_base`):** Run 11 `--fc-mode iffc-state-bon` **73/200 = 36.5%** vs IFFC+STATE Run 10 68/200 = 34.0% (**+2.5 pp** at ~7× wall / ~8.8× decode). Modest lift, **not free**. Details in `THESIS.md`. Best inference-time method on the **base** Gemma stack. **Do not** propose BoN as the next win on the SFT-v4 checkpoint (Run 16 STATE overlay already null/negative), the SFT-v5 checkpoint (Run 17 is a scored regression vs v4), **or** the SFT-v6 checkpoint (Run 18 is a scored miss vs v4).
- **Expected impact (pre-score):** +5–15% accuracy, N× latency increase
- **Effort:** Medium
- **Key trade-off story:** FunctionGemma at 204 tok/s makes N=5 cheap (~80ms); Qwen3.5 at 38 tok/s makes it expensive (~3.3s); Gemma BoN N=5 measured ~21.3 h vs STATE ~3.0 h — this *is* the trade-off narrative
- **Source:** Snell et al., "Scaling LLM Test-Time Compute," arXiv:2408.03314

### 3. xLAM-1b-fc-r as Benchmark Baseline (METHODS §15)

- **What:** Score published `Salesforce/xLAM-1b-fc-r-gguf` as a fine-tuned SLM baseline on BFCL v4 `multi_turn_base`
- **Why:** Provides a fine-tuned SLM comparison point. Answers: "how far can inference-time methods close the gap toward a purpose-built model?"
- **Registry:** key `xlam-1b` is in `benchmark/config.py` (`hf_repo` `Salesforce/xLAM-1b-fc-r-gguf`, `quant` `Q4_K_M`, distinct `api_name` / `slug`). Merged [PR #41](https://github.com/sskarz/Thesis/pull/41). JSON-in-text adapter `--fc-mode xlam` merged [PR #42](https://github.com/sskarz/Thesis/pull/42).
- **Measured (`multi_turn_base`, `--fc-mode nfc`):** Run 19 **0/200 = 0.0%** (Mac `run_id=19`; wall **13.55 min**; finished `2026-09-22T14:26:42Z` ≈ 10:26 AM ET). **All 200** eval errors: `Model response list is empty for turn 0`. Shapes: 196/200 all-empty `[[],…]`; 4/200 (`_33,_35,_36,_37`) have `["ERROR: "]` cells but still turn-0 empty. Avg `api_calls` **3.63**; gen_tps_mean ≈ **196.7** — **not** a dead server; model generated tokens; NFC harness recorded **no tool calls**. Same-harness contrast: Gemma NFC Run 4 = **54/200 = 27.0%**.
- **Measured (`multi_turn_base`, `--fc-mode xlam`):** Run 20 **6/200 = 3.0%** (Mac `run_id=20`; wall **~13.28 h**; `2026-09-22T15:28:35Z` → `2026-09-23T04:45:30Z` ≈ 11:28 AM ET Tue → 12:45 AM ET Wed). **Format-fixed fair-channel** after Run 19 NFC collapse. Smoke extracted calls (5/5). Full score still **very low**. 194/200 eval errors; `state_mismatch` **128**; under_call/missing exec **46**; empty_turn **20**. Flags: `hit_step_limit` **171/200**; `xlam_unparseable` **111/200**; `context_overflow` **42/200**. Avg `api_calls` **50.47** (min 2, max 101) — tools actually ran vs Run 19 **3.63**. gen_tps_mean ≈ **91.6** (n=50). Correct: `multi_turn_base_{50,119,133,139,144,161}`. Example: `_51` VehicleControl `lockDoors` loop; `_0` vs Run 19 empties is 20× GFS `cd`/`grep`/`sort` loops.
- **Framing:** Run 19 = **scored channel/format collapse** (NFC never parses `content`). Run 20 = **format-fixed fair-channel external baseline**, still **3.0%** — **not** a method win, **not** a training target, still **below CD 10.5%** and all Gemma/SFT/BoN. **Do not** claim xLAM scores 0 on BFCL overall. **Do not** equate 0.0% or 3.0% to published Salesforce overall BFCL ~**78.94%** (different metric / suite aggregation). Do **not** overwrite Run 19 or Run 20.
- **Published overall (not this pipeline):** ~78.94% on the BFCL leaderboard (Zhang et al.) — **keep as citation only**; do **not** write that this repo reproduced or refuted it.
- **Effort:** Registry + NFC + `--fc-mode xlam` **scored**. Tier-1 fair baseline is **closed at 3.0%**.
- **Source:** Zhang et al., "xLAM: Large Action Models," arXiv:2409.03215

### 4. MLX LoRA SFT of Gemma-4-E2B (METHODS — Experiment: MLX SFT)

- **What:** Frozen IFFC-format mix (all ToolACE + 25k oversampled xLAM-60k; seed=42; ~90/5/5 train/val/corpus_test) → mlx-lm LoRA on Apple Silicon → fuse → GGUF. Thesis test remains BFCL v4 `multi_turn_base` (never in the mix).
- **Why:** Training-time counterpart to inference-time IFFC / IFFC+STATE / BoN — same JSON-in-text channel, not native `tool_calls`
- **Measured (Gemma `multi_turn_base`):** Run 12 `--model gemma4-e2b-sft-iffc --fc-mode iffc` **1/200 = 0.5%** (only `multi_turn_base_50`). Run 13 `--model gemma4-e2b-sft-iffc-v2 --fc-mode iffc` **1/200 = 0.5%** (same entry; correct-set overlap identical). **Both scored negative** vs NFC 27.0% / IFFC 30.5% / CD 10.5% / GST 30.5% / STATE 34.0% / BoN 36.5%. Run 12 root cause: 5520 ToolACE assistant `[]` targets (78.1% of post-tool-result turns) teach stop-after-tools — **not** a parser bug (zero `iffc_unparseable`). Run 13: PR #24 dropped 334 non-terminal empty-after-exec (334→0) and kept 552 terminal; filter **did not recover**. Primary buckets Run 13: empty_turn **167**, state_mismatch **25**, under_call **7** (vs Run 12: 101 / 63 / 35). Run 14 `--fc-mode iffc --iffc-strip-empty-assistant` on the same v2 GGUF: **4/200 = 2.0%** (**DIAGNOSTIC** history bookkeeping — **not** a training method). Wins `_50`, `_161`, `_192`, `_193` (shared with Run 13: only `_50`). Primary buckets: empty_turn **160**, state_mismatch **28**, under_call **8**; empty-turn-0 **153** unchanged vs Run 13. Did **not** rescue the never-start collapse. **Do not present 2% as a method win.** Run 15 `--model gemma4-e2b-sft-iffc-v4 --fc-mode iffc` LoRA best@1600 (`val=0.196`): **30/200 = 15.0%** — **scored recovery** from that collapse (empty_turn **2** / state_mismatch **109** / under_call **59**), still **below IFFC 30.5% and below NFC 27.0%**. **Do not present 15% as beating IFFC.** Run 16 `--model gemma4-e2b-sft-iffc-v4 --fc-mode iffc-state` on the **same GGUF**: **26/200 = 13.0%** — **scored negative/null overlay** vs Run 15 (−4 / −2.0 pp). Empty still 3/174; residual state_mismatch **112** / under_call **59**. Contrast: base Gemma STATE (Run 10) was **+3.5 pp** vs IFFC. Run 17 `--model gemma4-e2b-sft-iffc-v5 --fc-mode iffc` LoRA best@1800 (`val=0.147`, [PR #35](https://github.com/sskarz/Thesis/pull/35), ~29850 train): **23/200 = 11.5%** — **scored regression vs v4** (−7 / −3.5 pp vs Run 15; also below Run 16 13.0%). Empty_turn **0** / state_mismatch **133** / under_call **44**. Overlap vs Run 15: both 18, only17 5 (`_12,_21,_42,_161,_169`), lost 12. Run 18 `--model gemma4-e2b-sft-iffc-v6 --fc-mode iffc` LoRA best@1800 (`val=0.142`, [PR #38](https://github.com/sskarz/Thesis/pull/38) recipe, [PR #39](https://github.com/sskarz/Thesis/pull/39) model key): **26/200 = 13.0%** — **scored miss vs v4** (−4 / −2.0 pp vs Run 15; modest +3 / +1.5 pp vs Run 17; ties Run 16). Empty_turn **1** (`_43`) / state_mismatch **118** / under_call **55** (turn0=38). Avg `api_calls` **13.34** (fixed v5’s 40.3 churn). Overlap vs Run 15: both 18, only18 8 (`_42,_47,_102,_144,_161,_169,_186,_192`), lost 12 (mostly TradingBot/VehicleControl state_mismatch; turn-0 under_call `_105,_114,_159,_166`). Details in `THESIS.md`.
- **Do not:** more epochs on v1 or the v2 filtered mix **or the v5 mix or the v6 mix**; BoN/STATE on those collapsed checkpoints **or on v5 or v6**; claim the `[]` filter fixed SFT; present Run 14’s 2.0% as a method win; present Run 15’s 15.0% as beating IFFC; present Run 16’s 13.0% as a STATE win on SFT-v4; present Run 17’s 11.5% as a method win; present Run 18’s 13.0% as a method win; **propose BoN as the next win on the SFT-v4, SFT-v5, or SFT-v6 checkpoint**; remake thin-filter-only ToolACE+xLAM mixes (v5/v6 **closed** as below v4); frame xLAM Runs 19–20 as a Tier-1 win (NFC **0.0%** / `--fc-mode xlam` **3.0%** are **negative external baselines**, not method wins)
- **Decision locked (2026-09-23):** pursue **TRAINING (SFT)** as the main path. Inference wrappers (BoN / STATE / GST / CD) are **secondary**, not the primary next bet. Best measured inference remains BoN **36.5%** (Run 11, base Gemma). Best measured SFT remains v4 IFFC **15.0%** (Run 15).
- **Next (planned, not scored):** SFT **v8 lever 1 / v8.1** (seed 48): same APIGen-MT core as v7; `--recipe v8` / `splits-v8`; observe→same-call drop (consecutive-only was too light) and mlx non-tool ~35%. **Run 21 (v7) collapsed unfinished** — do **not** finish-200 / BoN / more-epochs on v7. **Do not invent a v8 score.** Optional open data-gen complement: Data Turnstile (METHODS §24; **unscored**). Optional post-healthy-SFT local RL: Critical-State RL (METHODS §25; **unscored**; not blanket GRPO — **do not** unlock GRPO as the main path). Recipe v6 is a **measured miss vs v4** (Run 18 13.0% vs Run 15 15.0%). Do **not** BoN v6; do **not** more-epochs the v6 mix. Thin filter ablation on the same family is exhausted. v4 IFFC 15% remains best SFT recovery. IFFC+STATE on the v4 GGUF is **measured negative** (Run 16). Recipe v5 is **measured regression** (Run 17). BoN 36.5% still best inference-time method **on the base Gemma stack**.
- **Lit review:** vault note [`sskarz/thesis-vault` `📝 Notes/2026-09-23-SFT-training-lit-review.md`](https://github.com/sskarz/thesis-vault/blob/main/%F0%9F%93%9D%20Notes/2026-09-23-SFT-training-lit-review.md) (12-source synthesis for v7; vault owns the deep note — may land in a parallel vault PR before it is on `main`). Do **not** paste the full review here.
- **Code:** `sft/` — see `sft/README.md` for Mac commands. v8 scaffolding (unscored): `drop_identical_call.py` / `build_mix_v8.py` / `mix_stats_v8.py` / `student_filter_v8.py`; registry `gemma4-e2b-sft-iffc-v8`. Hub Accept on `Salesforce/APIGen-MT-5k` (CC-BY-NC) required before download.
- **Skipped v1:** APIGen-MT-5k (NC), Glaive, Hermes (docs-only optional backup). v7 **plans** to center on APIGen-MT (not scored).
- **Effort:** Medium-High (overnight train on Mac; not CI)
- **Hardware:** Apple Silicon (mlx-lm). RTX 4090 QLoRA of Qwen remains a later optional path (METHODS §18)
- **Source:** ToolACE arXiv:2409.00920; ToolACE-R arXiv:2504.01400; APIGen-MT arXiv:2504.03601; xLAM arXiv:2409.03215; Data Turnstile arXiv:2607.29250 (optional open data-gen complement, **unscored**); Critical-State RL arXiv:2609.24985 (post-healthy-SFT local RL, **unscored**; not blanket GRPO); mlx-lm LoRA docs

## Tier 2 — Should Do

Strengthens the thesis narrative and fills gaps in current evaluation.

### 5. Multi-Turn Evaluation

- **What:** Run `generate_multi_turn.py` + `evaluate_multi_turn.py` for all 4 multi-turn categories (800 test cases)
- **Why:** Thesis claims "sequential, context-dependent tool calling is critical" but has no multi-turn data yet
- **Effort:** Low (pipeline exists)

### 6. Context Compression for Multi-Turn (METHODS §5)

- **What:** Sliding window on `chat_history` — keep last 2 turns verbatim, summarize or truncate older turns
- **Why:** Directly addresses context overflow in multi-turn pipeline where `chat_history` grows indefinitely
- **Target categories:** All multi-turn, especially `multi_turn_long_context`
- **Expected impact:** -30–50% tokens, improved multi-turn accuracy
- **Effort:** Medium
- **Source:** Xu et al., "ACON," arXiv:2510.00615

### 7. Parallel Calling Investigation

- **What:** Test `tool_choice=required` + explicit GBNF grammar enforcing array-of-tool-calls structure
- **Why:** Even a partial unlock (10–20%) on the 0% categories is a publishable finding
- **Target categories:** `parallel`, `parallel_multiple`
- **Effort:** Medium
- **Note:** llama.cpp already uses grammar-constrained decoding for tool calls in lazy mode — investigate whether a stricter grammar for arrays can help

## Tier 3 — Nice to Have

### 8. Jetson Orin Nano Super Migration

- **What:** Repeat benchmarks on actual edge hardware (8GB shared CUDA memory)
- **Why:** Thesis mentions "resource-constrained hardware" — running on real edge hardware solidifies the claim
- **Effort:** High (hardware setup, CUDA build, full benchmark re-run)

### 9. RAG Tool Retrieval (METHODS §3)

- **What:** Embed BFCL function descriptions with `all-MiniLM-L6-v2`, build FAISS index, retrieve top-k tools per query
- **Why:** Only valuable with extended benchmark (50+ distractor tools). BFCL's 2–4 tools per test case won't show the benefit
- **Effort:** Medium-High (need to create extended benchmark with padded tool sets)
- **Complement (unscored):** Tool-interface granularity (METHODS §3a, MCP-GRANITE arXiv:2609.24161) — same capability set split into tools is a design knob; paper reports moderate ~4 task-class tools beat fine-grained and monolith on their edge/IoT suite. Redesign the API surface, don't only retrieve. **Not a scored thesis experiment.**
- **Source:** Gao et al., "RAG-MCP," arXiv:2505.03275

## Thesis Narrative

The experiment plan creates a three-way comparison:

```
Baseline (current)    ->  Inference-time optimized  ->  Fine-tuned
    |                           |                         |
Qwen2.5-3B: 76%        + templates + voting: ??%    QLoRA: ??%
Qwen3.5-4B: 82%        + templates + voting: ??%    QLoRA: ??%
xLAM-1b NFC (this harness, Run 19): 0.0%  ← format-channel collapse; NOT published ~78.94%
xLAM-1b --fc-mode xlam (Run 20): 3.0%  ← fair channel; still very low; NOT a method win; NOT ~78.94%
```

This answers the central question from METHODS.md: **does fine-tuning or inference-time optimization close the SLM-to-cloud-model gap more effectively for tool calling on edge devices?** **Decision locked (2026-09-23):** SFT is the main path; inference wrappers are secondary. Next planned method is SFT **v7** (**not scored**).

## Execution Order

1. Guided-Structured Templates (prompt change only)
2. Run multi-turn evaluation (pipeline exists)
3. Add xLAM-1b-fc-r to model registry and benchmark — **registry key `xlam-1b` exists** ([PR #41](https://github.com/sskarz/Thesis/pull/41)); **Run 19 NFC scored 0/200 = 0.0%** (format-channel collapse; **not** published overall ~78.94%). **Run 20 `--fc-mode xlam` scored 6/200 = 3.0%** (fair channel after [PR #42](https://github.com/sskarz/Thesis/pull/42); still **very low**; **not** a method win; still **below CD 10.5%**). Do **not** overwrite. **Do not claim published ~78.94% parity**
4. Best-of-N majority voting wrapper
5. Context compression for multi-turn
6. Parallel calling grammar investigation
7. MLX SFT v1 **and** v2 are **scored negative** (Run 12 **1/200 = 0.5%**; Run 13 **1/200 = 0.5%** — non-terminal-`[]` filter did not recover). Run 14 `--iffc-strip-empty-assistant` on the v2 GGUF is a **scored DIAGNOSTIC** (**4/200 = 2.0%**) — did **not** rescue; **do not present 2% as a method win**. Run 15 SFT-v4 is a **scored recovery** (**30/200 = 15.0%**) — empty-`[]` largely fixed; residual is state/under-call; still **below IFFC 30.5% and NFC 27.0%**. **Do not present 15% as beating IFFC.** Run 16 IFFC+STATE on that same v4 GGUF is a **scored negative/null overlay** (**26/200 = 13.0%**, −2.0 pp vs Run 15) — STATE helps base Gemma, not this ckpt. **Do not propose BoN as the next win on this checkpoint.** Run 17 SFT-v5 is a **scored regression** (**23/200 = 11.5%**, −3.5 pp / −7 vs Run 15) — empty still 0; residual state 133 / under-call 44. **Do not present 11.5% as a method win.** **No BoN / no more epochs on the v5 mix.** Recipe v5 is **closed as measured regression**. Run 18 SFT-v6 is a **scored miss vs v4** (**26/200 = 13.0%**, −2.0 pp / −4 vs Run 15; +1.5 pp vs Run 17) — empty still essentially 1; residual state 118 / under-call 55; avg api_calls 13.34. **Do not present 13.0% as a method win.** **No BoN / no more epochs on v6.** Recipe v6 is **closed as still below v4**. v4 IFFC 15% remains best SFT recovery. BoN 36.5% still best **on base Gemma**. Do not more epochs on v1/v2 / BoN on collapsed GGUFs / claim the filter fixed SFT. **Decision locked (2026-09-23):** SFT is the main path; next intended method is SFT **v7** (APIGen-MT verified MT trajs + calibrated stop/clarify + student-aware filter; **not scored**). Inference wrappers are secondary.
8. Jetson migration (if time permits)
