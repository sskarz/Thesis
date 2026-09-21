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
- **Measured (Gemma `multi_turn_base`):** Run 11 `--fc-mode iffc-state-bon` **73/200 = 36.5%** vs IFFC+STATE Run 10 68/200 = 34.0% (**+2.5 pp** at ~7× wall / ~8.8× decode). Modest lift, **not free**. Details in `THESIS.md`.
- **Expected impact (pre-score):** +5–15% accuracy, N× latency increase
- **Effort:** Medium
- **Key trade-off story:** FunctionGemma at 204 tok/s makes N=5 cheap (~80ms); Qwen3.5 at 38 tok/s makes it expensive (~3.3s); Gemma BoN N=5 measured ~21.3 h vs STATE ~3.0 h — this *is* the trade-off narrative
- **Source:** Snell et al., "Scaling LLM Test-Time Compute," arXiv:2408.03314

### 3. xLAM-1b-fc-r as Benchmark Baseline (METHODS §15)

- **What:** Run existing pipeline on `Salesforce/xLAM-1b-fc-r-gguf` — zero code change
- **Why:** Provides a fine-tuned SLM comparison point. Answers: "how far can inference-time methods close the gap toward a purpose-built model?"
- **Expected score:** ~78.94% overall (published BFCL result)
- **Effort:** Low (just add to config.py model registry, run benchmark)
- **Source:** Zhang et al., "xLAM: Large Action Models," arXiv:2409.03215

### 4. MLX LoRA SFT of Gemma-4-E2B (METHODS — Experiment: MLX SFT)

- **What:** Frozen IFFC-format mix (all ToolACE + 25k oversampled xLAM-60k; seed=42; ~90/5/5 train/val/corpus_test) → mlx-lm LoRA on Apple Silicon → fuse → GGUF. Thesis test remains BFCL v4 `multi_turn_base` (never in the mix).
- **Why:** Training-time counterpart to inference-time IFFC / IFFC+STATE / BoN — same JSON-in-text channel, not native `tool_calls`
- **Measured (Gemma `multi_turn_base`):** Run 12 `--model gemma4-e2b-sft-iffc --fc-mode iffc` **1/200 = 0.5%** (only `multi_turn_base_50`). Run 13 `--model gemma4-e2b-sft-iffc-v2 --fc-mode iffc` **1/200 = 0.5%** (same entry; correct-set overlap identical). **Both scored negative** vs NFC 27.0% / IFFC 30.5% / CD 10.5% / GST 30.5% / STATE 34.0% / BoN 36.5%. Run 12 root cause: 5520 ToolACE assistant `[]` targets (78.1% of post-tool-result turns) teach stop-after-tools — **not** a parser bug (zero `iffc_unparseable`). Run 13: PR #24 dropped 334 non-terminal empty-after-exec (334→0) and kept 552 terminal; filter **did not recover**. Primary buckets Run 13: empty_turn **167**, state_mismatch **25**, under_call **7** (vs Run 12: 101 / 63 / 35). Run 14 `--fc-mode iffc --iffc-strip-empty-assistant` on the same v2 GGUF: **4/200 = 2.0%** (**DIAGNOSTIC** history bookkeeping — **not** a training method). Wins `_50`, `_161`, `_192`, `_193` (shared with Run 13: only `_50`). Primary buckets: empty_turn **160**, state_mismatch **28**, under_call **8**; empty-turn-0 **153** unchanged vs Run 13. Did **not** rescue the never-start collapse. **Do not present 2% as a method win.** Run 15 `--model gemma4-e2b-sft-iffc-v4 --fc-mode iffc` LoRA best@1600 (`val=0.196`): **30/200 = 15.0%** — **scored recovery** from that collapse (empty_turn **2** / state_mismatch **109** / under_call **59**), still **below IFFC 30.5% and below NFC 27.0%**. **Do not present 15% as beating IFFC.** Details in `THESIS.md`.
- **Do not:** more epochs on v1 or the v2 filtered mix; BoN/STATE on those collapsed checkpoints; claim the `[]` filter fixed SFT; present Run 14’s 2.0% as a method win; present Run 15’s 15.0% as beating IFFC
- **Next (planned):** IFFC+STATE on the **same v4 GGUF** (cheap residual ablation — Thesis Bot). Empty pathology is largely fixed; residual is state/under-call. BoN 36.5% still best inference-time method.
- **Code:** `sft/` — see `sft/README.md` for Mac commands
- **Skipped v1:** APIGen-MT-5k (NC), Glaive, Hermes (docs-only optional backup)
- **Effort:** Medium-High (overnight train on Mac; not CI)
- **Hardware:** Apple Silicon (mlx-lm). RTX 4090 QLoRA of Qwen remains a later optional path (METHODS §18)
- **Source:** ToolACE arXiv:2409.00920; xLAM arXiv:2409.03215; mlx-lm LoRA docs

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
- **Source:** Gao et al., "RAG-MCP," arXiv:2505.03275

## Thesis Narrative

The experiment plan creates a three-way comparison:

```
Baseline (current)    ->  Inference-time optimized  ->  Fine-tuned
    |                           |                         |
Qwen2.5-3B: 76%        + templates + voting: ??%    QLoRA: ??%
Qwen3.5-4B: 82%        + templates + voting: ??%    QLoRA: ??%
xLAM-1b (fine-tuned baseline): ??%
```

This answers the central question from METHODS.md: **does fine-tuning or inference-time optimization close the SLM-to-cloud-model gap more effectively for tool calling on edge devices?**

## Execution Order

1. Guided-Structured Templates (prompt change only)
2. Run multi-turn evaluation (pipeline exists)
3. Add xLAM-1b-fc-r to model registry and benchmark
4. Best-of-N majority voting wrapper
5. Context compression for multi-turn
6. Parallel calling grammar investigation
7. MLX SFT v1 **and** v2 are **scored negative** (Run 12 **1/200 = 0.5%**; Run 13 **1/200 = 0.5%** — non-terminal-`[]` filter did not recover). Run 14 `--iffc-strip-empty-assistant` on the v2 GGUF is a **scored DIAGNOSTIC** (**4/200 = 2.0%**) — did **not** rescue; **do not present 2% as a method win**. Run 15 SFT-v4 is a **scored recovery** (**30/200 = 15.0%**) — empty-`[]` largely fixed; residual is state/under-call; still **below IFFC 30.5% and NFC 27.0%**. **Do not present 15% as beating IFFC.** Next planned: IFFC+STATE on this same v4 GGUF (Thesis Bot). BoN 36.5% still best. Do not more epochs on v1/v2 / BoN on collapsed GGUFs / claim the filter fixed SFT.
8. Jetson migration (if time permits)
