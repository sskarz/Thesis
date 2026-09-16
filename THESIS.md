# Optimizing Multiple Consecutive Tool Calling for Small Language Models in Agentic Applications

## Thesis Statement

Small language models (SLMs, <5B parameters) can be optimized for agentic workflows that require selecting and invoking the correct tool from a set of available tools across multiple consecutive steps. Through empirical benchmarking on resource-constrained hardware, this work evaluates how well quantized SLMs handle multi-tool selection — choosing the right function from multiple candidates — and identifies the accuracy, latency, and memory trade-offs that determine their viability for agentic applications where sequential, context-dependent tool calling is critical.

## Motivation & Importance

### Agentic AI demands reliable tool selection

Agentic applications require models to autonomously choose the correct tool from many available options across multiple consecutive steps. A wrong tool choice early in a sequence cascades into downstream failures. The model must not only understand the user's intent but also match it to the correct function signature, populate arguments with the right types and values, and — critically — know when *not* to call any tool at all.

### SLMs are the deployment target

Cloud-scale models (70B+) are well-studied for tool calling, but agentic applications on edge devices, embedded systems, and local-first architectures need SLMs (<5B parameters). Their tool-selection capabilities are underexplored. This work targets the gap between what large models can do and what small models actually achieve on standard benchmarks.

### Multiple tools in context is the hard problem

Single-function calling is largely solved at the 3B+ scale. The real challenge is when the model must reason over multiple available tools and select the correct one — the "multiple" and "irrelevance" categories in BFCL directly measure this. A model that scores well on `simple` (one function available) but poorly on `multiple` (several functions, pick the right one) reveals a fundamental limitation in tool discrimination.

### Consecutive calling enables agentic loops

An agent that can reliably call tool A, interpret the result, then call tool B based on that result unlocks autonomous workflows — from data pipelines to robotics to personal assistants. The `parallel` and `parallel_multiple` BFCL categories test a prerequisite for this: emitting multiple tool calls in a single turn.

### Resource constraints shape the design space

Edge devices (NVIDIA Jetson Orin Nano Super: 8 GB, smartphones: 6–8 GB) constrain model size, making quantization strategy and model architecture critical choices for agentic deployment. A model that fits in 4 GB with sub-second latency opens fundamentally different deployment scenarios than one requiring 12 GB with 700 ms time-to-first-token.

## Methodology

### Benchmark: BFCL v4

All models are evaluated on the Berkeley Function Calling Leaderboard (BFCL) v4. Single-turn evaluation uses 7 AST-scored categories totaling 1,390 test cases:

| Category | Test Cases | Description |
|----------|-----------|-------------|
| Simple | 400 | Single function available, call it correctly |
| Multiple | 200 | Multiple functions available, select the correct one |
| Parallel | 200 | Multiple functions need to be called in a single turn |
| Parallel Multiple | 200 | Multiple functions from multiple candidates in a single turn |
| Irrelevance | 240 | Functions available but the query does not require any call |
| Java | 100 | Function calling with Java-style type signatures |
| JavaScript | 50 | Function calling with JavaScript-style type signatures |

Multi-turn categories (800 test cases) evaluate sequential, stateful tool use. Scoring uses BFCL's **execution checker** rather than AST matching: the model enters an agent loop, tools are executed against a live backend, and the resulting state is compared to a reference trajectory.

| Category | Test Cases | Description |
|----------|-----------|-------------|
| Multi-turn base | 200 | Full tool set available; consecutive turns with execution feedback |
| Multi-turn miss_func | 200 | Some required functions are withheld until a later turn |
| Multi-turn miss_param | 200 | Required parameters are omitted from the user query |
| Multi-turn long_context | 200 | Large function-doc / history load; stress-tests context limits |

### Evaluation method

- **AST checker** (single-turn): BFCL's Abstract Syntax Tree checker compares model output against reference answers, allowing equivalent but differently-ordered arguments to match.
- **Execution checker** (multi-turn): BFCL executes the model's tool calls against a live backend and compares instance state to the ground-truth trajectory.
- **Deterministic output**: All models run with `--temp 0.0` to ensure reproducible results.
- **No reasoning mode** (single-turn baselines): Single-turn runs disable thinking/reasoning even if the model supports it (e.g., Qwen3.5), isolating raw tool-calling capability from chain-of-thought augmentation. The Gemma-4-E2B-it multi-turn runs (Run 4 NFC, Run 5 IFFC) left reasoning on.

### Infrastructure

- **Inference server**: llama.cpp build 8111 (commit `11c325c6e`), compiled with Metal GPU backend (`-DGGML_METAL=ON`).
- **Hardware**: Apple M4 Pro (Mac), Metal GPU acceleration, `-ngl 99` (all layers offloaded to GPU).
- **Context window**: 4,096 tokens for all models.
- **Serving**: OpenAI-compatible API via `llama-server` with `--jinja` for chat template support and `-fa on` for flash attention.

### FunctionGemma output format handling

FunctionGemma uses a non-standard output format with custom control tokens:

```
<start_function_call>call:get_weather{city:<escape>San Francisco<escape>}<end_function_call>
```

While llama-server b8111 successfully parses this into standard `tool_calls` via the Jinja template in most cases, a custom fallback parser (`benchmark/parse_functiongemma.py`) handles edge cases where the output arrives as raw text in `content`. The parser extracts function calls via regex, parses arguments using a character-level state machine that respects brace depth and `<escape>` regions, and converts them to the `[{"func_name": {"arg1": val1}}]` format expected by the BFCL evaluator. This is called automatically by `generate_responses.py` and is transparent to the rest of the pipeline.

## Models Tested

| Model | Architecture | Parameters | Quantization | GGUF Size | HuggingFace Source |
|-------|-------------|-----------|--------------|-----------|-------------------|
| Qwen2.5-3B-Instruct | Qwen 2.5 | 3B | Q4_K_M | ~1.8 GB | `bartowski/Qwen2.5-3B-Instruct-GGUF` |
| Llama-3.2-3B-Instruct | Llama 3.2 | 3.2B | Q4_K_M | ~2.0 GB | `bartowski/Llama-3.2-3B-Instruct-GGUF` |
| Qwen3.5-4B-Instruct | Qwen 3.5 | 4B | Q4_K_M | ~2.8 GB | `bartowski/Qwen_Qwen3.5-4B-GGUF` |
| FunctionGemma-270M-IT | Gemma 3 (specialist) | 270M | bf16 | ~512 MB | `ggml-org/functiongemma-270m-it-GGUF` |

Gemma-4-E2B-it (Q4_K_M) was additionally evaluated on **multi-turn categories only**: Run 4 NFC (`--only-multi-turn`, all four slices), Run 5 IFFC (`--categories multi_turn_base --fc-mode iffc`), Run 8 TagDispatch-style CD (`--categories multi_turn_base --fc-mode cd`; **21/200 = 10.5%**, a scored negative vs NFC/IFFC), Run 9 GST+NFC (`--categories multi_turn_base --fc-mode gst`; **61/200 = 30.5%**, ties IFFC, does not beat it), Run 10 IFFC+STATE (`--categories multi_turn_base --fc-mode iffc-state`; **68/200 = 34.0%**, first method above the 30.5% plateau), and Run 11 Best-of-N on IFFC+STATE (`--categories multi_turn_base --fc-mode iffc-state-bon`; **73/200 = 36.5%**, beats IFFC+STATE by +5 entries / +2.5 pp at ~7× wall). A shell-dialect format ablation ([PR #5](https://github.com/sskarz/Thesis/pull/5) harness; v2 after [PR #9](https://github.com/sskarz/Thesis/pull/9)) was run on the 13-entry `fs-only` GorillaFileSystem subset of `multi_turn_base`; that slice is **not** a full-200 score (v1 1/13, v2 0/13). Single-turn scores for this model are not reported here.

### Quantization notes

The three generalist models (Qwen2.5-3B, Llama-3.2-3B, Qwen3.5-4B) are all served at Q4_K_M quantization — a 4-bit mixed quantization scheme that balances model size against accuracy loss.

FunctionGemma-270M is served at bf16 (brain float 16) precision. This is the only available quantization for this model; no Q4_K_M or other quantized variants are published. At 270M parameters, aggressive quantization yields diminishing returns — the model is already small enough that the overhead of dequantization logic can negate size savings. Even at full bf16 precision, FunctionGemma's GGUF (~512 MB) is **3.5–5.5x smaller** than the Q4_K_M generalists, making it a realistic edge deployment target:

- **Smartphones** (iPhone, Pixel): 6–8 GB RAM with dedicated neural engines. A 512 MB model leaves ample headroom.
- **NVIDIA Jetson Orin Nano Super**: 8 GB shared CUDA memory. A 512 MB model uses ~6% of available VRAM.
- **Raspberry Pi 5 (8 GB)**: CPU-only inference is feasible at this model size.

The comparison remains controlled because the variable under study is **model architecture and specialization**, not quantization method.

## Accuracy Results

Single-turn results on BFCL v4, evaluated with the AST checker. **Bold** indicates best-in-class per category. These tables do not include Gemma-4-E2B-it (multi-turn only).

| Category | Qwen2.5-3B (Q4_K_M) | Llama-3.2-3B (Q4_K_M) | Qwen3.5-4B (Q4_K_M) | FunctionGemma-270M (bf16) |
|----------|---------------------|----------------------|---------------------|--------------------------|
| Simple (400) | **373/400 (93.25%)** | 207/400 (51.75%) | 361/400 (90.25%) | 205/400 (51.25%) |
| Multiple (200) | **185/200 (92.50%)** | 67/200 (33.50%) | 181/200 (90.50%) | 118/200 (59.00%) |
| Parallel (200)* | 0/200 (0.00%) | 0/200 (0.00%) | 0/200 (0.00%) | 0/200 (0.00%) |
| Parallel Multiple (200)* | 0/200 (0.00%) | 0/200 (0.00%) | 0/200 (0.00%) | 0/200 (0.00%) |
| Irrelevance (240) | 141/240 (58.75%) | 2/240 (0.83%) | **219/240 (91.25%)** | 48/240 (20.00%) |
| Java (100) | 37/100 (37.00%) | **54/100 (54.00%)** | 39/100 (39.00%) | 15/100 (15.00%) |
| JavaScript (50) | 17/50 (34.00%) | **35/50 (70.00%)** | 17/50 (34.00%) | 9/50 (18.00%) |

\* *Parallel scores invalidated by a llama-server grammar bug that truncated generation after the first tool call. See [Parallel calling](#parallel-calling-server-side-grammar-bug-not-a-model-limitation) for details. Fix implemented; re-run pending.*

### Overall accuracy (excluding parallel categories — affected by server bug, pending re-run)

| Model | Weighted Avg (5 categories, 990 cases) |
|-------|-----------------------------------------|
| Qwen2.5-3B | 753/990 (76.06%) |
| Qwen3.5-4B | 817/990 (82.53%) |
| Llama-3.2-3B | 365/990 (36.87%) |
| FunctionGemma-270M | 395/990 (39.90%) |

## Multi-Turn Accuracy Results

### NFC vs IFFC vs CD vs GST vs IFFC+STATE vs BoN on `multi_turn_base`

Primary metric: BFCL v4 `multi_turn_base` (200 execution-evaluated cases) on Gemma-4-E2B-it Q4_K_M, Mac Metal llama-server. CD is `--fc-mode cd`: a TagDispatch-style **two-pass approximation** on stock llama-server (pass 1: name GBNF enum; pass 2: `json_schema` for selected tools; never sends `tools` / `tool_choice`). **This is not XGrammar-2.** GST is `--fc-mode gst`: a **prompt-only overlay on NFC** (He et al. Detail template; still `tools` + `tool_choice=auto`). **This is not GBNF / XGrammar / IFFC.** IFFC+STATE is `--fc-mode iffc-state`: IFFC JSON-in-text plus a **harness-written `[STATE]`** snapshot from actual tool execution results. **This is not GST, not CD/GBNF, and not LLM summarization.** BoN is `--fc-mode iffc-state-bon`: **MV@N majority vote** (N=5, `temperature=0.7`) on the IFFC+STATE channel. **This is not NFC, not GST, not CD, and not a verifier / reward-model Best-of-N.**

| Mode | Run | Score | Accuracy | Δ vs NFC |
|------|-----|-------|----------|----------|
| NFC (native `tools` / `tool_calls`) | 4 | 54/200 | 27.0% | — |
| IFFC (Appendix B JSON-in-text) | 5 | 61/200 | 30.5% | +3.5 pp / +7 entries |
| CD (TagDispatch-style two-pass; **not** XGrammar-2) | 8 | 21/200 | 10.5% | −16.5 pp / −33 entries |
| GST+NFC (He Detail template on NFC; **not** GBNF) | 9 | 61/200 | 30.5% | +3.5 pp / +7 entries |
| IFFC+STATE (IFFC JSON-in-text + harness `[STATE]`) | 10 | 68/200 | 34.0% | +7.0 pp / +14 entries |
| **BoN N=5 on IFFC+STATE** (MV@N, `temp=0.7`) | **11** | **73/200** | **36.5%** | **+9.5 pp / +19 entries** |

Run 8 CD is a **scored negative** vs both NFC and IFFC (10.5% < 27.0% / 30.5%; 179 errors). Run 9 GST+NFC **ties IFFC at 61/200 = 30.5%** — a count tie, not the same 61 entries. Run 10 IFFC+STATE is the **first method above that 30.5% plateau** (+7 entries / +3.5 pp vs IFFC and vs GST). Run 11 BoN **beats IFFC+STATE** (+5 entries / +2.5 pp) at ~7× wall / ~N× decode — a modest lift, **not free**. Details below. The 13-entry BFCL-FS shell ablation is a separate slice and is **not** in this table.

### Gemma-4-E2B-it baseline (Run 4, 2026-09-10)

This is a multi-turn-only baseline: `--only-multi-turn`, 800 execution-evaluated cases, no single-turn categories. Serving was MacBook Pro M4 Pro, llama.cpp Metal, Q4_K_M, reasoning on, `-np 8` / `PARALLEL_SLOTS=8`.

| Category | Score | Accuracy | Flagged |
|----------|-------|----------|---------|
| `multi_turn_base` | 54/200 | 27.0% | 2 |
| `multi_turn_miss_func` | 16/200 | 8.0% | 5 |
| `multi_turn_miss_param` | 39/200 | 19.5% | 2 |
| `multi_turn_long_context` | 22/200 | 11.0% | 72 (mostly `context_overflow`) |

**`multi_turn_base` (full 200) remains the primary optimization target.** Run 4 NFC scored 27.0% on that slice; Run 5 IFFC (+3.5 pp), Run 8 CD (scored negative, 10.5%), Run 9 GST+NFC (ties IFFC at 30.5%), Run 10 IFFC+STATE (**68/200 = 34.0%**, first break past 30.5%), and Run 11 BoN on IFFC+STATE (**73/200 = 36.5%**, +2.5 pp vs STATE at ~7× wall) are reported below. The BFCL-FS shell eval is a 13-entry FS-only ablation, not a substitute for those 200-entry scores. `miss_func`, `miss_param`, and `long_context` are secondary stress slices (withheld tools, missing parameters, and long histories). They are not the headline success criterion.

Wall-clock generation throughput was ~24–37 s/entry at 8-way parallel, versus ~4–5 min/entry on the prior sequential path.

#### Failure modes on `multi_turn_base`

Among failures on the base slice, the dominant modes were approximately **64% state mismatch**, **23% empty/under-calling**, and **12% execution subsequence miss**. By environment, VehicleControl, FileSystem+Twitter, and Ticket+Travel were the weakest; TradingBot was relatively strong (~65%).

Eval-error rows (counts of error rows, not unique entries): `empty_turn` ≈ 118, `state_mismatch` ≈ 136, `missing_gt_exec` ≈ 38.

### Gemma-4-E2B-it IFFC (Run 5)

Run 5 evaluates Instruction-Followed Function Calling (IFFC; [arXiv:2608.22472](https://arxiv.org/abs/2608.22472) Appendix B) on the same model and hardware as Run 4, on `multi_turn_base` only. Serving: Mac M4 Pro, llama.cpp Metal, Gemma-4-E2B-it Q4_K_M, reasoning on, `-np 4` / `PARALLEL_SLOTS=4` (~16k ctx/slot). Run notes: `fc_mode=iffc`. Generation wall-clock ~2h17.

IFFC omits native `tools` / `tool_choice`; the model emits a JSON list in assistant text. Truncations = 0; no `context_overflow`. JSON-in-text held (no fallback to native tool_calls). Stored trajectories are BFCL func-strings (`mkdir(...)`, `mv(...)`), not raw OpenAI `tool_calls`.

#### NFC vs IFFC on `multi_turn_base`

| Mode | Run | Score | Accuracy | Δ vs NFC |
|------|-----|-------|----------|----------|
| NFC (native `tools` / `tool_calls`) | 4 | 54/200 | 27.0% | — |
| IFFC (Appendix B JSON-in-text) | 5 | 61/200 | 30.5% | +3.5 pp / +7 entries |

Pairwise over the same 200 entries: both fail = 124, IFFC rescues vs NFC = 22, IFFC regresses (NFC-pass → IFFC-fail) = 15.

Eval-error rows (counts of error rows, not unique entries):

| Bucket | Run 4 NFC | Run 5 IFFC |
|--------|-----------|------------|
| `empty_turn` | ≈118 | ≈41 |
| `state_mismatch` | ≈136 | ≈60 |
| `missing_gt_exec` | ≈38 | ≈38 |

Top Run 5 fail strings (by error-row count): VehicleControlAPI state 22, missing GT exec turn 0 21, empty turn 1 19, GorillaFileSystem state 18, empty turn 2 10, TravelAPI state 10.

#### Was the eval fair?

The harness matches arXiv:2608.22472 Appendix B system prompt (verbatim, including the curly apostrophe in "doesn’t"). IFFC omits `tools` / `tool_choice`; parses a JSON list of `{function_name, parameters}`; maps those dicts to BFCL func-strings; and feeds tool results back as **user** messages. Scoring uses the same BFCL `multi_turn_checker` as NFC. The run is **not misconfigured**. Residual failures are mostly agentic state mismatch and under-calling, not format parsing.

#### Why not paper-scale gains?

The paper ([*Small Reasoning Models are Instruction Followers in Function Calling*](https://arxiv.org/abs/2608.22472), arXiv:2608.22472) reports large IFFC lifts on some models. The +3.5 pp here is modest for four reasons:

1. **Different task.** Paper tables are BFCL V3 Live / Non-Live *single-turn* categories (`simple` / `multiple` / `parallel` / `parallel_multiple`). The primary metric here is BFCL v4 **`multi_turn_base`** (stateful consecutive tool use). Multi-turn agentic failures (state mismatch, empty turns) dominate; once NFC already produces valid calls, the format channel is not the main bottleneck.

2. **Partial IFFC.** The paper's full IFFC is Algorithm 1 dual-model decoupling: a dedicated small reasoning model (SRM) for function calling plus a separate QA LLM, with context isolation. This run implements Appendix B JSON-in-text on the *same* Gemma for the whole multi-turn loop — closest to their instruction-followed FC / PFC-style SRM path, **without** QA/SRM decoupling. Dual-model IFFC is deferred.

3. **Paper deltas vary with NFC strength.** Large jumps often occur where NFC is weak (e.g. Phi-4 Mini Live NFC ~30% → IFFC 44.3%; Non-Live NFC 9.5% → 77.4%). On already-strong NFC, gains are smaller (Qwen-3 4B Think Live avg NFC 81.5 → IFFC 86.7, ≈ +5 pp). The +3.5 pp on `multi_turn_base` sits in that modest band.

4. **What did improve.** Empty-turn / under-calling error rows dropped sharply (118 → 41); state mismatches also fell (136 → 60). Net +7 entries. Regressions exist (15 entries NFC-pass → IFFC-fail), often later-turn empty lists after early success. `missing_gt_exec` was unchanged (≈38).

IFFC reduced under-calling; it did not convert this into a single-turn-style accuracy jump. The remaining bottleneck is sequential state tracking, not JSON vs native tools.

A TagDispatch-style constrained-decoding ablation (`--fc-mode cd`) on the same 200-entry `multi_turn_base` is reported next; it is a **scored negative**. Guided-Structured Templates (`--fc-mode gst`, Run 9) follow; they **tie IFFC** and do not beat it. IFFC + harness STATE (`--fc-mode iffc-state`, Run 10) follows GST; it is the **first method above 30.5%**. Best-of-N majority vote on IFFC+STATE (`--fc-mode iffc-state-bon`, Run 11) follows STATE; it **beats STATE at 73/200 = 36.5%** with a ~7× wall-clock cost. A separate format channel — allowlisted shell text mapped onto GorillaFileSystem — is a **13-entry `fs-only` ablation** (two live evals, v1 and v2), not a replacement for the 200-entry scores.

### Gemma-4-E2B-it TagDispatch-style CD (Run 8) — scored negative

Run 8 evaluates `--fc-mode cd` on `multi_turn_base` only: a TagDispatch-style **two-pass approximation** of constrained decoding on **stock llama-server** (Dong et al., XGrammar-2 TagDispatch, [arXiv:2601.04426](https://arxiv.org/abs/2601.04426)). **Do not call this XGrammar-2.** XGrammar-2 TagDispatch switches sub-grammars inside one decode; llama.cpp cannot do that, so the harness uses two HTTP completions:

1. **Pass 1 (dispatch):** GBNF enum over live tool names (or stop with `[]`).
2. **Pass 2 (dispatched):** `json_schema` for the selected tool(s) arguments, compiled to GBNF by llama-server.

Requests **never** send `tools` / `tool_choice` (native tools GBNF is not this experiment). The persistent multi-turn transcript stays IFFC-shaped (JSON list in assistant text; user-role execution results). Scoring uses the same BFCL `multi_turn_checker` as NFC / IFFC.

Config: Gemma-4-E2B-it Q4_K_M, Mac Metal llama-server, `-np 4 -c 65536`, `run_id=8`. Overnight resume with per-entry checkpoints. Wall ~115 min (approx 10:46 PM → 12:41 AM ET). Optional figures: `benchmark/results/Gemma-4-E2B-it-Q4_K_M/figures/`.

Result: **21/200 = 10.5%** (179 errors).

#### NFC vs IFFC vs CD on `multi_turn_base`

Same category, same model family, same hardware class as the controls:

| Mode | Run | Score | Accuracy | Δ vs NFC |
|------|-----|-------|----------|----------|
| NFC (native `tools` / `tool_calls`) | 4 | 54/200 | 27.0% | — |
| IFFC (Appendix B JSON-in-text) | 5 | 61/200 | 30.5% | +3.5 pp / +7 entries |
| CD (TagDispatch-style two-pass; **not** XGrammar-2) | 8 | 21/200 | 10.5% | −16.5 pp / −33 entries |

CD is **−20.0 pp / −40 entries** vs IFFC (10.5% < 30.5%). This condition is **measured and negative**, not a pending overnight.

Top Run 8 fail strings (error-row counts, not unique entries):

| Fail string | Error rows |
|-------------|------------|
| missing GT execution turn 0 | 46 |
| VehicleControlAPI state mismatch | 29 |
| GorillaFileSystem | 20 |
| TravelAPI | 19 |
| TradingBot | 15 |
| empty turn 0 | 10 |

#### Interpretation

Two-pass CD on stock llama-server **hurt** `multi_turn_base` relative to both NFC and IFFC. Dominant errors remain agentic (missing ground-truth execution on turn 0, environment state mismatch), not a result that can be attributed to full XGrammar-2. This run does **not** test intra-request tag-triggered grammar switching, Cross-Grammar Cache, or an LLGuidance/`-DLLAMA_LLGUIDANCE=ON` rebuild. Do not treat two-pass `--fc-mode cd` as the D1 default given 10.5% < 27.0% / 30.5%.

### Gemma-4-E2B-it GST+NFC (Run 9)

Run 9 evaluates `--fc-mode gst` on `multi_turn_base` only: Guided-Structured Templates (He et al., [arXiv:2509.18076](https://arxiv.org/abs/2509.18076) Appendix A.2 **Detail** template) as a **prompt-only overlay on NFC**. Requests still send `tools` + `tool_choice=auto`. The curriculum is injected once as a system message (`benchmark/gst.py`). The model is instructed to write a terse `[GST]…[/GST]` block (steps 1–7), then emit native `tool_calls`. The BFCL execution checker runs **`tool_calls` only**; GST drafts are ignored. GST-only content with no `tool_calls` is treated as **ABSTAIN** (end of turn).

**Not** GBNF, XGrammar, or IFFC JSON-in-text. The paper template is single-turn; this harness adds prior tool observations into the curriculum, step-7 ABSTAIN, and a max of 8 GST lines. Function schemas stay on native `tools` — they are **not** dumped into the GST system prompt. [PR #14](https://github.com/sskarz/Thesis/pull/14) (`7ab0be5`) added `--fc-mode gst`.

Config: Gemma-4-E2B-it Q4_K_M, Mac 48 GB, llama.cpp Metal, `-np 4 -c 65536`, `run_id=9`. Started `2026-09-13T02:09:19Z`, finished `2026-09-13T04:26:49Z` (~137.5 min wall). 200/200 responses; 139 `eval_errors`. Notes: `GST+NFC overnight -np4 -c65536 vs NFC 27% / IFFC 30.5% / CD 10.5%; fc_mode=gst categories=multi_turn_base`.

Result: **61/200 = 30.5%** (ties IFFC; +7 vs NFC; +40 vs CD).

#### NFC vs IFFC vs CD vs GST on `multi_turn_base`

Same category, same model family, same hardware class as the controls:

| Mode | Run | Score | Accuracy | Δ vs NFC |
|------|-----|-------|----------|----------|
| NFC (native `tools` / `tool_calls`) | 4 | 54/200 | 27.0% | — |
| IFFC (Appendix B JSON-in-text) | 5 | 61/200 | 30.5% | +3.5 pp / +7 entries |
| CD (TagDispatch-style two-pass; **not** XGrammar-2) | 8 | 21/200 | 10.5% | −16.5 pp / −33 entries |
| GST+NFC (He Detail template on NFC; **not** GBNF) | 9 | 61/200 | 30.5% | +3.5 pp / +7 entries |

GST **ties IFFC** (61 = 61) and is **+40.0 pp / +40 entries vs CD**. The IFFC tie is a **count coincidence**, not the same 61 rows (see overlap). GST is **measured**; it does not beat IFFC on this slice.

#### Overlap (unique `entry_id`)

Pairwise over the same 200 entries:

| Pair | Both OK | First-only | Second-only | Both fail |
|------|---------|------------|-------------|-----------|
| GST vs NFC | 43 | GST-only 18 | NFC-only 11 | 128 |
| GST vs IFFC | 42 | GST-only 19 | IFFC-only 19 | 120 |

NFC ∩ IFFC ∩ GST all-OK: **34**. The exact 61 = 61 vs IFFC is a **19-row swap each way**, not a shared 61-entry set.

#### Failure buckets (one error per failed entry)

Thesis Bot taxonomy: one primary error per failed `entry_id` (not the eval-error *row* counts above, where one entry can contribute multiple `empty_turn` / `state_mismatch` rows). Failed entries: NFC 146, IFFC 139, CD 179, GST 139.

| Bucket | Run 4 NFC | Run 5 IFFC | Run 8 CD | Run 9 GST |
|--------|-----------|------------|----------|-----------|
| empty | 59 | 41 | 14 | **31** |
| under-call / missing GT exec | 19 | 38 | 70 | 29 |
| state mismatch | 68 | 60 | 95 | **79** |

GST vs NFC: fewer empty (59 → 31) but **more state mismatch** (68 → 79). Vs IFFC, GST is still worse on state (79 vs 60). The checklist made Gemma *call more*, not *track env state better*.

#### Inference metrics

From `inference_metrics` (means). IFFC Run 5 has **no** `inference_metrics` row.

| Metric | Run 4 NFC | Run 8 CD | Run 9 GST |
|--------|-----------|----------|-----------|
| Generation tok/s (mean) | 92.43 | 89.6 | 92.45 |
| Prompt tok/s (mean) | 852 | 591 | 769 |
| TTFT (mean) | 720 ms | 482 ms | 320 ms |
| Peak RSS | 15,591 MB | 15,531 MB | 15,160 MB |

GST `flags` are empty for all 200 rows — no stored GST-block compliance. `responses` store executed func-strings only, not raw `[GST]` text, so this run does **not** measure whether the model actually wrote the scaffold.

#### Interpretation

1. **Wrong lever for the remaining errors.** On this slice the failures are already state mismatch and under-calling, not parse/format. GST adds extra planning on the **same NFC channel**; it does not change how calls are emitted or scored.

2. **Paper template vs multi-turn harness.** He et al. Detail is a single-turn curriculum. Multi-turn BFCL needs prior observations and environment state. This harness keeps schemas on native `tools` rather than pasting them into the GST block (unlike IFFC's system-prompt dump).

3. **The checker ignores GST drafts.** A correct plan in `[GST]` plus wrong `tool_calls` still fails. ABSTAIN / fewer empty turns raised the number of attempted calls, which can produce *wrong* state changes (empty 59 → 31; state 68 → 79).

4. **The IFFC tie is a swap.** 19 entries GST-only vs 19 IFFC-only. IFFC changes the output channel (JSON-in-text, no native `tools`). GST only prepends pre-call prose on NFC.

5. **Paper gap, not a failed measurement.** He et al. is mostly single-turn and different models. This test is a multi-turn edge SLM with native tools and an ignored scaffold. Tie / no-lift vs IFFC is a **result** if the harness is correct ([PR #14](https://github.com/sskarz/Thesis/pull/14)). Do **not** treat GST-on-NFC as the D1 default given 30.5% = IFFC and worse state-mismatch than IFFC. IFFC+STATE (Run 10) is reported next; Best-of-N on that channel (Run 11) follows.

### Gemma-4-E2B-it IFFC+STATE (Run 10)

Run 10 evaluates `--fc-mode iffc-state` on `multi_turn_base` only: Instruction-Followed Function Calling (IFFC; [arXiv:2608.22472](https://arxiv.org/abs/2608.22472) Appendix B) plus a **harness-written STATE** snapshot from actual tool execution results. Requests omit `tools` / `tool_choice` (same as IFFC). After each executed tool result, `StateTracker` (`benchmark/iffc_state.py`) appends a compact `[STATE]` block to the IFFC user-role feedback. The last 2 turns stay verbatim; older raw dumps are dropped in favor of STATE.

**Not** GST (no 7-step curriculum; not on the NFC `tools` channel), **not** TagDispatch/CD GBNF, **not** LLM summarization. STATE is conversation context written by the harness from **executed** `execute_multi_turn_func_call` observations; it is not parsed as a tool call. Scoring uses the same BFCL `multi_turn_checker` as IFFC. [PR #16](https://github.com/sskarz/Thesis/pull/16) (`ca9a728`) added `--fc-mode iffc-state`.

Config: Gemma-4-E2B-it Q4_K_M, Mac M4 Pro 48 GB, llama.cpp Metal, `-np 4 -c 65536`, `run_id=10`. Started `2026-09-14T02:46:12Z`, finished `2026-09-14T05:47:17Z` (~181.1 min wall). 200 responses, 200 `multi_turn_timings`, 132 `eval_errors`, 1 scores row, 50 `metrics_per_request`. Notes: `IFFC+STATE overnight -np4 -c65536 vs NFC 27% / IFFC 30.5% / GST 30.5% / CD 10.5%; fc_mode=iffc-state categories=multi_turn_base`.

0/200 `responses.result` contain a literal `[STATE]` string — expected: STATE lives in harness conversation context, not in stored result blobs.

Result: **68/200 = 34.0%** (first method above the 30.5% plateau; +7 vs IFFC and vs GST; +14 vs NFC; +47 vs CD).

#### NFC vs IFFC vs CD vs GST vs IFFC+STATE on `multi_turn_base`

Same category, same model family, same hardware class as the controls:

| Mode | Run | Score | Accuracy | Δ vs NFC |
|------|-----|-------|----------|----------|
| NFC (native `tools` / `tool_calls`) | 4 | 54/200 | 27.0% | — |
| IFFC (Appendix B JSON-in-text) | 5 | 61/200 | 30.5% | +3.5 pp / +7 entries |
| CD (TagDispatch-style two-pass; **not** XGrammar-2) | 8 | 21/200 | 10.5% | −16.5 pp / −33 entries |
| GST+NFC (He Detail template on NFC; **not** GBNF) | 9 | 61/200 | 30.5% | +3.5 pp / +7 entries |
| **IFFC+STATE** (IFFC JSON-in-text + harness `[STATE]`) | **10** | **68/200** | **34.0%** | **+7.0 pp / +14 entries** |

IFFC+STATE is **+3.5 pp / +7 entries vs IFFC** and **vs GST**. This is the first clear break of the IFFC=GST 30.5% tie. IFFC+STATE is **measured**.

#### Overlap (unique `entry_id`)

Pairwise over the same 200 entries:

| Pair | Both OK | First-only (rescue) | Second-only (regress) | Both fail |
|------|---------|---------------------|----------------------|-----------|
| STATE vs IFFC | 48 | STATE-only 20 | IFFC-only 13 | 119 |
| STATE vs GST | 43 | STATE-only 25 | GST-only 18 | 114 |
| STATE vs NFC | 39 | STATE-only 29 | NFC-only 15 | 117 |
| STATE vs CD | 16 | STATE-only 52 | CD-only 5 | 127 |

Rescues vs IFFC (20): `multi_turn_base_3`, `_13`, `_40`, `_48`, `_51`, `_53`, `_79`, `_81`, `_112`, `_117`, `_128`, `_140`, `_141`, `_146`, `_152`, `_163`, `_166`, `_167`, `_172`, `_179`.

Regresses vs IFFC (13): `multi_turn_base_36`, `_42`, `_43`, `_52`, `_74`, `_76`, `_114`, `_120`, `_124`, `_139`, `_148`, `_157`, `_158`.

The net +7 vs IFFC is a **swap** (20 rescues / 13 regresses), not pure dominance.

#### Failure buckets (one error per failed entry)

Thesis Bot taxonomy: one primary error per failed `entry_id`. Failed entries: NFC 146, IFFC 139, CD 179, GST 139, IFFC+STATE 132.

| Bucket | Run 4 NFC | Run 5 IFFC | Run 8 CD | Run 9 GST | Run 10 IFFC+STATE |
|--------|-----------|------------|----------|-----------|-------------------|
| empty | 59 | 41 | 14 | 31 | **31** |
| under-call / missing GT exec | 19 | 38 | 70 | 29 | **37** |
| state mismatch | 68 | 60 | 95 | 79 | **64** |

Trend vs IFFC: empty 41→31 (tied GST’s empty cut); state 60→**64** (did not improve); under-call ≈38→37. The gain is **not** perfect state tracking.

State mismatch by environment (Run 10, 64 entries): VehicleControl 22, GorillaFileSystem 21, Travel 7, Twitter 6, Ticket 3, Message 3, TradingBot 2.

#### Inference metrics

From `inference_metrics` / `metrics_per_request` (50 requests). Mean turns 3.67; mean `api_calls` 8.44. IFFC Run 5 has **no** `inference_metrics` row. VRAM 0 (Metal).

| Metric | Run 4 NFC | Run 8 CD | Run 9 GST | Run 10 IFFC+STATE |
|--------|-----------|----------|-----------|-------------------|
| Generation tok/s (mean) | 92.43 | 89.6 | 92.45 | 89.7 |
| Prompt tok/s (mean) | 852 | 591 | 769 | 969 |
| TTFT (mean) | 720 ms | 482 ms | 320 ms | 312 ms |
| Peak RSS | 15,591 MB | 15,531 MB | 15,160 MB | 14,867 MB |

Run 10 additional percentiles: prompt tok/s mean/med/p95 ≈ 969 / 967 / 1066; generation tok/s ≈ 89.7 / 89.7 / 92.1; TTFT mean/med/p95 ≈ 312 / 125 / 1266 ms; RSS baseline/peak ≈ 14,463 / 14,867 MB.

#### Interpretation

The following points are **interpretation**, not additional measurements.

1. **Env-grounded STATE + IFFC channel.** The overlay cut empty turns (41→31) without GST’s checklist-on-NFC path and posted net +7 vs both 30.5% plateaus.

2. **State mismatch did not fall.** STATE may help “forgot to call” more than env-diff fidelity. VehicleControl (22) and GorillaFileSystem (21) still dominate the 64 state-mismatch failures.

3. **The IFFC lift is a swap.** 20 rescues / 13 regresses vs IFFC (net +7). Document the trade, not pure dominance.

4. **First clear break of the IFFC=GST 30.5% tie.** That supports additive working memory on the IFFC channel where GST-on-NFC was not additive.

5. **Remaining headroom is still agentic.** `state_mismatch` 64 + under-call 37. Next should target FS/Vehicle fidelity or under-call recovery, not another empty-only prompt trick. Best-of-N on this channel (Run 11) is reported next; MLX SFT remains the planned training-time follow-on.

**Paper / method gap.** ACON (Xu et al., [arXiv:2510.00615](https://arxiv.org/abs/2510.00615)) compresses older turns with LLM summarization. This harness writes a structured key-value `[STATE]` from executed tool results — **not** an LLM summary, **not** GST, **not** CD/GBNF. Dual-model IFFC (paper Algorithm 1 QA/SRM decoupling) is still deferred. Do **not** treat `--fc-mode iffc-state` as solved state tracking given state mismatch 60→64 vs IFFC.

### Gemma-4-E2B-it Best-of-N on IFFC+STATE (Run 11)

Run 11 evaluates `--fc-mode iffc-state-bon` on `multi_turn_base` only: **MV@N majority vote** on the IFFC+STATE channel (Snell et al., [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)). Requests omit `tools` / `tool_choice` (same as IFFC+STATE). At each agent step the harness samples N=5 candidates at `temperature=0.7`, majority-votes `(function_name, arguments)`, executes the **winner once**, then continues the IFFC+STATE loop (`[STATE]` from executed results; last 2 turns verbatim). Scoring uses the same BFCL `multi_turn_checker`. [PR #18](https://github.com/sskarz/Thesis/pull/18) (`6a82aaf`) added `--fc-mode iffc-state-bon`.

**Not** NFC, **not** GST, **not** TagDispatch/CD GBNF, **not** a verifier / reward-model Best-of-N. The overlay rides on Run 10’s IFFC+STATE path so the 34.0% channel is not given back.

Config: Gemma-4-E2B-it Q4_K_M, Mac M4 Pro 48 GB, llama.cpp Metal, `-np 4 -c 65536`, `run_id=11`, `fc_mode=iffc-state-bon`, `bon_n=5`, `temperature=0.7`. Started `2026-09-15T03:15:21Z`. **`finished_at` is NULL** in the DB at verify time: eval scored (1 `scores` row; 127 `eval_errors`), but `inference_metrics=0` and `metrics_per_request=0` — the pipeline metrics step may still be unfinished. Document that honestly; do not invent tok/s or TTFT. 200 responses, 200 `multi_turn_timings`. Notes: `BoN N=5 on IFFC+STATE overnight -np4 -c65536 vs 34% / IFFC 30.5% / GST 30.5% / NFC 27% / CD 10.5%; fc_mode=iffc-state-bon bon_n=5 categories=multi_turn_base`.

Result: **73/200 = 36.5%** (beats IFFC+STATE Run 10 68/200 = 34.0% by **+5 entries / +2.5 pp**; +12 vs IFFC/GST 30.5%; +19 vs NFC; +52 vs CD). Errors 127 vs Run 10’s 132.

#### NFC vs IFFC vs CD vs GST vs IFFC+STATE vs BoN on `multi_turn_base`

Same category, same model family, same hardware class as the controls:

| Mode | Run | Score | Accuracy | Δ vs NFC |
|------|-----|-------|----------|----------|
| NFC (native `tools` / `tool_calls`) | 4 | 54/200 | 27.0% | — |
| IFFC (Appendix B JSON-in-text) | 5 | 61/200 | 30.5% | +3.5 pp / +7 entries |
| CD (TagDispatch-style two-pass; **not** XGrammar-2) | 8 | 21/200 | 10.5% | −16.5 pp / −33 entries |
| GST+NFC (He Detail template on NFC; **not** GBNF) | 9 | 61/200 | 30.5% | +3.5 pp / +7 entries |
| IFFC+STATE (IFFC JSON-in-text + harness `[STATE]`) | 10 | 68/200 | 34.0% | +7.0 pp / +14 entries |
| **BoN N=5 on IFFC+STATE** (MV@N, `temp=0.7`) | **11** | **73/200** | **36.5%** | **+9.5 pp / +19 entries** |

BoN is **+2.5 pp / +5 entries vs IFFC+STATE**. BoN is **measured**. The lift is modest and **not free** (latency below).

#### Overlap vs Run 10 (unique `entry_id`)

Pairwise over the same 200 entries:

| Pair | Both OK | First-only (gained) | Second-only (lost) | Both fail |
|------|---------|---------------------|--------------------|-----------|
| BoN vs IFFC+STATE | 56 | BoN-only 17 | STATE-only 12 | 115 |

Gained under BoN (17): `multi_turn_base_5`, `_14`, `_29`, `_32`, `_33`, `_34`, `_42`, `_47`, `_52`, `_77`, `_108`, `_120`, `_123`, `_127`, `_131`, `_139`, `_151`.

Lost under BoN (12): `multi_turn_base_3`, `_40`, `_48`, `_53`, `_81`, `_103`, `_117`, `_141`, `_163`, `_167`, `_169`, `_195`.

Net +5 = 17 − 12. The lift is a **swap**, not pure dominance.

#### Failure buckets (one error per failed entry)

Thesis Bot taxonomy: one primary error per failed `entry_id`. Failed entries: NFC 146, IFFC 139, CD 179, GST 139, IFFC+STATE 132, BoN 127.

| Bucket | Run 4 NFC | Run 5 IFFC | Run 8 CD | Run 9 GST | Run 10 IFFC+STATE | Run 11 BoN |
|--------|-----------|------------|----------|-----------|-------------------|------------|
| empty | 59 | 41 | 14 | 31 | 31 | **36** |
| under-call / missing GT exec | 19 | 38 | 70 | 29 | 37 | **38** |
| state mismatch | 68 | 60 | 95 | 79 | 64 | **53** |

Trend vs IFFC+STATE: empty 31→**36** (rose); state 64→**53** (cut); under-call 37→38 (flat). BoN cut state mismatches but empty turns rose. The gain is **not** a free empty-turn cut.

State mismatch by environment (Run 11, 53 entries): VehicleControl 22, GorillaFileSystem 13 (was 21 on Run 10), Travel 9, Twitter 3, Ticket 3, Message 2, TradingBot 1.

Top error prefixes (Run 11): missing_exec ~38 rows; VehicleControl state 22; empty turn1 17; GFS state 13; empty turn2 9; Travel state 9; empty turn0 7.

#### Latency / cost (N=5)

From `multi_turn_timings` (200 entries). **No** `inference_metrics` / `metrics_per_request` rows at verify time (`finished_at` still NULL); do not invent tok/s or TTFT.

| Metric | Run 10 IFFC+STATE | Run 11 BoN N=5 | Ratio |
|--------|-------------------|----------------|-------|
| avg `predicted_ms` / entry | ~150 s | ~1318 s | ~8.8× |
| avg `prompt_ms` / entry | ~22 s | ~210 s | ~9.5× |
| avg `api_calls` | 8.4 | 42.7 | ~5.1× |
| avg turns | ~3.67 | ~3.67 | ~1× |
| Wall (generate) | ~3.0 h (full pipeline ~181 min) | ~21.3 h | ~7× |

Framing: modest **+2.5 pp** at **~7× wall / ~N× decode**. This is the edge cost/latency tradeoff the thesis promises. BoN is **not free**.

#### Interpretation

The following points are **interpretation**, not additional measurements.

1. **Test-time compute on the IFFC+STATE channel.** MV@N=5 beat greedy STATE (+5 / +2.5 pp) without changing the output channel, the STATE overlay, or the checker.

2. **State mismatch fell; empty turns rose.** 64→53 state errors (GFS 21→13; VehicleControl stayed 22). Empty 31→36. Majority vote helped env-diff consistency more than “call something this turn.”

3. **The BoN lift is a swap.** 17 rescues / 12 regresses vs Run 10 (net +5). Document the trade, not pure dominance.

4. **Cost is the story.** ~7× wall and ~8.8× decode for +2.5 pp is a real edge-device tradeoff, not a free accuracy bump. Do **not** treat N=5 as the default serving recipe.

5. **Remaining headroom is still agentic.** State mismatch 53 (VehicleControl 22, GFS 13, Travel 9) plus under-call 38. Next leverage: FS/Vehicle fidelity or under-call recovery — or training-time methods (MLX SFT) — not another empty-only prompt trick and not unbounded N.

**Paper / method gap.** Snell et al. ([arXiv:2408.03314](https://arxiv.org/abs/2408.03314)) frame test-time compute vs parameter scaling. This harness is **majority vote on tool-call tuples**, not tree search and not a learned verifier. Dual-model IFFC (paper Algorithm 1 QA/SRM decoupling) is still deferred. Do **not** treat `--fc-mode iffc-state-bon` as solved sequential calling given 127 remaining failures and ~7× wall.

### Gemma-4-E2B-it BFCL-FS shell dialect (fs-only live eval)

Harness: `experiments/vfs-shell/run_bfcl_fs.py` ([PR #5](https://github.com/sskarz/Thesis/pull/5)). The model emits allowlisted Unix-like text; the harness maps it to GorillaFileSystem func-strings (`cd notes` → `cd(folder='notes')`) and scores with the existing `execute_multi_turn_func_call` / `multi_turn_checker` path. No `tools` array is sent (the dialect lives in the system prompt). This does **not** change NFC or IFFC defaults.

Slice: `fs-only` = `involved_classes == ["GorillaFileSystem"]` → **n=13** on `multi_turn_base`. This is a format ablation on the FS subset only. **Do not compare 13-entry accuracy to the full 200.** Two live evals on the **same 13 IDs** are reported below.

#### Shell v1 (prior live eval)

Serving: Gemma-4-E2B-it Q4_K_M, llama.cpp Metal, `-np 1` / 32k context (lean restart after Jetsam). Same model family as Run 4 / Run 5.

Result: **1/13 = 7.7%** (only `multi_turn_base_12` passed). Artifact: `experiments/vfs-shell/results/shell_fs_only.jsonl`.

Ops: one Jetsam/memory kill mid-batch ~11:05pm ET after 2/13; lean auto-restart then clean finish ~11:17pm ET.

#### Shell v2 (negative ablation)

Harness after [PR #9](https://github.com/sskarz/Thesis/pull/9) (merged): shell observations show Unix-like text; JSONL stores `shell_emissions` (real shell like `ls -a`) plus BFCL-mapped `result` for scoring.

Model: Gemma-4-E2B-it Q4_K_M. llama-server: `-np 1 -c 16384`. Optional worktree: `/tmp/thesis-vfs-eval`. Output: `shell_fs_only_v2.jsonl`.

Result: **0/13 = 0.0%** on GorillaFileSystem-only `multi_turn_base` (same 13 IDs as the prior fs-only table).

Failures: state mismatch, empty turns, under-execution (example: `multi_turn_base_12` double `echo >` overwrite).

#### NFC vs IFFC vs shell on the same 13 FS-only IDs

Controls from `benchmark.db` (Run 4 NFC, Run 5 IFFC traces scored on these IDs) plus the two shell live evals, **not** the full 200:

| Mode | Run | Score | Accuracy | Pass IDs |
|------|-----|-------|----------|----------|
| NFC (native `tools` / `tool_calls`) | 4 | 3/13 | 23.1% | `multi_turn_base_3`, `_12`, `_29` |
| IFFC (Appendix B JSON-in-text) | 5 | 1/13 | 7.7% | `multi_turn_base_12` |
| Shell v1 (prior live eval) | — | 1/13 | 7.7% | `multi_turn_base_12` |
| Shell v2 (this run) | — | 0/13 | 0.0% | — |

Full `multi_turn_base` (200 entries): NFC **54/200 = 27.0%**, IFFC **61/200 = 30.5%**, CD **21/200 = 10.5%** (Run 8, scored negative), GST+NFC **61/200 = 30.5%** (Run 9, ties IFFC), IFFC+STATE **68/200 = 34.0%** (Run 10, first method above 30.5%), BoN N=5 on IFFC+STATE **73/200 = 36.5%** (Run 11, beats STATE by +2.5 pp at ~7× wall). Those remain the primary thesis numbers. The 13-ID table is an apples-to-apples format comparison on GorillaFileSystem-only entries.

#### Interpretation

Shell v1 did **not** beat NFC on FS-only (1/13 vs 3/13); it tied IFFC on this slice (both 1/13, shared pass `multi_turn_base_12`). Shell v2 is worse still (**0/13 = 0.0%**). Residual failures are state mismatch, empty turns, and under-execution — not a format-only gap that more shell→GFS mapping would close. Shell-as-action on BFCL-FS is a **negative** ablation: do not pursue as D1 default. The toy VFS harness stays for non-BFCL / future SFT only. The primary thesis metric remains full `multi_turn_base`.

No other model's multi-turn scores are reported yet.

## Inference Performance

All measurements on Apple M4 Pro with Metal GPU backend. Metrics sampled over 50 requests per model.

### Throughput

| Model | Prompt Processing (mean) | Prompt Processing (median) | Generation (mean) | Generation (median) |
|-------|-------------------------|---------------------------|-------------------|---------------------|
| Qwen2.5-3B | 192.5 tok/s | 86.9 tok/s | 93.5 tok/s | 93.6 tok/s |
| Llama-3.2-3B | 824.9 tok/s | 935.9 tok/s | 88.5 tok/s | 89.1 tok/s |
| Qwen3.5-4B | 620.5 tok/s | 623.8 tok/s | 37.9 tok/s | 38.0 tok/s |
| FunctionGemma-270M | 2,567.5 tok/s | 64.6 tok/s | **201.0 tok/s** | **204.2 tok/s** |

### Latency (Time to First Token)

| Model | Mean | Median | P95 | P99 |
|-------|------|--------|-----|-----|
| Qwen2.5-3B | 34.8 ms | 11.6 ms | 205.6 ms | 281.9 ms |
| Llama-3.2-3B | 249.1 ms | 218.2 ms | 520.2 ms | 569.1 ms |
| Qwen3.5-4B | 723.6 ms | 656.3 ms | 1,051.4 ms | 1,099.6 ms |
| FunctionGemma-270M | **19.9 ms** | **16.2 ms** | **34.0 ms** | **37.9 ms** |

### Memory (RSS)

| Model | Baseline | Peak | Delta |
|-------|----------|------|-------|
| Qwen2.5-3B | 8,512.8 MB | 8,546.9 MB | 34.1 MB |
| Llama-3.2-3B | 10,778.2 MB | 12,383.7 MB | 1,605.5 MB |
| Qwen3.5-4B | **3,974.8 MB** | **4,075.4 MB** | 100.6 MB |
| FunctionGemma-270M | 6,389.2 MB | 6,562.2 MB | 173.0 MB |

## Analysis & Key Findings

### Qwen2.5-3B: Best raw tool-calling accuracy

Qwen2.5-3B leads on `simple` (93.25%) and `multiple` (92.50%), the two categories most directly relevant to agentic tool selection. When the model has one or several functions available and must pick the right one with correct arguments, Qwen2.5-3B gets it right over 9 times out of 10. Its weakness is `irrelevance` (58.75%) — it calls a function when it shouldn't about 40% of the time — and cross-language function signatures (Java 37%, JavaScript 34%).

### Qwen3.5-4B: Best overall with strongest irrelevance detection

Qwen3.5-4B achieves the highest weighted accuracy across non-parallel categories (82.53%), driven by its dominant irrelevance detection (91.25%). This is critical for agentic applications: an agent that hallucinates tool calls when none are needed will execute unintended actions. Its core tool-calling accuracy (90.25% simple, 90.50% multiple) trails Qwen2.5-3B only slightly. The trade-off is performance: Qwen3.5-4B is the slowest model tested at 38.0 tok/s generation and 656.3 ms median TTFT. However, it is the most memory-efficient generalist at 4,075.4 MB peak RSS — fitting comfortably within 8 GB edge devices.

### Llama-3.2-3B: Strongest cross-language support

Llama-3.2-3B is the weakest model on core Python tool calling (51.75% simple, 33.50% multiple) and nearly fails irrelevance detection entirely (0.83%). However, it dominates cross-language function calling: Java 54% and JavaScript 70%, substantially outperforming all other models. This suggests Llama 3.2's training data or fine-tuning gave it stronger multi-language type signature understanding, even as its Python tool-calling logic lags behind. Its memory footprint is the largest (12,383.7 MB peak), making it the hardest to deploy on constrained devices.

### FunctionGemma-270M: Fastest inference, competitive for its size

FunctionGemma is the clear performance winner: 204.2 tok/s median generation (2.2x faster than the next model), 16.2 ms median TTFT (14x faster than Qwen3.5-4B), and the smallest GGUF at ~512 MB. Despite being ~11x smaller than the 3B generalists, its `multiple` category accuracy (59.00%) beats Llama-3.2-3B (33.50%), demonstrating that specialist fine-tuning can compensate for parameter count on the core tool-selection task. Its format incompatibility (requiring a custom parser) and weaker irrelevance/cross-language scores reflect the limitations of specialist training.

### Llama's "multiple" failure is structural, not semantic

A diagnostic investigation of Llama-3.2-3B's 33.50% score on `multiple` (200 test cases, 2–4 tools each) revealed that the model selects the **correct function in nearly every case**. The failures are entirely caused by **argument type serialization** — Llama wraps values in JSON strings where the schema expects native types:

| Expected | Llama produces | Example |
|----------|---------------|---------|
| `integer` | `string` | `"5"` instead of `5` |
| `number`/`float` | `string` | `"3.14"` instead of `3.14` |
| `array`/`tuple` | `string` | `"[3, 4]"` instead of `[3, 4]` |
| `object`/`dict` | `string` | `"{'a': 1}"` instead of `{"a": 1}` |

The BFCL AST checker strictly rejects these type mismatches (e.g., *"Incorrect type for parameter 'side1'. Expected type integer, got str"*), turning what are semantically correct tool calls into evaluation failures. The Qwen models do not exhibit this behavior — they produce correctly-typed JSON natively.

#### Mitigation: system prompt + schema-aware type coercion

Two complementary techniques were tested, both applied at inference time without modifying the model:

1. **System prompt injection**: A system message instructing the model to use correct JSON types for tool arguments. This nudges the model to produce correct types during generation.

2. **Schema-aware type coercion**: A post-processing step that coerces each argument value to the type declared in the BFCL function schema. For each argument, the coercion function examines the schema's `type` field and attempts to convert string values — `int()` for integers, `float()` for numbers, `json.loads()` for arrays and objects. Values that already have the correct type pass through unchanged.

#### Results

Both techniques were evaluated on the full 200-entry `multiple` category across all three generalist models:

| Method | Llama 3.2 3B | Qwen2.5 3B | Qwen3.5 4B |
|--------|-------------|------------|------------|
| Baseline | 41.5% | 93.0% | 94.5% |
| System prompt only | 57.0% (+15.5pp) | 94.5% (+1.5pp) | 94.5% (±0) |
| Type coercion only | 87.5% (+46.0pp) | 93.0% (±0) | 94.5% (±0) |
| **System prompt + coercion** | **92.0% (+50.5pp)** | **94.5% (+1.5pp)** | **94.5% (±0)** |

Key observations:

- **Type coercion is the dominant factor**: It alone raises Llama from 41.5% to 87.5% — a 46 percentage point improvement. The system prompt adds another 4.5pp on top.
- **No regressions on Qwen models**: Both techniques are safe to apply unconditionally. Qwen models already produce correct types, so coercion is a no-op. The system prompt gives Qwen2.5 a marginal +1.5pp boost.
- **Llama matches Qwen after mitigation**: At 92.0%, Llama's `multiple` accuracy is within 2.5pp of Qwen2.5's 94.5% — demonstrating that Llama's tool *selection* capability was never the bottleneck.
- **The gap was formatting, not understanding**: This is a significant finding for SLM deployment. What appeared to be a 60pp accuracy gap between model families was primarily a superficial serialization difference, not a fundamental capability difference.

#### Implications for SLM tool calling

This analysis reveals that **benchmark scores can significantly understate a model's true tool-calling capability** when the evaluation is sensitive to argument formatting. For edge deployment, a lightweight post-processing layer — schema-aware type coercion — can close the gap between model families without any fine-tuning, additional inference cost, or architectural changes. The system prompt adds a small further improvement at the cost of a few extra input tokens.

Both techniques are now integrated into the benchmark pipeline (`benchmark/coerce_types.py` and a tool-calling system prompt in `benchmark/generate_responses.py`) and applied to all models uniformly.

### Parallel calling: Server-side grammar bug, not a model limitation

All four models initially scored 0% on `parallel` and `parallel_multiple` (400 test cases total). This appeared to be a fundamental SLM limitation — but investigation revealed the root cause is a **llama-server bug in grammar-constrained decoding**, not a model capability gap.

#### Diagnosis

When `tools` are passed in the OpenAI-compatible API payload, llama-server generates a PEG grammar to constrain the model's output to valid tool-call syntax. For the Qwen chat template, the grammar produced is:

```
tool-call ::= "<tool_call>\n" (tool-func) ("," (tool-func))* "</tool_call>"
```

This grammar expects multiple tool calls **inside a single `<tool_call>` block separated by commas**. But the Qwen template's actual format uses **separate `<tool_call>...</tool_call>` blocks per call**. After the first `</tool_call>` tag, the grammar is satisfied and generation terminates — the model never gets to produce the 2nd or 3rd call.

Evidence from a controlled test (Qwen2.5-3B, prompt: "Find the factorial of 5, 10, and 15"):

| Mode | Tool calls produced | Tokens generated |
|------|-------------------|-----------------|
| With `tools` param (grammar active) | 1 (only `factorial(5)`) | 21 |
| Without `tools` (raw text, no grammar) | 3 (all correct: 5, 10, 15) | 65 |

The model generates all three calls correctly when unconstrained. The grammar kills the 2nd and 3rd calls by stopping generation early.

#### Fix

For `parallel` and `parallel_multiple` categories, the pipeline now passes `tool_choice: "none"` instead of `"auto"`. This bypasses grammar-constrained decoding while still including the tool definitions in the Jinja-templated prompt (the model sees the same tools, just isn't grammar-constrained). A content parser extracts `<tool_call>` blocks from the raw response text. Non-parallel categories continue using `tool_choice: "auto"` with full grammar constraining.

#### Implications

This finding has significant implications for SLM benchmarking infrastructure:

1. **Benchmark validity**: Any llama.cpp-based evaluation reporting 0% parallel tool-calling accuracy should be scrutinized — the failure may be in the serving layer, not the model.
2. **Grammar-constrained decoding is a double-edged sword**: It guarantees valid JSON output for single tool calls but silently breaks multi-call scenarios. The grammar's repetition structure must match the model's trained output format exactly.
3. **SLMs can do parallel tool calling**: The 3B-parameter Qwen2.5 correctly emits 2–4 parallel function calls when unconstrained, suggesting that parallel tool calling is within reach for sub-5B models — contrary to the initial 0% scores.

### Accuracy vs. speed trade-off

There is a clear inverse relationship between accuracy and inference speed across the tested models:

| Model | Weighted Accuracy | Generation Speed | TTFT (median) |
|-------|------------------|-----------------|---------------|
| Qwen3.5-4B | 82.53% | 38.0 tok/s | 656.3 ms |
| Qwen2.5-3B | 76.06% | 93.6 tok/s | 11.6 ms |
| FunctionGemma-270M | 39.90% | 204.2 tok/s | 16.2 ms |
| Llama-3.2-3B | 36.87% | 89.1 tok/s | 218.2 ms |

The most accurate model (Qwen3.5-4B) is 5.4x slower at generation than the fastest (FunctionGemma). Application designers must choose where on this curve their use case falls — a robotics controller may need 16 ms TTFT even at lower accuracy, while a financial agent may need 91% irrelevance detection even at 656 ms latency.

### Specialist vs. generalist

FunctionGemma-270M is a purpose-built tool-calling model fine-tuned from Gemma 3. With only 270M parameters, it achieves `multiple`-category accuracy (59.00%) that beats the 3.2B-parameter Llama-3.2-3B (33.50%). This validates the specialist hypothesis: targeted fine-tuning on tool-calling data can yield competitive accuracy at a fraction of the model size, with dramatically better inference performance. The trade-off is generality — FunctionGemma struggles on tasks outside its training distribution (irrelevance, cross-language).

## Remaining Work

- **Multi-turn optimization**: Gemma-4-E2B-it on `multi_turn_base`: NFC Run 4 **54/200 = 27.0%**; IFFC Run 5 **61/200 = 30.5%** (+3.5 pp); TagDispatch-style CD Run 8 **21/200 = 10.5%** (179 errors); GST+NFC Run 9 **61/200 = 30.5%** (ties IFFC; +7 vs NFC); IFFC+STATE Run 10 **68/200 = 34.0%** (first method above the 30.5% plateau; +7 vs IFFC and vs GST); BoN N=5 on IFFC+STATE Run 11 **73/200 = 36.5%** (beats STATE by +5 / +2.5 pp at ~7× wall). CD is **measured and negative**. Guided templates are **measured**: they **tie IFFC and do not beat it** — the 61=61 is a 19-row swap, not the same 61 entries. GST vs NFC: empty 59→31 but state mismatch 68→79. IFFC+STATE is **measured** and **beats** the IFFC/GST plateau: empty 41→31 (tied GST’s empty cut); state mismatch 60→**64** (did not improve); under-call ≈38→37. The +7 vs IFFC is a 20-rescue / 13-regress swap, not pure dominance. BoN is **measured** and **beats STATE**: empty 31→**36** (rose); state 64→**53** (cut); under-call 37→38 (flat). The +5 vs STATE is a 17-rescue / 12-regress swap. Wall generate ~21.3 h vs Run 10 ~3.0 h; avg `predicted_ms` ~1318 s vs ~150 s (~8.8×). Modest +2.5 pp is **not free**. Honesty: `--fc-mode iffc-state-bon` is MV@N on the IFFC+STATE channel — **not** NFC, **not** GST, **not** CD/GBNF, **not** a verifier/reward-model BoN; `--fc-mode iffc-state` is IFFC JSON-in-text plus harness-written STATE from executed tool results — **not** GST, **not** CD/GBNF, **not** LLM summarization; `--fc-mode gst` is prompt-only on the NFC channel (not GBNF/IFFC); `--fc-mode cd` is a two-pass approximation on stock llama-server — **not** full XGrammar-2. Do **not** pursue this CD condition, another GST Detail overlay on NFC, or unbounded N as D1 default. Residual errors are still agentic state mismatch (VehicleControl 22 / GorillaFileSystem 13 / Travel 9) and under-calling, not format parsing. Next leverage: FS/Vehicle state fidelity or under-call recovery — or training-time methods — not another empty-only prompt trick and not more test-time N without a cost budget. BFCL-FS / VFS shell→GorillaFileSystem fs-only is a **negative ablation** (v1 **1/13 = 7.7%**, v2 **0/13 = 0.0%** vs NFC **3/13 = 23.1%** vs IFFC **1/13 = 7.7%** on the same 13 GorillaFileSystem-only IDs; not the full 200). Do **not** pursue shell-as-action as D1 default. The toy VFS harness stays for non-BFCL / future SFT only. Next methodology candidate: MLX SFT — not more shell→GFS BFCL chasing, not an unscored CD overnight, not another GST-on-NFC curriculum, and not treating N=5 as free accuracy. Dual-model IFFC (paper Algorithm 1 QA/SRM decoupling) is deferred. `miss_func` / `miss_param` / `long_context` remain secondary stress tests. Run 11 `finished_at` was NULL at verify time (`inference_metrics=0`, `metrics_per_request=0`); eval scored.
- **NVIDIA Jetson Orin Nano Super migration**: Repeat all benchmarks on CUDA with 8 GB shared memory to measure real edge-device performance.
- **RAG augmentation for large tool sets**: The `multiple` category only presents 2–4 tools per test case, so RAG-based tool filtering provides no benefit. However, real-world agentic applications may expose 50–100+ tools. Test whether retrieval-augmented generation (embedding-based tool retrieval to pre-filter the tool set before prompting) improves accuracy and reduces prompt length for large tool registries.
- **Additional models and quantizations**: Expand the model matrix (e.g., Phi-3.5-mini, Gemma-2-2B) and test alternative quantization levels (Q5_K_M, Q8_0) to map the accuracy-size Pareto frontier.
- **Re-run parallel benchmarks**: Now that the grammar-constraining bypass is in place, re-run `parallel` and `parallel_multiple` categories across all models to measure true SLM parallel tool-calling accuracy.
