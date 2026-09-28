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

Gemma-4-E2B-it family (multi-turn only; not in the single-turn table above):

| Checkpoint | `model_key` | Quantization | Source |
|------------|-------------|--------------|--------|
| Gemma-4-E2B-it (base instruct) | `gemma4-e2b` | Q4_K_M | `bartowski/google_gemma-4-E2B-it-GGUF` |
| Gemma-4-E2B-it SFT-IFFC best1000 | `gemma4-e2b-sft-iffc` | Q4_K_M fused | local `sft/fused/gemma-4-e2b-it-iffc-best1000-Q4_K_M.gguf` |
| Gemma-4-E2B-it SFT-IFFC-v2 filtered-[] best2000 | `gemma4-e2b-sft-iffc-v2` | Q4_K_M fused | local `sft/fused/gemma-4-e2b-it-iffc-v2-best2000-Q4_K_M.gguf` |
| Gemma-4-E2B-it SFT-IFFC-v4 best1600 | `gemma4-e2b-sft-iffc-v4` | Q4_K_M fused | local `sft/fused/gemma-4-e2b-it-iffc-v4-best1600-Q4_K_M.gguf` |
| Gemma-4-E2B-it SFT-IFFC-v5 best1800 | `gemma4-e2b-sft-iffc-v5` | Q4_K_M fused | local `sft/fused/gemma-4-e2b-it-iffc-v5-best1800-Q4_K_M.gguf` |
| Gemma-4-E2B-it SFT-IFFC-v6 best1800 | `gemma4-e2b-sft-iffc-v6` | Q4_K_M fused | local `sft/fused/gemma-4-e2b-it-iffc-v6-best1800-Q4_K_M.gguf` |

External xLAM-1b-fc-r baseline (not Gemma; PLAN Tier 1 §3 / METHODS §15; registry [PR #41](https://github.com/sskarz/Thesis/pull/41); JSON-in-text adapter [PR #42](https://github.com/sskarz/Thesis/pull/42)):

| Checkpoint | `model_key` | Quantization | Source |
|------------|-------------|--------------|--------|
| xLAM-1b-fc-r | `xlam-1b` | Q4_K_M | `Salesforce/xLAM-1b-fc-r-gguf` |

Gemma-4-E2B-it (Q4_K_M) was additionally evaluated on **multi-turn categories only**: Run 4 NFC (`--only-multi-turn`, all four slices), Run 5 IFFC (`--categories multi_turn_base --fc-mode iffc`), Run 8 TagDispatch-style CD (`--categories multi_turn_base --fc-mode cd`; **21/200 = 10.5%**, a scored negative vs NFC/IFFC), Run 9 GST+NFC (`--categories multi_turn_base --fc-mode gst`; **61/200 = 30.5%**, ties IFFC, does not beat it), Run 10 IFFC+STATE (`--categories multi_turn_base --fc-mode iffc-state`; **68/200 = 34.0%**, first method above the 30.5% plateau), Run 11 Best-of-N on IFFC+STATE (`--categories multi_turn_base --fc-mode iffc-state-bon`; **73/200 = 36.5%**, beats IFFC+STATE by +5 entries / +2.5 pp at ~7× wall), Run 12 SFT-IFFC LoRA best@1000 (`--model gemma4-e2b-sft-iffc --categories multi_turn_base --fc-mode iffc`; **1/200 = 0.5%**, a scored negative vs every inference-time baseline), and Run 13 SFT-IFFC-v2 filtered-[] LoRA best@2000 (`--model gemma4-e2b-sft-iffc-v2 --categories multi_turn_base --fc-mode iffc`; **1/200 = 0.5%**, same collapse as Run 12 — the non-terminal-`[]` filter did **not** recover). Run 14 is a **DIAGNOSTIC** history-bookkeeping ablation on the **same** v2 GGUF (`--fc-mode iffc --iffc-strip-empty-assistant`; **4/200 = 2.0%**): stripping trailing assistant `[]` from `chat_history` did **not** rescue the never-start collapse — **do not present 2% as a method win**. Run 15 SFT-IFFC-v4 LoRA best@1600 (`--model gemma4-e2b-sft-iffc-v4 --categories multi_turn_base --fc-mode iffc`; **30/200 = 15.0%**) is a **scored recovery** from that collapse (Runs 12–14 ≈ 0.5–2%), still **below IFFC 30.5% and below NFC 27.0%** — **do not present 15% as beating IFFC**. Run 16 is a **scored negative/null ablation** of IFFC+STATE on that **same** SFT-v4 GGUF (`--model gemma4-e2b-sft-iffc-v4 --categories multi_turn_base --fc-mode iffc-state`; **26/200 = 13.0%**): STATE is **−4 entries / −2.0 pp vs Run 15 plain IFFC** — the overlay that helped base Gemma (Run 10 +3.5 pp vs IFFC) does **not** help this checkpoint. **Do not propose BoN as the next win on this ckpt.** Run 17 SFT-IFFC-v5 LoRA best@1800 (`--model gemma4-e2b-sft-iffc-v5 --categories multi_turn_base --fc-mode iffc`; **23/200 = 11.5%**) is a **scored regression vs v4** (−7 entries / −3.5 pp vs Run 15 15.0%; also below Run 16 13.0%). Empty-`[]` is fully suppressed (0 empty errors); residual is `state_mismatch` 133 / `under_call` 44. **Do not present 11.5% as a method win.** **No BoN on this ckpt; no more epochs on the v5 mix.** Run 18 SFT-IFFC-v6 LoRA best@1800 (`--model gemma4-e2b-sft-iffc-v6 --categories multi_turn_base --fc-mode iffc`; **26/200 = 13.0%**) is a **scored miss vs v4** (−4 entries / −2.0 pp vs Run 15 15.0%; modest +3 / +1.5 pp vs Run 17; ties Run 16 13.0%). Recipe v6 ([PR #38](https://github.com/sskarz/Thesis/pull/38)): v4 empty/recovery + only `drop_state_inconsistent`; turn-0 underspec/synth **off**. Model key [PR #39](https://github.com/sskarz/Thesis/pull/39). Empty still essentially suppressed (1 empty on `_43`). Residual `state_mismatch` 118 + `under_call` 55. Avg `api_calls` **13.34** (fixed v5’s 40.3 churn). **Do not present 13.0% as a method win.** **No BoN / no more epochs on v6.** v4 IFFC 15% remains best SFT recovery; BoN 36.5% remains best inference on **base** Gemma. Run 19 xLAM-1b-fc-r NFC (`--model xlam-1b --categories multi_turn_base --fc-mode nfc`; **0/200 = 0.0%**) is a **scored channel/format collapse on our multi_turn NFC harness** (NFC never parses assistant `content`). Registry [PR #41](https://github.com/sskarz/Thesis/pull/41). **Not** a claim that xLAM scores 0 on BFCL overall; **not** published Salesforce overall BFCL ~78.94% (different metric); **not** a dead server (gen_tps_mean ≈ **196.7**; avg `api_calls` **3.63** — the model generated tokens; NFC recorded **no tool calls**); **not** a method win; **not** an ability floor. Run 20 (`--model xlam-1b --categories multi_turn_base --fc-mode xlam`; **6/200 = 3.0%**) is the **format-fixed fair-channel external baseline** after [PR #42](https://github.com/sskarz/Thesis/pull/42): smoke extracted calls; full score is still **very low**. Tools actually ran (avg `api_calls` **50.47** vs Run 19 **3.63**). **Not** a method win; **not** published ~78.94%; still **below CD 10.5%** and all Gemma/SFT/BoN. Run 27 (`--model gemma4-e2b --categories multi_turn_base --fc-mode iffc-stop-repeat`; **54/200 = 27.0%**) is a **BASE Gemma** IFFC harness lever ([PR #54](https://github.com/sskarz/Thesis/pull/54), commit `5b49d7a`): stop-on-repeat after executed calls. **Scored miss vs Run 5 IFFC 30.5%** (−3.5 pp / −7) and **−9.5 pp vs BoN 36.5%**. Count-ties NFC 27.0% (do **not** claim the same 54 entries). **Not** smoke81 / v7 / v8 SFT / BoN. **Not** a method win. Full LoRA on failed v8 mixes is **not started**. A shell-dialect format ablation ([PR #5](https://github.com/sskarz/Thesis/pull/5) harness; v2 after [PR #9](https://github.com/sskarz/Thesis/pull/9)) was run on the 13-entry `fs-only` GorillaFileSystem subset of `multi_turn_base`; that slice is **not** a full-200 score (v1 1/13, v2 0/13). Single-turn scores for this model are not reported here.

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

### NFC vs IFFC vs CD vs GST vs IFFC+STATE vs BoN vs SFT-IFFC vs xLAM NFC vs xLAM `--fc-mode xlam` vs IFFC-stop-repeat on `multi_turn_base`

Primary metric: BFCL v4 `multi_turn_base` (200 execution-evaluated cases) on Gemma-4-E2B-it Q4_K_M, Mac Metal llama-server, plus **external** xLAM-1b-fc-r Q4_K_M baselines on the same slice (**not** the same model; NFC Run 19 was format-channel collapse; `--fc-mode xlam` Run 20 is the fair-channel score). Run 27 is the same **base** Gemma GGUF on `--fc-mode iffc-stop-repeat` ([PR #54](https://github.com/sskarz/Thesis/pull/54)): a harness force-stop after repeated IFFC calls, **not** a retrain. CD is `--fc-mode cd`: a TagDispatch-style **two-pass approximation** on stock llama-server (pass 1: name GBNF enum; pass 2: `json_schema` for selected tools; never sends `tools` / `tool_choice`). **This is not XGrammar-2.** GST is `--fc-mode gst`: a **prompt-only overlay on NFC** (He et al. Detail template; still `tools` + `tool_choice=auto`). **This is not GBNF / XGrammar / IFFC.** IFFC+STATE is `--fc-mode iffc-state`: IFFC JSON-in-text plus a **harness-written `[STATE]`** snapshot from actual tool execution results. **This is not GST, not CD/GBNF, and not LLM summarization.** BoN is `--fc-mode iffc-state-bon`: **MV@N majority vote** (N=5, `temperature=0.7`) on the IFFC+STATE channel. **This is not NFC, not GST, not CD, and not a verifier / reward-model Best-of-N.** SFT-IFFC is `--model gemma4-e2b-sft-iffc --fc-mode iffc`: MLX LoRA on the same IFFC JSON-in-text channel (fused Q4_K_M, best val @ iter 1000). SFT-IFFC-v2 is `--model gemma4-e2b-sft-iffc-v2 --fc-mode iffc`: the same channel after [PR #24](https://github.com/sskarz/Thesis/pull/24) dropped non-terminal empty-after-exec `[]` (LoRA best val @ iter 2000). **Neither is an inference-time overlay or a parser change.** Run 14 (`--fc-mode iffc --iffc-strip-empty-assistant` on the same v2 GGUF) is a **DIAGNOSTIC** history-bookkeeping ablation: omit trailing assistant `[]` from `chat_history` before the next BFCL user turn. **Not** a training method, **not** a new `fc-mode`, and **not** a claim that standard IFFC improved. SFT-IFFC-v4 is `--model gemma4-e2b-sft-iffc-v4 --fc-mode iffc`: [PR #31](https://github.com/sskarz/Thesis/pull/31) v4 unroll (v3 empty-after-exec drop + recovery, plus drop first-assistant-empty and clarifying `[]`; LoRA best val @ iter 1600). **Scored recovery from collapse, still below IFFC and NFC.** Run 16 is `--model gemma4-e2b-sft-iffc-v4 --fc-mode iffc-state` on that **same GGUF**: a **scored negative/null STATE overlay** versus Run 15 plain IFFC (13.0% < 15.0%). **Not** a retrain, **not** BoN, and **not** a claim that STATE failed on base Gemma. SFT-IFFC-v5 is `--model gemma4-e2b-sft-iffc-v5 --fc-mode iffc`: [PR #35](https://github.com/sskarz/Thesis/pull/35) v5 unroll (keep v4 empty-recovery; drop underspecified turn-0; synthetic complete first-episode; drop state-inconsistent continue-after-exec; LoRA best val @ iter 1800, val 0.147). **Scored regression versus Run 15 (11.5% < 15.0%).** **Not** a method win; **not** BoN; **not** more epochs on this mix. SFT-IFFC-v6 is `--model gemma4-e2b-sft-iffc-v6 --fc-mode iffc`: [PR #38](https://github.com/sskarz/Thesis/pull/38) v6 unroll (v4 empty/recovery + only `drop_state_inconsistent`; turn-0 underspec drop and synthetic complete turn-0 **off**; LoRA best val @ iter 1800, val 0.142). Model key [PR #39](https://github.com/sskarz/Thesis/pull/39). **Scored miss versus Run 15 (13.0% < 15.0%).** **Not** a method win; **not** BoN; **not** more epochs on this mix. Run 19 is `--model xlam-1b --fc-mode nfc` on the published Salesforce xLAM-1b-fc-r Q4_K_M GGUF (registry [PR #41](https://github.com/sskarz/Thesis/pull/41)): a **scored channel/format collapse** on **this** multi_turn NFC harness (**0/200 = 0.0%**). **Not** a Gemma method, **not** published overall BFCL ~78.94%, **not** an ability floor, **not** a claim that xLAM scores 0 on BFCL overall. Run 20 is `--model xlam-1b --fc-mode xlam` on that **same GGUF** after [PR #42](https://github.com/sskarz/Thesis/pull/42): the **format-fixed fair-channel** score (**6/200 = 3.0%**). Smoke extracted calls; full score is still **very low**. **Not** a method win, **not** published ~78.94%, still **below CD 10.5%** and all Gemma/SFT/BoN. Run 27 is `--model gemma4-e2b --fc-mode iffc-stop-repeat` on **base** Gemma-4-E2B-it Q4_K_M ([PR #54](https://github.com/sskarz/Thesis/pull/54), commit `5b49d7a`): IFFC JSON-in-text plus a harness force-stop when the proposed call set repeats already-executed calls. **Scored miss vs Run 5 IFFC (27.0% < 30.5%).** **Not** smoke81 / v7 / v8 SFT / BoN. **Not** a method win. Accuracy row is authoritative (do **not** invent tok/s).

| Mode | Run | Score | Accuracy | Δ vs NFC |
|------|-----|-------|----------|----------|
| NFC (native `tools` / `tool_calls`) | 4 | 54/200 | 27.0% | — |
| IFFC (Appendix B JSON-in-text) | 5 | 61/200 | 30.5% | +3.5 pp / +7 entries |
| CD (TagDispatch-style two-pass; **not** XGrammar-2) | 8 | 21/200 | 10.5% | −16.5 pp / −33 entries |
| GST+NFC (He Detail template on NFC; **not** GBNF) | 9 | 61/200 | 30.5% | +3.5 pp / +7 entries |
| IFFC+STATE (IFFC JSON-in-text + harness `[STATE]`) | 10 | 68/200 | 34.0% | +7.0 pp / +14 entries |
| BoN N=5 on IFFC+STATE (MV@N, `temp=0.7`) | 11 | 73/200 | 36.5% | +9.5 pp / +19 entries |
| SFT-IFFC LoRA best@1000 (IFFC channel) | 12 | 1/200 | 0.5% | −26.5 pp / −53 entries |
| SFT-IFFC-v2 filtered-[] LoRA best@2000 (IFFC channel) | 13 | 1/200 | 0.5% | −26.5 pp / −53 entries |
| SFT-IFFC-v2 + strip trailing `[]` (**DIAGNOSTIC** history bookkeeping; **not** a training method) | 14 | 4/200 | 2.0% | −25.0 pp / −50 entries |
| SFT-IFFC-v4 LoRA best@1600 (IFFC channel; scored recovery) | 15 | 30/200 | 15.0% | −12.0 pp / −24 entries |
| SFT-IFFC-v4 + IFFC+STATE (same GGUF; **null/negative vs Run 15**) | 16 | 26/200 | 13.0% | −14.0 pp / −28 entries |
| SFT-IFFC-v5 LoRA best@1800 (IFFC channel; **scored regression vs Run 15**) | 17 | 23/200 | 11.5% | −15.5 pp / −31 entries |
| **SFT-IFFC-v6 LoRA best@1800** (IFFC channel; **scored miss vs Run 15**) | **18** | **26/200** | **13.0%** | **−14.0 pp / −28 entries** |
| **xLAM-1b-fc-r NFC** (external; **format-channel collapse**; **not** published BFCL) | **19** | **0/200** | **0.0%** | **−27.0 pp / −54 entries** vs Gemma NFC |
| **xLAM-1b-fc-r `--fc-mode xlam`** (external; **fair channel**; still **very low**; **not** published BFCL) | **20** | **6/200** | **3.0%** | **−24.0 pp / −48 entries** vs Gemma NFC |
| **IFFC-stop-repeat** (base Gemma; harness force-stop; **scored miss vs Run 5 IFFC**) | **27** | **54/200** | **27.0%** | same accuracy level as NFC; **−3.5 pp / −7** vs IFFC |

Order on this slice: **xLAM NFC 0.0%** < **xLAM `--fc-mode xlam` 3.0%** < CD 10.5% < SFT-v5 11.5% < **SFT-v6 13.0%** = SFT-v4+STATE 13.0% < **SFT-v4 IFFC 15.0%** < Gemma NFC 27.0% = **IFFC-stop-repeat 27.0%** (**same accuracy level** as Run 4 NFC; **not** an identical method) < IFFC/GST 30.5% < base STATE 34.0% < **BoN 36.5%**. Run 8 CD is a **scored negative** vs both NFC and IFFC (10.5% < 27.0% / 30.5%; 179 errors). Run 9 GST+NFC **ties IFFC at 61/200 = 30.5%** — a count tie, not the same 61 entries. Run 10 IFFC+STATE is the **first method above that 30.5% plateau** (+7 entries / +3.5 pp vs IFFC and vs GST) **on base Gemma**. Run 11 BoN **beats IFFC+STATE** (+5 entries / +2.5 pp) at ~7× wall / ~N× decode — a modest lift, **not free**. Run 12 SFT-IFFC is a **scored negative** vs every inference-time baseline on this slice (0.5% < 10.5% CD; 199 errors; only `multi_turn_base_50` correct). Root cause is the training mix (ToolACE stop-after-tools `[]`), not the IFFC parser. Run 13 SFT-IFFC-v2 is the **same collapse** (1/200 = 0.5%; only `multi_turn_base_50`; correct-set overlap identical with Run 12): dropping 334 non-terminal empty-after-exec `[]` rows ([PR #24](https://github.com/sskarz/Thesis/pull/24)) did **not** recover accuracy. Run 14 is a **DIAGNOSTIC** history-bookkeeping ablation on the same v2 GGUF (**4/200 = 2.0%**; wins `_50`, `_161`, `_192`, `_193` — shared with Run 13: only `_50`). Stripping trailing assistant `[]` from chat history does **not** rescue the never-start collapse. **Do not present 2% as a method win.** Run 15 SFT-IFFC-v4 is a **scored recovery** from that collapse (**30/200 = 15.0%**; +29 entries / +14.5 pp vs Run 13) that **does not beat IFFC** (−15.5 pp / −31 vs Run 5) **or NFC** (−12.0 pp / −24 vs Run 4). Empty-`[]` self-conditioning is largely fixed (2/170 empty errors); residual is `state_mismatch` 109 and `under_call` 59. **Do not present 15% as a method win over IFFC.** Run 16 is a **scored negative/null ablation** of `--fc-mode iffc-state` on that **same v4 GGUF** (**26/200 = 13.0%**; −4 entries / −2.0 pp vs Run 15). STATE helped base Gemma (+3.5 pp) and **hurt / did not help SFT-v4**. Residual remains `state_mismatch` 112 / `under_call` 59 (empty still 3/174). **Do not propose BoN as the next win on this checkpoint.** Run 17 SFT-IFFC-v5 is a **scored regression vs v4** (**23/200 = 11.5%**; −7 entries / −3.5 pp vs Run 15; also −3 / −1.5 pp vs Run 16). Empty-`[]` is fully suppressed (**0/177** empty errors); residual is `state_mismatch` 133 (~75%) / `under_call` 44 (~25%, turn 0 = 27). **Do not present 11.5% as a method win.** **No BoN on this ckpt; no more epochs on the v5 mix.** Recipe v5’s turn-0 / state-consistency filters did **not** lift `multi_turn_base` vs v4 recovery. Run 18 SFT-IFFC-v6 is a **scored miss vs v4** (**26/200 = 13.0%**; −4 entries / −2.0 pp vs Run 15; +3 / +1.5 pp vs Run 17; ties Run 16). Empty still essentially suppressed (**1** empty on `_43` turn 2); residual is `state_mismatch` 118 (~68%) / `under_call` 55 (~32%, turn 0 = 38). Avg `api_calls` **13.34** (vs Run 15 17.3, Run 17 40.3) — v6 stopped the v5 call-churn blowup. **Do not present 13.0% as a method win.** **No BoN / no more epochs on v6.** Thin filter ablation on the same family is likely exhausted. Run 19 xLAM-1b NFC is a **scored format-channel collapse** on this harness (**0/200 = 0.0%**; all 200 `Model response list is empty for turn 0`): NFC never parses `content`. Run 20 `--fc-mode xlam` is the **format-fixed fair-channel** score (**6/200 = 3.0%**; +6 / +3.0 pp vs Run 19): tools actually ran (avg `api_calls` **50.47** vs 3.63) but the full score is still **very low** — **not** a method win, **not** published ~78.94%, still **below CD 10.5%**. Run 27 IFFC-stop-repeat is a **scored miss vs plain IFFC on base Gemma** (**54/200 = 27.0%**; `scores.id=20`; **200/200 responses** complete; −7 entries / −3.5 pp vs Run 5; −19 / −9.5 pp vs BoN). Same accuracy level as Run 4 NFC 27.0% — **not** an identical method. Stop flags **37/200**, all among the 146 failures (**0 stopped and correct**). VehicleControlAPI state mismatch **22→33** vs Run 5. Stop-on-repeat helped collapsed smoke81 over-call in a 10-ID probe but did **not** improve `multi_turn_base` vs IFFC on **healthy base** Gemma. **Thursday lock:** tradeoff for **loop-dominated cases only**; **not** a method improvement. **Not** smoke81 / v7 / v8 SFT / BoN. **Do not present 27.0% as a method win.** Inference-time BoN **36.5%** remains the best measured score **on the base Gemma stack**; IFFC+STATE 34.0%; IFFC Run 5 **30.5%**. v4 IFFC **15.0%** remains the best **SFT** recovery. Details below. The 13-entry BFCL-FS shell ablation is a separate slice and is **not** in this table.

### Gemma-4-E2B-it baseline (Run 4, 2026-09-10)

This is a multi-turn-only baseline: `--only-multi-turn`, 800 execution-evaluated cases, no single-turn categories. Serving was MacBook Pro M4 Pro, llama.cpp Metal, Q4_K_M, reasoning on, `-np 8` / `PARALLEL_SLOTS=8`.

| Category | Score | Accuracy | Flagged |
|----------|-------|----------|---------|
| `multi_turn_base` | 54/200 | 27.0% | 2 |
| `multi_turn_miss_func` | 16/200 | 8.0% | 5 |
| `multi_turn_miss_param` | 39/200 | 19.5% | 2 |
| `multi_turn_long_context` | 22/200 | 11.0% | 72 (mostly `context_overflow`) |

**`multi_turn_base` (full 200) remains the primary optimization target.** Run 4 NFC scored 27.0% on that slice; Run 5 IFFC (+3.5 pp), Run 8 CD (scored negative, 10.5%), Run 9 GST+NFC (ties IFFC at 30.5%), Run 10 IFFC+STATE (**68/200 = 34.0%**, first break past 30.5% **on base Gemma**), Run 11 BoN on IFFC+STATE (**73/200 = 36.5%**, +2.5 pp vs STATE at ~7× wall), Run 12 SFT-IFFC LoRA best@1000 (**1/200 = 0.5%**, scored negative), and Run 13 SFT-IFFC-v2 filtered-[] LoRA best@2000 (**1/200 = 0.5%**, same collapse — filter did not recover) are reported below. Run 14 (`--iffc-strip-empty-assistant` on the same v2 GGUF) is a **DIAGNOSTIC** history-bookkeeping ablation (**4/200 = 2.0%**): it did **not** rescue the collapse — **do not present 2% as a method win**. Run 15 SFT-IFFC-v4 LoRA best@1600 is a **scored recovery** (**30/200 = 15.0%**) from that 0.5–2% collapse, still **below IFFC 30.5% and below NFC 27.0%** — **do not present 15% as beating IFFC**. Run 16 IFFC+STATE on that **same v4 GGUF** is a **scored negative/null overlay** (**26/200 = 13.0%**, −2.0 pp vs Run 15) — **do not propose BoN as the next win on this ckpt**. Run 17 SFT-IFFC-v5 LoRA best@1800 is a **scored regression vs v4** (**23/200 = 11.5%**, −3.5 pp / −7 vs Run 15) — **do not present 11.5% as a method win**; **no BoN / no more epochs on this mix**. Run 18 SFT-IFFC-v6 LoRA best@1800 is a **scored miss vs v4** (**26/200 = 13.0%**, −2.0 pp / −4 vs Run 15; +1.5 pp vs Run 17) — **do not present 13.0% as a method win**; **no BoN / no more epochs on v6**. Run 19 xLAM-1b-fc-r NFC is a **scored format-channel collapse** (**0/200 = 0.0%**) on this harness — NFC never parses `content`; **not** published overall BFCL ~78.94%. Run 20 `--fc-mode xlam` is the **format-fixed fair-channel** score (**6/200 = 3.0%**) — tools actually ran; still **very low**; **not** a method win; **not** published ~78.94%; still **below CD 10.5%**. Run 27 `--fc-mode iffc-stop-repeat` on **base** Gemma is a **scored miss vs IFFC** (**54/200 = 27.0%**, −3.5 pp vs Run 5; −9.5 pp vs BoN) — **not** smoke81 / v7 / v8 SFT / BoN; **not** a method win. The BFCL-FS shell eval is a 13-entry FS-only ablation, not a substitute for those 200-entry scores. `miss_func`, `miss_param`, and `long_context` are secondary stress slices (withheld tools, missing parameters, and long histories). They are not the headline success criterion.

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

A TagDispatch-style constrained-decoding ablation (`--fc-mode cd`) on the same 200-entry `multi_turn_base` is reported next; it is a **scored negative**. Guided-Structured Templates (`--fc-mode gst`, Run 9) follow; they **tie IFFC** and do not beat it. IFFC + harness STATE (`--fc-mode iffc-state`, Run 10) follows GST; it is the **first method above 30.5%**. Best-of-N majority vote on IFFC+STATE (`--fc-mode iffc-state-bon`, Run 11) follows STATE; it **beats STATE at 73/200 = 36.5%** with a ~7× wall-clock cost. MLX SFT-IFFC LoRA best@1000 (`--model gemma4-e2b-sft-iffc --fc-mode iffc`, Run 12) follows BoN; it is a **scored negative at 1/200 = 0.5%** (training-mix stop-after-tools, not a parser bug). MLX SFT-IFFC-v2 filtered-[] LoRA best@2000 (`--model gemma4-e2b-sft-iffc-v2 --fc-mode iffc`, Run 13) follows Run 12; it is the **same 0.5% collapse** — the non-terminal-`[]` filter was insufficient. Run 14 (`--iffc-strip-empty-assistant` on that same v2 GGUF) follows Run 13; it is a **DIAGNOSTIC** history-bookkeeping ablation at **4/200 = 2.0%** — stripping trailing `[]` from history did **not** rescue the never-start collapse (**not** a method win). Run 15 SFT-IFFC-v4 LoRA best@1600 (`--model gemma4-e2b-sft-iffc-v4 --fc-mode iffc`) follows Run 14; it is a **scored recovery at 30/200 = 15.0%** from that collapse, still **below IFFC 30.5% and below NFC 27.0%** (**not** a win over IFFC). Run 16 (`--model gemma4-e2b-sft-iffc-v4 --fc-mode iffc-state` on that same GGUF) follows Run 15; it is a **scored negative/null STATE overlay at 26/200 = 13.0%** versus Run 15 plain IFFC (**not** a BoN-on-this-ckpt next win). Run 17 SFT-IFFC-v5 LoRA best@1800 (`--model gemma4-e2b-sft-iffc-v5 --fc-mode iffc`) follows Run 16; it is a **scored regression at 23/200 = 11.5%** versus Run 15 (**not** a method win; **no** BoN / more epochs on this mix). Run 18 SFT-IFFC-v6 LoRA best@1800 (`--model gemma4-e2b-sft-iffc-v6 --fc-mode iffc`) follows Run 17; it is a **scored miss at 26/200 = 13.0%** versus Run 15 15.0% (**not** a method win; **no** BoN / more epochs on v6). Run 19 xLAM-1b-fc-r NFC (`--model xlam-1b --fc-mode nfc`) follows Run 18; it is a **scored format-channel collapse at 0/200 = 0.0%** on this NFC harness (**not** published overall BFCL ~78.94%; NFC never parses `content`; **not** a method win). Run 20 (`--model xlam-1b --fc-mode xlam`) follows Run 19; it is the **format-fixed fair-channel** score at **6/200 = 3.0%** after [PR #42](https://github.com/sskarz/Thesis/pull/42) (**not** a method win; **not** published ~78.94%; still **below CD 10.5%**). Run 27 (`--model gemma4-e2b --fc-mode iffc-stop-repeat`) follows Run 20 as a **BASE Gemma** harness lever ([PR #54](https://github.com/sskarz/Thesis/pull/54)); it is a **scored miss at 54/200 = 27.0%** versus Run 5 IFFC 30.5% (**not** a method win; **not** smoke81 / v7 / v8 SFT / BoN). A separate format channel — allowlisted shell text mapped onto GorillaFileSystem — is a **13-entry `fs-only` ablation** (two live evals, v1 and v2), not a replacement for the 200-entry scores.

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

5. **Remaining headroom is still agentic.** `state_mismatch` 64 + under-call 37. Next should target FS/Vehicle fidelity or under-call recovery, not another empty-only prompt trick. Best-of-N on this channel (Run 11) is reported next; MLX SFT v1 (Run 12) and the filtered-[] mix rebuild (Run 13) are both **scored negatives** at 0.5% (not more epochs on the same data). The strip-trailing-`[]` history ablation (Run 14) is a **DIAGNOSTIC** on the same v2 GGUF at 2.0% — **not** a method win. SFT-IFFC-v4 (Run 15) is a **scored recovery** at 15.0% from that collapse, still **below IFFC and NFC**.

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

5. **Remaining headroom is still agentic.** State mismatch 53 (VehicleControl 22, GFS 13, Travel 9) plus under-call 38. Next leverage: FS/Vehicle fidelity or under-call recovery — not another empty-only prompt trick and not unbounded N. The first training-time follow-on (MLX SFT v1, Run 12) is reported next; it is a **scored negative**. The mix rebuild (Run 13) is the same collapse. The strip-trailing-`[]` history ablation (Run 14) is a **DIAGNOSTIC** on the same v2 GGUF — **not** a method win. SFT-IFFC-v4 (Run 15) is a **scored recovery** at 15.0%, still below IFFC/NFC; BoN **36.5%** remains the best inference-time method.

**Paper / method gap.** Snell et al. ([arXiv:2408.03314](https://arxiv.org/abs/2408.03314)) frame test-time compute vs parameter scaling. This harness is **majority vote on tool-call tuples**, not tree search and not a learned verifier. Dual-model IFFC (paper Algorithm 1 QA/SRM decoupling) is still deferred. Do **not** treat `--fc-mode iffc-state-bon` as solved sequential calling given 127 remaining failures and ~7× wall.

### Gemma-4-E2B-it SFT-IFFC LoRA best@1000 (Run 12) — scored negative

Run 12 evaluates the first training-time counterpart to inference-time IFFC: MLX LoRA SFT of Gemma-4-E2B-it on the frozen IFFC JSON-in-text mix (`sft/`; ToolACE-all + 25k xLAM), fused at **best val @ iter 1000**, served as Q4_K_M GGUF, then scored with the same `--fc-mode iffc` harness as Run 5. Requests omit `tools` / `tool_choice`. Scoring uses the same BFCL `multi_turn_checker`. Distinct `model_key` so results do **not** overwrite base Gemma.

**Not** an inference-time overlay, **not** GST, **not** CD/GBNF, **not** IFFC+STATE / BoN on this checkpoint. The channel is the same Appendix B JSON-in-text as Run 5; only the weights changed.

#### Checkpoint identity

| Field | Value |
|-------|-------|
| `run_id` | 12 (Mac `benchmark.db`, verified) |
| `model_display` | Gemma-4-E2B-it SFT-IFFC best1000 (Q4_K_M) |
| `model_key` | `gemma4-e2b-sft-iffc` |
| Adapter | MLX LoRA, mlx-lm 0.32, best val @ iter 1000 (`val=0.206`) |
| Serve | fused Q4_K_M; `llama-server -m sft/fused/gemma-4-e2b-it-iffc-best1000-Q4_K_M.gguf` |
| Mode | `--fc-mode iffc`, `--categories multi_turn_base` |
| Notes | `SFT-IFFC LoRA best@1000 (val0.206); mlx-lm 0.32; Q4_K_M fused; fc_mode=iffc categories=multi_turn_base` |
| Started / finished | `2026-09-20T12:19:15Z` / `2026-09-20T12:33:00Z` (~14 min wall) |
| Rows | 200 responses, 199 `eval_errors`, 200 `multi_turn_timings` |
| Flags | all 200 `flags` empty — **zero** `iffc_unparseable` |

Result: **1/200 = 0.5%** (`correct=1`, `total=200`, `accuracy=0.005`). Only win: `multi_turn_base_50`. This is **−26.5 pp / −53 entries vs NFC**, **−30.0 pp / −60 vs IFFC**, and **−36.0 pp / −72 vs BoN**. Worse than CD’s already-negative 10.5%.

#### Same-slice baselines

| Method | Run | Score |
|--------|-----|-------|
| NFC | 4 | 54/200 = 27.0% |
| IFFC (base) | 5 | 61/200 = 30.5% |
| CD | 8 | 21/200 = 10.5% |
| GST | 9 | 61/200 = 30.5% |
| IFFC+STATE | 10 | 68/200 = 34.0% |
| BoN N=5 | 11 | 73/200 = 36.5% |
| **SFT-IFFC LoRA best@1000** | **12** | **1/200 = 0.5%** |

#### Error mix (two views — do not conflate)

**Primary BFCL bucket per failed entry** (one error per failed `entry_id`; n=199; sum 101+63+35=199):

| Bucket | Count |
|--------|-------|
| `empty_turn` | **101** |
| `state_mismatch` | **63** |
| `under_call` | **35** |

**Thesis Bot trajectory audit** (turn-level empty rate and early-stop trajectories — a different metric from the 101 `empty_turn` primary bucket):

| Audit figure | Run 5 IFFC | Run 12 SFT-IFFC |
|--------------|------------|-----------------|
| Empty turns | 13.2% | **69.9%** |
| Early-stop trajectories | 56/200 | **158/200** |

Both views support the same conclusion: **early abstain dominates**. The 101 `empty_turn` primaries are the checker’s first failure label; the 69.9% / 158/200 figures describe how often the model emits `[]` and how often trajectories stop early. They are complementary, not interchangeable.

Qualitative: entries IFFC wins (e.g. `multi_turn_base_101`) the SFT checkpoint gets turn 0 + turn 1 right, then emits `[]` on turn 2 (`view_messages_sent` missing). Secondary: when it does call, some bad order (e.g. `mv` before `mkdir` on `multi_turn_base_0`) then empties.

#### Root-cause audit (training mix, not serve/parser)

Documented as **audit / root cause**. This is **not** a llama-server, chat-template, or IFFC-parser bug.

1. **Format + chat template OK.** A live single-step IFFC probe on the served GGUF emits valid `[{"function_name","parameters"}]`. Zero `iffc_unparseable` on all 200 Run 12 responses.

2. **Dominant failure = early stop with `[]`.** Empty turns 13.2% → 69.9%; early-stop trajectories 56/200 → 158/200. The model often completes the first tool loop, then abstains instead of continuing.

3. **Training-data smoking gun.** The mlx train split has **5520** assistant `[]` targets, **all from ToolACE** (32.5% of ToolACE assistant turns; xLAM 0%). After a `"Function execution results"` user message, **78.1%** of assistant targets are `[]` (2121 empty vs 596 continue). That teaches **stop-after-tools**. BFCL `multi_turn_base` needs continued calls on later user turns.

4. **Live multi-turn probe reproduces it.** After stock→send, the next user ask `"show messages sent"` → the SFT model returns `[]`.

5. **Secondary call-order errors** (e.g. `mv` before `mkdir` on `multi_turn_base_0`) then empties — not the mass of the 199 failures.

#### Anti-recommendations

Do **not**:

- train more epochs on the **same** mix
- run BoN or IFFC+STATE on **this** checkpoint
- treat “fuse `final@2000`” as the first fix

Those would amplify or paper over a stop-after-tools prior, not remove it.

#### Next (Sanskar to pick — scored as Run 13)

The mix rebuild that **dropped non-terminal `[]` after tool results** ([PR #24](https://github.com/sskarz/Thesis/pull/24)) was trained and scored as Run 13. Result: **1/200 = 0.5%** — same collapse as Run 12; filter insufficient. See the Run 13 subsection.

**Paper / method gap.** ToolACE (Liu et al., [arXiv:2409.00920](https://arxiv.org/abs/2409.00920)) and xLAM (Zhang et al., [arXiv:2409.03215](https://arxiv.org/abs/2409.03215)) are the v1 sources; IFFC Appendix B is the chat shape. Run 12 shows that copying ToolACE’s terminal-`[]` density into an unrolled multi-turn IFFC adapter **hurts** BFCL sequential calling. Do **not** treat SFT-on-IFFC as a free accuracy bump over Run 5.

### Gemma-4-E2B-it SFT-IFFC-v2 filtered-[] LoRA best@2000 (Run 13) — scored negative

Run 13 evaluates the first mix rebuild after Run 12: [PR #24](https://github.com/sskarz/Thesis/pull/24) dropped **334** mid-trajectory empty-after-exec `[]` rows (non-terminal **334→0**) while **keeping 552** terminal empty-after-exec. Same IFFC JSON-in-text harness as Run 5 / Run 12 (`--fc-mode iffc`). Requests omit `tools` / `tool_choice`. Scoring uses the same BFCL `multi_turn_checker`. Distinct `model_key` `gemma4-e2b-sft-iffc-v2` so results do **not** overwrite base Gemma or Run 12.

**Not** an inference-time overlay, **not** GST, **not** CD/GBNF, **not** IFFC+STATE / BoN on this checkpoint. The channel is the same Appendix B JSON-in-text; only the unroll filter and LoRA weights changed.

#### Checkpoint identity

| Field | Value |
|-------|-------|
| `run_id` | 13 (Mac `benchmark.db`, verified) |
| `model_display` | Gemma-4-E2B-it SFT-IFFC-v2 filtered-[] (Q4_K_M) |
| `model_key` | `gemma4-e2b-sft-iffc-v2` |
| Adapter | MLX LoRA, mlx-lm 0.32, best val @ iter 2000 (`val=0.211`) |
| Serve | fused Q4_K_M; `llama-server -m sft/fused/gemma-4-e2b-it-iffc-v2-best2000-Q4_K_M.gguf` |
| Mode | `--fc-mode iffc`, `--categories multi_turn_base` |
| Notes | `SFT-IFFC-v2 filtered nonterminal-[] LoRA best@2000 val0.211; mlx-lm 0.32; Q4_K_M; fc_mode=iffc categories=multi_turn_base vs Run12 0.5% / BoN 36.5%` |
| Wall | ~14 min |
| Rows | 200 responses, 199 `eval_errors`, 200 `multi_turn_timings` |
| Flags | all 200 `flags` empty — **zero** `iffc_unparseable` |
| Figures | `benchmark/results/Gemma-4-E2B-it-SFT-IFFC-v2-Q4_K_M/figures` |

Result: **1/200 = 0.5%**. Only win: `multi_turn_base_50` — the **same** entry as Run 12; correct-set overlap is identical. This is **−26.5 pp / −53 entries vs NFC**, **−30.0 pp / −60 vs IFFC**, and **−36.0 pp / −72 vs BoN**. Same score as Run 12. The non-terminal-`[]` filter did **not** recover accuracy.

#### Same-slice baselines

| Method | Run | Score |
|--------|-----|-------|
| NFC | 4 | 54/200 = 27.0% |
| IFFC (base) | 5 | 61/200 = 30.5% |
| CD | 8 | 21/200 = 10.5% |
| GST | 9 | 61/200 = 30.5% |
| IFFC+STATE | 10 | 68/200 = 34.0% |
| BoN N=5 | 11 | 73/200 = 36.5% |
| SFT-IFFC LoRA best@1000 | 12 | 1/200 = 0.5% |
| **SFT-IFFC-v2 filtered-[] LoRA best@2000** | **13** | **1/200 = 0.5%** |

Inference-time baselines are **unchanged**: BoN **36.5%** remains the best inference result; IFFC+STATE 34.0%; IFFC 30.5%; Run 12 SFT v1 0.5%.

#### Error mix (two views — do not conflate)

**Primary BFCL bucket per failed entry** (one error per failed `entry_id`; n=199; sum 167+25+7=199):

| Bucket | Run 12 | Run 13 |
|--------|--------|--------|
| `empty_turn` | 101 | **167** |
| `state_mismatch` | 63 | **25** |
| `under_call` | 35 | **7** |

**Thesis Bot trajectory audit** (turn-level empty rate and early-stop trajectories — a different metric from the 167 `empty_turn` primaries):

| Audit figure | Run 12 SFT-IFFC v1 | Run 13 SFT-IFFC-v2 |
|--------------|--------------------|--------------------|
| Empty turns | ~70% | still **~70%** (522/734 error rows) |
| Early-stop trajectories | 158/200 | **67/200** |

Document the audit **alongside** the verified primary buckets; do not swap them. Early-stop trajectories fell 158→67, but empty-labeled **primary** fails rose 101→167. That is **more empty turns overall as first-fail labels**, not a win. Empty turns still dominate ~70% of error rows.

#### Audit probes (Thesis Bot — not primary buckets)

Label as **audit probes**. These do not replace the 167/25/7 primary buckets or the 1/200 score. Only win remains `multi_turn_base_50`. Do **not** claim the `[]` filter fixed multi-turn.

**1. Misleading short probe vs BFCL IFFC shape.** A short live multi-turn that returned `view_messages_sent` on the v2 GGUF bundled the next user ask into the tool-result user message. Under true BFCL IFFC shape (exec results → model emits `[]` to end the within-turn loop → a **separate** next user turn), the same v2 GGUF returns `[]` on turn 2 for `multi_turn_base_101` — matching scored Run 13.

**2. Self-conditioning on `[]`.** Controlled probes with a 3-tool IFFC system prompt:

| Probe | History | v2 result |
|-------|---------|-----------|
| A | BFCL-shaped history with an intermediate `[]` | turn 2 `[]` |
| B | next-ask bundled into results; no `[]` in history | `view_messages_sent` |
| C | no `[]` assistant turns | works |
| D | prior `[]` in history even if the last user message bundles the ask | still `[]` |

Once SFT emits `[]`, later turns stay empty.

**3. Pattern shift** (C = non-empty tool-call turn, E = empty `[]` turn). Run 5 IFFC is mostly `CCC` / `CC` / `CCCC`. Run 12 is `CE` / `CEE` / `CEEE` (start then stop). Run 13 is `EE` / `EEEE` / `EEE` dominant (turn-0 refusal). The filter cut mid-trajectory `[]` targets but **worsened leading empties**: `first_empty` at index 0 rose **40→153**.

**4. Not tool-catalog size alone.** Short vs full tool lists both fail BFCL-shaped turn 2. Turn-0-empty vs turn-0-call tool counts are similar (~27–30).

#### Filter vs result

PR #24 dropped 334 mid-trajectory empty-after-exec rows (non-terminal 334→0) and still kept 552 terminal empty-after-exec. The live score is the same 0.5% as the unfiltered v1 mix. Format is not the gap: zero `iffc_unparseable` on all 200 Run 13 responses (same as Run 12).

#### Anti-recommendations

Do **not**:

- train more epochs on **this** filtered mix
- run BoN or IFFC+STATE on **this** v2 checkpoint
- claim the `[]` filter **fixed** SFT or multi-turn

Those would amplify or paper over the same collapse.

#### Next (Sanskar decides — scored as Run 14 then Run 15)

The history-bookkeeping diagnostic (`--iffc-strip-empty-assistant` on this v2 GGUF) was scored as Run 14: **4/200 = 2.0%** — did **not** rescue the never-start collapse. See the Run 14 subsection. **Do not present 2% as a method win.**

The v4 curriculum ([PR #31](https://github.com/sskarz/Thesis/pull/31): drop all empty-after-exec including terminal + drop first-assistant-empty and clarifying `[]`) was trained and scored as Run 15: **30/200 = 15.0%** — a **scored recovery**, still below IFFC/NFC. See the Run 15 subsection.

**Paper / method gap.** The v2 filter was the measured test of “drop non-terminal ToolACE `[]` after tool results.” Run 13 shows that cut is **insufficient** for BFCL sequential calling. Do **not** treat filtered-`[]` SFT as a recovery over Run 12 or over Run 5 IFFC.

### Gemma-4-E2B-it SFT-IFFC-v2 + strip trailing `[]` (Run 14) — DIAGNOSTIC history-bookkeeping ablation

**DIAGNOSTIC — not a training method.** Run 14 is a harness history-bookkeeping ablation on the **same** fused GGUF as Run 13 (`gemma4-e2b-sft-iffc-v2`, SFT-IFFC-v2 filtered-[] Q4_K_M). `--fc-mode iffc` plus `--iffc-strip-empty-assistant`: omit trailing assistant `[]` stop turns from `chat_history` before the next BFCL user turn. Notes tag `iffc_strip_empty_assistant=1` plus the DIAGNOSTIC phrase. Scoring uses the same BFCL `multi_turn_checker`. Distinct `runs` row; figures nest under `iffc-strip-empty-assistant/` so they do **not** overwrite Run 13.

**Not** a new `fc-mode`, **not** a mix rebuild, **not** GST / CD / IFFC+STATE / BoN, and **not** a claim that standard IFFC improved. **Do not present 2.0% as a method win.**

#### Checkpoint identity

| Field | Value |
|-------|-------|
| `run_id` | 14 (Mac `benchmark.db`, verified) |
| `model_display` | Gemma-4-E2B-it SFT-IFFC-v2 filtered-[] (Q4_K_M) |
| `model_key` | `gemma4-e2b-sft-iffc-v2` (same as Run 13) |
| Adapter | MLX LoRA, mlx-lm 0.32, best val @ iter 2000 (`val=0.211`) — **same weights as Run 13** |
| Serve | fused Q4_K_M; `llama-server -m sft/fused/gemma-4-e2b-it-iffc-v2-best2000-Q4_K_M.gguf` |
| Mode | `--fc-mode iffc --iffc-strip-empty-assistant`, `--categories multi_turn_base` |
| Notes | `iffc_strip_empty_assistant=1` + DIAGNOSTIC history-bookkeeping phrase |
| Wall | ~10 min |
| Rows | 200 responses, 196 `eval_errors`, 200 `multi_turn_timings` |
| Flags | all 200 `flags` empty — **zero** `iffc_unparseable` |
| Figures | `benchmark/results/Gemma-4-E2B-it-SFT-IFFC-v2-Q4_K_M/iffc-strip-empty-assistant/figures` |

Result: **4/200 = 2.0%** (correct=4, accuracy=0.02). Wins: `multi_turn_base_50`, `_161`, `_192`, `_193`. Shared with Run 13: only `_50`; **+3 new**. vs Run 13: 1/200 = 0.5%; vs BoN 36.5%; vs IFFC 30.5%. This is **−25.0 pp / −50 entries vs NFC**, **−28.5 pp / −57 vs IFFC**, **−34.5 pp / −69 vs BoN**. The +3 vs Run 13 is **not** a method win — the never-start collapse remains.

#### Same-slice baselines

| Method | Run | Score |
|--------|-----|-------|
| NFC | 4 | 54/200 = 27.0% |
| IFFC (base) | 5 | 61/200 = 30.5% |
| CD | 8 | 21/200 = 10.5% |
| GST | 9 | 61/200 = 30.5% |
| IFFC+STATE | 10 | 68/200 = 34.0% |
| BoN N=5 | 11 | 73/200 = 36.5% |
| SFT-IFFC LoRA best@1000 | 12 | 1/200 = 0.5% |
| SFT-IFFC-v2 filtered-[] LoRA best@2000 | 13 | 1/200 = 0.5% |
| **SFT-IFFC-v2 + strip trailing `[]` (DIAGNOSTIC)** | **14** | **4/200 = 2.0%** |

Inference-time baselines are **unchanged**: BoN **36.5%** remains the best inference result; IFFC+STATE 34.0%; IFFC 30.5%. Run 12 / Run 13 SFT remain 0.5%.

#### Error mix

**Primary BFCL bucket per failed entry** (one error per failed `entry_id`; n=196; sum 160+28+8=196):

| Bucket | Run 13 | Run 14 (DIAGNOSTIC) |
|--------|--------|---------------------|
| `empty_turn` | 167 | **160** |
| `state_mismatch` | 25 | **28** |
| `under_call` | 7 | **8** |

Empty-turn error rows at turn 0: **153** (unchanged vs Run 13 `first_empty` at index 0). Any-empty failed entries ~160. The mass of first-turn refusals did **not** move.

Format is not the gap: **zero** `iffc_unparseable` on all 200 Run 14 responses (same as Run 12 / Run 13).

#### Takeaway

Stripping trailing assistant `[]` from chat history does **NOT** rescue the SFT v2 never-start collapse. Root cause remains the data/harness recipe (post-tool `[]` conditioning / turn-0 refusal), not merely keeping those tokens in history. **Do not present 2% as a method win.** Label as **DIAGNOSTIC / history bookkeeping ablation**.

#### Anti-recommendations

Do **not**:

- treat 4/200 = 2.0% as an inference-time method win or a training-time recovery
- claim `--iffc-strip-empty-assistant` fixed SFT v2 or multi-turn
- run BoN or IFFC+STATE on this checkpoint
- overwrite Run 13 (or Run 12) with this diagnostic `runs` row

**Paper / method gap.** Run 13’s self-conditioning probes asked whether keeping `[]` in history was sufficient to explain the collapse. Run 14 removes those trailing stop tokens from `chat_history` and the model still refuses at turn 0 (153 empty-turn-0 rows unchanged). The pathology is in the weights / mix, not the bookkeeping of stop tokens in the transcript. SFT-IFFC-v4 (Run 15) retrains on a stronger empty-target curriculum and is reported next.

### Gemma-4-E2B-it SFT-IFFC-v4 best@1600 (Run 15) — scored recovery (still below IFFC)

Run 15 evaluates the first mix rebuild that actually moved the score off the 0.5–2% floor: [PR #31](https://github.com/sskarz/Thesis/pull/31) SFT v4 unroll keeps the v3 empty-after-exec drop (including terminal) and recovery curriculum, **plus** drops first-assistant-empty and clarifying empty-after-user `[]` targets. Same IFFC JSON-in-text harness as Run 5 (`--fc-mode iffc`). Requests omit `tools` / `tool_choice`. Scoring uses the same BFCL `multi_turn_checker`. Distinct `model_key` `gemma4-e2b-sft-iffc-v4` so results do **not** overwrite base Gemma or Runs 12–14.

**Not** an inference-time overlay, **not** GST, **not** CD/GBNF, **not** IFFC+STATE / BoN on this checkpoint. The channel is the same Appendix B JSON-in-text; only the unroll curriculum and LoRA weights changed.

**Framing:** this is a **scored recovery from SFT collapse**, not a win over IFFC. 15.0% sits **below NFC 27.0% and below IFFC 30.5%**. Do **not** present Run 15 as beating IFFC.

#### Checkpoint identity

| Field | Value |
|-------|-------|
| `run_id` | 15 (Mac `benchmark.db`, verified) |
| `model_display` | Gemma-4-E2B-it SFT-IFFC-v4 (Q4_K_M) |
| `model_key` | `gemma4-e2b-sft-iffc-v4` |
| Adapter | MLX LoRA, best val @ iter 1600 (`val=0.196`) |
| Recipe | SFT v4: v3 empty-after-exec drop + recovery, plus drop first-assistant-empty and clarifying `[]` ([PR #31](https://github.com/sskarz/Thesis/pull/31)) |
| Serve | fused Q4_K_M; `llama-server -m sft/fused/gemma-4-e2b-it-iffc-v4-best1600-Q4_K_M.gguf` |
| Mode | `--fc-mode iffc`, `--categories multi_turn_base` |
| Hardware | MacBook Pro M4 Pro, llama.cpp Metal, 48 GB |
| Notes | `fc_mode=iffc categories=multi_turn_base` |
| Started / finished | `2026-09-21T16:52:59Z` / **`finished_at` still NULL** at verify |
| Rows | 200 responses, 170 `eval_errors`, 200 `multi_turn_timings` |
| Inference metrics | **`inference_metrics=0`** — do **not** invent tok/s or TTFT |

Result: **30/200 = 15.0%**. This is **−12.0 pp / −24 entries vs NFC**, **−15.5 pp / −31 vs IFFC**, and **−21.5 pp / −43 vs BoN**. Versus SFT v2 (Run 13): **+14.5 pp / +29 entries**. Versus the Run 14 diagnostic: **+13.0 pp / +26**. Empty-`[]` self-conditioning is largely fixed (2 empty-turn primaries of 170 failures). Residual is agentic state mismatch and under-call. **Do not present 15% as beating IFFC.**

#### Same-slice baselines

| Method | Run | Score |
|--------|-----|-------|
| NFC | 4 | 54/200 = 27.0% |
| IFFC (base) | 5 | 61/200 = 30.5% |
| CD | 8 | 21/200 = 10.5% |
| GST | 9 | 61/200 = 30.5% |
| IFFC+STATE | 10 | 68/200 = 34.0% |
| BoN N=5 | 11 | 73/200 = 36.5% |
| SFT-IFFC LoRA best@1000 | 12 | 1/200 = 0.5% |
| SFT-IFFC-v2 filtered-[] LoRA best@2000 | 13 | 1/200 = 0.5% |
| SFT-IFFC-v2 + strip trailing `[]` (**DIAGNOSTIC**) | 14 | 4/200 = 2.0% |
| **SFT-IFFC-v4 LoRA best@1600** | **15** | **30/200 = 15.0%** |

Order: CD 10.5% < **SFT-v4 15.0%** < NFC 27.0% < IFFC/GST 30.5% < STATE 34.0% < **BoN 36.5%**. Inference-time BoN **36.5%** remains the best measured result. SFT-v4 recovered from the 0.5–2% collapse and still sits **below NFC**.

#### Correct set (30)

`multi_turn_base_50`, `_52`, `_64`, `_78`, `_101`, `_105`, `_106`, `_107`, `_109`, `_114`, `_116`, `_119`, `_122`, `_129`, `_133`, `_135`, `_136`, `_139`, `_142`, `_149`, `_154`, `_156`, `_159`, `_163`, `_166`, `_182`, `_184`, `_190`, `_191`, `_193`.

`_50` still wins (shared with the collapsed SFT runs) — VehicleControl `lockDoors` / `setHeadlights` path.

#### Overlap vs IFFC Run 5

Unique `entry_id` versus the 61 IFFC wins:

| Set | Count |
|-----|-------|
| both (shared with IFFC) | 23 |
| only Run 15 (SFT-v4 wins IFFC missed) | 7 |
| only Run 5 (lost vs IFFC) | 38 |
| net vs IFFC | **−31** (7 − 38) |

SFT-v4-only wins: `multi_turn_base_64`, `_149`, `_154`, `_156`, `_159`, `_163`, `_166`.

Lost vs IFFC (38): `_7`, `_8`, `_12`, `_23`, `_36`, `_42`, `_43`, `_45`, `_57`, `_60`, `_74`, `_76`, `_83`, `_91`, `_96`, `_100`, `_102`, `_103`, `_111`, `_113`, `_115`, `_120`, `_124`, `_130`, `_137`, `_138`, `_144`, `_148`, `_150`, `_157`, `_158`, `_161`, `_164`, `_169`, `_186`, `_192`, `_195`, `_198`.

The 7 new wins do **not** offset the 38 IFFC entries this checkpoint still fails. This is a swap with a large net loss versus IFFC, not a dominance.

#### Failure buckets (one error per failed entry)

Thesis Bot primary buckets; DB-checked. Failed entries: **170**. Sum 2+109+59=170.

| Bucket | Run 13 | Run 14 (DIAGNOSTIC) | Run 15 SFT-v4 |
|--------|--------|---------------------|---------------|
| `empty_turn` | 167 | 160 | **2** (1.2% of failures) |
| `state_mismatch` | 25 | 28 | **109** (64%) |
| `under_call` / `missing_exec` | 7 | 8 | **59** (35%) |

Empty residual still exists but is rare: `_43` empty on turn 2; `_199` empty on turn 4. That is **not** the mass empty-after-tools collapse of Runs 12–14 (101 / 167 / 160 empty primaries).

State mismatch by environment (Run 15, 109 entries): VehicleControl **27**, Travel **26**, TradingBot **20**, GorillaFileSystem **19**, Twitter **7**, Message **7**, Ticket **3**. Example: `_51` (and `_53`, `_54`) — the model emits VehicleControl tool calls but instance state diverges.

Under-call / missing GT execution by first failed turn: turn0=**35**, turn1=13, turn2=7, turn3=2, turn4=2 (59 total).

#### `hit_step_limit` and wall (no tok/s)

`hit_step_limit:turn_0` is present on **57** `multi_turn_timings` rows (exact match to the handoff). Some entries also hit multi-turn step limits on later turns.

From `multi_turn_timings` only (`inference_metrics=0`, `finished_at` NULL — do **not** invent tok/s or TTFT):

| Metric | Run 15 |
|--------|--------|
| Sum `predicted_ms` | ≈ 18.8 generate-minutes |
| Avg `predicted_ms` / entry | ≈ 5631 ms |
| Avg turns | 3.67 |
| Avg `api_calls` | 17.32 |

The higher call volume versus collapsed SFT (Runs 12–14 finished in ~10–14 min wall by emptying early) is consistent with the empty-`[]` pathology being largely fixed: the model now *calls*, then loses on state / under-call.

#### Takeaways

1. **Recovery, not a win.** 15.0% is +29 entries versus SFT v2’s 0.5% and +26 versus the Run 14 diagnostic. It remains **−24 vs NFC and −31 vs IFFC**. Do **not** write that SFT-v4 beat IFFC, NFC, STATE, or BoN.

2. **Empty-`[]` self-conditioning is largely fixed.** Primary empty_turn 167 → **2**. The v4 curriculum (drop first-assistant-empty + clarifying `[]` on top of v3 empty-after-exec) did the job Runs 13–14 did not. Residual empties `_43` / `_199` are late-turn and rare.

3. **Residual is agentic.** 64% state mismatch + 35% under-call. VehicleControl `_51` is the representative state-divergence: the model uses the right API family and still fails the env checker. 57 entries hit `hit_step_limit` on turn 0.

4. **Inference-time stack still wins.** BoN 36.5% > STATE 34.0% > IFFC/GST 30.5% > NFC 27.0% > **SFT-v4 15.0%** > CD 10.5%. Training-time IFFC on this mix has not overtaken the base IFFC channel.

5. **STATE overlay on this GGUF is measured and negative.** Run 16 `--fc-mode iffc-state` on the **same** v4 weights scores **26/200 = 13.0%** (−4 / −2.0 pp vs this run). Contrast: base Gemma STATE (Run 10) was **+3.5 pp** vs IFFC. Do **not** BoN this checkpoint as the next win. Recipe v5 (Run 17) is a **scored regression** at **11.5%** vs this 15.0%. Recipe v6 (Run 18) is a **scored miss** at **13.0%** vs this 15.0% (−2.0 pp / −4). Inference-time BoN **36.5%** remains best on the **base** Gemma stack. This 15.0% remains the best **SFT** recovery.

#### Anti-recommendations

Do **not**:

- present 30/200 = 15.0% as beating IFFC, NFC, STATE, or BoN
- overwrite Runs 12–14 with this `runs` row
- treat v4 as closing the SFT gap to the inference-time ladder
- invent tok/s or TTFT (`inference_metrics=0`)
- treat IFFC+STATE on this GGUF as an open win (Run 16 scored it **negative**)
- propose BoN as the next win on this checkpoint

**Paper / method gap.** SFT v1/v2 copied ToolACE’s empty-list density into an unrolled IFFC adapter and collapsed. v4 shows that dropping first-turn and clarifying `[]` (on top of empty-after-exec) restores *calling*, but not *stateful sequential correctness*. The remaining gap to IFFC is environment fidelity and under-call, not JSON format. Run 16 shows the harness `[STATE]` overlay does **not** close that gap on these weights (null/negative vs plain IFFC). Run 17 shows the v5 turn-0 / state-consistency mix does **not** close it either (scored regression vs this run). Run 18 shows a thinner v4 + `drop_state_inconsistent`-only mix does **not** beat this 15.0% either (scored miss at 13.0%). Next SFT lever is TBD — likely not another thin filter on the same family, not BoN-on-this-ckpt, and not more epochs on v5/v6.

### Gemma-4-E2B-it SFT-IFFC-v4 + IFFC+STATE (Run 16) — scored negative/null ablation

Run 16 is the cheap residual ablation that Run 15 called for: **same SFT-IFFC-v4 GGUF**, same Mac Metal serving stack, **`--fc-mode iffc-state`** instead of plain IFFC. Distinct `runs` row (`run_id=16`); `model_key` remains `gemma4-e2b-sft-iffc-v4` so this does **not** overwrite Run 15 (or Runs 12–14 / base Gemma). Scoring uses the same BFCL `multi_turn_checker`. STATE is the harness-written `[STATE]` snapshot from **executed** tool results (METHODS `--fc-mode iffc-state`). **Not** a retrain, **not** GST, **not** CD/GBNF, **not** LLM summarization, **not** BoN.

**Framing:** this is a **scored negative/null overlay** of IFFC+STATE on the SFT-v4 checkpoint. **26/200 = 13.0%** sits **below Run 15 plain IFFC 15.0%** (−4 entries / −2.0 pp). Contrast: on **base** Gemma, STATE **helped** (Run 10 34.0% vs IFFC 30.5%, +3.5 pp). Empty-`[]` collapse stays fixed (3/174 empty). Residual is still **state_mismatch + under_call**, which is **not** harness-fixable the way STATE was for base Gemma. **Do not propose BoN as the next win on this checkpoint.** Recipe v5 (Run 17) is a **scored regression** at 11.5% vs Run 15. Recipe v6 (Run 18) is a **scored miss** at 13.0% vs Run 15. BoN **36.5%** remains the best inference-time method on the **base** Gemma stack.

#### Checkpoint identity

| Field | Value |
|-------|-------|
| `run_id` | 16 (Mac `benchmark.db`, verified) |
| `model_display` | Gemma-4-E2B-it SFT-IFFC-v4 (Q4_K_M) |
| `model_key` | `gemma4-e2b-sft-iffc-v4` (**same weights as Run 15**) |
| Adapter | MLX LoRA, best val @ iter 1600 (`val=0.196`) — **same as Run 15** |
| Recipe | SFT v4 GGUF unchanged; overlay is `--fc-mode iffc-state` |
| Serve | fused Q4_K_M; `llama-server -m sft/fused/gemma-4-e2b-it-iffc-v4-best1600-Q4_K_M.gguf` |
| Mode | `--fc-mode iffc-state`, `--categories multi_turn_base` |
| Hardware | MacBook Pro M4 Pro, llama.cpp Metal, 48 GB |
| Started / finished | `2026-09-21T18:39:35Z` / `2026-09-21T20:52:58Z` (~133.4 min wall; ~16:39–16:53 ET finish window) |
| Rows | 200 responses, 174 `eval_errors`, 200 `multi_turn_timings` |
| Inference metrics | row exists but **all zeros** — do **not** invent tok/s or TTFT |

Result: **26/200 = 13.0%**. Versus Run 15 plain IFFC: **−2.0 pp / −4 entries**. Versus base STATE (Run 10): **−21.0 pp**. Versus IFFC (Run 5): **−17.5 pp / −35**. Versus BoN (Run 11): **−23.5 pp**. Versus NFC (Run 4): **−14.0 pp / −28**. Empty still fixed (3/174 = 2%). Residual `state_mismatch` **112** / `under_call` **59**. **Do not present 13% as a method win.** **Do not BoN this ckpt as the next experiment.**

#### Same-slice baselines

| Method | Run | Score |
|--------|-----|-------|
| NFC | 4 | 54/200 = 27.0% |
| IFFC (base) | 5 | 61/200 = 30.5% |
| CD | 8 | 21/200 = 10.5% |
| GST | 9 | 61/200 = 30.5% |
| IFFC+STATE (base Gemma) | 10 | 68/200 = 34.0% |
| BoN N=5 | 11 | 73/200 = 36.5% |
| SFT-IFFC LoRA best@1000 | 12 | 1/200 = 0.5% |
| SFT-IFFC-v2 filtered-[] LoRA best@2000 | 13 | 1/200 = 0.5% |
| SFT-IFFC-v2 + strip trailing `[]` (**DIAGNOSTIC**) | 14 | 4/200 = 2.0% |
| SFT-IFFC-v4 LoRA best@1600 (plain IFFC) | 15 | 30/200 = 15.0% |
| **SFT-IFFC-v4 + IFFC+STATE** | **16** | **26/200 = 13.0%** |

Order: CD 10.5% < **SFT-v4+STATE 13.0%** < **SFT-v4 IFFC 15.0%** < NFC 27.0% < IFFC/GST 30.5% < base STATE 34.0% < **BoN 36.5%**. STATE on SFT-v4 is **below** plain IFFC on the same weights. Inference-time BoN **36.5%** remains the best measured result **on base Gemma**.

#### Correct set (26)

`multi_turn_base_42`, `_47`, `_50`, `_64`, `_101`, `_102`, `_104`, `_105`, `_106`, `_109`, `_114`, `_119`, `_122`, `_129`, `_136`, `_139`, `_140`, `_142`, `_154`, `_158`, `_163`, `_169`, `_179`, `_191`, `_192`, `_193`.

`_50` still wins (shared with collapsed SFT and Run 15) — VehicleControl `lockDoors` / `setHeadlights` path.

#### Overlap vs Run 15 (same weights, IFFC → IFFC+STATE)

Unique `entry_id` versus the 30 Run 15 wins:

| Set | Count |
|-----|-------|
| both (shared with Run 15) | 17 |
| only Run 16 (STATE-only wins) | 9 |
| only Run 15 (lost under STATE) | 13 |
| net vs Run 15 | **−4** (9 − 13) |

STATE-only wins: `multi_turn_base_42`, `_47`, `_102`, `_104`, `_140`, `_158`, `_169`, `_179`, `_192`.

Lost vs plain IFFC (13): `_52`, `_78`, `_107`, `_116`, `_133`, `_135`, `_149`, `_156`, `_159`, `_166`, `_182`, `_184`, `_190`.

The 9 STATE-only wins do **not** offset the 13 entries that plain IFFC already had. This is a **swap with a net loss**, not a rescue. Several of the losses (`_149`, `_156`, `_159`, `_166`) were **SFT-v4-only wins versus base IFFC** in Run 15 — STATE dropped them.

#### Failure buckets (one error per failed entry)

Thesis Bot primary buckets; DB-checked. Failed entries: **174**. Sum 3+112+59=174.

| Bucket | Run 15 SFT-v4 IFFC | Run 16 SFT-v4+STATE |
|--------|--------------------|---------------------|
| `empty_turn` | 2 (1.2%) | **3** (2%) |
| `state_mismatch` | 109 (64%) | **112** (64%) |
| `under_call` / `missing_exec` | 59 (35%) | **59** (34%) |

Empty residual still rare: `_175` empty on turn 0; `_185` empty on turn 0; `_108` empty on turn 4. That is **not** the mass empty-after-tools collapse of Runs 12–14. STATE did **not** re-break the v4 empty-recovery.

State mismatch by environment (Run 16, 112 entries): VehicleControl **25**, GorillaFileSystem **23**, Travel **21**, TradingBot **20**, Twitter **10**, Message **9**, Ticket **4**. Slight reshuffle versus Run 15 (VehicleControl 27 / Travel 26 / TradingBot 20 / GFS 19) — still the same env family, not a harness parse failure.

Under-call / missing GT execution by first failed turn: turn0=**31**, turn1=18, turn2=8, turn3=1, turn4=1 (59 total). Turn-0 under-call is still the bulk (31/59), slightly down from Run 15’s turn0=35.

#### Wall (no tok/s)

From `multi_turn_timings` only (`inference_metrics` all zeros — do **not** invent tok/s or TTFT):

| Metric | Run 16 |
|--------|--------|
| Wall | ~133.4 min (`18:39:35Z` → `20:52:58Z`) |
| Sum `predicted_ms` | ≈ 15.9 generate-minutes |
| Avg turns | 3.67 |
| Avg `api_calls` | 14.83 |

Call volume is in the same band as Run 15 (avg `api_calls` 17.32 there): the model still *calls*, then loses on state / under-call. The ~133 min wall is expected for a full 200-entry IFFC+STATE loop, not an early-empty collapse (~10–14 min on Runs 12–14).

#### Takeaways

1. **Null/negative overlay, not a win.** 13.0% is −4 entries versus the same GGUF under plain IFFC. Do **not** write that STATE on SFT-v4 beat IFFC, NFC, base STATE, or BoN.

2. **STATE is checkpoint-dependent.** Base Gemma: Run 10 **+3.5 pp** vs IFFC. SFT-v4: Run 16 **−2.0 pp** vs IFFC. The same harness `[STATE]` does not transfer.

3. **Empty-`[]` stays fixed.** 3/174 empty primaries (2%). Residual empties `_175` / `_185` (turn 0) and `_108` (turn 4) are rare. Keep the v4 empty-recovery curriculum.

4. **Residual is still agentic.** 64% state mismatch + 34% under-call. STATE did not cut either bucket the way it cut empty turns on base Gemma. This is **not** a harness-bookkeeping leftover.

5. **Do not BoN this checkpoint as the next win.** BoN 36.5% remains best inference on the **base** Gemma stack. Recipe v5 (Run 17) is a **scored regression** at 11.5% vs Run 15 15.0% — not more test-time N on these weights and not more epochs on that mix. Recipe v6 (Run 18) is a **scored miss** at 13.0% vs this checkpoint’s 15.0%.

#### Anti-recommendations

Do **not**:

- present 26/200 = 13.0% as beating Run 15, IFFC, NFC, STATE, or BoN
- overwrite Run 15 (or Runs 12–14) with this `runs` row
- propose BoN / `--fc-mode iffc-state-bon` as the next win on this SFT-v4 checkpoint
- invent tok/s or TTFT (`inference_metrics` all zeros)
- treat “STATE on v4” as still-open (it is **measured negative**)
- treat recipe v5 as still-open as a planned win (Run 17 scored it a **regression**)

**Paper / method gap.** Harness STATE helps the **base** instruct checkpoint track forgotten calls. After SFT-v4 restored calling, the remaining errors are env-state divergence and turn-0 incompleteness — not the empty-list / forgotten-call pattern STATE was designed for. Recipe v5 taught turn-0 completeness and env-state consistency while keeping the v4 empty-recovery and **regressed** versus v4 IFFC (Run 17 11.5% < Run 15 15.0%). Recipe v6 kept only `drop_state_inconsistent` on the v4 family and still **missed** v4 (Run 18 13.0% < Run 15 15.0%). Inference-time BoN stays on the **base** Gemma ladder. Next SFT lever is TBD (likely not another thin filter on the same family).

### Gemma-4-E2B-it SFT-IFFC-v5 best@1800 (Run 17) — scored regression vs v4

Run 17 evaluates the mix rebuild that Runs 15–16 called for: [PR #35](https://github.com/sskarz/Thesis/pull/35) SFT v5 unroll keeps every v4 empty-target drop and the recovery-after-`[]` curriculum, **plus** drop underspecified turn-0, synthetic complete first-episode, and drop state-inconsistent continue-after-exec. Train **~29850** rows after prepare. Same IFFC JSON-in-text harness as Run 5 / Run 15 (`--fc-mode iffc`). Requests omit `tools` / `tool_choice`. Scoring uses the same BFCL `multi_turn_checker`. Distinct `model_key` `gemma4-e2b-sft-iffc-v5` so results do **not** overwrite base Gemma or Runs 12–16.

**Not** an inference-time overlay, **not** GST, **not** CD/GBNF, **not** IFFC+STATE / BoN on this checkpoint. The channel is the same Appendix B JSON-in-text; only the unroll curriculum and LoRA weights changed.

**Framing:** this is a **scored regression versus SFT-v4**, not a method win. **23/200 = 11.5%** sits **below Run 15 v4 IFFC 15.0%** (−7 entries / −3.5 pp) and **below Run 16 v4+STATE 13.0%** (−3 / −1.5 pp). Empty-`[]` collapse is still fully suppressed (**0 empty errors** — even cleaner than v4’s 2/170). Residual is still **state_mismatch + under_call**. Recipe v5’s turn-0 / state-consistency filters did **not** lift `multi_turn_base` versus v4 recovery. **Do not present 11.5% as a method win.** **No BoN on this ckpt; no more epochs on the v5 mix.** v4 IFFC **15.0%** remains the best SFT recovery; BoN **36.5%** remains the best inference-time method on **base** Gemma.

#### Checkpoint identity

| Field | Value |
|-------|-------|
| `run_id` | 17 (Mac `benchmark.db`, verified) |
| `model_display` | Gemma-4-E2B-it SFT-IFFC-v5 (Q4_K_M) |
| `model_key` | `gemma4-e2b-sft-iffc-v5` |
| Adapter | MLX LoRA, best val @ iter 1800 (`val=0.147`) |
| Recipe | SFT v5: keep v4 empty-drops + recovery-after-`[]`; drop underspecified turn-0; synthetic complete first-episode; drop state-inconsistent continue-after-exec ([PR #35](https://github.com/sskarz/Thesis/pull/35)). Train ~29850 after prepare. |
| Serve | fused Q4_K_M; `llama-server -m sft/fused/gemma-4-e2b-it-iffc-v5-best1800-Q4_K_M.gguf` |
| Mode | `--fc-mode iffc`, `--categories multi_turn_base` |
| Hardware | MacBook Pro M4 Pro, llama.cpp Metal, 48 GB |
| Started / finished | `2026-09-21T23:00:31Z` / **`finished_at` still NULL** at verify |
| Rows | 200 responses, **177** `eval_errors`, 200 `multi_turn_timings` |
| Inference metrics | **`inference_metrics=0`** — do **not** invent tok/s or TTFT |

Result: **23/200 = 11.5%**. Versus Run 15 v4 IFFC: **−3.5 pp / −7 entries**. Versus Run 16 v4+STATE: **−1.5 pp / −3**. Versus NFC (Run 4): **−15.5 pp / −31**. Versus IFFC (Run 5): **−19.0 pp / −38**. Versus BoN (Run 11): **−25.0 pp / −50**. Empty-`[]` fully suppressed (0 empty-turn primaries of 177 failures). Residual is agentic state mismatch and under-call. **Do not present 11.5% as a method win.**

#### Same-slice baselines

| Method | Run | Score |
|--------|-----|-------|
| NFC | 4 | 54/200 = 27.0% |
| IFFC (base) | 5 | 61/200 = 30.5% |
| CD | 8 | 21/200 = 10.5% |
| GST | 9 | 61/200 = 30.5% |
| IFFC+STATE (base Gemma) | 10 | 68/200 = 34.0% |
| BoN N=5 | 11 | 73/200 = 36.5% |
| SFT-IFFC LoRA best@1000 | 12 | 1/200 = 0.5% |
| SFT-IFFC-v2 filtered-[] LoRA best@2000 | 13 | 1/200 = 0.5% |
| SFT-IFFC-v2 + strip trailing `[]` (**DIAGNOSTIC**) | 14 | 4/200 = 2.0% |
| SFT-IFFC-v4 LoRA best@1600 (plain IFFC) | 15 | 30/200 = 15.0% |
| SFT-IFFC-v4 + IFFC+STATE | 16 | 26/200 = 13.0% |
| **SFT-IFFC-v5 LoRA best@1800** | **17** | **23/200 = 11.5%** |

Order: CD 10.5% < **SFT-v5 11.5%** < SFT-v4+STATE 13.0% < **SFT-v4 IFFC 15.0%** < NFC 27.0% < IFFC/GST 30.5% < base STATE 34.0% < **BoN 36.5%**. v5 sits **below** v4 IFFC and **below** v4+STATE. Inference-time BoN **36.5%** remains the best measured result **on base Gemma**. v4 IFFC **15.0%** remains the best **SFT** recovery.

#### Correct set (23)

`multi_turn_base_12`, `_21`, `_42`, `_50`, `_52`, `_64`, `_78`, `_101`, `_107`, `_109`, `_119`, `_122`, `_133`, `_139`, `_149`, `_154`, `_161`, `_163`, `_166`, `_169`, `_182`, `_184`, `_193`.

`_50` still wins (shared with collapsed SFT and Runs 15–16) — VehicleControl `lockDoors` / `setHeadlights` path.

#### Overlap vs Run 15 (v4 IFFC → v5 IFFC)

Unique `entry_id` versus the 30 Run 15 wins:

| Set | Count |
|-----|-------|
| both (shared with Run 15) | 18 |
| only Run 17 (v5-only wins) | 5 |
| only Run 15 (lost vs v4) | 12 |
| net vs Run 15 | **−7** (5 − 12) |

Shared 18: `multi_turn_base_50`, `_52`, `_64`, `_78`, `_101`, `_107`, `_109`, `_119`, `_122`, `_133`, `_139`, `_149`, `_154`, `_163`, `_166`, `_182`, `_184`, `_193`.

v5-only wins: `multi_turn_base_12`, `_21`, `_42`, `_161`, `_169`.

- `_12`, `_42`, `_161`, `_169` were **IFFC Run 5 wins that v4 missed** (in Run 15’s 38 lost-vs-IFFC). v5 recovered those four.
- `_21` is **new versus both v4 and IFFC** (not in the Run 15 correct set and not in Run 15’s 38 IFFC-only losses).

Lost vs v4 (12): `multi_turn_base_105`, `_106`, `_114`, `_116`, `_129`, `_135`, `_136`, `_142`, `_156`, `_159`, `_190`, `_191`.

- `_156` and `_159` were **SFT-v4-only wins versus base IFFC** in Run 15 — v5 dropped them.

The 5 v5-only wins do **not** offset the 12 entries that v4 already had. This is a **swap with a net loss**, not a rescue. Recipe v5 did not dominate v4 on this slice.

#### Failure buckets (one error per failed entry)

Thesis Bot primary buckets; DB-checked. Failed entries: **177**. Sum 0+133+44=177.

| Bucket | Run 15 SFT-v4 IFFC | Run 16 SFT-v4+STATE | Run 17 SFT-v5 IFFC |
|--------|--------------------|---------------------|--------------------|
| `empty_turn` | 2 (1.2%) | 3 (2%) | **0** (0%) |
| `state_mismatch` | 109 (64%) | 112 (64%) | **133** (~75%) |
| `under_call` / `missing_exec` | 59 (35%) | 59 (34%) | **44** (~25%) |

Empty-`[]` collapse is **fully suppressed** — even cleaner than v4’s 2 residual empties (`_43` / `_199`). That is **not** the mass empty-after-tools collapse of Runs 12–14 (101 / 167 / 160 empty primaries).

State mismatch by environment (Run 17, 133 entries): Travel **32**, TradingBot **30**, VehicleControl **28**, GorillaFileSystem **23**, Twitter **8**, Message **8**, Ticket **4**. Versus Run 15 (VehicleControl 27 / Travel 26 / TradingBot 20 / GFS 19) the same env family **grew**, especially Travel and TradingBot. The turn-0 completeness / state-consistency filters did **not** cut env-diff failures.

Under-call / missing GT execution by first failed turn: turn0=**27**, turn1=8, turn2=8, turn3=1 (44 total). Turn-0 under-call is still the bulk (27/44), down from Run 15’s turn0=35 / 59 total — a partial under-call cut that does **not** offset the +24 state-mismatch rise (109→133).

#### Wall (no tok/s)

From `multi_turn_timings` only (`inference_metrics=0`, `finished_at` NULL — do **not** invent tok/s or TTFT):

| Metric | Run 17 |
|--------|--------|
| Sum `predicted_ms` | ≈ 51.9 generate-minutes |
| Avg turns | 3.67 |
| Avg `api_calls` | 40.27 |

Call volume is **higher** than v4 (Run 15 avg `api_calls` 17.32; Run 16 14.83): the model still *calls* — more, not less — then loses on state / under-call. The ~52 generate-minutes is consistent with a full 200-entry IFFC loop that is **not** an early-empty collapse (~10–14 min on Runs 12–14).

#### Takeaways

1. **Regression, not a win.** 11.5% is −7 entries versus the v4 IFFC recovery and −3 versus v4+STATE. Do **not** write that SFT-v5 beat v4, IFFC, NFC, STATE, or BoN.

2. **Empty-`[]` stays fully fixed.** Primary empty_turn 2 → **0**. The v4 empty-recovery curriculum still holds under v5. Residual is **not** the stop-after-tools collapse.

3. **Residual is still agentic — and state mismatch got worse.** ~75% state mismatch (133) + ~25% under-call (44). Under-call improved (59→44, turn0 35→27) while state mismatch rose (109→133). Net −7. Travel 32 / TradingBot 30 / VehicleControl 28 / GFS 23.

4. **Turn-0 / state-consistency filters did not lift this slice.** The v5 recipe targeted exactly those residuals and still lost 12 of v4’s 30 wins, including two v4-only-vs-IFFC entries (`_156`, `_159`). Four recovered IFFC wins (`_12`, `_42`, `_161`, `_169`) plus `_21` do not offset that.

5. **Do not BoN this checkpoint; do not more-epochs the v5 mix.** BoN 36.5% remains best inference on the **base** Gemma stack. v4 IFFC **15.0%** remains the best **SFT** recovery. Recipe v6 (Run 18) is a **scored miss** at 13.0% vs v4 — not another thin filter on this family as a planned win.

#### Anti-recommendations

Do **not**:

- present 23/200 = 11.5% as beating Run 15, Run 16, IFFC, NFC, STATE, or BoN
- overwrite Runs 12–16 with this `runs` row
- propose BoN / `--fc-mode iffc-state-bon` as the next win on this SFT-v5 checkpoint
- train more epochs on the v5 mix
- invent tok/s or TTFT (`inference_metrics=0`)
- treat “recipe v5” as still-open as a planned win (it is **measured regression**)

**Paper / method gap.** v4 restored calling by dropping empty-`[]` teachers. v5 tried to teach turn-0 completeness and env-state consistency on top of that recovery. Empty stayed fixed (0); under-call fell; state mismatch rose enough to net a **−3.5 pp** regression. Filtering / rewriting ToolACE+xLAM first-episode gold did not transfer to BFCL `multi_turn_base` env-diff. Recipe v6 dropped the suspected-harm turn-0 rewrites and kept only `drop_state_inconsistent`; Run 18 scored **13.0%** — a modest lift vs this 11.5% that still **misses v4 15.0%**. Inference-time BoN stays on the **base** Gemma ladder. Next SFT lever is TBD (likely not another thin filter on the same family).

### Gemma-4-E2B-it SFT-IFFC-v6 best@1800 (Run 18) — scored miss vs v4

Run 18 evaluates the mix that Run 17 called for: [PR #38](https://github.com/sskarz/Thesis/pull/38) SFT v6 unroll keeps every v4 empty-target drop and the recovery-after-`[]` curriculum, **plus only** `drop_state_inconsistent`. Turn-0 underspec drop and synthetic complete turn-0 are **off**. Same IFFC JSON-in-text harness as Run 5 / Run 15 (`--fc-mode iffc`). Requests omit `tools` / `tool_choice`. Scoring uses the same BFCL `multi_turn_checker`. Distinct `model_key` `gemma4-e2b-sft-iffc-v6` ([PR #39](https://github.com/sskarz/Thesis/pull/39)) so results do **not** overwrite base Gemma or Runs 12–17.

**Not** an inference-time overlay, **not** GST, **not** CD/GBNF, **not** IFFC+STATE / BoN on this checkpoint. The channel is the same Appendix B JSON-in-text; only the unroll curriculum and LoRA weights changed.

**Framing:** this is a **scored miss versus SFT-v4**, not a method win. **26/200 = 13.0%** sits **below Run 15 v4 IFFC 15.0%** (−4 entries / −2.0 pp). Modest lift vs **Run 17 v5 11.5%** (+3 / +1.5 pp). Ties Run 16 v4+STATE at 13.0% (different ckpt/mode). Empty-`[]` collapse is still essentially suppressed (**1** empty on `_43` turn 2 — vs v4’s 2 and v5’s 0). Residual is still **state_mismatch + under_call**. Avg `api_calls` **13.34** (vs Run 15 **17.3**, Run 17 **40.3**) — v6 stopped the v5 call-churn blowup. **Do not present 13.0% as a method win.** **No BoN on this ckpt; no more epochs on the v6 mix.** v4 IFFC **15.0%** remains the best SFT recovery; BoN **36.5%** remains the best inference-time method on **base** Gemma.

#### Checkpoint identity

| Field | Value |
|-------|-------|
| `run_id` | 18 (Mac `benchmark.db`, verified) |
| `model_display` | Gemma-4-E2B-it SFT-IFFC-v6 (Q4_K_M) |
| `model_key` | `gemma4-e2b-sft-iffc-v6` |
| Adapter | MLX LoRA, best val @ iter 1800 (`val=0.142`); from `adapters-v6-recovery` |
| Recipe | SFT v6: v4 empty/recovery + only `drop_state_inconsistent`; turn-0 underspec drop and synthetic complete turn-0 **off** ([PR #38](https://github.com/sskarz/Thesis/pull/38)). Model key [PR #39](https://github.com/sskarz/Thesis/pull/39). |
| Serve | fused Q4_K_M; `llama-server -m sft/fused/gemma-4-e2b-it-iffc-v6-best1800-Q4_K_M.gguf` |
| Mode | `--fc-mode iffc`, `--categories multi_turn_base` |
| Hardware | MacBook Pro M4 Pro, llama.cpp Metal, 48 GB |
| Started / finished | `2026-09-22T12:50:47Z` / `2026-09-22T14:01:37Z` (~**70.8 min** wall) |
| Rows | 200 responses, **174** `eval_errors`, 200 `multi_turn_timings` |
| Inference metrics | present (n=50 sampled requests): gen_tps_mean ≈ **103.1**; prompt_tps_mean ≈ **741.4**; peak_rss ≈ **9170 MB** |

Result: **26/200 = 13.0%**. Versus Run 15 v4 IFFC: **−2.0 pp / −4 entries**. Versus Run 17 v5: **+1.5 pp / +3**. Versus Run 16 v4+STATE: **0.0 pp**. Versus NFC (Run 4): **−14.0 pp**. Versus IFFC (Run 5): **−17.5 pp**. Versus BoN (Run 11): **−23.5 pp**. Empty-`[]` still essentially suppressed (1 empty-turn primary of 174 failures). Residual is agentic state mismatch and under-call. **Do not present 13.0% as a method win.**

#### Same-slice baselines

| Method | Run | Score |
|--------|-----|-------|
| NFC | 4 | 54/200 = 27.0% |
| IFFC (base) | 5 | 61/200 = 30.5% |
| CD | 8 | 21/200 = 10.5% |
| GST | 9 | 61/200 = 30.5% |
| IFFC+STATE (base Gemma) | 10 | 68/200 = 34.0% |
| BoN N=5 | 11 | 73/200 = 36.5% |
| SFT-IFFC LoRA best@1000 | 12 | 1/200 = 0.5% |
| SFT-IFFC-v2 filtered-[] LoRA best@2000 | 13 | 1/200 = 0.5% |
| SFT-IFFC-v2 + strip trailing `[]` (**DIAGNOSTIC**) | 14 | 4/200 = 2.0% |
| SFT-IFFC-v4 LoRA best@1600 (plain IFFC) | 15 | 30/200 = 15.0% |
| SFT-IFFC-v4 + IFFC+STATE | 16 | 26/200 = 13.0% |
| SFT-IFFC-v5 LoRA best@1800 | 17 | 23/200 = 11.5% |
| **SFT-IFFC-v6 LoRA best@1800** | **18** | **26/200 = 13.0%** |

Order: CD 10.5% < SFT-v5 11.5% < **SFT-v6 13.0%** = SFT-v4+STATE 13.0% < **SFT-v4 IFFC 15.0%** < NFC 27.0% < IFFC/GST 30.5% < base STATE 34.0% < **BoN 36.5%**. v6 sits **below** v4 IFFC, **ties** v4+STATE, and **slightly above** v5. Inference-time BoN **36.5%** remains the best measured result **on base Gemma**. v4 IFFC **15.0%** remains the best **SFT** recovery.

#### Correct set (26)

`multi_turn_base_42`, `_47`, `_50`, `_101`, `_102`, `_116`, `_119`, `_122`, `_133`, `_135`, `_139`, `_142`, `_144`, `_149`, `_154`, `_156`, `_161`, `_163`, `_169`, `_182`, `_184`, `_186`, `_190`, `_191`, `_192`, `_193`.

`_50` still wins (shared with collapsed SFT and Runs 15–17) — VehicleControl `lockDoors` / `setHeadlights` path.

#### Overlap vs Run 15 (v4 IFFC → v6 IFFC)

Unique `entry_id` versus the 30 Run 15 wins:

| Set | Count |
|-----|-------|
| both (shared with Run 15) | 18 |
| only Run 18 (v6-only wins) | 8 |
| only Run 15 (lost vs v4) | 12 |
| net vs Run 15 | **−4** (8 − 12) |

Shared 18: `multi_turn_base_50`, `_101`, `_116`, `_119`, `_122`, `_133`, `_135`, `_139`, `_142`, `_149`, `_154`, `_156`, `_163`, `_182`, `_184`, `_190`, `_191`, `_193`.

v6-only wins: `multi_turn_base_42`, `_47`, `_102`, `_144`, `_161`, `_169`, `_186`, `_192`.

Lost vs v4 (12): `multi_turn_base_52`, `_64`, `_78`, `_105`, `_106`, `_107`, `_109`, `_114`, `_129`, `_136`, `_159`, `_166`.

- Mostly TradingBot / VehicleControl **state_mismatch**.
- Turn-0 **under_call** on `_105`, `_114`, `_159`, `_166`.

The 8 v6-only wins do **not** offset the 12 entries that v4 already had. This is a **swap with a net loss**, not a rescue. Recipe v6 did not dominate v4 on this slice.

#### Overlap vs Run 17 (v5 IFFC → v6 IFFC)

Unique `entry_id` versus the 23 Run 17 wins:

| Set | Count |
|-----|-------|
| both (shared with Run 17) | 15 |
| only Run 18 (v6-only vs v5) | 11 |
| only Run 17 (lost vs v5) | 8 |
| net vs Run 17 | **+3** (11 − 8) |

Shared 15: `multi_turn_base_42`, `_50`, `_101`, `_119`, `_122`, `_133`, `_139`, `_149`, `_154`, `_161`, `_163`, `_169`, `_182`, `_184`, `_193`.

v6-only vs v5: `multi_turn_base_47`, `_102`, `_116`, `_135`, `_142`, `_144`, `_156`, `_186`, `_190`, `_191`, `_192`.

Lost vs v5 (8): `multi_turn_base_12`, `_21`, `_52`, `_64`, `_78`, `_107`, `_109`, `_166`.

Removing v5’s turn-0 synth/underspec filters helped a little (+3 entries) but did **not** recover to v4.

#### Failure buckets (one error per failed entry)

Thesis Bot primary buckets; DB-checked. Failed entries: **174**. Sum 1+118+55=174.

| Bucket | Run 15 SFT-v4 IFFC | Run 16 SFT-v4+STATE | Run 17 SFT-v5 IFFC | Run 18 SFT-v6 IFFC |
|--------|--------------------|---------------------|--------------------|--------------------|
| `empty_turn` | 2 (1.2%) | 3 (2%) | **0** (0%) | **1** (~0.6%) |
| `state_mismatch` | 109 (64%) | 112 (64%) | **133** (~75%) | **118** (~68%) |
| `under_call` / `missing_exec` | 59 (35%) | 59 (34%) | **44** (~25%) | **55** (~32%) |

Empty-`[]` collapse is **still essentially suppressed** — 1 residual empty (`_43` turn 2), matching one of v4’s two leftovers (`_43` / `_199`). That is **not** the mass empty-after-tools collapse of Runs 12–14 (101 / 167 / 160 empty primaries).

State mismatch by environment (Run 18, 118 entries): VehicleControl **31**, TradingBot **26**, GorillaFileSystem **22**, Travel **19**, Twitter **10**, Ticket **5**, Message **5**. Versus Run 15 (VehicleControl 27 / Travel 26 / TradingBot 20 / GFS 19) VehicleControl and TradingBot **grew**. Versus Run 17 (Travel 32 / TradingBot 30 / VehicleControl 28 / GFS 23) the v5 state-mismatch spike **eased** (133→118) but did not fall below v4’s 109.

Under-call / missing GT execution by first failed turn: turn0=**38**, turn1=8, turn2=6, turn3=2, turn4=1 (55 total). Turn-0 under-call is still the bulk (38/55), up from Run 17’s turn0=27 / 44 total and similar to Run 15’s turn0=35 / 59 total.

#### Wall / throughput

From `multi_turn_timings` plus `inference_metrics` (n=50 sampled requests):

| Metric | Run 18 |
|--------|--------|
| Wall | ~70.8 min |
| Avg turns | 3.67 |
| Avg `api_calls` | **13.34** |
| gen_tps_mean | ≈ 103.1 |
| prompt_tps_mean | ≈ 741.4 |
| peak_rss | ≈ 9170 MB |

Call volume is **sane** versus v5 (Run 17 avg `api_calls` **40.27**; Run 15 **17.32**; Run 16 **14.83**): v6 stopped the v5 call-churn blowup. The model still *calls*, then loses on state / under-call — not an early-empty collapse.

#### Takeaways

1. **Miss versus v4, not a win.** 13.0% is −4 entries versus the v4 IFFC recovery, +3 versus v5, and a tie versus v4+STATE. Do **not** write that SFT-v6 beat v4, IFFC, NFC, STATE, or BoN.

2. **Empty-`[]` stays essentially fixed.** Primary empty_turn 0→**1** (`_43` turn 2). The v4 empty-recovery curriculum still holds under v6. Residual is **not** the stop-after-tools collapse.

3. **Residual is still agentic.** ~68% state mismatch (118) + ~32% under-call (55). VehicleControl 31 / TradingBot 26 / GFS 22 / Travel 19. Lost vs v4 is mostly TradingBot/VehicleControl state_mismatch, plus turn-0 under_call on `_105`, `_114`, `_159`, `_166`.

4. **Thin state-inconsistent-only filter did not beat v4.** Turning off v5’s turn-0 synth/underspec recovered some of the v5 regression (11.5%→13.0%) and cut api_calls 40.3→13.34, but the 8 v6-only wins do not offset the 12 lost v4 entries.

5. **Do not BoN this checkpoint; do not more-epochs the v6 mix.** BoN 36.5% remains best inference on the **base** Gemma stack. v4 IFFC **15.0%** remains the best **SFT** recovery. Next SFT lever is TBD — likely not another thin filter on the same family.

#### Anti-recommendations

Do **not**:

- present 26/200 = 13.0% as beating Run 15, IFFC, NFC, STATE, or BoN
- overwrite Runs 12–17 with this `runs` row
- propose BoN / `--fc-mode iffc-state-bon` as the next win on this SFT-v6 checkpoint
- train more epochs on the v6 mix
- treat “recipe v6” as still-open as a planned win (it is a **measured miss vs v4**)

**Paper / method gap.** v4 restored calling by dropping empty-`[]` teachers. v5 added turn-0 completeness and env-state consistency and **regressed**. v6 kept only the suspected-keep (`drop_state_inconsistent`) and still **missed v4** (13.0% < 15.0%). Empty stays fixed; api_calls returned to a sane range; residual is still env-diff + turn-0 under-call. Thin filter ablation on this ToolACE+xLAM family looks exhausted. Inference-time BoN stays on the **base** Gemma ladder. Next SFT lever is TBD. Run 19 xLAM-1b NFC is a **separate** scored format-channel collapse on this harness (0.0%) — not an SFT mix result. Run 20 `--fc-mode xlam` is the **format-fixed fair-channel** score on that same published GGUF (3.0%) — still **very low**, **not** a method win, **not** published ~78.94%.

### xLAM-1b-fc-r NFC (Run 19) — scored format-channel collapse (not an ability floor)

Run 19 evaluates the published Salesforce xLAM-1b-fc-r Q4_K_M GGUF (`Salesforce/xLAM-1b-fc-r-gguf`) on the **same** BFCL v4 `multi_turn_base` NFC harness as Gemma Run 4 (`--fc-mode nfc`; native `tools` / `tool_calls` via llama-server `--jinja`). Registry key `xlam-1b` is in `benchmark/config.py` ([PR #41](https://github.com/sskarz/Thesis/pull/41)); distinct `api_name` / `slug` so results do **not** overwrite Gemma or SFT rows. Scoring uses the same BFCL `multi_turn_checker`. **No new method.** PLAN Tier 1 §3 / METHODS §15.

**Framing:** this is a **channel/format collapse on our multi_turn NFC harness**, useful as a **negative external baseline** versus Gemma NFC Run 4 **54/200 = 27.0%**. **Do not** claim xLAM scores 0 on BFCL overall. **Do not** equate to published Salesforce xLAM overall BFCL ~**78.94%** (different metric / suite aggregation). **Do not** present 0.0% as a method win or as an ability floor. The fair-channel score is Run 20 `--fc-mode xlam` (**6/200 = 3.0%**), documented next — still **very low**, **not** a method win, **not** published ~78.94%.

**Not a dead server.** `inference_metrics` gen_tps_mean ≈ **196.7** tok/s (median 196.7); prompt_tps_mean ≈ **1931.6**; ttft_mean_ms ≈ **37.6**; peak_rss_mb ≈ **18560.9** (n=50 sampled requests). Avg `api_calls` = **3.63** (min 1, max 7, n=200). The model generated tokens; the NFC harness recorded **no tool calls**.

#### Checkpoint identity

| Field | Value |
|-------|-------|
| `run_id` | 19 (Mac `benchmark.db`, verified) |
| `model_display` | xLAM-1b-fc-r (Q4_K_M) |
| `model_key` | `xlam-1b` |
| Slug | `xLAM-1b-fc-r-Q4_K_M` |
| Serve | `-hf Salesforce/xLAM-1b-fc-r-gguf:Q4_K_M` (HF file `xLAM-1b-fc-r.Q4_K_M.gguf`) |
| Mode | `--fc-mode nfc`, `--categories multi_turn_base` |
| Hardware | MacBook Pro M4 Pro, llama.cpp Metal |
| Started / finished | `2026-09-22T14:13:09Z` / `2026-09-22T14:26:42Z` (~**13.55 min** wall; ≈ **10:26 AM ET**) |
| Pipeline | `BFCL_RUN19_EXIT:0` / `PIPELINE_DONE:0` |
| Registry | Thesis [PR #41](https://github.com/sskarz/Thesis/pull/41) (merged) |
| Rows | 200 responses, **200** `eval_errors` |
| Inference metrics | present (n=50): gen_tps_mean ≈ **196.7**; prompt_tps_mean ≈ **1931.6**; ttft_mean_ms ≈ **37.6**; peak_rss ≈ **18560.9 MB** |

Result: **0/200 = 0.0%**. Versus Gemma NFC (Run 4): **−27.0 pp / −54 entries**. Versus IFFC (Run 5): **−30.5 pp**. Versus BoN (Run 11): **−36.5 pp**. Versus CD (Run 8): **−10.5 pp**. **Do not present 0.0% as a method win or as an ability floor.**

#### Same-slice baselines

| Method | Run | Score |
|--------|-----|-------|
| **xLAM-1b-fc-r NFC** (external; format-channel collapse) | **19** | **0/200 = 0.0%** |
| **xLAM-1b-fc-r `--fc-mode xlam`** (external; fair channel; still very low) | **20** | **6/200 = 3.0%** |
| CD | 8 | 21/200 = 10.5% |
| SFT-IFFC-v5 LoRA best@1800 | 17 | 23/200 = 11.5% |
| SFT-IFFC-v6 LoRA best@1800 | 18 | 26/200 = 13.0% |
| SFT-IFFC-v4 + IFFC+STATE | 16 | 26/200 = 13.0% |
| SFT-IFFC-v4 LoRA best@1600 (plain IFFC) | 15 | 30/200 = 15.0% |
| Gemma NFC | 4 | 54/200 = 27.0% |
| IFFC (base) / GST | 5 / 9 | 61/200 = 30.5% |
| IFFC+STATE (base Gemma) | 10 | 68/200 = 34.0% |
| BoN N=5 | 11 | 73/200 = 36.5% |

Order: **xLAM NFC 0.0%** < **xLAM `--fc-mode xlam` 3.0%** < CD 10.5% < SFT-v5 11.5% < SFT-v6 13.0% = SFT-v4+STATE 13.0% < **SFT-v4 IFFC 15.0%** < Gemma NFC 27.0% < IFFC/GST 30.5% < base STATE 34.0% < **BoN 36.5%**. Inference-time BoN **36.5%** remains the best measured result **on base Gemma**. v4 IFFC **15.0%** remains the best **SFT** recovery.

#### Failure taxonomy (one error per failed entry)

**All 200** eval errors: `Model response list is empty for turn 0` (classified empty). Turn-0 empty on **all 200** responses.

| Bucket | n |
|--------|---|
| `Model response list is empty for turn 0` | **200 / 200** |

#### Response shapes

- **196/200**: all turns empty lists, e.g. `multi_turn_base_0` → `[[], [], [], []]`
- **4/200** have non-empty cells, all `["ERROR: "]` style (still turn-0 empty, so the **same** error bucket):
  - `_33`: `[[], [["ERROR: "]], [["ERROR: "]], [], []]`
  - `_35`: `[[], [["ERROR: "]], [["ERROR: "]]]`
  - `_36`: `[[], [["ERROR: "]], [["ERROR: "]]]`
  - `_37`: `[[], [], [["ERROR: "]]]`

Sample errors (all identical): `multi_turn_base_0`, `_1`, `_10`, `_100`, `_101`, `_102`, `_103`, `_104` → `Model response list is empty for turn 0`.

#### Wall / throughput

From `multi_turn_timings` plus `inference_metrics` (n=50 sampled requests):

| Metric | Run 19 |
|--------|--------|
| Wall | **13.55 min** (`14:13:09Z` → `14:26:42Z`) |
| Avg `api_calls` | **3.63** (min 1, max 7, n=200) |
| gen_tps_mean | ≈ **196.7** |
| prompt_tps_mean | ≈ **1931.6** |
| ttft_mean_ms | ≈ **37.6** |
| peak_rss | ≈ **18560.9 MB** |

The ~13.55 min wall and ~197 tok/s generation are consistent with a live 200-entry NFC loop that is **not** a hung or empty server. The model produced tokens; our NFC harness recorded **no tool calls**.

#### Takeaways

1. **Format-channel collapse, not an ability floor.** 0/200 is a harness/format miss on **this** NFC path. Do **not** write that xLAM cannot call tools, that it scores 0 on BFCL overall, or that published ~78.94% is falsified.

2. **Useful negative external baseline.** Same harness, same slice: Gemma NFC Run 4 = **27.0%**. xLAM NFC = **0.0%**. That contrast is about **channel compatibility**, not Gemma vs published xLAM overall.

3. **Not a dead server.** gen_tps_mean ≈ 196.7; avg `api_calls` 3.63. Tokens were generated. The checker saw empty turn-0 tool-call lists.

4. **Do not treat 0.0% as a method win.** This is not an SFT mix, not IFFC, not STATE, not BoN. Best inference remains BoN **36.5%** (base Gemma). Best SFT remains v4 IFFC **15.0%**.

5. **Fair channel is Run 20, not an IFFC diagnostic.** `--fc-mode xlam` (PR #42) is the format-fixed path. It scores **6/200 = 3.0%** — still **very low**. Do **not** treat Run 19 0.0% as an ability floor.

#### Anti-recommendations

Do **not**:

- claim xLAM scores 0 on BFCL overall
- equate 0.0% to published Salesforce xLAM overall BFCL ~78.94% (different metric)
- present 0/200 as a method win or as an ability floor
- invent that Run 20 scored published ~78.94%
- invent tok/s beyond verified gen_tps_mean ≈ **196.7**
- overwrite this `runs` row
- treat a hung/dead llama-server as the explanation (metrics contradict that)

**Paper / method gap.** Specialist fine-tuned SLMs can still collapse on a **stock NFC** multi-turn harness if their native emission format does not land in llama-server `tool_calls`. Gemma NFC 27% shows the same channel is usable for a different model. Run 20 (`--fc-mode xlam`) is the **format-fixed fair-channel** score on this GGUF: **3.0%**, still **very low** — channel fixed vs this 0.0%, not a win vs Gemma/SFT/BoN.

### xLAM-1b-fc-r `--fc-mode xlam` (Run 20) — format-fixed fair-channel baseline (still 3.0%; not a method win)

Run 20 scores the **same** published Salesforce xLAM-1b-fc-r Q4_K_M GGUF as Run 19 (`Salesforce/xLAM-1b-fc-r-gguf`; registry key `xlam-1b`, [PR #41](https://github.com/sskarz/Thesis/pull/41)) on BFCL v4 `multi_turn_base` with `--fc-mode xlam` ([PR #42](https://github.com/sskarz/Thesis/pull/42)): no native OpenAI `tools` / `tool_choice`; Salesforce-style system prompt; parse JSON-in-text `{"tool_calls":[...]}` in assistant `content`. Scoring uses the same BFCL `multi_turn_checker`. **No new training method.** PLAN Tier 1 §3 / METHODS §15.

**Framing:** Run 19 NFC was **0/200 = 0.0%** because NFC never parses `content` (format-channel collapse). Run 20 is the **format-fixed fair-channel external baseline**. A pre-run smoke extracted calls (5/5). The full score **6/200 = 3.0%** is still **very low** — an ability + behavior miss on this harness, **not** a method win. **Do not** equate to published Salesforce xLAM overall BFCL ~**78.94%** (different metric / suite aggregation). Place on the scoreboard **above** Run 19 0.0% and **below** CD 10.5% / all SFT and Gemma baselines. Best inference remains BoN **36.5%** (base Gemma). Best SFT remains v4 IFFC **15.0%**.

**Tools actually ran.** Avg `api_calls` = **50.47** (min 2, max 101, n=200) versus Run 19 **3.63** (empty channel). Dominant failure is 20-step tool loops / over-calling plus parse fails and residual state drift — **not** empty turn-0 lists.

#### Checkpoint identity

| Field | Value |
|-------|-------|
| `run_id` | 20 (Mac `benchmark.db`, verified) |
| `model_display` | xLAM-1b-fc-r (Q4_K_M) |
| `model_key` | `xlam-1b` |
| Slug | `xLAM-1b-fc-r-Q4_K_M` |
| Serve | `-hf Salesforce/xLAM-1b-fc-r-gguf:Q4_K_M` (HF file `xLAM-1b-fc-r.Q4_K_M.gguf`) |
| Mode | `--fc-mode xlam`, `--categories multi_turn_base` |
| Hardware | MacBook Pro M4 Pro, llama.cpp Metal |
| Started / finished | `2026-09-22T15:28:35Z` / `2026-09-23T04:45:30Z` (~**13.28 h** wall; ≈ **11:28 AM ET Tue → 12:45 AM ET Wed**) |
| Pipeline | Exit 0 / `PIPELINE_DONE` (per Thesis Bot handoff) |
| Adapter | Thesis [PR #42](https://github.com/sskarz/Thesis/pull/42) (merged; xLAM JSON-in-text channel) |
| Registry | Thesis [PR #41](https://github.com/sskarz/Thesis/pull/41) (merged) |
| Rows | 200 responses, **194** `eval_errors` |
| Inference metrics | present (n=50): gen_tps_mean ≈ **91.6**; prompt_tps_mean ≈ **1354.9**; ttft_mean_ms ≈ **1853.4**; peak_rss ≈ **16621.8 MB** |

Result: **6/200 = 3.0%**. Versus Run 19 NFC: **+3.0 pp / +6 entries**. Versus Gemma NFC (Run 4): **−24.0 pp / −48 entries**. Versus CD (Run 8): **−7.5 pp**. Versus IFFC (Run 5): **−27.5 pp**. Versus BoN (Run 11): **−33.5 pp**. **Do not present 3.0% as a method win. Do not equate to published ~78.94%.**

#### Same-slice baselines

| Method | Run | Score |
|--------|-----|-------|
| **xLAM-1b-fc-r NFC** (external; format-channel collapse) | **19** | **0/200 = 0.0%** |
| **xLAM-1b-fc-r `--fc-mode xlam`** (external; fair channel; still very low) | **20** | **6/200 = 3.0%** |
| CD | 8 | 21/200 = 10.5% |
| SFT-IFFC-v5 LoRA best@1800 | 17 | 23/200 = 11.5% |
| SFT-IFFC-v6 LoRA best@1800 | 18 | 26/200 = 13.0% |
| SFT-IFFC-v4 + IFFC+STATE | 16 | 26/200 = 13.0% |
| SFT-IFFC-v4 LoRA best@1600 (plain IFFC) | 15 | 30/200 = 15.0% |
| Gemma NFC | 4 | 54/200 = 27.0% |
| IFFC (base) / GST | 5 / 9 | 61/200 = 30.5% |
| IFFC+STATE (base Gemma) | 10 | 68/200 = 34.0% |
| BoN N=5 | 11 | 73/200 = 36.5% |

Order:

```
xLAM NFC 0.0% < xLAM fc-mode xlam 3.0% < CD 10.5% < SFT-v5 11.5% < SFT-v6 13.0% = SFT-v4+STATE 13.0%
  < SFT-v4 IFFC 15.0% < Gemma NFC 27.0% < IFFC/GST 30.5% < base STATE 34.0% < BoN 36.5%
```

Inference-time BoN **36.5%** remains the best measured result **on base Gemma**. v4 IFFC **15.0%** remains the best **SFT** recovery.

#### Failure taxonomy (one error per failed entry)

**194/200** eval errors (194 distinct entries).

| Family | n |
|--------|---|
| `state_mismatch` (instance ≠ GT for API) | **128** |
| under-call / missing GT exec results | **46** |
| empty turn (turn 0/1/2) | **20** |

`state_mismatch` by API: VehicleControlAPI **34** · TravelAPI **30** · TradingBot **23** · GorillaFileSystem **22** · MessageAPI **7** · TicketAPI **7** · TwitterAPI **5**.

#### Response flags (entries with ≥1; n=200)

| Flag | entries |
|------|---------|
| `hit_step_limit` | **171 / 200** (445 turn occurrences) |
| `xlam_unparseable` | **111 / 200** (201 turn occurrences) |
| `context_overflow` | **42 / 200** (72 turn occurrences) |
| (no flags) | 4 |

Dominant pattern: **20-step tool loops / over-calling** (`hit_step_limit`) plus frequent parse fails (`xlam_unparseable`), with residual state drift. Unlike Run 19, the model is **in the tool loop**.

#### Correct entries (6)

`multi_turn_base_{50,119,133,139,144,161}` — several still hit step limits or unparseable flags but passed overall score.

#### Example failure (`multi_turn_base_51`, VehicleControl `state_mismatch`)

Turn 0: 19× repeated `lockDoors(...unlock=False...)` → `hit_step_limit`; later turns over-call `find_nearest_tire_shop` / `message_login`.

#### Example vs Run 19 (`multi_turn_base_0`)

Run 19 `_0`: `[[],[],[],[]]` empty. Run 20 `_0`: 20× GorillaFileSystem calls per early turns (`cd`, `grep`, `sort` loops) + `xlam_unparseable` later — **channel works; policy/state fails**.

#### Wall / throughput

From `multi_turn_timings` plus `inference_metrics` (n=50 sampled requests):

| Metric | Run 20 |
|--------|--------|
| Wall | **~13.28 h** (`2026-09-22T15:28:35Z` → `2026-09-23T04:45:30Z`) |
| Avg `api_calls` | **50.47** (min 2, max 101, n=200) |
| gen_tps_mean | ≈ **91.6** (median **86.5**) |
| prompt_tps_mean | ≈ **1354.9** |
| ttft_mean_ms | ≈ **1853.4** |
| peak_rss | ≈ **16621.8 MB** |

Slower generation than Run 19 ~197 tok/s is expected: real multi-turn tool loops, ~13 h wall, not empty-channel NFC. Do **not** invent other tok/s.

#### Takeaways

1. **Format channel is fixed; ability is still weak.** Run 19 0.0% was NFC never parsing `content`. Run 20 3.0% is the fair JSON-in-text channel. Tools execute. Score remains **very low**.

2. **Not a method win.** 3.0% sits **above** Run 19 0.0% and **below** CD 10.5% and every Gemma/SFT/BoN number on this slice. Do **not** present as beating CD, SFT-v4, IFFC, NFC, STATE, or BoN.

3. **Not published Salesforce overall BFCL ~78.94%.** Different metric / suite aggregation. Do **not** equate 3.0% (or 0.0%) with that leaderboard figure.

4. **Over-calling dominates.** `hit_step_limit` on **171/200**; `xlam_unparseable` on **111/200**; `state_mismatch` **128**. Contrast Run 19: all empty turn-0 lists.

5. **Best numbers unchanged.** Best inference remains BoN **36.5%** (base Gemma). Best SFT remains v4 IFFC **15.0%**.

#### Anti-recommendations

Do **not**:

- claim ~78.94% parity or that this pipeline reproduced published xLAM overall BFCL
- present 6/200 = 3.0% as a method win
- overwrite Run 19 or this `runs` row
- invent tok/s beyond verified gen_tps_mean ≈ **91.6** (n=50)
- treat Run 19 0.0% as the fair xLAM baseline going forward (the fair channel is this run)
- BoN this GGUF as the next win

**Paper / method gap.** Fixing the format channel (PR #42) moved xLAM from empty-list NFC (0.0%) into a real tool loop (avg `api_calls` 50.47; 3.0%). Residual is over-calling, parse fails, and env-state drift — not “NFC never saw a call.” That is still far from Gemma NFC 27% / IFFC 30.5% / BoN 36.5% on this harness, and it is **not** the published overall BFCL figure.

### Gemma-4-E2B-it IFFC-stop-repeat (Run 27) — scored miss vs IFFC on base Gemma

Run 27 scores **base** Gemma-4-E2B-it Q4_K_M (`model_key` `gemma4-e2b`; slug `Gemma-4-E2B-it-Q4_K_M`) on BFCL v4 `multi_turn_base` with `--fc-mode iffc-stop-repeat` ([PR #54](https://github.com/sskarz/Thesis/pull/54), commit `5b49d7a`). Same IFFC JSON-in-text channel as Run 5 (`tools` / `tool_choice` omitted); the harness force-stops a turn when the proposed IFFC call set is a subset of already-executed calls (or consecutive identical call sets after an exec observation). Scoring uses the same BFCL `multi_turn_checker`. **No retrain. Not BoN. Not SFT v7/v8. Not smoke81.** Distinct `runs` row (`run_id=27`); does **not** overwrite Run 5 IFFC or any SFT row.

**Framing:** stop-on-repeat passed a 10-ID smoke gate on **collapsed smoke81** (avg `api_calls` 21.3→6.3; `hit_step_limit` 4→0) but on **healthy base Gemma** did **not** improve `multi_turn_base` versus plain IFFC. Teacher-side anti-loop filters on v8/v8.1 are exhausted as a story; this harness lever is **not a free win** on base. **Thursday lock:** a tradeoff for **loop-dominated cases only** — **not** a method improvement. Best remains BoN **36.5%** / IFFC Run 5 **30.5%**. **Do not present 27.0% as a method win.** Accuracy row is authoritative; do **not** invent tok/s (metrics may still be collecting).

#### Checkpoint identity

| Field | Value |
|-------|-------|
| `run_id` | 27 (Mac `benchmark.db`, `scores.id=20`, verified) |
| `model_display` | Gemma-4-E2B-it (Q4_K_M) **BASE** |
| `model_key` | `gemma4-e2b` |
| Slug | `Gemma-4-E2B-it-Q4_K_M` |
| Mode | `--fc-mode iffc-stop-repeat`, `--categories multi_turn_base` |
| Hardware | Mac Metal llama-server, `-np 4 -c 65536`, `temp=0` |
| Started | `2026-09-28T04:32:18+00:00` (overnight) |
| Harness | [PR #54](https://github.com/sskarz/Thesis/pull/54), commit `5b49d7a` |
| Entries | 200 |
| Responses | **200/200 complete** |
| Rows | 146 `eval_errors` (failed entries = 146 → 54 correct) |
| Stop flags | **37/200**; all 37 among the 146 failures (**0 stopped and correct**) |

Result: **54/200 = 27.0%** (accuracy 0.27; `scores.id=20`). Versus Run 5 IFFC: **−3.5 pp / −7 entries**. Versus BoN (Run 11): **−9.5 pp / −19 entries**. Versus NFC (Run 4): **same accuracy level** (also **54/200 = 27.0%**) — **not** an identical method and **not** claimed as the same 54 entries. **Do not present 27.0% as a method win.**

#### Same-slice baselines

| Method | Run | Score |
|--------|-----|-------|
| xLAM-1b-fc-r NFC (external; format-channel collapse) | 19 | 0/200 = 0.0% |
| xLAM-1b-fc-r `--fc-mode xlam` (external; fair channel) | 20 | 6/200 = 3.0% |
| CD | 8 | 21/200 = 10.5% |
| SFT-IFFC-v5 LoRA best@1800 | 17 | 23/200 = 11.5% |
| SFT-IFFC-v6 / SFT-v4+STATE | 18 / 16 | 26/200 = 13.0% |
| SFT-IFFC-v4 LoRA best@1600 | 15 | 30/200 = 15.0% |
| Gemma NFC | 4 | 54/200 = 27.0% |
| **IFFC-stop-repeat (base Gemma)** | **27** | **54/200 = 27.0%** (same accuracy level as NFC; **not** the same method) |
| IFFC (base) / GST | 5 / 9 | 61/200 = 30.5% |
| IFFC+STATE (base Gemma) | 10 | 68/200 = 34.0% |
| BoN N=5 | 11 | 73/200 = 36.5% |

Inference-time BoN **36.5%** remains the best measured result **on base Gemma**. v4 IFFC **15.0%** remains the best **SFT** recovery. Full LoRA on failed v8 mixes is **not started**.

#### Failure taxonomy (`eval_errors` run_id=27, n=146)

Top buckets (verified). State mismatch + empty/under-call dominate — **not** primarily unconstrained decode loops.

| Bucket | Count |
|--------|-------|
| VehicleControlAPI state mismatch | **33** (Run 5: **22** → **33**) |
| GorillaFileSystem state mismatch | 19 |
| empty response list turn1 | 18 |
| missing GT exec turn0 | 17 |
| empty response list other turn | 14 |
| TravelAPI state mismatch | 12 |
| missing GT exec other | 8 |
| empty response list turn0 | 8 |
| Other API state mismatch | 7 |
| missing GT exec turn1 | 6 |
| TicketAPI state mismatch | 4 |

Aggregated (sums of the verified buckets above; 146/146): **state mismatch 75** (33+19+12+7+4); **empty response list 40** (18+14+8); **missing GT exec / under-call 31** (17+8+6). VehicleControlAPI state mismatch **rose vs Run 5 IFFC (22→33)**.

Stop flags: **37/200** entries. All 37 are among the 146 failures — **0 stopped and correct**. Force-stop did not salvage a passing trajectory on this slice.

Example entry_ids: empty turn1 `multi_turn_base_121` / `_125` / `_126`; VehicleControl state `multi_turn_base_51` / `_52` / `_53`.

#### Takeaways

1. **Harness stop-on-repeat is not a method improvement on healthy base Gemma.** 27.0% sits **below** Run 5 IFFC 30.5% (−3.5 pp) and **below** BoN 36.5% (−9.5 pp). Run 4 NFC is also **54/200 = 27.0%** — **same accuracy level, not an identical method.** **Thursday lock:** tradeoff for **loop-dominated cases only**.

2. **Residual is still agentic, and VehicleControl got worse vs IFFC.** State mismatch (VehicleControl **33**, up from Run 5 **22**; GorillaFileSystem 19; Travel 12) plus empty/under-call (empty turn1 18; missing GT exec turn0 17; empty other 14). Stop flags on **37/200**, all failures (**0 stopped and correct**). Not primarily unconstrained decode loops — the smoke81 over-call pathology was not the base-Gemma bottleneck.

3. **Not SFT.** This is BASE inference-time harness (`gemma4-e2b`). Do **not** confuse with smoke81 / v7 / v8 LoRA. Full LoRA on failed v8 mixes is **not started**.

4. **Best numbers unchanged.** Best inference remains BoN **36.5%** (base Gemma). Best IFFC remains Run 5 **30.5%**. Best SFT remains v4 IFFC **15.0%**.

#### Anti-recommendations

Do **not**:

- present 54/200 = 27.0% as a method win over IFFC, NFC, STATE, or BoN
- equate this run with smoke81 / SFT v7 / SFT v8 / BoN
- invent tok/s or wall-clock (accuracy row is authoritative)
- overwrite Run 5 IFFC or this `runs` row
- treat harness stop-on-repeat as a method improvement or the next default on base Gemma (Thursday lock: **loop-dominated cases only**)
- unlock GRPO or start v8 full LoRA from this result

**Paper / method gap.** A decode/harness force-stop that cut collapsed-smoke81 over-call (10-ID probe) **regressed** the healthy IFFC channel on base Gemma (−3.5 pp). Stop flags fired on 37/200, all among the 146 failures (**0 stopped and correct**). VehicleControlAPI state mismatch rose vs Run 5 (**22→33**). Remaining `multi_turn_base` headroom on this stack is still VehicleControl / GorillaFileSystem / Travel state fidelity and empty/under-call, not another unconstrained-loop gate on base. **Thursday lock:** tradeoff for **loop-dominated cases only**; **not** a method improvement. Best remains BoN **36.5%** / IFFC Run 5 **30.5%**.

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

Full `multi_turn_base` (200 entries): NFC **54/200 = 27.0%**, IFFC **61/200 = 30.5%**, CD **21/200 = 10.5%** (Run 8, scored negative), GST+NFC **61/200 = 30.5%** (Run 9, ties IFFC), IFFC+STATE **68/200 = 34.0%** (Run 10, first method above 30.5% **on base Gemma**), BoN N=5 on IFFC+STATE **73/200 = 36.5%** (Run 11, beats STATE by +2.5 pp at ~7× wall), SFT-IFFC LoRA best@1000 **1/200 = 0.5%** (Run 12, scored negative; mix stop-after-tools, not parser), SFT-IFFC-v2 filtered-[] LoRA best@2000 **1/200 = 0.5%** (Run 13, same collapse; filter did not recover), SFT-IFFC-v2 + strip trailing `[]` **4/200 = 2.0%** (Run 14, **DIAGNOSTIC** history bookkeeping — **not** a method win; did not rescue the never-start collapse), SFT-IFFC-v4 LoRA best@1600 **30/200 = 15.0%** (Run 15, **scored recovery** from that collapse, still **below IFFC 30.5% and below NFC 27.0%** — **do not present as beating IFFC**), SFT-IFFC-v4 + IFFC+STATE **26/200 = 13.0%** (Run 16, **scored negative/null overlay** vs Run 15 — **do not propose BoN on this ckpt**), SFT-IFFC-v5 LoRA best@1800 **23/200 = 11.5%** (Run 17, **scored regression vs v4** — **do not present as a method win**), SFT-IFFC-v6 LoRA best@1800 **26/200 = 13.0%** (Run 18, **scored miss vs v4 15.0%** — **do not present as a method win**; **no BoN / no more epochs on v6**), xLAM-1b-fc-r NFC **0/200 = 0.0%** (Run 19, **scored format-channel collapse** on this harness — **not** published overall BFCL ~78.94%; **not** a method win), xLAM-1b-fc-r `--fc-mode xlam` **6/200 = 3.0%** (Run 20, **format-fixed fair-channel** after [PR #42](https://github.com/sskarz/Thesis/pull/42) — tools actually ran; still **very low**; **not** a method win; **not** published ~78.94%; still **below CD 10.5%**). Those remain the primary thesis numbers. The 13-ID table is an apples-to-apples format comparison on GorillaFileSystem-only entries.

#### Interpretation

Shell v1 did **not** beat NFC on FS-only (1/13 vs 3/13); it tied IFFC on this slice (both 1/13, shared pass `multi_turn_base_12`). Shell v2 is worse still (**0/13 = 0.0%**). Residual failures are state mismatch, empty turns, and under-execution — not a format-only gap that more shell→GFS mapping would close. Shell-as-action on BFCL-FS is a **negative** ablation: do not pursue as D1 default. The toy VFS harness stays for non-BFCL / future SFT only. The primary thesis metric remains full `multi_turn_base`.

No other model's multi-turn scores are reported yet besides Gemma-4-E2B-it (Runs 4–18) and the xLAM-1b-fc-r external baselines (Run 19 NFC 0.0% format-channel collapse; Run 20 `--fc-mode xlam` 3.0% fair-channel — **not** published overall BFCL).

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

- **MLX SFT (v1/v2 collapse; v4 scored recovery; v4+STATE scored negative/null; v5 scored regression; v6 scored miss vs v4):** Frozen mix was **all** Team-ACE/ToolACE (~11.3k multi-turn) + stratified **25k** Salesforce/xlam-function-calling-60k (oversample parallel / multi-answer), IFFC JSON-in-text only (same `--fc-mode iffc` shape; no native `tool_calls`). Splits by conversation id, seed=42, stratify on source / is_multi_turn / n_tools_in_gold (1 vs ≥2), ~90/5/5 → ~32.7k / 1.8k / 1.8k train/val/corpus_test. Thesis test remains BFCL v4 `multi_turn_base` — never in those files. Base: Gemma-4-E2B-it. **Run 12** (`gemma4-e2b-sft-iffc`, fused Q4_K_M best val @ iter 1000): **1/200 = 0.5%** (only `multi_turn_base_50`; 199 errors; zero `iffc_unparseable`). Primary buckets: empty_turn **101**, state_mismatch **63**, under_call **35**. Audit: empty turns 13.2% (IFFC) → **69.9%**; early-stop 56/200 → **158/200**. Root cause: 5520 ToolACE assistant `[]` targets (32.5% of ToolACE assistant turns; xLAM 0%); after `"Function execution results"`, **78.1%** of targets are `[]` (2121 empty vs 596 continue) — teaches stop-after-tools. **Run 13** (`gemma4-e2b-sft-iffc-v2`, LoRA best@2000 `val=0.211`, mlx-lm 0.32): [PR #24](https://github.com/sskarz/Thesis/pull/24) dropped 334 mid-trajectory empty-after-exec rows (non-terminal 334→0) and still kept 552 terminal empty-after-exec. Score **1/200 = 0.5%** — same collapse as Run 12; only win `multi_turn_base_50` (correct-set overlap identical); 200 responses, 199 `eval_errors`, 200 timings; ~14 min wall; all flags empty (zero `iffc_unparseable`). Primary buckets: empty_turn **167**, state_mismatch **25**, under_call **7**. Audit (do not conflate): empty turns still ~70% (522/734 error rows); early-stop 158→**67** (down) but empty-labeled primary fails 101→**167** — more empty turns overall, not a win. Filter did **not** recover accuracy. Audit probes (not primary buckets): bundled-ask short probe that returned `view_messages_sent` is **not** BFCL IFFC shape — true shape returns `[]` on turn 2 for `multi_turn_base_101`; once SFT emits `[]`, later turns stay empty; pattern shift Run 5 `CCC` → Run 12 `CE`/`CEE` → Run 13 `EE`/`EEEE` (`first_empty` idx 0 **40→153**); not tool-catalog size (~27–30). Figures: `benchmark/results/Gemma-4-E2B-it-SFT-IFFC-v2-Q4_K_M/figures`. **Run 14** (same `gemma4-e2b-sft-iffc-v2` GGUF, `--fc-mode iffc --iffc-strip-empty-assistant`): **DIAGNOSTIC** history-bookkeeping ablation — **4/200 = 2.0%** (correct=4; wins `_50`, `_161`, `_192`, `_193`; shared with Run 13: only `_50`). 200 responses, 196 `eval_errors`, 200 timings; ~10 min wall; all flags empty (zero `iffc_unparseable`). Primary buckets: empty_turn **160**, state_mismatch **28**, under_call **8**. Empty-turn-0 error rows **153** (unchanged vs Run 13 `first_empty` idx 0); any-empty failed entries ~160. Stripping trailing assistant `[]` from history did **not** rescue the never-start collapse. **Do not present 2% as a method win.** Figures: `benchmark/results/Gemma-4-E2B-it-SFT-IFFC-v2-Q4_K_M/iffc-strip-empty-assistant/figures`. **Run 15** (`gemma4-e2b-sft-iffc-v4`, SFT v4 LoRA best@1600 `val=0.196`, fused `sft/fused/gemma-4-e2b-it-iffc-v4-best1600-Q4_K_M.gguf`, `--fc-mode iffc`, Mac `run_id=15`): **30/200 = 15.0%** — **scored recovery** from the 0.5–2% collapse, still **−12.0 pp / −24 vs NFC** and **−15.5 pp / −31 vs IFFC**. **Do not present 15% as beating IFFC.** 200 responses, 170 `eval_errors`, 200 timings; started `2026-09-21T16:52:59Z`; **`finished_at` NULL**, **`inference_metrics=0`** (do not invent tok/s). Correct set 30; overlap vs IFFC: both 23, only15 7 (`_64,_149,_154,_156,_159,_163,_166`), only5 38. Primary buckets: empty_turn **2** (1.2%; residual `_43` turn2, `_199` turn4), state_mismatch **109** (64%; VehicleControl 27 / Travel 26 / TradingBot 20 / GFS 19; example `_51`), under_call **59** (35%; turn0=35). `hit_step_limit:turn_0` on **57** timing rows. Empty-`[]` pathology is largely fixed; residual is state/under-call. **Run 16** (same `gemma4-e2b-sft-iffc-v4` GGUF, `--fc-mode iffc-state`, Mac `run_id=16`): **26/200 = 13.0%** — **scored negative/null overlay** vs Run 15 (−4 / −2.0 pp). Started `2026-09-21T18:39:35Z`, finished `2026-09-21T20:52:58Z` (~133.4 min wall). 200 responses, 174 `eval_errors`, 200 timings; **`inference_metrics` all zeros** (do not invent tok/s). Correct set 26; overlap vs Run 15: both 17, only16 9 (`_42,_47,_102,_104,_140,_158,_169,_179,_192`), lost 13 (`_52,_78,_107,_116,_133,_135,_149,_156,_159,_166,_182,_184,_190`). Primary buckets: empty_turn **3** (2%; `_175` turn0, `_185` turn0, `_108` turn4), state_mismatch **112** (64%; VehicleControl 25 / GFS 23 / Travel 21 / TradingBot 20 / Twitter 10 / Message 9 / Ticket 4), under_call **59** (34%; turn0=31). Sum `predicted_ms` ≈ 15.9 gen-min; avg turns 3.67; avg `api_calls` 14.83. STATE helped base Gemma (+3.5 pp) and **hurt / did not help SFT-v4**. **Run 17** (`gemma4-e2b-sft-iffc-v5`, SFT v5 LoRA best@1800 `val=0.147`, fused `sft/fused/gemma-4-e2b-it-iffc-v5-best1800-Q4_K_M.gguf`, `--fc-mode iffc`, Mac `run_id=17`, [PR #35](https://github.com/sskarz/Thesis/pull/35) mix, ~29850 train rows): **23/200 = 11.5%** — **scored regression vs v4** (−7 / −3.5 pp vs Run 15; also −3 / −1.5 pp vs Run 16). Started `2026-09-21T23:00:31Z`; **`finished_at` NULL**, **`inference_metrics=0`** (do not invent tok/s). 200 responses, 177 `eval_errors`, 200 timings. Correct set 23; overlap vs Run 15: both 18, only17 5 (`_12,_21,_42,_161,_169`), lost 12 (`_105,_106,_114,_116,_129,_135,_136,_142,_156,_159,_190,_191`). Primary buckets: empty_turn **0** (0%), state_mismatch **133** (~75%; Travel 32 / TradingBot 30 / VehicleControl 28 / GFS 23 / Twitter 8 / Message 8 / Ticket 4), under_call **44** (~25%; turn0=27). Sum `predicted_ms` ≈ 51.9 gen-min; avg turns 3.67; avg `api_calls` 40.27. **Do not present 11.5% as a method win.** **No BoN on this ckpt; no more epochs on the v5 mix.** **Run 18** (`gemma4-e2b-sft-iffc-v6`, SFT v6 LoRA best@1800 `val=0.142`, fused `sft/fused/gemma-4-e2b-it-iffc-v6-best1800-Q4_K_M.gguf`, `--fc-mode iffc`, Mac `run_id=18`, [PR #38](https://github.com/sskarz/Thesis/pull/38) mix, model key [PR #39](https://github.com/sskarz/Thesis/pull/39)): **26/200 = 13.0%** — **scored miss vs v4** (−4 / −2.0 pp vs Run 15; +3 / +1.5 pp vs Run 17; ties Run 16). Started `2026-09-22T12:50:47Z`, finished `2026-09-22T14:01:37Z` (~70.8 min wall). 200 responses, 174 `eval_errors`, 200 timings; `inference_metrics` present (gen_tps_mean ≈ 103.1; prompt_tps_mean ≈ 741.4; peak_rss ≈ 9170 MB). Correct set 26; overlap vs Run 15: both 18, only18 8 (`_42,_47,_102,_144,_161,_169,_186,_192`), lost 12 (`_52,_64,_78,_105,_106,_107,_109,_114,_129,_136,_159,_166`). Overlap vs Run 17: both 15, only18 11, only17 8. Primary buckets: empty_turn **1** (`_43` turn 2), state_mismatch **118** (~68%; VehicleControl 31 / TradingBot 26 / GFS 22 / Travel 19 / Twitter 10 / Ticket 5 / Message 5), under_call **55** (~32%; turn0=38). Avg turns 3.67; avg `api_calls` **13.34** (vs Run 15 17.3, Run 17 40.3). **Do not present 13.0% as a method win.** **No BoN on this ckpt; no more epochs on the v6 mix.** Do **not** more epochs on v1/v2, BoN/STATE on the collapsed checkpoints, claim the v2 `[]` filter fixed SFT, present Run 14 as a method win, present Run 15 as beating IFFC, present Run 16 as a method win, present Run 17 as a method win, present Run 18 as a method win, or **propose BoN as the next win on v4 or v5 or v6**. Recipe v5 is **measured regression**. Recipe v6 is **measured miss vs v4**. v4 IFFC **15.0%** remains the best SFT recovery. BoN **36.5%** remains the best inference-time method **on the base Gemma stack**. **Run 19** (`xlam-1b`, published Salesforce xLAM-1b-fc-r Q4_K_M, `--fc-mode nfc`, Mac `run_id=19`, registry [PR #41](https://github.com/sskarz/Thesis/pull/41)): **0/200 = 0.0%** — **scored format-channel collapse** on this NFC harness (all 200 `Model response list is empty for turn 0`; 196/200 all-empty `[[],…]`; 4/200 `_33,_35,_36,_37` have `["ERROR: "]` cells but still turn-0 empty). Wall **13.55 min** (`2026-09-22T14:13:09Z` → `14:26:42Z` ≈ 10:26 AM ET). Avg `api_calls` **3.63**; gen_tps_mean ≈ **196.7** — **not** a dead server. **Do not** claim xLAM scores 0 on BFCL overall; **do not** equate to published ~78.94%; **do not** present as a method win or ability floor. **Run 20** (same GGUF, `--fc-mode xlam`, Mac `run_id=20`, adapter [PR #42](https://github.com/sskarz/Thesis/pull/42)): **6/200 = 3.0%** — **format-fixed fair-channel** after Run 19 NFC collapse. Wall **~13.28 h** (`2026-09-22T15:28:35Z` → `2026-09-23T04:45:30Z`). 194/200 eval errors; `state_mismatch` **128**; under_call/missing exec **46**; empty_turn **20**. Flags: `hit_step_limit` **171/200**; `xlam_unparseable` **111/200**; `context_overflow` **42/200**. Avg `api_calls` **50.47** (min 2, max 101) — tools actually ran (vs Run 19 3.63). gen_tps_mean ≈ **91.6** (n=50). Correct: `multi_turn_base_{50,119,133,139,144,161}`. **Do not** present 3.0% as a method win; **do not** equate to published ~78.94%; still **below CD 10.5%**. **Run 27** (`gemma4-e2b`, **BASE**, `--fc-mode iffc-stop-repeat`, Mac `run_id=27`, `scores.id=20`, harness [PR #54](https://github.com/sskarz/Thesis/pull/54) commit `5b49d7a`): **54/200 = 27.0%** — **200/200 responses** complete; **scored miss vs IFFC** (−3.5 pp vs Run 5 30.5%; −9.5 pp vs BoN 36.5%). Same accuracy level as Run 4 NFC 27.0% — **not** an identical method. Stop flags **37/200**, all among 146 failures (**0 stopped and correct**). VehicleControlAPI **22→33** vs Run 5. **Thursday lock:** tradeoff for **loop-dominated cases only**; **not** a method improvement. Best still BoN **36.5%** / IFFC Run 5 **30.5%**. **Not** smoke81 / v7 / v8 SFT / BoN. **Not** a method win. Full LoRA on failed v8 mixes is **not started**. Do **not** invent tok/s. Pipeline: `sft/` (see `sft/README.md`).
- **Multi-turn optimization**: Gemma-4-E2B-it on `multi_turn_base`: NFC Run 4 **54/200 = 27.0%**; IFFC Run 5 **61/200 = 30.5%** (+3.5 pp); TagDispatch-style CD Run 8 **21/200 = 10.5%** (179 errors); GST+NFC Run 9 **61/200 = 30.5%** (ties IFFC; +7 vs NFC); IFFC+STATE Run 10 **68/200 = 34.0%** (first method above the 30.5% plateau; +7 vs IFFC and vs GST); BoN N=5 on IFFC+STATE Run 11 **73/200 = 36.5%** (beats STATE by +5 / +2.5 pp at ~7× wall); SFT-IFFC LoRA best@1000 Run 12 **1/200 = 0.5%** (scored negative; 199 errors); SFT-IFFC-v2 filtered-[] LoRA best@2000 Run 13 **1/200 = 0.5%** (same collapse; filter did not recover); SFT-IFFC-v2 + strip trailing `[]` Run 14 **4/200 = 2.0%** (**DIAGNOSTIC** history bookkeeping — **not** a method win); SFT-IFFC-v4 LoRA best@1600 Run 15 **30/200 = 15.0%** (**scored recovery** from collapse, still **below IFFC 30.5% and below NFC 27.0%** — **do not present as beating IFFC**); SFT-IFFC-v4 + IFFC+STATE Run 16 **26/200 = 13.0%** (**scored negative/null overlay** vs Run 15 — **do not propose BoN on this ckpt**); SFT-IFFC-v5 LoRA best@1800 Run 17 **23/200 = 11.5%** (**scored regression vs v4** — **do not present as a method win**; **no BoN / no more epochs on this mix**); SFT-IFFC-v6 LoRA best@1800 Run 18 **26/200 = 13.0%** (**scored miss vs v4 15.0%** — **do not present as a method win**; **no BoN / no more epochs on v6**); xLAM-1b-fc-r NFC Run 19 **0/200 = 0.0%** (**scored format-channel collapse** on this harness — **not** published overall BFCL ~78.94%; **not** a method win); xLAM-1b-fc-r `--fc-mode xlam` Run 20 **6/200 = 3.0%** (**format-fixed fair-channel** — tools actually ran; still **very low**; **not** a method win; **not** published ~78.94%; still **below CD 10.5%**); IFFC-stop-repeat Run 27 **54/200 = 27.0%** (**BASE** Gemma harness lever, [PR #54](https://github.com/sskarz/Thesis/pull/54) — **scored miss vs IFFC 30.5%** (−3.5 pp / −7); **−9.5 pp vs BoN**; **200/200 responses**; stop flags **37/200**, all failures (**0 stopped and correct**); VehicleControlAPI **22→33** vs Run 5; same accuracy level as Run 4 NFC 27.0% — **not** an identical method; **Thursday lock:** tradeoff for **loop-dominated cases only**; **not** a method improvement; **not** smoke81 / v7 / v8 SFT / BoN; **not** a method win). CD is **measured and negative**. Guided templates are **measured**: they **tie IFFC and do not beat it** — the 61=61 is a 19-row swap, not the same 61 entries. GST vs NFC: empty 59→31 but state mismatch 68→79. IFFC+STATE is **measured** and **beats** the IFFC/GST plateau **on base Gemma**: empty 41→31 (tied GST’s empty cut); state mismatch 60→**64** (did not improve); under-call ≈38→37. The +7 vs IFFC is a 20-rescue / 13-regress swap, not pure dominance. BoN is **measured** and **beats STATE** **on base Gemma**: empty 31→**36** (rose); state 64→**53** (cut); under-call 37→38 (flat). The +5 vs STATE is a 17-rescue / 12-regress swap. Wall generate ~21.3 h vs Run 10 ~3.0 h; avg `predicted_ms` ~1318 s vs ~150 s (~8.8×). Modest +2.5 pp is **not free**. SFT-IFFC v1 is **measured and negative**: empty_turn 101 / state_mismatch 63 / under_call 35; audit empty turns 13.2%→69.9%, early-stop 56→158 of 200. SFT-IFFC-v2 is **measured and the same collapse**: empty_turn **167** / state_mismatch **25** / under_call **7**; audit empty turns still ~70% (522/734 error rows); early-stop 158→67 (down) but empty-labeled primary fails 101→167 — more empty turns overall, not a win. Run 14 strip-trailing-`[]` is **measured as a DIAGNOSTIC**, not a method: **4/200 = 2.0%**; empty_turn **160** / state_mismatch **28** / under_call **8**; empty-turn-0 **153** unchanged vs Run 13; did **not** rescue turn-0 refusal. Run 15 SFT-v4 is **measured as a scored recovery, not an IFFC win**: **30/200 = 15.0%**; empty_turn **2** / state_mismatch **109** / under_call **59**; overlap vs IFFC both 23 / only15 7 / only5 38; `hit_step_limit:turn_0` on 57 rows; `finished_at` NULL, `inference_metrics=0` (do not invent tok/s). Run 16 SFT-v4+STATE is **measured as a negative/null overlay, not a STATE win on this ckpt**: **26/200 = 13.0%**; empty_turn **3** / state_mismatch **112** / under_call **59**; overlap vs Run 15 both 17 / only16 9 / lost 13; ~133.4 min wall; `inference_metrics` all zeros (do not invent tok/s). Run 17 SFT-v5 is **measured as a scored regression vs v4, not a method win**: **23/200 = 11.5%**; empty_turn **0** / state_mismatch **133** / under_call **44**; overlap vs Run 15 both 18 / only17 5 / lost 12; `finished_at` NULL, `inference_metrics=0` (do not invent tok/s). Run 18 SFT-v6 is **measured as a scored miss vs v4, not a method win**: **26/200 = 13.0%**; empty_turn **1** / state_mismatch **118** / under_call **55**; overlap vs Run 15 both 18 / only18 8 / lost 12; wall ~70.8 min; avg `api_calls` 13.34; gen_tps_mean ≈ 103.1. Honesty: `--fc-mode iffc-state-bon` is MV@N on the IFFC+STATE channel — **not** NFC, **not** GST, **not** CD/GBNF, **not** a verifier/reward-model BoN; `--fc-mode iffc-state` is IFFC JSON-in-text plus harness-written STATE from executed tool results — **not** GST, **not** CD/GBNF, **not** LLM summarization; `--fc-mode gst` is prompt-only on the NFC channel (not GBNF/IFFC); `--fc-mode cd` is a two-pass approximation on stock llama-server — **not** full XGrammar-2; Run 12 SFT is weight change on the IFFC channel — **not** a parser fix and **not** IFFC+STATE/BoN on the SFT GGUF; Run 13 SFT-v2 is the same IFFC channel after dropping 334 non-terminal empty-after-exec `[]` (552 terminal kept) — **not** a recovery and **not** IFFC+STATE/BoN on the v2 GGUF; `--iffc-strip-empty-assistant` (Run 14) is history bookkeeping on that same v2 GGUF — **DIAGNOSTIC**, **not** a training method, **not** a 2% method win; Run 15 SFT-v4 is the v4 unroll (drop first-assistant-empty + clarifying `[]` on top of v3) on a new GGUF — **scored recovery**, **not** an IFFC win; Run 16 is `--fc-mode iffc-state` on that same v4 GGUF — **scored negative/null**, **not** a BoN-on-this-ckpt next win; Run 17 SFT-v5 is the v5 unroll (turn-0 completeness + env-state consistency on top of v4) on a new GGUF — **scored regression vs v4**, **not** a method win; Run 18 SFT-v6 is the v6 unroll (v4 empty/recovery + only `drop_state_inconsistent`) on a new GGUF — **scored miss vs v4**, **not** a method win; Run 19 xLAM-1b NFC is the published Salesforce GGUF on the stock NFC harness — **scored format-channel collapse**, **not** published overall BFCL; Run 20 `--fc-mode xlam` is the **format-fixed fair-channel** score on the same GGUF — **6/200 = 3.0%**, still **very low**, **not** a method win, **not** published ~78.94%; Run 27 `--fc-mode iffc-stop-repeat` is IFFC JSON-in-text plus a harness force-stop on **base** Gemma ([PR #54](https://github.com/sskarz/Thesis/pull/54)) — **scored miss vs IFFC**, **not** a method improvement, **not** smoke81/v7/v8; **Thursday lock:** **loop-dominated cases only**; best still BoN **36.5%** / IFFC Run 5 **30.5%**. Do **not** pursue this CD condition, another GST Detail overlay on NFC, unbounded N, more epochs on the v1 mix **or** the v2 filtered mix **or** the v5 mix **or** the v6 mix, BoN/STATE on the Run 12 or Run 13 checkpoint, fuse `final@2000` as D1 default, claim the `[]` filter fixed SFT, present Run 14’s 2.0% as a method win, present Run 15’s 15.0% as beating IFFC/NFC, present Run 16’s 13.0% as a method win, present Run 17’s 11.5% as a method win, present Run 18’s 13.0% as a method win, or **BoN the SFT-v4 or SFT-v5 or SFT-v6 checkpoint as the next experiment**. Residual errors on the inference-time channel are still agentic state mismatch (VehicleControl 22 / GorillaFileSystem 13 / Travel 9) and under-calling, not format parsing. On SFT-v4 the residual is the same agentic pair after empty-`[]` was largely fixed (Run 15: state 109 / under-call 59; Run 16: state 112 / under-call 59). On SFT-v5 empty is fully suppressed (0) while state mismatch **rose** (133) and under-call fell (44). On SFT-v6 empty is still essentially suppressed (1) while state mismatch is 118 and under-call 55; avg api_calls 13.34 (fixed v5’s 40.3 churn). Next leverage on the inference-time channel: FS/Vehicle state fidelity or under-call recovery — not another empty-only prompt trick, not more test-time N without a cost budget, and **not** treating harness stop-on-repeat as a method improvement on base (Run 27 **54/200 = 27.0%**, miss vs IFFC; Thursday lock: **loop-dominated cases only**). **Next SFT lever:** SFT **v7** (planned, **not scored**) — APIGen-MT verified multi-turn trajs + calibrated stop/clarify + student-aware filter. Optional open data-gen complement: Data Turnstile (METHODS §24; **unscored**). Optional post-healthy-SFT local RL: Critical-State RL (METHODS §25; **unscored**; not blanket GRPO). **SFT locked as main path** (2026-09-23); inference wrappers secondary. Vault: [`sskarz/thesis-vault` `📝 Notes/2026-09-23-SFT-training-lit-review.md`](https://github.com/sskarz/thesis-vault/blob/main/%F0%9F%93%9D%20Notes/2026-09-23-SFT-training-lit-review.md). Recipe v5 is **measured regression**; recipe v6 is **measured miss vs v4**; not another thin filter on the same family; not BoN on v5/v6; not more epochs on those mixes. v4 IFFC 15.0% remains best SFT recovery. BoN 36.5% still best **on base Gemma**. BFCL-FS / VFS shell→GorillaFileSystem fs-only is a **negative ablation** (v1 **1/13 = 7.7%**, v2 **0/13 = 0.0%** vs NFC **3/13 = 23.1%** vs IFFC **1/13 = 7.7%** on the same 13 GorillaFileSystem-only IDs; not the full 200). Do **not** pursue shell-as-action as D1 default. The toy VFS harness stays for non-BFCL / future SFT only. Dual-model IFFC (paper Algorithm 1 QA/SRM decoupling) is deferred. `miss_func` / `miss_param` / `long_context` remain secondary stress tests. Run 11 `finished_at` was NULL at verify time (`inference_metrics=0`, `metrics_per_request=0`); eval scored. Run 12 finished `2026-09-20T12:33:00Z`. Run 13 finished ~14 min wall. Run 14 finished ~10 min wall. Run 15 started `2026-09-21T16:52:59Z`; `finished_at` still NULL at verify (`inference_metrics=0`). Run 16 started `2026-09-21T18:39:35Z`, finished `2026-09-21T20:52:58Z`; `inference_metrics` all zeros. Run 17 started `2026-09-21T23:00:31Z`; `finished_at` still NULL at verify (`inference_metrics=0`). Run 18 started `2026-09-22T12:50:47Z`, finished `2026-09-22T14:01:37Z` (~70.8 min wall); inference_metrics present (gen_tps_mean ≈ 103.1). Run 19 started `2026-09-22T14:13:09Z`, finished `2026-09-22T14:26:42Z` (~13.55 min wall); inference_metrics present (gen_tps_mean ≈ 196.7); **all 200** `Model response list is empty for turn 0`. Run 20 started `2026-09-22T15:28:35Z`, finished `2026-09-23T04:45:30Z` (~13.28 h wall); inference_metrics present (gen_tps_mean ≈ 91.6); 194/200 eval errors; avg `api_calls` **50.47**. Run 27 started `2026-09-28T04:32:18+00:00`; Mac `scores.id=20` **54/200 = 27.0%**; **200/200 responses** complete; 146 `eval_errors`; stop flags **37/200** (all failures; **0 stopped and correct**); VehicleControlAPI **22→33** vs Run 5; accuracy row authoritative (do **not** invent tok/s).
- **NVIDIA Jetson Orin Nano Super migration**: Repeat all benchmarks on CUDA with 8 GB shared memory to measure real edge-device performance.
- **RAG augmentation for large tool sets**: The `multiple` category only presents 2–4 tools per test case, so RAG-based tool filtering provides no benefit. However, real-world agentic applications may expose 50–100+ tools. Test whether retrieval-augmented generation (embedding-based tool retrieval to pre-filter the tool set before prompting) improves accuracy and reduces prompt length for large tool registries. Complementary (unscored) knob: MCP-GRANITE (METHODS §3a) — redesign how the same capability set is split into tools, don't only retrieve.
- **Additional models and quantizations**: Expand the model matrix (e.g., Phi-3.5-mini, Gemma-2-2B) and test alternative quantization levels (Q5_K_M, Q8_0) to map the accuracy-size Pareto frontier. xLAM-1b-fc-r is **already scored** on this harness: Run 19 NFC **0/200 = 0.0%** (format-channel collapse) and Run 20 `--fc-mode xlam` **6/200 = 3.0%** (fair-channel; still **very low**; **not** a method win; **not** published overall BFCL ~78.94%). Do **not** overwrite either run.
- **Re-run parallel benchmarks**: Now that the grammar-constraining bypass is in place, re-run `parallel` and `parallel_multiple` categories across all models to measure true SLM parallel tool-calling accuracy.
