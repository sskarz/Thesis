# Methods for Improving SLM Tool Calling

A survey of techniques to improve small language model (SLM, <5B parameter) performance on agentic multi-step tool calling — specifically on the BFCL v4 benchmark. Part I covers inference-time methods (prompt engineering, structured decoding, retrieval augmentation, tool-interface design, context management, compute scaling). Part II covers training-time methods (supervised fine-tuning, reinforcement learning, knowledge distillation, synthetic data generation).

# Part I: Inference-Time Methods

Each method in this section operates at inference time without modifying model weights.

## Table of Contents

- [1. Guided-Structured Templates](#1-guided-structured-templates)
- [2. Constrained Decoding and Grammar-Guided Generation](#2-constrained-decoding-and-grammar-guided-generation)
- [27. PTC-Decoder: Plan-to-Act plus Tool-Name DFA](#27-ptc-decoder-plan-to-act-plus-tool-name-dfa)
- [3. RAG-Based Tool Retrieval](#3-rag-based-tool-retrieval) (literature only; **out of scope** for this thesis paper)
- [3a. Tool-Interface Granularity (MCP-GRANITE)](#3a-tool-interface-granularity-mcp-granite)
- [4. Inference-Time Compute Scaling](#4-inference-time-compute-scaling)
- [28. Looped Language Models for Compositional Tool Calling](#28-looped-language-models-for-compositional-tool-calling)
- [5. Context Compression for Multi-Turn](#5-context-compression-for-multi-turn)
- [6. ReAct and ReWOO Planning Patterns](#6-react-and-rewoo-planning-patterns)
- [7. Prompt-Only Tool Calling](#7-prompt-only-tool-calling)
- [8. Few-Shot Example Retrieval](#8-few-shot-example-retrieval)
- [9. Verification and Self-Correction Loops](#9-verification-and-self-correction-loops)
- [10. DSPy Automatic Prompt Optimization](#10-dspy-automatic-prompt-optimization)
- [11. Parallel Function Calling via SimpleTool](#11-parallel-function-calling-via-simpletool)
- [Key Finding: llama.cpp Already Uses Grammar-Constrained Decoding](#key-finding-llamacpp-already-uses-grammar-constrained-decoding)
- [Experiment: TagDispatch-style CD (`--fc-mode cd`)](#experiment-tagdispatch-style-cd---fc-mode-cd)
- [Experiment: Guided-Structured Templates (`--fc-mode gst`)](#experiment-guided-structured-templates---fc-mode-gst)
- [Experiment: IFFC + harness STATE (`--fc-mode iffc-state`)](#experiment-iffc--harness-state---fc-mode-iffc-state)
- [Experiment: IFFC stop-on-repeat (`--fc-mode iffc-stop-repeat`)](#experiment-iffc-stop-on-repeat---fc-mode-iffc-stop-repeat) (Run 27 ungated miss 27.0%; Run 30 tighten **32.5%** scored win vs IFFC on BASE; **do not overwrite**; Run 42 OODA-card is now higher Acc)
- [Experiment: IFFC OODA card (`--fc-mode iffc-ooda-card`)](#experiment-iffc-ooda-card---fc-mode-iffc-ooda-card) (Run 42 **71/200 = 35.5%**; +3.0 pp vs Run 30; `state_mismatch` gate **MISS**; still below BoN 36.5%)
- [Experiment: IFFC PTC-A (`--fc-mode iffc-ptc-a`)](#experiment-iffc-ptc-a---fc-mode-iffc-ptc-a) (Run 46 **27/200 = 13.5%** scored FAIL / Acc regression; **bank and close**; PTC-B **HOLD**; next = DPO)
- [Experiment: IFFC Decision sidecar (`--fc-mode iffc-decision`)](#experiment-iffc-decision-sidecar---fc-mode-iffc-decision) (Run 58 **50/200 = 25.0%** scored FAIL_METHOD / no lift; park Decision Acc; Run 61 Decision+OODA **52/200 = 26.0%** scored FAIL vs R5–R42; park Decision(+OODA) Acc; Runs 56/57 both **0/10**; Runs 59/60 **1/10 vs 0/10** — **not** N=200; compose/hygiene smoke PASS)
- [Experiment: Best-of-N majority voting (`--fc-mode iffc-state-bon`)](#experiment-best-of-n-majority-voting---fc-mode-iffc-state-bon)
- [Experiment: MLX SFT v1 (Gemma-4-E2B, IFFC mix)](#experiment-mlx-sft-v1-gemma-4-e2b-iffc-mix)
- [SFT v9.2 smoke-gate FAIL (Run 36; HOLD / park)](#sft-v9--v91--v92--lever-2-action-quotas-10-id-smoke-gate-fail--hold) (plain-IFFC 10-ID; **not** a `multi_turn_base` %; quota remix **HOLD**)
- [DPO β0.2 smoke100 10-ID PASS (Run 49; health gate)](#dpo-β02-smoke100-10-id-pass-run-49--health-gate-not-a-bfcl-) (plain-IFFC 10-ID; **not** a BFCL %; full DPO **scored** Run 51 **53/200 = 26.5%** — **not** a method win)
- [Prioritized Recommendations](#prioritized-recommendations)
- [Post–Run27 consensus (2026-09-28)](#post-run27-consensus-2026-09-28) (Part II; Run 46 iffc-ptc-a **13.5%** scored FAIL / closed; Run 42 **35.5%** still best non-BoN; Run 30 **32.5%** untouched; SFT HOLD; DPO β0.2 smoke100 Run 49 **PASS**; DPO-full Run 51 **26.5%** scored / not a win; Run 61 Decision+OODA **26.0% FAIL**; Run 58 Decision **25.0% FAIL**; PR #57 stays draft, **not** more PTC/OODA remix)
- [References](#references)

---

## 1. Guided-Structured Templates

### Overview

A structured, template-based reasoning framework that guides the model through a fixed 7-step curriculum before generating any tool call, replacing free-form chain-of-thought with discrete reasoning stages.

### Mechanism

The template enforces the following steps in order:

1. Identify which function or set of functions best fit the user query
2. Decide on relevancy of provided functions
3. Examine relevant function documentation
4. Extract and validate parameters from the user query
5. Perform type conversions or resolve implicit values
6. Draft the function call(s)
7. Revalidate the composed function calls against the documentation

By embedding this template into the system prompt, the model follows a structured curriculum before producing tool calls, creating explicit intermediate reasoning chains that serve as interpretable, step-by-step justification for identifying, selecting, examining, and parameterizing functions.

### Evidence

Evaluated on BFCL v2 across both closed-source and open-source models:

| Model | With Template | With CoT | No Thought |
|-------|---------------|----------|------------|
| GPT-4o-FC | **80.26** | 78.83 | 74.79 |
| LLaMA-3-70B | **76.81** | 75.05 | 71.50 |
| LLaMA-3.1-8B | **69.65** | 66.70 | 65.75 |
| Mistral-7B | 64.12 | 63.50 | 61.89 |

Template-based prompting achieves +3–12% relative improvement over baselines without fine-tuning and consistently outperforms free-form chain-of-thought. When combined with fine-tuning, template training yields +2.8 on BFCL v2 and +1.7 on Nexus over CoT-trained models.

### Applicability

- **BFCL categories:** `multi_turn_base` first (Gemma-4-E2B-it GST+NFC Run 9 **61/200 = 30.5%**, ties IFFC; vs scored NFC 27.0% / IFFC 30.5% / CD 10.5%); also relevant to `multiple`, `irrelevance`, `parallel`
- **Implementation:** Multi-turn ablation is `--fc-mode gst` (GST reasoning on the NFC `tools`/`tool_choice` channel; see [Experiment: Guided-Structured Templates](#experiment-guided-structured-templates---fc-mode-gst)). Prompt-only; **not** GBNF/XGrammar. Do **not** treat this as CD.
- **Complexity:** Low
- **Limitation:** Increases generated tokens (terse 7-step block each step). Template must stay small; this harness does **not** dump function schemas into the GST system prompt (those stay on native `tools`). On Gemma `multi_turn_base`, GST **ties IFFC and does not beat it** (see `THESIS.md`).

### Source

He, Y., et al. "Improving Large Language Models Function Calling and Interpretability via Guided-Structured Templates." arXiv:2509.18076, 2025. https://arxiv.org/abs/2509.18076

---

## 2. Constrained Decoding and Grammar-Guided Generation

### Overview

Grammar-constrained decoding restricts the LLM's token sampling at each step so that only tokens valid under a formal grammar receive nonzero probability. This guarantees syntactically valid output (e.g., well-formed JSON matching a schema) without modifying model weights.

### Mechanism

llama.cpp implements this via GBNF (GGML BNF) grammars:

1. A grammar (either hand-written GBNF or auto-converted from JSON Schema) defines the set of valid output strings
2. At each decoding step, tokens that would violate the grammar are masked (set to -infinity logits)
3. The model can only produce tokens that advance the grammar to a valid state

llama.cpp's built-in JSON Schema to GBNF converter supports: `type` (string, integer, number, boolean, null, object, array), `properties`, `required`, `additionalProperties`, `enum`, `const`, `anyOf`/`oneOf`, `pattern`, `minimum`/`maximum`, `minLength`/`maxLength`, `minItems`/`maxItems`, and local `$ref`.

### Evidence

- Grammar-constrained sampling eliminates JSON parse failures entirely (100% valid JSON) with minimal impact on generation speed when token masks are pre-cached (Willard & Louf, 2023).
- However, constrained decoding can degrade accuracy, and **smaller models are 2–3x more sensitive** to format constraints than larger models (Tam et al., 2024). Key mitigations:
  - Include leading whitespace in grammars (5–10% improvement)
  - Prefer less-restrictive grammars
  - Test multiple format variants per model family
- For tool calling specifically (where the output format is well-defined and the model is trained for it), the effect is positive or neutral. Grammar constraints help small models more than large models because SLMs are more likely to produce malformed JSON, truncated output, or wrong field names without constraints.

### LLGuidance Alternative

llama.cpp optionally supports LLGuidance (`-DLLAMA_LLGUIDANCE=ON`), a Rust-based constrained decoding library from Microsoft Research:

- **Performance:** ~50μs average, 0.5ms p99 token mask computation on a 128K vocabulary
- **Better JSON Schema coverage:** Follows the spec more faithfully than native GBNF conversion
- **Lexer/parser split:** ~10x more efficient than pure GBNF because LLM tokens usually align with lexemes
- **Build requirement:** Needs the Rust compiler; enable with `-DLLAMA_LLGUIDANCE=ON`

### Applicability

- **BFCL categories:** All, but especially `parallel` and `parallel_multiple` (0% across all models) where a GBNF grammar enforcing an array-of-tool-calls structure could unlock scores
- **Implementation:** The multi-turn CD ablation is `--fc-mode cd` (TagDispatch-style two-pass on stock llama-server; see [Experiment: TagDispatch-style CD](#experiment-tagdispatch-style-cd---fc-mode-cd)). Do **not** treat NFC `tools`/`tool_choice` GBNF as that experiment. `grammar` and `json_schema` are the llama-server fields used; LLGuidance (`-DLLAMA_LLGUIDANCE=ON`) is optional and not required.
- **Complexity:** Medium (must keep CD payloads free of `tools` so native tools GBNF never activates)
- **Limitation:** Guarantees syntactic validity only, not semantic correctness. The model could produce perfectly formatted JSON with wrong function names or argument values — except that CD pass 1 *does* reject unknown names at the grammar level.

### Sources

- Willard, B. T. & Louf, R. "Efficient Guided Generation for Large Language Models." arXiv:2307.09702, 2023. https://arxiv.org/abs/2307.09702
- Tam, Z., et al. "Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models." arXiv:2408.02442, 2024. https://arxiv.org/abs/2408.02442
- Jain, N., et al. "Lost in Space: Optimizing Tokens for Grammar-Constrained Decoding." arXiv:2502.14969, 2025. https://arxiv.org/abs/2502.14969
- llama.cpp GBNF documentation: https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
- llama.cpp LLGuidance documentation: https://github.com/ggml-org/llama.cpp/blob/master/docs/llguidance.md
- **Related (paper still theirs; harness PTC-A scored FAIL):** [§27 PTC-Decoder](#27-ptc-decoder-plan-to-act-plus-tool-name-dfa) — training-free Plan-to-Act + tool-name DFA (args free). Paper numbers are theirs (not BFCL `multi_turn_base`). Our live-name DFA channel `--fc-mode iffc-ptc-a` is **scored** Run 46 **27/200 = 13.5%** — **FAIL / Acc regression**; **bank and close**. See [IFFC PTC-A](#experiment-iffc-ptc-a---fc-mode-iffc-ptc-a). Do **not** equate Yu et al. F1 0.359 to this harness.

---

## 27. PTC-Decoder: Plan-to-Act plus Tool-Name DFA

### Overview

PTC-Decoder (Yu, Mu, & Wu, 2026) is a **training-free** decoder for offline resource-constrained edge SLMs. It combines **Plan-to-Act** (plan as a forced first tool) with a **TC-Decoder** DFA that hard-constrains **tool names** only (arguments remain free). The paper evaluates edge SLMs including **Gemma-4-2B Q4** on Jetson Orin NX in an offline satellite-agent setting. llama.cpp GBNF / LLGuidance is the natural integration hypothesis for this repo.

**Train vs infer: inference-only** (plug-and-play; no retraining). Yu et al. paper numbers remain **theirs** (Jetson / satellite LLM-judge + F1) — do **not** equate +1.21 / 0.359 F1 to a thesis harness score. A **separate** live tool-name DFA harness (`--fc-mode iffc-ptc-a`; args free; [PR #63](https://github.com/sskarz/Thesis/pull/63) @ `3569311`) is **scored** Run 46 **27/200 = 13.5%** on BFCL v4 `multi_turn_base` — **FAIL / Acc regression**; **bank and close**. PTC-B **HOLD**. Next path = **DPO** (β0.2 smoke100 10-ID Run 49 **PASS** health gate — **not** a BFCL %; DPO-full **scored** Run 51 **53/200 = 26.5%** — **not** a method win), **not** more PTC remix. See [IFFC PTC-A](#experiment-iffc-ptc-a---fc-mode-iffc-ptc-a) and [DPO β0.2 smoke100 PASS](#dpo-β02-smoke100-10-id-pass-run-49--health-gate-not-a-bfcl-).

### Mechanism

1. **Plan-to-Act:** elevate planning to an atomic tool and force its invocation at the first inference step. The paper finds that prompt-only plan-then-execute is what their SLMs almost entirely disregard.
2. **TC-Decoder:** a deterministic finite automaton that imposes token-level hard constraints on **tool names** while leaving parameter generation unconstrained, so the model still chooses argument values.

Complements, does not replace, IFFC / stop-repeat. Closest peer to [§2 Constrained Decoding](#2-constrained-decoding-and-grammar-guided-generation) and `--fc-mode cd` name-GBNF.

### Evidence

Paper results (Yu, Mu, & Wu, 2026; cite as **theirs**, not ours):

- **7 SLMs × 200** remote-sensing tasks. Mean Overall LLM-judge **+1.21** (p<0.01).
- Gemma-4-2B(Q4) F1 baseline **0.279 → PTC 0.359**.
- Ablation: Plan without TC-Decoder is much weaker; **TC-Decoder is the primary driver**.
- Token overhead ~**2.07×**. R.A. remains the weakest dimension.
- Also ToolAlpaca / Seal-Tools / API-Bank F1 gains.

**Not** BFCL `multi_turn_base`. Do **not** equate +1.21 / 0.359 F1 to a thesis harness score.

### Applicability

- **Thesis role:** Logged 2026-09-28 as a paper method. Yu et al. paper remains **theirs** (not our execution checker). The live-name DFA harness `--fc-mode iffc-ptc-a` is **scored FAIL** (Run 46 **27/200 = 13.5%**) — **bank and close**. PTC-B **HOLD**. Do **not** treat PTC as the next Acc win.
- **BFCL categories:** `--fc-mode iffc-ptc-a` was measured on `multi_turn_base` only (Run 46). Parallel / other slices **untested**. Paper F1 ≠ this score.
- **Complexity:** Medium (DFA / GBNF over live tool names; plan-as-first-tool prompt).
- **Limitation:** Paper setting is offline satellite / Jetson Orin NX with LLM-judge overall + F1 — not our execution checker. Token overhead ~2×.

### Sources

- Yu, M., Mu, K., & Wu, G. "PTC-Decoder: Towards Intelligent SLMs on Offline Resource-Constrained Edge Devices." arXiv:2609.30836, 2026. https://arxiv.org/abs/2609.30836
- Code: https://github.com/yuminghui/llm-tool-constrained-decoder

---

## 3. RAG-Based Tool Retrieval

**Out of scope for this thesis paper (decision 2026-09-29).** RAG appeared in an earlier registered-title framing. This section is **literature only**. Do not implement, score, or roadmap RAG as a planned experiment before December 2026.

### Overview

When the available tool set is large, Retrieval-Augmented Generation (RAG) reduces prompt bloat by embedding tool descriptions into a vector index and retrieving only the top-k most relevant tools per query, rather than presenting all tools to the LLM at once.

### Mechanism (RAG-MCP)

The RAG-MCP framework uses a three-step pipeline:

1. **Offline indexing:** Embed each tool's description (name, docstring, parameter schema) using a lightweight LLM-based retriever (e.g., Qwen). Store embeddings in a vector index.
2. **Online retrieval:** At inference, embed the user query with the same model. Perform semantic search over the tool index, returning the top-k candidate tools.
3. **Prompt injection:** Only the retrieved tool descriptions are injected into the LLM's prompt, drastically reducing context length.

### Evidence

RAG-MCP results (Gao et al., 2025):

| Method | Tool Selection Accuracy | Avg. Prompt Tokens |
|--------|------------------------|--------------------|
| All tools (baseline) | 13.62% | 2,134 |
| Keyword matching | 18.20% | — |
| **RAG-MCP** | **43.13%** | **1,084** |

RAG-MCP triples tool selection accuracy and cuts prompt tokens by over 50%. Tested with Qwen-max-0125 as the base LLM and Qwen as the retriever.

### Prior Work

- **Gorilla** (Patil et al., 2023): Fine-tuned LLaMA-7B with a retriever (BM25 or GPT-Index) that fetches relevant API documentation at inference. Retrieval reduced hallucination (calling non-existent APIs) from 40–60% to <5%.
- **ToolBench** (Qin et al., 2024): Used sentence-BERT to embed 16,464 APIs from RapidAPI, retrieved top-5 per query. Retrieval-augmented ToolLLaMA outperformed ChatGPT when the API pool exceeded ~100 APIs. With pools <10, retrieval provided no benefit.
- **CRAFT** (Yuan et al., 2024): Contrastive learning for tool retrieval. Key finding: tool retrieval is most beneficial when the total tool set is large (>20 tools) AND the model is small (<7B parameters).
- **ToolkenGPT** (Hao et al., 2023): Represents each tool as a special "toolken" token with a learned embedding. Tool retrieval is based on embedding similarity. Scaled to 13,000+ tools. NeurIPS 2023 oral presentation. Note: the toolken embedding approach requires training, but the retrieval component (embedding-based tool filtering) is separable and usable without fine-tuning.

### Applicability

- **BFCL categories:** Limited benefit for single-turn categories (2–4 tools per test case). Potentially significant for multi-turn categories where all function docs for involved classes are loaded (dozens of tools), and for `multi_turn_long_context` where large tool doc sets can exceed the 8192-token context window.
- **Real-world deployment:** Essential. A production agent with 100+ tools cannot fit all descriptions in an SLM's context window.
- **Thesis contribution:** **None.** Out of scope for this thesis paper — not a planned experiment and not a deferred near-term claim.
- **Implementation:** Not planned. (Literature implementations use a small embedding model and a vector index; do not treat that as a thesis deliverable.)
- **Complexity:** Medium
- **Complement:** Retrieving a subset of tools is one knob. **How the same capability set is split into tools** is another — see [§3a Tool-Interface Granularity (MCP-GRANITE)](#3a-tool-interface-granularity-mcp-granite). That paper is **not** a scored thesis experiment.

### Sources

- Gao, Y., et al. "RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation." arXiv:2505.03275, 2025. https://arxiv.org/abs/2505.03275
- Patil, S., et al. "Gorilla: Large Language Model Connected with Massive APIs." arXiv:2305.15334, 2023. https://arxiv.org/abs/2305.15334
- Qin, Y., et al. "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs." ICLR 2024, arXiv:2307.16789. https://arxiv.org/abs/2307.16789
- Yuan, S., et al. "CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets." arXiv:2309.17428, 2024. https://arxiv.org/abs/2309.17428
- Hao, S., et al. "ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings." NeurIPS 2023, arXiv:2305.11554. https://arxiv.org/abs/2305.11554

---

## 3a. Tool-Interface Granularity (MCP-GRANITE)

### Overview

How the **same capability set** is split into tools is a design knob. MCP leaves interface design unspecified; MCP-GRANITE holds capabilities fixed and varies only the granularity of the tool schema. In the literature this is complementary to tool RAG; **RAG itself is out of scope** for this thesis paper.

**Not a scored thesis experiment.** Do not invent a `multi_turn_base` number for this method. Paper numbers below are theirs.

### Mechanism

Four interface levels over an identical capability set:

| Level | Split | Typical tool count |
|-------|-------|--------------------|
| L4 | Fine-grained primitives | 8–10 tools |
| L3 | Task-class | ~**4** tools |
| L2 | Read / write | 2 tools |
| L1 | Monolith | 1 tool |

The benchmark: **81** scenarios × **9** edge/IoT domains × **9** local models (**268M–20.9B**) × **8,748** trials.

### Evidence

Paper results (Paschalides et al., 2026; cite as theirs, not ours):

- **L3 (~4 task-class tools)** is the best trade-off: task completion **+16.4%** vs L4 fine-grained and **+33.6%** vs L1 monolith. Argument accuracy nearly doubles from L4→L3 (**0.20 → 0.40**).
- L1 over-consolidation: **28.2%** zero-tool-call rate.
- Model size is weakly correlated with task completion (ρ≈**0.285**) and strongly with latency (ρ≈**0.946**).
- A **3.2B** model (Llama 3.2) at the right granularity outperforms a mismatched **20.9B** (GPT-OSS) on the paper’s TC/efficiency trade-off narrative.

### Applicability

- **BFCL categories:** System-design knob, not a generation-loop ablation. Relevant whenever the harness exposes a class of APIs (multi-turn function docs, padded tool sets, production MCP catalogs). Complements [§3 RAG-Based Tool Retrieval](#3-rag-based-tool-retrieval) as **literature only** (RAG is out of scope for this thesis paper): that literature discusses retrieval; this section discusses API-surface cardinality.
- **Edge SLM relevance:** High. Moderate ~4 task-class tools beat both a fine-grained dump and a single monolith on their suite.
- **Implementation:** Redesign how BFCL / production tools are grouped (task-class wrappers vs primitive APIs). No weight change. No `--fc-mode`.
- **Complexity:** Medium (schema / catalog design, not a decode-time trick)
- **Limitation:** Paper suite is MCP edge/IoT, **not** our BFCL v4 `multi_turn_base`. Do **not** equate +16.4% / +33.6% TC to a thesis harness score. Unscored.
- **Complement (unscored, 2026-09-28):** [§30 HyperTool](#30-hypertool-code-block-execution-granularity) changes **execution granularity** (code-block folds multi-tool chains into one outer call) rather than schema cardinality. MCP-Universe numbers are **theirs**; not BFCL `multi_turn_base`.

### Source

Paschalides, D., Symeonides, M., Pallis, G., & Dikaiakos, M. D. "MCP-GRANITE Benchmark: GRANularity Interface TEsting for MCP-Based LLM Agents." arXiv:2609.24161, 2026. https://arxiv.org/abs/2609.24161

---

## 4. Inference-Time Compute Scaling

### Overview

Instead of relying on a single greedy decode, generate N candidate responses and select the best one. Strategies include majority voting, best-of-N with a verifier, and tree search. Research shows that smaller models with advanced inference strategies can outperform larger models with naive decoding.

### Strategies

- **Majority voting (MV@N):** Generate N responses, parse each into tool calls, select the most frequent `(function_name, arguments)` tuple. Most consistent for accuracy but highest token cost.
- **Best-of-N with verifier:** Generate N candidates, score each with a reward model or schema validator, select the highest-scoring one.
- **Self-consistency:** Sample multiple reasoning chains (at temperature > 0), take the majority answer. Originally proposed for chain-of-thought reasoning but applicable to tool selection.
- **Tree search:** Explore multiple decoding paths with branching and backtracking. More complex but can find better solutions.

### Evidence

Snell et al. (2024) demonstrated that "scaling LLM test-time compute optimally can be more effective than scaling model parameters." Specifically, Llemma-7B with tree search **consistently outperformed Llemma-34B** across all tested inference strategies on the MATH benchmark.

A large-scale study spanning 30B+ generated tokens across 8 open-source LLMs (7B–235B) found (Qu et al., 2025):

- No single test-time scaling strategy universally dominates
- For a given model type, optimal performance scales monotonically with compute budget
- **Short-horizon models** (e.g., QwQ-32B): prefer larger sample counts with first-finish search at low compute, scaling to MV@N at high compute
- **Long-horizon models** (e.g., Qwen3-32B): prefer simple decoding at low compute, MV@N at high compute
- Beam search consistently underperforms, showing "inverse scaling" where larger beams degrade accuracy

### Applicability

- **BFCL categories:** All. Particularly attractive for models with fast inference (Qwen2.5-3B at 93.6 tok/s, FunctionGemma at 204 tok/s) where generating 5 candidates is still fast on an RTX 4090.
- **Implementation:** Multi-turn ablation is `--fc-mode iffc-state-bon` (MV@N overlay on **IFFC+STATE**, not NFC/GST/CD; see [Experiment: Best-of-N majority voting](#experiment-best-of-n-majority-voting---fc-mode-iffc-state-bon)). Samples N candidates at `temperature=0.7` (default N=5), majority-votes `(function_name, arguments)` **at each turn/step**, then executes the winner once and continues the IFFC+STATE agent loop. Latency ~N×.
- **Complexity:** Medium
- **Tradeoff:** Multiplies inference time by N. With FunctionGemma's 16ms TTFT, 5 samples take ~80ms. With Qwen3.5's 656ms TTFT, 5 samples take ~3.3 seconds. On Gemma-4-E2B IFFC+STATE Run 10 (~181 min / ~3.0 h for 200 `multi_turn_base` entries), BoN N=5 Run 11 measured **~21.3 h generate (~7× wall)** and ~8.8× avg `predicted_ms` for **+2.5 pp** (73/200 = 36.5% vs 68/200 = 34.0%). Modest accuracy at ~N× decode — **not free**.

### Sources

- Snell, C., et al. "Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters." arXiv:2408.03314, 2024. https://arxiv.org/abs/2408.03314
- Qu, C., et al. "The Art of Scaling Test-Time Compute for Large Language Models." arXiv:2512.02008, 2025. https://arxiv.org/abs/2512.02008
- Wang, X., et al. "Self-Consistency Improves Chain of Thought Reasoning in Language Models." ICLR 2023, arXiv:2203.11171. https://arxiv.org/abs/2203.11171
- Brown, B., et al. "Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models." ICLR 2025, arXiv:2408.00724. https://arxiv.org/abs/2408.00724
- **Related (unscored, 2026-09-28):** [§28 Looped Language Models](#28-looped-language-models-for-compositional-tool-calling) — recurrent latent depth as a test-time compute knob for compositional tool calling. Paper BFCL is static single-turn AST (live/multi-turn left future); not our `multi_turn_base`.

---

## 28. Looped Language Models for Compositional Tool Calling

### Overview

Popescu, Sáez de Ocáriz Borde, and Liò (2026) study **recurrent latent depth** for compositional / dependency-aware tool use. Native **Ouro 1.4B/2.6B** or retrofit Llama-3.2-1B / OLMo-2-1B. Helps compositional tool use (BFCL parallel/nested + NESTful); smaller gains on isolated API-Bank. Adaptive exit improves the compute–performance trade-off.

**Train vs infer: mostly inference / test-time compute** (fixed or adaptive recurrent depth). The paper uses matched SFT recipes so looped vs non-looped is a fair architecture comparison — natively looped ≫ retrofit on nested workflows. **Not a scored thesis experiment.** Do not invent a `multi_turn_base` number.

### Mechanism

Instead of a single forward pass, the model repeatedly refines latent representations before emitting each token (recurrent depth). Depth can be fixed or stopped adaptively. This is a test-time compute knob, orthogonal to decode heuristics (IFFC / CD / stop-repeat / BoN) and to [§4 majority-vote scaling](#4-inference-time-compute-scaling).

### Evidence

Paper results (Popescu et al., 2026; cite as **theirs**, not ours): BFCL AST / NESTful Win Rate / API-Bank under matched SFT. Accuracy on multi-step generally rises with recurrent depth; adaptive stopping matches deep fixed depth at lower average iterations.

**Limitation (paper):** static single-turn BFCL categories in the main study; live/multi-turn BFCL left as future work. Do **not** invent our MT scores.

### Applicability

- **Thesis role:** Logged 2026-09-28; **unscored**. Would need a looped backbone or retrofit recipe this repo does not currently run.
- **BFCL categories:** Paper is compositional single-turn AST / nested workflows, **not** our `multi_turn_base` execution slice.
- **Complexity:** High (architecture / pretrain; not a `--fc-mode`).
- **Limitation:** No looped Gemma-4-E2B in this harness. Retrofit ≠ native Ouro on nested workflows.

### Source

Popescu, A. C., Sáez de Ocáriz Borde, H., & Liò, P. "Looped Language Models Improve Compositional Tool Calling." arXiv:2608.18171, 2026. https://arxiv.org/abs/2608.18171

---

## 5. Context Compression for Multi-Turn

### Overview

As LLM agents interact across multiple turns, actions and observations accumulate into ever-growing contexts that exceed the context window and distract the model. Context compression reduces the volume of information in the agent's working memory while preserving task-relevant details.

### Mechanism (ACON)

Agent Context Optimization (ACON) uses two complementary strategies:

1. **History compression:** When conversation history exceeds a threshold (e.g., 4096 tokens), summarize older turns into a concise state description while preserving the most recent turns verbatim.
2. **Observation compression:** When individual tool results exceed a threshold (e.g., 1024 tokens), condense them using context-aware summarization that considers the prior interaction history.

ACON optimizes compression guidelines through natural language feedback using contrastive task analysis — comparing successful vs. failed executions to identify which information was actually needed.

### Evidence

ACON results (Kang et al., 2025):

| Benchmark | Baseline Accuracy | ACON Accuracy | Peak Token Reduction |
|-----------|-------------------|---------------|---------------------|
| AppWorld | 56.0% | 56.5% | 26% (9.93K → 7.33K) |
| OfficeBench | 72.6% | 74.7% | 30% |
| 8-Objective QA | 0.336 EM | 0.373 EM | 54.5% |

For smaller agents specifically, ACON delivered dramatic improvements: Qwen3-14B improved **32% on AppWorld** and **46% on multi-objective QA** through cleaner contexts that reduce distraction effects. Distillation to Qwen3-8B and Phi-4 preserved >95% of teacher accuracy.

### Practical Approaches

Three complementary compression techniques used by production agent systems (LangChain, Google ADK, JetBrains):

1. **Offload large tool results** to the filesystem whenever they occur
2. **Offload old write/edit arguments** from tool calls when context crosses a threshold
3. **LLM-based summarization** of older turns when no more context is eligible for offloading

The most recent ~10% of the context window is preserved verbatim as active working memory. At a critical threshold (e.g., 60% capacity), compress conversation history by generating structured summaries capturing: task context, actions taken (tool calls with results), outcomes, next steps, and important references.

### Applicability

- **BFCL categories:** `multi_turn_base` first (Gemma-4-E2B-it BoN Run 11 **73/200 = 36.5%** on IFFC+STATE, +2.5 pp vs Run 10 **68/200 = 34.0%** at ~7× wall; vs NFC 27.0% / IFFC 30.5% / GST 30.5% / CD 10.5%); also `multi_turn_miss_func`, `multi_turn_miss_param`, `multi_turn_long_context`. The current `generate_multi_turn.py` appends every tool call and result to `chat_history` indefinitely. By turn 4–5 with multiple tool calls per turn, this can exceed 8192 tokens (triggering the `context_overflow` flag).
- **Implementation:** The multi-turn ablation is `--fc-mode iffc-state` (IFFC JSON-in-text plus a harness-written STATE snapshot from actual tool execution results; see [Experiment: IFFC + harness STATE](#experiment-iffc--harness-state---fc-mode-iffc-state)). Keep the last 2 turns verbatim; older raw dumps are dropped in favor of the compact STATE block. **Not** LLM summarization, **not** GST, **not** CD/GBNF.
- **Complexity:** Medium (sliding window: Low)
- **Caveat:** Summarization itself requires an LLM call. For SLMs, the summarization quality may be poor. A simpler structured state tracker (explicit key-value record of tool results) avoids this — that is what `--fc-mode iffc-state` implements. On **base** Gemma `multi_turn_base`, STATE **beat IFFC** but did **not** cut state mismatch (60→64); BoN on that channel then cut state 64→53 at ~7× wall (see `THESIS.md`). On **SFT-v4** the same overlay is a **scored negative** (Run 16 13.0% vs Run 15 15.0%). Recipe v5 on a new mix is a **scored regression** (Run 17 11.5% vs Run 15 15.0%). Recipe v6 (v4 empty/recovery + only `drop_state_inconsistent`; turn-0 rewrites off) is a **scored miss vs v4** (Run 18 13.0% vs Run 15 15.0%). Next leverage is SFT **v7** (planned, **not scored** — not another thin filter on the same family), not BoN-on-v4/v5/v6 and not more empty-only prompts / more epochs on v5/v6. **SFT path locked 2026-09-23** as the main path.

### Sources

- Kang et al. "ACON: Optimizing Context Compression for Long-horizon LLM Agents." arXiv:2510.00615, 2025. https://arxiv.org/abs/2510.00615
- LangChain. "Context Management for Deep Agents." 2025. https://blog.langchain.com/context-management-for-deepagents/
- JetBrains Research. "Cutting Through the Noise: Smarter Context Management for LLM-Powered Agents." 2025. https://blog.jetbrains.com/research/2025/12/efficient-context-management/
- Google. "Context Compression — Agent Development Kit (ADK)." 2025. https://google.github.io/adk-docs/context/compaction/

---

## 6. ReAct and ReWOO Planning Patterns

### ReAct (Reasoning + Acting)

The model alternates between Thought (reasoning), Action (tool call), and Observation (tool result) steps in an interleaved loop:

```
Thought: I need to find the weather in Paris. I'll use the get_weather function.
Action: get_weather(location="Paris")
Observation: {"temperature": 22, "condition": "sunny"}
Thought: I now have the weather information.
Answer: The weather in Paris is 22°C and sunny.
```

ReAct decomposes the complex task of "understand query + select tool + format arguments" into discrete, simpler steps. Shown to be effective on multi-step tasks (HotpotQA, FEVER, ALFWorld, WebShop).

**Caveat for SLMs:** ReAct adds overhead from repeated context per step and requires parsing tool calls from free-text output. For single-turn BFCL categories, it adds overhead without the benefit of iterative refinement. For multi-turn categories, the existing `generate_multi_turn.py` already implements a ReAct-like loop.

### ReWOO (Reasoning Without Observation)

ReWOO decouples reasoning from execution into three distinct phases:

1. **Planner:** Given the user query, generate a comprehensive multi-step plan at once, with placeholders for tool results
2. **Worker:** Execute each planned step, filling in results
3. **Solver:** Use the completed plan to generate the final response

**Advantage:** 5–10x token reduction vs. ReAct by eliminating the internal monologue during tool execution and avoiding repetitive prompt overhead of calling the LLM multiple times.

**Tradeoff:** Fragile — if the initial plan is flawed or a tool returns an unexpected error, the agent has no built-in mechanism to self-correct, often leading to hard failure.

### Applicability

- **BFCL categories:** Multi-turn categories are a natural fit. A planning prompt before each turn ("Given the conversation so far, what tool calls are needed?") could improve accuracy without changing the loop structure.
- **Complexity:** Medium (single-model planning) to High (two-pass generation)
- **Limitation for SLMs:** Wei et al. (2022) noted that CoT "does not positively impact performance for small models." Full planning may not help 1–3B models, but a lighter variant (listing needed functions before calling them) could provide modest benefits.

### Sources

- Yao, S., et al. "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR 2023, arXiv:2210.03629. https://arxiv.org/abs/2210.03629
- Xu, B., et al. "ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models." arXiv:2305.18323, 2023. https://arxiv.org/abs/2305.18323

---

## 7. Prompt-Only Tool Calling

### Overview

A method that enables LLMs to achieve stable tool calling using only prompt engineering, targeting models that lack native tool calling support.

### Mechanism

The system prompt contains three elements:

1. **Tool example:** A trivial demonstration tool (e.g., number increment/decrement) showing the model how tool calling works, without confusing it with the actual usable tools
2. **Tool instructions:** Dynamically generated human-readable descriptions of available tools listing each tool's name, description, parameters, and required fields
3. **Return format:** The exact JSON structure for tool calls: `{"tool": "tool_name", "parameters": {"param": "value"}}`

Output is parsed via regex to extract tool names and parameters. Tool results are returned to the model as observation messages, enabling multi-turn tool usage.

### Evidence

Tested on quantized 7B–9B models from Ollama:

| Model | Success Rate |
|-------|-------------|
| gemma2-9b | 100% (all 7 tasks) |
| llama3-8b | 96% |
| qwen2-7b | 99% |
| mistral-7b | 97% |

### Applicability

- **BFCL relevance:** Low for the current pipeline (llama-server already handles tool calling natively via `--jinja`). However, the **demonstration tool technique** is relevant — including a simple worked example in the system prompt could help models that struggle with the output format.
- **Complexity:** Low

### Source

He, S. "Achieving Tool Calling Functionality in LLMs Using Only Prompt Engineering Without Fine-Tuning." arXiv:2407.04997, 2024. https://arxiv.org/abs/2407.04997

---

## 8. Few-Shot Example Retrieval

### Overview

Instead of static few-shot examples, dynamically retrieve the most similar worked examples from a database of correct tool call examples, based on semantic similarity to the current query.

### Mechanism

1. Build a database of (query, correct_tool_call) pairs — BFCL's reference answers can seed this
2. Embed all examples using a sentence embedding model
3. At inference, embed the test query, retrieve K nearest examples by cosine similarity
4. Include retrieved examples as few-shot demonstrations in the prompt

### Evidence

- Rubin et al. (2022) showed that retrieved (vs. random) few-shot examples yield significantly better in-context learning.
- Qin et al. (2024, ToolLLM) used a "demo retriever" and achieved higher pass rates than random demo selection.
- For SLMs specifically, few-shot gains are proportionally larger on structured output tasks because these models learn output format primarily from in-context patterns rather than instruction following.
- Estimated impact: +5–20pp on structured output tasks for 1–3B models.

### Applicability

- **BFCL categories:** High potential for `parallel`/`parallel_multiple` (showing the model the expected multi-call array format), `irrelevance` (showing "no tool needed" responses), and cross-language categories (Java/JavaScript examples).
- **Implementation:** Build a vector store of representative examples per category. Inject retrieved examples as additional user/assistant message turns before the test query. Use different function names than the test tools to avoid confusion.
- **Complexity:** Medium
- **Caveat:** Must not exceed the 8192-token context window. Few-shot examples compete for space with tool definitions.

### Sources

- Rubin, O., et al. "Learning To Retrieve Prompts for In-Context Learning." NAACL 2022, arXiv:2112.08633. https://arxiv.org/abs/2112.08633
- Qin, Y., et al. "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs." ICLR 2024, arXiv:2307.16789. https://arxiv.org/abs/2307.16789

---

## 9. Verification and Self-Correction Loops

### Overview

After the model generates a tool call, validate it against the function schema before execution. If invalid, re-prompt the model with the error for correction.

### Mechanism

```
Model generates: {"get_weather": {"location": 42}}
Verifier catches: parameter 'location' expects string, got integer
Re-prompt: "Your tool call had an error: parameter 'location' expects
            string but got integer. Please fix."
Model generates: {"get_weather": {"location": "42nd Street, NY"}}
```

### Evidence

- **Reflexion** (Shinn et al., 2023): Agents verbally reflect on task feedback and maintain reflective text in episodic memory. Improved accuracy by ~14% on reasoning tasks.
- **Self-Refine** (Madaan et al., 2023): Iterative refinement through self-feedback.
- **However:** Recent work shows mixed results for SLMs. Huang et al. (2024) demonstrated that "small language models need strong verifiers to self-correct reasoning" — SLMs cannot reliably self-correct without external verification. Naive "reflect and retry" loops can amplify errors rather than resolve them.
- **Implication:** Self-correction for SLM tool calling should use **external schema validation** (JSON Schema check against the function definition) rather than asking the model to verify its own output.

### Applicability

- **BFCL categories:** All categories where the model gets the function right but mangles arguments. The existing `coerce_types.py` is a static version of this; a dynamic version would re-prompt the model.
- **Implementation:** Add JSON Schema validation before writing results, with a retry loop (max 2–3 retries) in `generate_for_category()`.
- **Complexity:** Medium
- **Tradeoff:** Each retry costs another full inference pass. On edge devices, may be prohibitive.

### Sources

- Shinn, N., et al. "Reflexion: Language Agents with Verbal Reinforcement Learning." NeurIPS 2023, arXiv:2303.11366. https://arxiv.org/abs/2303.11366
- Madaan, A., et al. "Self-Refine: Iterative Refinement with Self-Feedback." NeurIPS 2023, arXiv:2303.17651. https://arxiv.org/abs/2303.17651
- Huang, J., et al. "Small Language Models Need Strong Verifiers to Self-Correct Reasoning." arXiv:2404.17140, 2024. https://arxiv.org/abs/2404.17140

---

## 10. DSPy Automatic Prompt Optimization

### Overview

DSPy is a framework for programmatically optimizing LLM prompts. Instead of hand-crafting system prompts, define the pipeline as a typed program with a metric function, and let DSPy's optimizers automatically search the space of instructions and few-shot examples to maximize the metric.

### Mechanism

1. Define a DSPy module with typed signatures (e.g., `query: str, tools: list[dict] -> tool_calls: list[dict]`)
2. Provide a metric function (e.g., BFCL AST checker accuracy)
3. Provide a small training set (subset of BFCL test cases with reference answers)
4. DSPy's MIPROv2 optimizer generates candidate instructions and few-shot examples, evaluates them against the metric, and uses Bayesian optimization to search the space

DSPy includes `dspy.ReAct` for agentic tool use, making it directly applicable to tool calling pipelines.

### Evidence

DSPy has demonstrated significant improvements over hand-crafted prompts across diverse tasks. The MIPROv2 optimizer generates data-aware and demonstration-aware instructions using Bayesian optimization. No BFCL-specific benchmarks are published, but the framework is general-purpose and has been validated on structured output tasks.

### Applicability

- **BFCL categories:** All. Could automatically discover model-specific prompt optimizations that beat hand-crafted templates.
- **Implementation:** Build a DSPy wrapper around llama-server API calls, use BFCL accuracy as the metric, and let the optimizer run on a subset of test cases.
- **Complexity:** High (requires restructuring the pipeline into DSPy's programming model)
- **Value:** Could produce a model-specific optimized prompt per model (Qwen2.5-3B, Llama-3.2-3B, etc.), demonstrating that automatic optimization can outperform manual prompt engineering for SLMs.

### Source

Khattab, O., et al. "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." ICLR 2024. https://dspy.ai/

---

## 11. Parallel Function Calling via SimpleTool

### Overview

SimpleTool exploits structural properties of function calling to achieve parallel tool call generation without additional parameters or draft models.

### Mechanism

Rather than generating multiple tool calls sequentially in a single response (which SLMs universally fail at on BFCL), SimpleTool decouples the process by exploiting the fact that function calls have predictable structure. The approach achieves parallelism at the decoding level rather than requiring the model to emit a complete array of calls in one pass.

### Applicability

- **BFCL categories:** `parallel`, `parallel_multiple` (currently 0% across all tested models)
- **Note:** The paper was published March 2026 (arXiv:2603.00030). Code is available at https://github.com/HaxxorCialtion/SimpleTool.
- **Complexity:** Would require adapting the approach to work with llama.cpp's serving infrastructure

### Source

Shi, X., et al. "SimpleTool: Parallel Decoding for Real-Time LLM Function Calling." arXiv:2603.00030, 2026. https://arxiv.org/abs/2603.00030

---

## Key Finding: llama.cpp Already Uses Grammar-Constrained Decoding

Analysis of the local llama.cpp source code (`common/chat.cpp`) reveals that **grammar-constrained decoding is already active** during tool calling when using `--jinja` with the `tools` parameter.

### How It Works

1. **Grammar construction:** For each tool in the `tools` array, the function's `parameters` JSON schema is converted to a GBNF grammar via `json-schema-to-grammar.cpp`. This constrains argument names, types, and structure to match the schema.

2. **Lazy mode:** With `tool_choice = "auto"` (the current benchmark setting), `grammar_lazy = true`. The grammar is **not enforced from token 0**. Instead, the model generates freely until a trigger token is detected:
   - Hermes 2 Pro (Qwen2.5): triggers on `[TOOL_CALLS]`
   - Llama 3.x: triggers on `<|python_tag|>`

3. **Required mode:** With `tool_choice = "required"`, `grammar_lazy = false` — the grammar is enforced from token 0, and `min_calls = 1`.

4. **Parallel support:** The `parallel_tool_calls` parameter is supported but depends on the model's Jinja template.

### Implications

The 0% parallel score across all models is **not due to missing grammar support** — grammar is already active. The failure is likely:

1. The grammar may not enforce the array-of-calls structure for parallel output
2. Models may never produce the trigger token for a multi-call response
3. The `parallel_tool_calls` parameter may need to be explicitly enabled in the API payload

This changes the thesis narrative from "grammar-constrained decoding could help" to **"grammar-constrained decoding is already active — the question is why small models still fail, and what additional inference-time techniques can close the gap."**

### Actionable Experiments

1. **Test `tool_choice = "required"`** for non-irrelevance categories
2. **Test `parallel_tool_calls: true`** in the API payload
3. **Build with LLGuidance** for stricter JSON Schema enforcement
4. **Check server logs** to verify grammar activation during benchmark runs
5. **Error analysis:** Determine whether parallel failures are format failures (model attempts multi-call but mangles it) or semantic failures (model doesn't understand it should call multiple functions)
6. **Run `--fc-mode cd`** (TagDispatch-style two-pass; see below) on Gemma `multi_turn_base` vs NFC 27.0% / IFFC 30.5% — this is a separate condition from native `tools` GBNF

### Source

Local source code analysis: `llama.cpp/common/chat.cpp`, `llama.cpp/common/json-schema-to-grammar.cpp`, `llama.cpp/grammars/README.md`

---

## Experiment: TagDispatch-style CD (`--fc-mode cd`)

This is the **constrained-decoding ablation** for the BFCL multi-turn harness. It is **not** NFC, and it does **not** treat llama-server native `tools` GBNF as the CD condition.

### Why NFC tools-GBNF is not the CD experiment

NFC (`--fc-mode nfc`) sends `tools` + `tool_choice=auto`. llama-server `--jinja` may compile a **lazy** tools grammar from the chat template (trigger tokens, then argument GBNF). That path is already the baseline (Gemma-4-E2B-it Q4_K_M `multi_turn_base` Run 4: 54/200 = 27.0%). It also has a known **parallel truncation** bug on some templates (grammar satisfied after the first `</tool_call>`), which is why single-turn `parallel` / `parallel_multiple` bypass grammar with `tool_choice=none`.

Rebranding that native grammar as “CD” would not be a new independent variable.

### What `--fc-mode cd` actually does

CD approximates XGrammar-2 **TagDispatch** (Dong et al., arXiv:2601.04426) on **stock llama-server**, without XGrammar and without rebuilding with LLGuidance.

XGrammar-2 TagDispatch is an intra-request engine primitive: decoding starts in a *dispatching* mode; emitting tag \(t_i\) switches to sub-grammar \(G_i\); after \(G_i\) completes, control returns to dispatching. llama.cpp cannot do that in one sampler (no XGrammar; optional LLGuidance is a different library and is **OFF** unless you pass `-DLLAMA_LLGUIDANCE=ON`).

The harness therefore uses a **two-pass approximation** of the same control flow:

| Pass | Role (TagDispatch analogue) | llama-server constraint | Output |
|------|------------------------------|-------------------------|--------|
| 1. Dispatch | Select tags from the live tool list, or stop | `grammar` GBNF: `[]` or `[{"function_name": <enum>}, ...]` | Names only (`additionalProperties: false`) |
| 2. Dispatched | Constrain arguments to the *selected* tool(s) | `json_schema` (server compiles to GBNF via `json-schema-to-grammar`) | IFFC-shaped `[{function_name, parameters}, ...]` |

The persistent multi-turn transcript stays IFFC-shaped (JSON list in assistant text; user-role execution results). **Only the completion requests** carry `grammar` / `json_schema`. Requests **never** include `tools` / `tool_choice`, so native tools GBNF is not applied.

A static tagged-union `oneOf` over *all* tools (`build_tagged_union_schema`) is the EBNF encoding XGrammar-2 calls cumbersome; it is implemented for tests/docs but **not** used at runtime. Two-pass keeps the pass-2 grammar as small as the selected tool.

### llama-server API (what we actually send)

Investigated against current llama-server OpenAI-compatible `POST /v1/chat/completions`:

| Field | Effect | Used by CD? |
|-------|--------|-------------|
| `tools` / `tool_choice` | Chat-template tools GBNF (NFC) | **No** |
| `grammar` | Sampling-level GBNF (template-independent) | Pass 1 |
| `json_schema` | Converted to GBNF; **takes precedence** if both `grammar` and `json_schema` are set | Pass 2 |
| `response_format: {type: json_schema}` | OpenAI-shaped wrapper; some templates historically ignored it | Not required (Gemma-4 handlers already understand `json_schema`) |

Pass 1 sends **only** `grammar` so `json_schema` cannot override the name enum. Pass 2 sends **only** `json_schema` so llama.cpp’s converter handles property permutation / nested BFCL parameter schemas.

### Honesty vs full XGrammar-2

| XGrammar-2 | This harness (`cd`) |
|-----------|---------------------|
| Intra-request tag-triggered grammar switch | **Two HTTP requests** (dispatch, then args) |
| Cross-Grammar Cache / Earley mask cache / JIT | Not available on llama-server GBNF |
| Ships in MLC / vLLM / SGLang | Not used (stock llama-server) |
| Strict structural tags in one decode | Approximation: name GBNF, then per-tool JSON Schema |

Do **not** report this as “we ran XGrammar-2”. Score tables in `THESIS.md` stay unchanged until a Gemma `multi_turn_base` CD run is scored.

### LLGuidance (optional; not required)

llama.cpp can be rebuilt with Microsoft’s LLGuidance backend:

```bash
# macOS (Apple Silicon) — requires a Rust toolchain
cmake -B build -DGGML_METAL=ON -DLLAMA_LLGUIDANCE=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)

# Linux CUDA (not the CD default machine)
# cmake -B build -DGGML_CUDA=ON -DLLAMA_OPENSSL=ON -DLLAMA_LLGUIDANCE=ON
```

The current Mac Metal build is assumed **without** this flag. CD does not need it.

### How to run (Gemma-4-E2B-it, `multi_turn_base`)

Same server command as NFC / IFFC (no extra llama-server flags):

```bash
./llama.cpp/build/bin/llama-server --jinja -fa on \
  -hf bartowski/google_gemma-4-E2B-it-GGUF:Q4_K_M \
  --host 0.0.0.0 --port 8080 -ngl 99 -c 65536 --temp 0.0 \
  -np 4 -cb -ub 1024

uv run python -m benchmark.smoke_cd   # hermetic; add --live if the server is up

uv run python -m benchmark.run_all --model gemma4-e2b --only-multi-turn \
  --categories multi_turn_base --fc-mode cd \
  --notes "TagDispatch-style CD vs NFC 27% / IFFC 30.5%"
```

Comparison target (already scored, do not overwrite): NFC Run 4 = 54/200 = 27.0%; IFFC Run 5 = 61/200 = 30.5%; CD Run 8 = 21/200 = 10.5%; GST+NFC Run 9 = 61/200 = 30.5%; IFFC+STATE Run 10 = 68/200 = 34.0%; BoN N=5 Run 11 = 73/200 = 36.5%; SFT-IFFC LoRA best@1000 Run 12 = 1/200 = 0.5% (scored negative); SFT-IFFC-v2 filtered-[] LoRA best@2000 Run 13 = 1/200 = 0.5% (same collapse; filter insufficient); SFT-IFFC-v2 + strip trailing `[]` Run 14 = 4/200 = 2.0% (**DIAGNOSTIC** history bookkeeping — **not** a method win); SFT-IFFC-v4 LoRA best@1600 Run 15 = 30/200 = 15.0% (**scored recovery** from collapse, still **below IFFC 30.5% and below NFC 27.0%** — **do not present as beating IFFC**); SFT-IFFC-v4 + IFFC+STATE Run 16 = 26/200 = 13.0% (**scored negative/null overlay** vs Run 15 — **do not propose BoN on this ckpt**); SFT-IFFC-v5 LoRA best@1800 Run 17 = 23/200 = 11.5% (**scored regression vs Run 15** — **do not present as a method win**; **no BoN / no more epochs on this mix**); SFT-IFFC-v6 LoRA best@1800 Run 18 = 26/200 = 13.0% (**scored miss vs Run 15 15.0%** — **do not present as a method win**; **no BoN / no more epochs on v6**); xLAM-1b-fc-r NFC Run 19 = 0/200 = 0.0% (**scored format-channel collapse** on this multi_turn NFC harness — **not** published overall BFCL ~78.94%; **not** an ability floor; **not** a method win; gen_tps_mean ≈ **196.7**; avg `api_calls` **3.63**); xLAM-1b-fc-r `--fc-mode xlam` Run 20 = 6/200 = 3.0% (**format-fixed fair-channel** after [PR #42](https://github.com/sskarz/Thesis/pull/42) — tools actually ran, avg `api_calls` **50.47**; still **very low**; **not** a method win; **not** published ~78.94%; still **below CD 10.5%**; gen_tps_mean ≈ **91.6**); IFFC-stop-repeat Run 27 = 54/200 = 27.0% (**BASE** Gemma ungated, [PR #54](https://github.com/sskarz/Thesis/pull/54) commit `5b49d7a` — **scored miss vs IFFC 30.5%** (−3.5 pp); **not** smoke81 / v7 / v8 SFT / BoN; **not** a method win); IFFC-stop-repeat **tighten** Run 30 = 65/200 = **32.5%** (**BASE** Gemma, PR #54 commit `78c22a0`, MIN=6 `observe_same`; `scores.id=21`; **200/200 responses** — **first scored win vs Run 5 IFFC 30.5%** (**+2.0 percentage points; 65−61 correct** — **not** “+4pp”); **+5.5 pp vs Run 27**; still **below STATE 34.0% and BoN 36.5%**; **not** best overall inference; do **not** overwrite this row / BoN / IFFC / SFT-v4; **Codex CONFIRM_PASS**; do **not** invent tok/s); IFFC OODA-card Run 42 = 71/200 = **35.5%** (**BASE** Gemma, `--fc-mode iffc-ooda-card`, [PR #61](https://github.com/sskarz/Thesis/pull/61) @ `1dea341`; **200/200 responses**; finished `2026-09-29T05:51:02.787870+00:00`; **+6 / +3.0 pp vs Run 30**; still **below BoN 36.5% (−1.0 pp)** and **above STATE 34.0% (+1.5 pp)**; `state_mismatch` **71** gate **MISS** vs <60; empty+missing **58** MET vs ≤75; TTFT **517.8 ms** vs Run 30 **630.5 ms** (≤1.25× **MET**); **not** a clean gate sweep; **not** 4-pass OODA / **not** iffc-state / **not** a flag on stop-repeat; do **not** overwrite Run 30; calib Run 40 ≠ BFCL %; Run 41 orphan ignore; SFT parked; no BoN; do **not** invent tok/s); IFFC PTC-A Run 46 = 27/200 = **13.5%** (**BASE** Gemma, `--fc-mode iffc-ptc-a`, [PR #63](https://github.com/sskarz/Thesis/pull/63) @ `3569311`; **200/200 responses**; finished `2026-09-29T08:15:24.061478+00:00`; **scored FAIL / Acc regression** vs Run 42 **−22.0 pp** / Run 30 **−19.0 pp** / IFFC **−17.0 pp**; `state_mismatch` **102** gate **MISS** vs ≤60; empty+missing **71** MET vs ≤75; wall ~42.1 min / ratio vs Run 30 **0.234485** (≤1.25× **MET**); **bank and close**; PTC-B **HOLD**; calib Run 45 ≠ BFCL %; do **not** overwrite Run 30 / Run 42; no BoN; do **not** invent tok/s); DPO-full Run 51 = **53/200 = 26.5%** (plain IFFC; **not** a method win; other fields unverified); IFFC-Decision Run 58 = **50/200 = 25.0%** (**BASE** Gemma, `--fc-mode iffc-decision` + tev1, treatment-only; **FAIL_METHOD**; **−10.5 pp vs Run 42** / **−7.5 pp vs Run 30** / **−5.5 pp vs IFFC**; sidecar ~**92.7%** HTTP 200 — **not** dead; park Decision Acc; do **not** overwrite BoN / Run 42 / Run 30 / IFFC); IFFC-Decision+OODA Run 61 = **52/200 = 26.0%** (**BASE** Gemma, `--fc-mode iffc-decision-ooda` + tev1, treatment-only; **FAIL vs R5–R42**; **−9.5 pp vs Run 42** / **−6.5 pp vs Run 30** / **−4.5 pp vs IFFC**; only **+1.0 pp vs Run 58**; park Decision(+OODA) Acc; compose/hygiene smoke PASS — Acc is this row, not invented from smoke; Runs 59/60 10-ID **1/10 vs 0/10** — **not** N=200).

### Pause / resume (kill overnight, continue the same run)

Multi-turn generation (NFC / IFFC / CD) checkpoints **each completed entry** into `benchmark/results/benchmark.db`: `responses` and `multi_turn_timings` for the current `run_id`. Killing the process (Ctrl-C, machine sleep, overnight shutdown) does **not** discard finished entries. In-flight entries (still only in RAM) are regenerated.

This continues the **same** `runs` row (`finished_at` stays NULL until the pipeline finishes). Do not start a new run and copy results.

```bash
# First launch (creates run_id N)
uv run python -m benchmark.run_all --model gemma4-e2b --only-multi-turn \
  --categories multi_turn_base --fc-mode cd \
  --notes "TagDispatch-style CD vs NFC 27% / IFFC 30.5%"

# Kill the process whenever. Rerun the same flags plus --resume:
uv run python -m benchmark.run_all --model gemma4-e2b --only-multi-turn \
  --categories multi_turn_base --fc-mode cd --resume
```

`--resume` selects the latest unfinished run for that `--model` whose notes record the same `--fc-mode`. Pin a row with `--resume-run-id N`. On each category, entry IDs already present in `responses` for that run/category are skipped. Generation-only: `python -m benchmark.generate_multi_turn --fc-mode cd --run-id N`.

### Implementation notes

- Module: `benchmark/cd.py`. Smoke: `uv run python -m benchmark.smoke_cd`.
- Hermetic tests prove the name enum / per-tool schema **reject** unknown names, extra wrapper keys, and missing required args. Live `--live` only checks structure against a real server.
- Two-pass **doubles API calls** on steps that invoke a tool (name + args). End-of-turn `[]` is a single dispatch request.
- NFC and IFFC use the same multi-turn loop; only the completion request differs (see above). Checkpoint/resume is shared across NFC/IFFC/CD.

### Sources

- Dong, Y., et al. "XGrammar-2: Efficient Dynamic Structured Generation Engine for Agentic LLMs." arXiv:2601.04426, 2026. https://arxiv.org/abs/2601.04426
- llama.cpp server `json_schema` / `grammar` fields: `tools/server/server-schema.cpp`

---

## Experiment: Guided-Structured Templates (`--fc-mode gst`)

This is the **prompt-only GST ablation** for the BFCL multi-turn harness (He et al., arXiv:2509.18076). It is **not** IFFC JSON-in-text and **not** TagDispatch/CD GBNF.

Scored on Gemma-4-E2B-it Q4_K_M `multi_turn_base` (Mac 48 GB, llama.cpp, `-np 4 -c 65536`): GST+NFC Run 9 **61/200 = 30.5%** (started `2026-09-13T02:09:19Z`, finished `2026-09-13T04:26:49Z`, ~137.5 min wall; 200/200 responses; 139 `eval_errors`). Ties IFFC Run 5 **61/200 = 30.5%**; +7 vs NFC Run 4 **54/200 = 27.0%**; +40 vs CD Run 8 **21/200 = 10.5%**. The 61=61 vs IFFC is a **swap** (19 GST-only, 19 IFFC-only), not the same 61 rows. Full tables and interpretation: `THESIS.md`. Do **not** treat GST as a lift over IFFC.

### Contrast vs NFC / IFFC / CD

| Mode | Channel | Constraint | What the model writes |
|------|---------|------------|------------------------|
| NFC | `tools` + `tool_choice=auto` | llama-server lazy tools GBNF (baseline) | Native `tool_calls` |
| IFFC | no `tools` / `tool_choice` | none (prompt) | JSON list in assistant text |
| CD | no `tools` / `tool_choice` | pass-1 name GBNF, pass-2 `json_schema` | IFFC-shaped JSON, two HTTP passes |
| **GST** | **same as NFC** (`tools` + `tool_choice=auto`) | **none extra** (prompt overlay) | `[GST]…[/GST]` (steps 1–7) in **content**, then native `tool_calls` |

GST keeps NFC scoring plumbing: BFCL execution uses `tool_calls` converted to func-strings. GST text is parsed only for smoke/tests and **stripped/ignored** for the checker. A draft inside the GST block is not executed.

No llama.cpp rebuild. No `grammar` / `json_schema` fields.

### Paper Detail template (encoded)

He et al. Appendix A.2 **Detail** (the prompting template), with multi-turn step-7 ABSTAIN:

1. Identify which function(s) best fit the user query from descriptions
2. Pick that function/set to fulfill the request
3. Examine selected function documentation
4. Analyze parameters (types, required vs optional)
5. Extract values from query (type conversions if needed)
6. Draft the function call(s)
7. Revalidate against docs + user request (or abstain / note missing params)

The paper is single-turn. This harness adds three multi-turn rules **outside** extra PE:

- Steps must use the full transcript (prior tool observations / env feedback), not only the latest user utterance.
- Step 7 may **ABSTAIN** (no `tool_calls`) when no further call is appropriate or required params are missing.
- GST output is **terse**: one short bullet per step.

### Max guidance (context)

| Piece | Bound | Why |
|-------|-------|-----|
| System curriculum | one system message, curriculum only | Tool schemas stay on native `tools` (unlike IFFC, which dumps JSON schemas into the system prompt ~6k tok) |
| Model `[GST]` block | **max 8 lines**, one bullet per step | Limit multi-turn blow-up; later turns still see prior GST in the transcript |
| Extra PE | none beyond the template + the three multi-turn rules above | Paper: template is the intervention, not free-form CoT |

Do not paste function docs into `[GST]`. If a live run still overflows (`context_overflow` flags), shrink further or add context compression — that is a **separate** method.

### How to run (Gemma-4-E2B-it, `multi_turn_base`)

Same server command as NFC / IFFC / CD (no extra llama-server flags):

```bash
./llama.cpp/build/bin/llama-server --jinja -fa on \
  -hf bartowski/google_gemma-4-E2B-it-GGUF:Q4_K_M \
  --host 0.0.0.0 --port 8080 -ngl 99 -c 65536 --temp 0.0 \
  -np 4 -cb -ub 1024

uv run python -m benchmark.smoke_gst   # hermetic; add --live if the server is up

uv run python -m benchmark.run_all --model gemma4-e2b --only-multi-turn \
  --categories multi_turn_base --fc-mode gst \
  --notes "GST+NFC vs NFC 27% / IFFC 30.5% / CD 10.5%"
```

Already scored (do not overwrite): GST+NFC Run 9 = 61/200 = 30.5% (ties IFFC); IFFC+STATE Run 10 = 68/200 = 34.0% (beats IFFC/GST); BoN N=5 Run 11 = 73/200 = 36.5% (beats STATE). Details in `THESIS.md`.

### Pause / resume

Checkpoint/resume is the same as NFC / IFFC / CD (PR #12): each finished entry lands in `responses` / `multi_turn_timings` for the current `run_id`.

```bash
# Kill anytime. Continue the unfinished gst run:
uv run python -m benchmark.run_all --model gemma4-e2b --only-multi-turn \
  --categories multi_turn_base --fc-mode gst --resume
```

`--resume` selects the latest unfinished run for that `--model` whose notes record `fc_mode=gst`. Pin a row with `--resume-run-id N`. Generation-only: `python -m benchmark.generate_multi_turn --fc-mode gst --run-id N`.

### Implementation notes

- Module: `benchmark/gst.py`. Smoke: `uv run python -m benchmark.smoke_gst`.
- `_prepare_step_messages` injects the GST system prompt when `fc_mode=gst`; NFC/IFFC/CD paths are unchanged.
- `_build_chat_payload` still attaches `tools` / `tool_choice=auto` (GST is **not** in `FC_JSON_TEXT_MODES`).
- Missing `[GST]` tags do not fail the entry: `tool_calls` still execute. GST-only content (no `tool_calls`) ends the turn (ABSTAIN).
- Single-turn `generate_responses.py` is not switched by `--fc-mode gst`; use `--only-multi-turn` for this ablation.
- Run 9 stored `flags` empty for all 200 rows (no GST-block compliance bit). `responses` store executed func-strings only, not raw `[GST]` text.

### Sources

- He, Y., Dang, H., et al. "Improving Large Language Models Function Calling and Interpretability via Guided-Structured Templates." arXiv:2509.18076, 2025. https://arxiv.org/abs/2509.18076 (Appendix A.2 Detail template)

---

## Experiment: IFFC + harness STATE (`--fc-mode iffc-state`)

This is the **IFFC + verified working-memory** ablation for the BFCL multi-turn harness (METHODS §5 / PLAN #6). It is **not** GST (no 7-step curriculum; not on the NFC `tools` channel), **not** TagDispatch/CD GBNF, **not** LLM summarization, and **not** shell→GFS.

Scored on Gemma-4-E2B-it Q4_K_M `multi_turn_base` (Mac M4 Pro 48 GB, llama.cpp, `-np 4 -c 65536`): IFFC+STATE Run 10 **68/200 = 34.0%** (started `2026-09-14T02:46:12Z`, finished `2026-09-14T05:47:17Z`, ~181.1 min wall; 200/200 responses; 132 `eval_errors`). Beats IFFC Run 5 **61/200 = 30.5%** and GST Run 9 **61/200 = 30.5%** by +7 entries / +3.5 pp — the first method above the 30.5% plateau **on base Gemma**. Vs NFC Run 4 **54/200 = 27.0%** (+14); vs CD Run 8 **21/200 = 10.5%** (+47). The +7 vs IFFC is a **swap** (20 STATE-only, 13 IFFC-only), not the same 61 rows plus seven. Empty 41→31 (tied GST’s empty cut); state mismatch 60→**64** (did not improve); under-call ≈38→37. **On the SFT-v4 GGUF the same overlay is a scored negative:** Run 16 **26/200 = 13.0%** vs Run 15 plain IFFC **30/200 = 15.0%** (−4 / −2.0 pp). Full tables and interpretation: `THESIS.md`. Do **not** treat this as perfect state tracking. Do **not** BoN the SFT-v4, SFT-v5, or SFT-v6 checkpoint as the next win. Harness: [PR #16](https://github.com/sskarz/Thesis/pull/16).

### Contrast vs NFC / IFFC / GST / CD

| Mode | Channel | Constraint | What the model writes | Extra context |
|------|---------|------------|------------------------|---------------|
| NFC | `tools` + `tool_choice=auto` | llama-server lazy tools GBNF (baseline) | Native `tool_calls` | none |
| IFFC | no `tools` / `tool_choice` | none (prompt) | JSON list in assistant text | none |
| CD | no `tools` / `tool_choice` | pass-1 name GBNF, pass-2 `json_schema` | IFFC-shaped JSON, two HTTP passes | none |
| GST | **same as NFC** | **none extra** (prompt overlay) | `[GST]…[/GST]` then native `tool_calls` | model-written plan (unverified) |
| **iffc-state** | **same as IFFC** | **none extra** | IFFC JSON list in assistant text | harness-written `[STATE]` from **executed** tool results |

`iffc-state` keeps IFFC scoring plumbing: JSON-in-text parse → BFCL func-strings → `multi_turn_checker`. STATE is injected in **user** messages after execution; it is not parsed as a tool call.

Lesson from GST: extra model-written planning on NFC did not fix env state. This mode injects a **verified** compact snapshot instead. Run 10 confirms the empty-turn cut (41→31) but state mismatch did **not** fall (60→64) — the gain is not perfect state tracking.

No llama.cpp rebuild. No `grammar` / `json_schema` fields. No native `tools`.

### What the harness writes

After each executed tool result, `StateTracker` updates from the **real** `execute_multi_turn_func_call` observations (plus the func-string that was executed), then appends a short block to the IFFC user-role feedback:

```
[STATE]
cwd: /docs
files: notes.txt, todo.md
listing: notes.txt, todo.md
fields: engineOn=True; fuelLevel=12.5
last: cd(folder=docs)->cwd=docs; ls()->[notes.txt, todo.md]
[/STATE]
```

Fields are whatever BFCL backends already return (GFS `current_working_directory` / `current_directory_content`; Vehicle/Travel/Twitter dict keys that actually changed). No second LLM, no new executable backend.

History policy: keep the last **2** user/assistant+tool turns verbatim; drop older `Function execution results` dumps. The accumulated STATE on the latest observation replaces those dumps.

### How to run (Gemma-4-E2B-it, `multi_turn_base`)

Same server command as NFC / IFFC / GST / CD (no extra llama-server flags):

```bash
./llama.cpp/build/bin/llama-server --jinja -fa on \
  -hf bartowski/google_gemma-4-E2B-it-GGUF:Q4_K_M \
  --host 0.0.0.0 --port 8080 -ngl 99 -c 65536 --temp 0.0 \
  -np 4 -cb -ub 1024

uv run python -m benchmark.smoke_iffc_state   # hermetic; add --live if the server is up

uv run python -m benchmark.run_all --model gemma4-e2b --only-multi-turn \
  --categories multi_turn_base --fc-mode iffc-state \
  --notes "IFFC+STATE vs NFC 27% / IFFC 30.5% / GST 30.5% / CD 10.5%"
```

Already scored (do not overwrite): IFFC+STATE Run 10 = 68/200 = 34.0% (beats IFFC/GST; first method above 30.5% **on base Gemma**); BoN N=5 Run 11 = 73/200 = 36.5% (beats STATE by +2.5 pp at ~7× wall); SFT-v4 + IFFC+STATE Run 16 = 26/200 = 13.0% (**null/negative vs Run 15 15.0%** — do not overwrite; do **not** BoN this ckpt). SFT-v5 Run 17 = 23/200 = 11.5% (**scored regression vs Run 15** — do not BoN that ckpt either). SFT-v6 Run 18 = 26/200 = 13.0% (**scored miss vs Run 15 15.0%** — do not BoN that ckpt either). Details in `THESIS.md`.

### Pause / resume

Checkpoint/resume is the same as NFC / IFFC / CD / GST (PR #12): each finished entry lands in `responses` / `multi_turn_timings` for the current `run_id`.

```bash
# Kill anytime. Continue the unfinished iffc-state run:
uv run python -m benchmark.run_all --model gemma4-e2b --only-multi-turn \
  --categories multi_turn_base --fc-mode iffc-state --resume
```

`--resume` selects the latest unfinished run for that `--model` whose notes record `fc_mode=iffc-state`. Pin a row with `--resume-run-id N`. Generation-only: `python -m benchmark.generate_multi_turn --fc-mode iffc-state --run-id N`.

### Implementation notes

- Module: `benchmark/iffc_state.py`. Smoke: `uv run python -m benchmark.smoke_iffc_state`.
- `_prepare_step_messages` injects Appendix B IFFC + a short STATE-trust addendum when `fc_mode=iffc-state`; NFC / IFFC / GST / CD paths are unchanged.
- `_build_chat_payload` omits `tools` / `tool_choice` (`iffc-state` is in `FC_JSON_TEXT_MODES`, like IFFC/CD).
- `_append_execution_feedback` updates STATE from execution results, attaches `[STATE]`, then compresses dumps older than the last two turns.
- Single-turn `generate_responses.py` is not switched by `--fc-mode iffc-state`; use `--only-multi-turn` for this ablation.
- Run 10 stored 0/200 `responses.result` with a literal `[STATE]` string (STATE is harness conversation context, not stored in result blobs).

### Sources

- Kang et al. "ACON: Optimizing Context Compression for Long-horizon LLM Agents." arXiv:2510.00615, 2025. https://arxiv.org/abs/2510.00615
- Instruction-Followed Function Calling: arXiv:2608.22472 Appendix B (the IFFC channel this overlay sits on)

---

## Experiment: IFFC stop-on-repeat (`--fc-mode iffc-stop-repeat`)

This is the **harness force-stop** ablation on the IFFC JSON-in-text channel ([PR #54](https://github.com/sskarz/Thesis/pull/54)). After a tool-exec observation, if the proposed IFFC call set is a **subset of already-executed** calls, the harness force-stops the turn (`[]`, do not re-execute). It also breaks **consecutive identical** assistant call sets (still only after an exec observation, so a fresh BFCL user ask is not blocked). Baseline `--fc-mode iffc` is **unchanged**. **Not** NFC, **not** GST, **not** CD/GBNF, **not** STATE, **not** BoN, **not** a retrain.

Two measured variants on **BASE** Gemma-4-E2B-it Q4_K_M `multi_turn_base` (`-np 4 -c 65536`, `temp=0`, **200/200 responses** complete). Do **not** overwrite either `runs` row. Accuracy rows are authoritative; do **not** invent tok/s.

### Run 27 — ungated (scored miss vs IFFC)

Ungated observe-same (commit `5b49d7a`; Mac `run_id=27`, `scores.id=20`, overnight 2026-09-28): **54/200 = 27.0%**. Versus Run 5 IFFC **61/200 = 30.5%**: **−3.5 pp / −7 entries**. Versus BoN Run 11 **36.5%**: **−9.5 pp**. Run 4 NFC is also **54/200 = 27.0%** — **same accuracy level, not an identical method**. **Scored miss vs plain IFFC. Not a method win.**

**Not** smoke81 / v7 / v8 SFT / BoN. Full LoRA on failed v8 mixes is **not started**.

Stop flags: **37/200**. All 37 are among the 146 failures — **0 stopped and correct**. VehicleControlAPI state mismatch **22→33** vs Run 5.

A 10-ID smoke probe on **collapsed smoke81** passed the over-call gate (avg `api_calls` 21.3→6.3; `hit_step_limit` 4→0). That probe is **not** a `multi_turn_base` score. **Thursday lock on the ungated gate:** tradeoff for **loop-dominated cases only**; **not** a method improvement.

### Run 30 — tighten MIN=6 (first scored win vs IFFC on BASE)

**Codex CONFIRM_PASS.** Tighten @ commit `78c22a0` (PR #54 — gate `observe_same_call` on **MIN=6** assistant tool-turns per ask; `consecutive_identical` stays immediate). Mac `run_id=30`, `scores.id=21`, started `2026-09-28T11:48:01Z`. BASE `gemma4-e2b` **bartowski** Gemma-4-E2B-it Q4_K_M GGUF; `fc_mode=iffc-stop-repeat`. **`multi_turn_base` only:** **65/200 = 0.325**. **200/200 responses + 200/200 `multi_turn_timings`**. `entry_id` set **identical** to Run 5 and Run 27. Stop flags **37→3**. Relaunch used `--resume-run-id 30` after orphan; **N=200 complete**.

**Thursday one-liner:** BASE + iffc-stop-repeat (`observe_same` gated after 6 tool-turns) → **32.5%** `multi_turn_base` vs plain IFFC **30.5%** / ungated stop **27%**.

**Measured: 65/200 = 32.5%.** Versus Run 5 IFFC **61/200 = 30.5%**: **+2.0 percentage points (65−61 correct)** — **not** “+4pp”. Versus Run 27 ungated **54/200 = 27.0%**: **+5.5 pp**. Versus Run 10 IFFC+STATE **68/200 = 34.0%**: **−1.5 pp** (still below). Versus BoN Run 11 **73/200 = 36.5%**: **−4.0 pp** (BoN still best inference). **First scored win of tightened stop-repeat over plain IFFC on BASE.** **Celebrate / park SFT-v9 prepare: YES.** Do **not** claim best overall inference. Do **not** overwrite BoN **36.5%**, Run 5 IFFC **30.5%** (keep as baseline), or SFT-v4 **15.0%**.

**Caveats (must appear):**
1. Harness/decode **guardrail on BASE** — **not** an SFT/method win.
2. `runs.finished_at` still **NULL** — metrics 4/5 **soft bookkeeping**; accuracy is locked in `scores`.
3. Status JSON is **stale** — trust **DB + log**.
4. Relaunch used `--resume-run-id 30` after orphan; **N=200 complete**.

Stop flags: **3/200**, **all FAIL** (0 stopped and correct): `stop_on_repeat_observe_same_call:turn_1` ×1; `stop_on_repeat_consecutive_identical:turn_0` ×1; `stop_on_repeat_consecutive_identical:turn_4` ×1.

Avg `api_calls`: Run 5 **8.28** / Run 27 **7.92** / Run 30 **8.30**; avg turns **3.67** all three.

**Not** smoke81 / v7 / v8 / v9 SFT / BoN. SFT v9.2 10-ID smoke-gate **FAILed** (Run 36 avg `api_calls` **25.7**, `hit_step_limit` **6/10**) — **HOLD / park**; **not** a `multi_turn_base` %; **do not invent** a v9 BFCL %. Do **not** merge [PR #57](https://github.com/sskarz/Thesis/pull/57) (stays draft @`cde93a3`). **No BoN, no v7 resume, no full LoRA on failed mixes** as the reaction. Pre-gates (Thesis Bot; do not invent): A/B PASS — smoke81 Run 28 avg_api 7.8; base10 VC Run 29 soft healthy.

### Failure modes

State mismatch + empty/under-call dominate — **not** primarily unconstrained decode loops.

| Bucket | Run 5 IFFC | Run 27 ungated | **Run 30 tighten** |
|--------|------------|----------------|--------------------|
| total errors | 139 | 146 | **135** |
| state_mismatch (any API) | 60 | 75 | **60** (back to Run 5) |
| VehicleControlAPI state | 22 | 33 | **22** (recovered) |
| GorillaFileSystem state | 18 | 19 | 19 |
| TravelAPI state | 10 | 12 | 9 |
| empty response list | 41 | 40 | **38** |
| missing GT exec results | 38 | 31 | 37 |

Empty pattern Run 30: turn1×16, turn2×12, turn0×6, turn3×2, turn4×2.

Run 27 details (146/146): state mismatch **75**, empty **40**, missing GT exec **31**. VehicleControl **rose vs Run 5 (22→33)**. Examples: empty turn1 `multi_turn_base_121` / `_125` / `_126`; VehicleControl `multi_turn_base_51` / `_52` / `_53`.

Run 30 examples: fail VC state `multi_turn_base_52`, `_58`, `_63`, `_64`, `_65`; correct `multi_turn_base_100` (5 api / 2 turns), `_101`, `_102`, `_103`, `_105`.

Best inference remains BoN **36.5%** (base Gemma). Best IFFC remains Run 5 **30.5%** (baseline). Tightened stop-repeat Run 30 **32.5%** is a **scored win vs that IFFC baseline**, still **below STATE 34.0% and BoN 36.5%**. Run 42 `--fc-mode iffc-ooda-card` is now **higher Acc** (**71/200 = 35.5%**, +3.0 pp vs this row) with a `state_mismatch` gate **MISS** — see [IFFC OODA card](#experiment-iffc-ooda-card---fc-mode-iffc-ooda-card). Do **not** overwrite this `runs` row. Best SFT remains v4 IFFC **15.0%**. Details: `THESIS.md`. Do **not** overwrite Run 27 or Run 30. Do **not** make ungated `iffc-stop-repeat` the default on base.

**GAR framing (lit, not a new score):** Zhao et al., *Calibration is the Bottleneck*, [arXiv:2609.00949](https://arxiv.org/abs/2609.00949) (2026) — Acc ≤ GAR; SRI/CRI-like decode can **hurt Acc**. Ungated force-stop cut Acc **30.5% → 27.0%** (VehicleControl **22→33**). **Gated recover:** tighten MIN=6 recovered past IFFC (**27.0% → 32.5%**; VehicleControl **33→22**, back to Run 5). Evidence writeup: [PR #55](https://github.com/sskarz/Thesis/pull/55) (Run 27 miss) + this Run 30 add. **Tightened stop-repeat PASSES** (Run 30 **32.5%**). **SFT v9.2 10-ID smoke-gate FAIL** (Run 36 avg **25.7**, hit **6/10**) → [HOLD / park](#sft-v9--v91--v92--lever-2-action-quotas-10-id-smoke-gate-fail--hold). Next after advisor: Stage B pref pairs, **not** another quota remix. Do **not** invent tok/s. Do **not** invent a v9 BFCL %.

### How to run

Harness lives in [PR #54](https://github.com/sskarz/Thesis/pull/54) (`benchmark/stop_on_repeat.py`; tighten MIN=6 @ `78c22a0`). Serve **base** Gemma (not an SFT GGUF):

```bash
uv run python -m benchmark.run_all --model gemma4-e2b --only-multi-turn \
  --categories multi_turn_base --fc-mode iffc-stop-repeat \
  --notes "BASE gemma4-e2b iffc-stop-repeat vs Run5 IFFC 30.5%; NOT smoke81/v7/v8"
```

Already scored (do not overwrite): Run 27 ungated = 54/200 = 27.0%; Run 30 tighten = 65/200 = 0.325. Relaunch of Run 30 used `--resume-run-id 30` after orphan; **N=200 complete**. Do **not** invent a second Run 30 row. Run 42 is a **new** `--fc-mode iffc-ooda-card` run — **never resume Run 30**.

---

## Experiment: IFFC OODA card (`--fc-mode iffc-ooda-card`)

This is a **dedicated** `--fc-mode iffc-ooda-card` (PR [#61](https://github.com/sskarz/Thesis/pull/61) draft branch `cursor/iffc-ooda-card-e5c5` @ `1dea341`). **Not** a flag on `--fc-mode iffc-stop-repeat`. **Not** `--fc-mode iffc-state`. **Not** Guo et al. Specialized 4-pass OODA (METHODS [§26](#26-ooda-tool-typed-observe-orient-decide-act) paper method remains distinct). One-pass `[OODA]` card (`goal` / `slots` / `last` / `next`) on a **working copy** of messages; stacked on stop-repeat **MIN=6**; `next` is **advisory**. Baseline `--fc-mode iffc` and `--fc-mode iffc-stop-repeat` are **unchanged**. **Not** NFC, **not** GST, **not** CD/GBNF, **not** STATE, **not** BoN, **not** a retrain.

Harness code lives in PR #61 — **leave that PR alone**; this writeup is docs-only. Do **not** overwrite Run 30. Do **not** invent tok/s beyond verified TTFT. Do **not** invent a second score. Calib is **not** a BFCL accuracy claim. SFT still parked (best SFT v4 **15.0%**). No BoN.

### Calib Run 40 (10-ID, unscored) — PASS, not a BFCL %

Mac `run_id=40`; 10 responses; **no** `scores` row. Artifacts: `ooda_card_calib_10id.json` + `ooda_card_calib_status.md`. **PASS:** avg `api_calls` **6.1**, `hit_step_limit` **0/10**, `iffc_unparseable` **0/10**, `turn0_tool` **10/10**. Finished `2026-09-29T02:06:15.918626+00:00`. **Not** a `multi_turn_base` accuracy. Do **not** cite 10-ID as 35.5%.

### Orphan Run 41 — ignore

Mac `run_id=41`: **0 responses**, no score, no `finished_at`. Abort / ignore note only. Do **not** invent a Run 41 %.

### Run 42 — full-200 `multi_turn_base` (71/200 = 35.5%)

**NEW run** (never `--resume-run-id 30`). Mac `run_id=42`; BASE `gemma4-e2b` / slug `Gemma-4-E2B-it-Q4_K_M`; `fc_mode=iffc-ooda-card`; `--categories multi_turn_base` **only**; notes `BASE`, `-np4 -c65536`, `temp=0`. **200/200 responses.** Finished DB `2026-09-29T05:51:02.787870+00:00` = **2026-09-29 01:51 ET**.

**Measured: 71/200 = 35.5%** (accuracy **0.355**). Versus Run 30 tighten **65/200 = 32.5%**: **+6 correct / +3.0 pp**. Versus Run 5 IFFC **61/200 = 30.5%**: **+10 / +5.0 pp**. Versus Run 4 NFC **54/200 = 27.0%**: **+17 / +8.5 pp**. Versus Run 10 IFFC+STATE **68/200 = 34.0%**: **+3 / +1.5 pp** (above STATE). Versus BoN Run 11 **73/200 = 36.5%**: **−2 / −1.0 pp** (BoN still best inference). **Accuracy win vs Run 30 and vs IFFC / NFC; still below BoN.** Do **not** claim a clean gate sweep (see `state_mismatch` **MISS** below). Do **not** overwrite Run 30 / BoN / IFFC / SFT-v4.

#### `eval_errors` (Run 42)

Total error rows **129** (failed entries = 129 → 71 correct). Other = 0.

| Bucket | n | Gate |
|--------|--:|------|
| `state_mismatch` | **71** | **MISS** vs <60 |
| `empty_response` | **17** | — |
| `missing_gt_exec` | **41** | — |
| empty + missing GT | **58** | **MET** vs ≤75 |
| total `eval_errors` | **129** | — |

#### Pre-registered gates

| Gate | Verified | Status |
|------|----------|--------|
| Accuracy ≥65/200 | 71/200 | **MET** |
| `state_mismatch` <60 | 71 | **MISS** |
| empty + missing GT ≤75 | 17+41=58 | **MET** |
| Latency ≤1.25× Run 30 | TTFT mean **517.8 ms** vs Run 30 **630.5 ms** (SQL 517.7544 / 630.4518; ratio 0.82; 1.25× threshold 788.06 ms) | **MET** |

Do **not** invent generation tok/s. Call out the `state_mismatch` miss: 71 is **not** <60. Three of four pre-registered gates MET; the state gate did **not**.

#### N=200 scoreboard after Run 42

- Best inference: BoN **36.5%** (Run 11) — **still best**
- Run 42 iffc-ooda-card: **35.5%** (new **second** inference)
- Run 10 IFFC+STATE: **34.0%**
- Run 30 iffc-stop-repeat tighten: **32.5%**
- Run 5 IFFC: **30.5%**
- Best SFT: v4 **15.0%**

SFT still parked. No BoN on this channel. Calib Run 40 ≠ 35.5%. Run 30 untouched. Acc ranking of winners still holds after Run 46 PTC-A **FAIL** (13.5%) — BoN **36.5%** / this row **35.5%**.

### How to run

Harness lives in [PR #61](https://github.com/sskarz/Thesis/pull/61) (draft; `cursor/iffc-ooda-card-e5c5` @ `1dea341`). Serve **base** Gemma (not an SFT GGUF):

```bash
uv run python -m benchmark.run_all --model gemma4-e2b --only-multi-turn \
  --categories multi_turn_base --fc-mode iffc-ooda-card \
  --notes "BASE gemma4-e2b iffc-ooda-card vs Run30 32.5%; NOT iffc-state / NOT 4-pass OODA"
```

Already scored (do not overwrite): Run 40 calib 10-ID (unscored PASS); Run 41 orphan 0 responses (ignore); Run 42 full-200 = 71/200 = 0.355. Do **not** resume Run 30. Do **not** invent a second Run 42 row. Run 46 `--fc-mode iffc-ptc-a` is a **later** scored FAIL (**27/200 = 13.5%**) — **not** a replacement for this row; see [IFFC PTC-A](#experiment-iffc-ptc-a---fc-mode-iffc-ptc-a).

---

## Experiment: IFFC PTC-A (`--fc-mode iffc-ptc-a`)

This is a **dedicated** `--fc-mode iffc-ptc-a` (PR [#63](https://github.com/sskarz/Thesis/pull/63) draft branch `pr-63-ptc-a` @ `3569311`). **Live tool-name DFA; args free.** Inspired by Yu et al. PTC-Decoder (METHODS [§27](#27-ptc-decoder-plan-to-act-plus-tool-name-dfa)); **not** their Jetson/satellite paper score; **not** full Plan-to-Act+TC-Decoder; **not** `--fc-mode cd` two-pass; **not** `--fc-mode iffc-ooda-card`; **not** `--fc-mode iffc-state`; **not** BoN; **not** a retrain. Baseline `--fc-mode iffc` / `--fc-mode iffc-stop-repeat` / `--fc-mode iffc-ooda-card` are **unchanged**.

Harness code lives in PR #63 — **leave that PR alone**; this writeup is docs-only. Do **not** overwrite Run 30 / Run 42. Do **not** invent tok/s. Do **not** invent a second score. Calib is **not** a BFCL accuracy claim. SFT still parked (best SFT v4 **15.0%**). No BoN. **PTC-B HOLD.** Next path = **DPO** (β0.2 smoke100 10-ID Run 49 **PASS** health gate — **not** a BFCL %; DPO-full **scored** Run 51 **53/200 = 26.5%** — **not** a method win; [PR #57](https://github.com/sskarz/Thesis/pull/57) stays **draft**), **not** more PTC/OODA remix. See [DPO β0.2 smoke100 PASS](#dpo-β02-smoke100-10-id-pass-run-49--health-gate-not-a-bfcl-).

### Calib Run 45 (10-ID, unscored) — PASS, not a BFCL %

Mac `run_id=45`; 10 responses; **no** `scores` row. Notes: `CALIB UNSCORED iffc-ptc-a 10id BASE @ooda-worktree`; parent=BASE IFFC+ptc-a solo; ids=`multi_turn_base_{4,38,46,64,84,95,30,82,3,163}`. Artifacts: `ptc_a_calib_status.md` + `ptc_a_calib_10id.exit` **0**. **PASS:** avg `api_calls` **5.4**, `hit_step_limit` **0/10**, `iffc_unparseable` **0/10**, `turn0_tool` **10/10**. JSON also records `all_10_present: true`, `turn0_tool_ge_6: true`, `unparseable_le_1: true`, `no_total_collapse_avg_ge_4: true`. Finished `2026-09-29T07:32:17.570451+00:00`. **Not** a `multi_turn_base` accuracy. Calib did **not** predict full-200. Do **not** cite 10-ID as 13.5%.

### Run 46 — full-200 `multi_turn_base` (27/200 = 13.5%) — scored FAIL

**NEW run** (never `--resume-run-id 30` / `42`). Mac `run_id=46`; BASE `gemma4-e2b` / slug `Gemma-4-E2B-it-Q4_K_M`; `fc_mode=iffc-ptc-a`; `--categories multi_turn_base` **only**; notes `BASE @Thesis-ptc-a`, `-np4 -c65536`, `temp=0`. **200/200 responses** (`mode=multi`). Finished DB `2026-09-29T08:15:24.061478+00:00` = **2026-09-29 04:15:24 ET**. Started `2026-09-29T07:33:18.893471+00:00`.

**Measured: 27/200 = 13.5%** (accuracy **0.135**). Versus Run 42 OODA-card **71/200 = 35.5%**: **−44 correct / −22.0 pp**. Versus Run 30 tighten **65/200 = 32.5%**: **−38 / −19.0 pp**. Versus Run 5 IFFC **61/200 = 30.5%**: **−34 / −17.0 pp**. **Large Acc regression vs all three.** Near SFT-v4 **15.0%** in magnitude — **do not equate methods**. Do **not** overwrite Run 30 / Run 42 / BoN / IFFC / SFT-v4.

#### `eval_errors` (Run 46)

Total error rows **173** (failed entries = 173 → 27 correct).

| Bucket | n | Gate |
|--------|--:|------|
| `state_mismatch` | **102** | **MISS** vs ≤60 |
| `empty_response` | **17** | — |
| `missing_gt_exec` | **54** | — |
| empty + missing GT | **71** | **MET** vs ≤75 |
| total `eval_errors` | **173** | — |

Among state mismatches: VehicleControlAPI **25** / GorillaFileSystem **23** / TradingBot **18** / TravelAPI **17** (also TwitterAPI 7 / MessageAPI 7 / TicketAPI 5). Empty: turn0×15, turn1×1, turn4×1. Missing GT exec: turn0×33, turn1×12, turn2×5, turn3×1, turn4×3.

#### Pre-registered gates

| Gate | Verified | Status |
|------|----------|--------|
| Accuracy ≥71/200 (beat Run 42) | 27/200 | **MISS** |
| `state_mismatch` ≤60 | 102 | **MISS** |
| empty + missing GT ≤75 | 17+54=71 | **MET** |
| Latency ≤1.25× Run 30 | wall ~**42.1 min**; wall ratio vs Run 30 **0.234485**; predicted-ms ratio **0.133812** (SQL limits 13461.251 s / 38163240.86 predicted-ms) | **MET** |

Avg `api_calls` **7.62** (Run 30 **8.305** / Run 42 **8.72** / calib Run 45 **5.4**). Do **not** invent generation tok/s. Two of four pre-registered gates MET (empty+missing; latency); accuracy and `state_mismatch` did **not**.

#### Verdict

**PTC-A solo is a scored regression; bank and close.** PTC-B **HOLD**. Next path = **DPO** (β0.2 smoke100 10-ID Run 49 **PASS** health gate — **not** a BFCL %; DPO-full **scored** Run 51 **53/200 = 26.5%** — **not** a method win; [PR #57](https://github.com/sskarz/Thesis/pull/57) stays **draft**), **not** more PTC/OODA remix. No BoN. SFT still parked. Run 30 / Run 42 **untouched**. See [DPO β0.2 smoke100 PASS](#dpo-β02-smoke100-10-id-pass-run-49--health-gate-not-a-bfcl-).

#### N=200 scoreboard after Run 46

Acc ranking of winners **unchanged**:

- Best inference: BoN **36.5%** (Run 11) — **still best**
- Best non-BoN harness: Run 42 iffc-ooda-card **35.5%**
- Run 10 IFFC+STATE: **34.0%**
- Run 30 iffc-stop-repeat tighten: **32.5%**
- Run 5 IFFC: **30.5%**
- Run 46 iffc-ptc-a: **13.5%** — scored **FAIL / regression** (near SFT-v4 15% level — **do not equate methods**)
- Best SFT: v4 **15.0%**

SFT still parked. No BoN on this channel. Calib Run 45 ≠ 13.5%. Run 30 / Run 42 untouched. Do **not** invent scores.

### How to run

Harness lives in [PR #63](https://github.com/sskarz/Thesis/pull/63) (draft; `pr-63-ptc-a` @ `3569311`). Serve **base** Gemma (not an SFT GGUF):

```bash
uv run python -m benchmark.run_all --model gemma4-e2b --only-multi-turn \
  --categories multi_turn_base --fc-mode iffc-ptc-a \
  --notes "BASE gemma4-e2b iffc-ptc-a vs Run42 35.5%; NOT resume 30/42"
```

Already scored (do not overwrite): Run 45 calib 10-ID (unscored PASS); Run 46 full-200 = 27/200 = 0.135. Do **not** resume Run 30 or Run 42. Do **not** invent a second Run 46 row. Do **not** run PTC-B as a reaction. Do **not** BoN this channel.

---

## Experiment: IFFC Decision sidecar (`--fc-mode iffc-decision`)

A **second small model** (Decision / Choice; here `tev1:latest` on Ollama) ranks the next tool name on the IFFC JSON-in-text channel and may execute, soft-noop, hard-noop, or bypass. Hints are **non-persisted** working-copy injections. **Not** `--fc-mode iffc-ooda-card`. **Not** `--fc-mode iffc-stop-repeat`. **Not** `--fc-mode iffc-state`. **Not** BoN. **Not** a retrain.

### Runs 56/57 — 10-ID paired MF6 bank (both 0/10; not a full-200 Acc)

Paired probe on `multi_turn_base_{4,38,46,64,84,95,30,82,3,163}`: Run 56 treatment `iffc-decision` + tev1 **0/10**; Run 57 control plain `iffc` **0/10**. Sidecar was healthy (113/113 HTTP 200). The bank is a historical **fail slice** (also 0/10 on Run 5 IFFC 61/200 and Run 30 65/200) — **EVAL/CONTROL confound**. Parked, then Sanskar override → full-200 Run 58. **Do not invent Acc beyond 0/10.** **Do not** treat 0/10 as N=200.

### Run 58 — full-200 `multi_turn_base` (50/200 = 25.0%) — scored FAIL_METHOD

**Measured: 50/200 = 25.0%** (accuracy **0.25**). BASE `gemma4-e2b` Q4_K_M; `--fc-mode iffc-decision` + `tev1:latest`; **treatment-only** (no same-session plain-IFFC full-200 twin). Thesis `04348af`; probe `07c7be1`. Levers: `DESC=short` `CLAMP=relevant` `SOFT=soft_once` `ARGMAX=1` `HARD_BREAK=0`. Versus Run 42 OODA-card **71/200 = 35.5%**: **−21 / −10.5 pp**. Versus Run 30 tighten **65/200 = 32.5%**: **−15 / −7.5 pp**. Versus Run 5 IFFC **61/200 = 30.5%**: **−11 / −5.5 pp**. ID flips vs Run 42: lost 33 / gained 12 / net **−21**. **FAIL_METHOD / no lift.** Do **not** overwrite BoN / Run 42 / STATE / Run 30 / IFFC / SFT-v4.

Primary WHY: **`MODEL_WEAK_RANKING`** (turn-0 gold top-1 ≈ **38.8%** when Choice returned) with strong **`RECIPE_CONFOUND`** (no OODA / no stop-repeat on this arm) and **noop/empty residual**. Sidecar **alive** (HTTP 200 ≈ **92.7%** of Decision events; `timeout_warm` ≈ **7.3%**, ~174/200 turn-0 *first* calls timed out) — **not** a dead helper. Empty-response errors **48** vs Run 42 **17**. Gold in menu ≈ **97.7%**; gold∩dropped ≈ **2.3%** (not clamp-primary). Policies: execute **1436** / noop_end_turn **756** / bypass **261**.

| Class | R58 | R42 | R30 | R5 |
|-------|-----|-----|-----|-----|
| empty_response | **48** | 17 | 38 | 41 |
| missing_gt_exec | 34 | 41 | 37 | 38 |
| **Total fails** | **150** | 129 | 135 | 139 |

**Park Decision Acc** for tev1 @ these levers on full-200 BASE. Cross-run deltas are the evidence; do **not** over-claim a paired A/B.

### Recipe-align / compose (smoke PASS; Acc = Run 61 26.0% FAIL)

As shipped on the Run 58 pin, `iffc-decision` **cannot** share Run 42’s OODA-card + stop-repeat MIN=6 recipe (**NO**). Compose mode **`iffc-decision-ooda`** **landed** as Thesis commit **`50b743f`** (OODA source **`1dea341`**). STATUS `/tmp/sft_logs/codex_iffc_decision_ooda_compose_STATUS.md`: **smoke PASS**. Turn-0 Decision timeout hygiene landed Thesis **`ee4d877`** / probe **`98f1b56`** (STATUS smoke/hygiene **PASS**; Acc not invented from smoke). Full-200 compose Acc is **Run 61 52/200 = 26.0%** (FAIL vs R5–R42). **Do not** resume 30/42/51–61.

### Runs 59/60 — 10-ID paired trail (1/10 vs 0/10; not a full-200 Acc)

Same bank as Runs 56/57: Run 59 treatment `iffc-decision-ooda` + tev1 **1/10** (pass `_64`); Run 60 control `iffc-ooda-card` **0/10**. Thesis `ee4d877`; probe `98f1b56`. **Do not invent Acc beyond 1/10 vs 0/10.** **Do not** treat as N=200. Pointer to Run 61.

### Run 61 — full-200 `iffc-decision-ooda` (52/200 = 26.0%) — scored FAIL vs R5–R42

**Measured: 52/200 = 26.0%** (accuracy **0.26**; DB scores id=34). BASE `gemma4-e2b` Q4_K_M; `--fc-mode iffc-decision-ooda` + `tev1:latest`; **treatment-only** (no same-session control). Thesis `ee4d877`; probe `98f1b56`. Levers: `DESC=short` `CLAMP=relevant` `SOFT=soft_once` `ARGMAX=1` `HARD_BREAK=0`. Versus Run 42 OODA-card **71/200 = 35.5%**: **−19 / −9.5 pp**. Versus Run 30 tighten **65/200 = 32.5%**: **−13 / −6.5 pp**. Versus Run 5 IFFC **61/200 = 30.5%**: **−9 / −4.5 pp**. Versus Run 58 Decision **50/200 = 25.0%**: **+2 / +1.0 pp**. **FAIL** vs R5–R42. Acc lever stays weak. **Park Decision(+OODA) Acc** for tev1 @ these levers on full-200 BASE. Do **not** overwrite BoN / Run 42 / STATE / Run 30 / IFFC / SFT-v4. Do **not** invent tok/s (metrics still running at CONFIRM). Compose/hygiene smoke PASS — Acc is this row.

#### N=200 scoreboard after Run 61

- Best inference: BoN **36.5%** (Run 11) — **still best**
- Best non-BoN harness: Run 42 iffc-ooda-card **35.5%**
- Run 10 IFFC+STATE: **34.0%**
- Run 30 iffc-stop-repeat tighten: **32.5%**
- Run 5 IFFC: **30.5%**
- Run 51 DPO-full plain IFFC: **26.5%** — scored; **not** a method win
- Run 61 iffc-decision-ooda: **26.0%** — scored **FAIL vs R5–R42**
- Run 58 iffc-decision: **25.0%** — scored **FAIL / no lift**
- Best SFT: v4 **15.0%**
- Run 46 iffc-ptc-a: **13.5%** — scored **FAIL** (closed)

### How to run

Already scored (do **not** overwrite): Run 56/57 10-ID both 0/10; Run 58 full-200 = 50/200 = 0.25; Run 59/60 10-ID 1/10 vs 0/10; Run 61 full-200 = 52/200 = 0.26. Do **not** resume 30/42/51–61. Do **not** invent a second Run 58 or Run 61 row. Compose/hygiene smoke PASS is **not** an Acc claim.

---

## Experiment: Best-of-N majority voting (`--fc-mode iffc-state-bon`)

This is the **test-time compute / MV@N** ablation for the BFCL multi-turn harness (METHODS §4 / PLAN #2; Snell et al., [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)). It **rides on IFFC+STATE** so the 34.0% channel is not given back. It is **not** NFC, **not** GST, **not** TagDispatch/CD GBNF, and **not** a verifier / reward-model Best-of-N.

Scored on Gemma-4-E2B-it Q4_K_M `multi_turn_base` (Mac M4 Pro 48 GB, llama.cpp, `-np 4 -c 65536`): BoN N=5 Run 11 **73/200 = 36.5%** (started `2026-09-15T03:15:21Z`; **`finished_at` NULL** at verify time; 200/200 responses; 127 `eval_errors`; `inference_metrics=0` / `metrics_per_request=0` — eval scored, metrics may still be unfinished). Beats IFFC+STATE Run 10 **68/200 = 34.0%** by +5 entries / +2.5 pp. Vs IFFC/GST **61/200 = 30.5%** (+12); vs NFC Run 4 **54/200 = 27.0%** (+19); vs CD Run 8 **21/200 = 10.5%** (+52). The +5 vs STATE is a **swap** (17 BoN-only, 12 STATE-only), not the same 68 rows plus five. Empty 31→**36** (rose); state mismatch 64→**53** (cut); under-call 37→38 (flat). Wall generate ~21.3 h vs Run 10 ~3.0 h (**~7×**); avg `predicted_ms` ~1318 s vs ~150 s (~8.8×); avg `api_calls` 42.7 vs 8.4. Modest +2.5 pp at ~7× wall / ~N× decode — **not free**. Full tables and interpretation: `THESIS.md`. Harness: [PR #18](https://github.com/sskarz/Thesis/pull/18).

### Contrast vs NFC / IFFC / GST / CD / iffc-state

| Mode | Channel | Constraint | Sampling | Extra context |
|------|---------|------------|----------|---------------|
| NFC | `tools` + `tool_choice=auto` | llama-server lazy tools GBNF (baseline) | greedy `temperature=0.0` | none |
| IFFC | no `tools` / `tool_choice` | none (prompt) | greedy `temperature=0.0` | none |
| CD | no `tools` / `tool_choice` | pass-1 name GBNF, pass-2 `json_schema` | greedy `temperature=0.0` | none |
| GST | **same as NFC** | **none extra** (prompt overlay) | greedy `temperature=0.0` | model-written plan (unverified) |
| iffc-state | **same as IFFC** | **none extra** | greedy `temperature=0.0` | harness-written `[STATE]` |
| **iffc-state-bon** | **same as IFFC+STATE** | **none extra** | **MV@N at `temperature=0.7`** (default N=5) | same `[STATE]` as iffc-state |

Scoring is unchanged: IFFC JSON-in-text parse → BFCL func-strings → `multi_turn_checker`. Only the **executed winner** is stored in `responses.result`, like other modes.

### What the harness does (per step)

At each agent step, **before** `execute_multi_turn_func_call`:

1. Sample N completions on the IFFC+STATE transcript at `temperature=0.7` (request payload overrides the server `--temp 0.0` default).
2. Parse each with the IFFC JSON-in-text parser.
3. Majority-vote on the step's `(function_name, arguments)` tuples (a parallel list of calls is one ballot).
4. Execute the winner **once**, then update `[STATE]` and compress older dumps exactly as `iffc-state`.

Losing samples are **not** written to `chat_history`.

**Empty / unparseable:** parseable `[]` is **ABSTAIN** (a real ballot). Unparseable candidates are **excluded** from the vote. If every sample is unparseable, the first sample is kept so the existing IFFC retry path can fire.

**Tie-break (deterministic):** highest count; if still tied, prefer any non-empty tool-call key over ABSTAIN; if still tied, first-seen key in sample order.

Default **N=5**, configurable with `--bon-n`. **Latency ~N×** vs greedy IFFC+STATE is **measured** on Run 11: ~7× wall (~21.3 h generate vs Run 10 ~3.0 h) and ~8.8× avg `predicted_ms` for +2.5 pp. Do not treat N=5 as a free accuracy bump.

No llama.cpp rebuild. No `grammar` / `json_schema`. No native `tools`. NFC / IFFC / GST / CD / `iffc-state` paths are unchanged.

### How to run (Gemma-4-E2B-it, `multi_turn_base`)

Same server command as IFFC+STATE (no extra llama-server flags). Already scored on `multi_turn_base` (do not overwrite Run 11):

```bash
./llama.cpp/build/bin/llama-server --jinja -fa on \
  -hf bartowski/google_gemma-4-E2B-it-GGUF:Q4_K_M \
  --host 0.0.0.0 --port 8080 -ngl 99 -c 65536 --temp 0.0 \
  -np 4 -cb -ub 1024

uv run python -m benchmark.smoke_bon   # hermetic; add --live if the server is up

uv run python -m benchmark.run_all --model gemma4-e2b --only-multi-turn \
  --categories multi_turn_base --fc-mode iffc-state-bon \
  --notes "BoN N=5 on IFFC+STATE vs 34% / IFFC 30.5% / GST 30.5% / NFC 27% / CD 10.5%"
```

Optional `--bon-n N` (must be ≥ 1). Notes record `fc_mode=iffc-state-bon bon_n=N`.

Already scored (do not overwrite): BoN N=5 Run 11 = 73/200 = 36.5% (beats IFFC+STATE 34.0% by +2.5 pp at ~7× wall). Best inference-time method on the **base** Gemma stack. **Do not** BoN the SFT-v4 checkpoint as the next win (Run 16 STATE overlay already null/negative). **Do not** BoN the SFT-v5 checkpoint (Run 17 is a scored regression vs v4). **Do not** BoN the SFT-v6 checkpoint (Run 18 is a scored miss vs v4). Details in `THESIS.md`.

### Pause / resume

Checkpoint/resume is the same as other multi-turn modes (PR #12): each finished entry lands in `responses` / `multi_turn_timings` for the current `run_id`.

```bash
# Kill anytime. Continue the unfinished iffc-state-bon run:
uv run python -m benchmark.run_all --model gemma4-e2b --only-multi-turn \
  --categories multi_turn_base --fc-mode iffc-state-bon --resume
```

`--resume` selects the latest unfinished run for that `--model` whose notes record `fc_mode=iffc-state-bon`. It restores `bon_n` from those notes (mismatch with `--bon-n` is an error). Pin a row with `--resume-run-id N`. Generation-only: `python -m benchmark.generate_multi_turn --fc-mode iffc-state-bon --run-id N`.

### Implementation notes

- Module: `benchmark/bon.py`. Smoke: `uv run python -m benchmark.smoke_bon`.
- `_complete_chat_sync` / async path sample N times when `fc_mode=iffc-state-bon`, then vote; NFC / IFFC / GST / CD / `iffc-state` stay single-sample greedy.
- `_prepare_step_messages` injects Appendix B IFFC + STATE-trust addendum (`FC_IFFC_STATE_MODES`); `_append_execution_feedback` updates STATE from the **executed winner** only.
- `_build_chat_payload` omits `tools` / `tool_choice` (`iffc-state-bon` is in `FC_JSON_TEXT_MODES`).
- Single-turn `generate_responses.py` is not switched by `--fc-mode iffc-state-bon`; use `--only-multi-turn` for this ablation.

### Sources

- Snell, C., et al. "Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters." arXiv:2408.03314, 2024. https://arxiv.org/abs/2408.03314
- Wang, X., et al. "Self-Consistency Improves Chain of Thought Reasoning in Language Models." ICLR 2023, arXiv:2203.11171. https://arxiv.org/abs/2203.11171
- Instruction-Followed Function Calling: arXiv:2608.22472 Appendix B (the IFFC channel)
- IFFC+STATE: METHODS experiment `--fc-mode iffc-state` (the STATE overlay this vote sits on)

---

## Experiment: MLX SFT (Gemma-4-E2B, IFFC mix)

Training-time counterpart to the inference-time IFFC channel. **v1 and v2 both scored negative** on BFCL v4 `multi_turn_base`. Run 14 is a **DIAGNOSTIC** history-bookkeeping ablation on the same v2 GGUF (`--iffc-strip-empty-assistant`): **4/200 = 2.0%** — did **not** rescue the never-start collapse. **Do not present 2% as a method win.** **v4 (Run 15) is a scored recovery: 30/200 = 15.0%** — empty-`[]` largely fixed, still **below IFFC 30.5% and below NFC 27.0%**. **Do not present 15% as beating IFFC.** **Run 16 (IFFC+STATE on that same v4 GGUF) is a scored negative/null overlay: 26/200 = 13.0%** (−2.0 pp vs Run 15). STATE helps **base** Gemma, not this SFT-v4 checkpoint. **Do not propose BoN as the next win on this ckpt.** **v5 (Run 17) is a scored regression: 23/200 = 11.5%** (−3.5 pp / −7 vs Run 15; state_mismatch 109→133, under_call 59→44, empty 2→0, avg api_calls/entry 17.3→40.3). Hypothesis: synthetic complete turn-0 taught parallel call dumps. **Do not present 11.5% as a method win.** **No BoN on this ckpt; no more epochs on the v5 mix.** **v6 (Run 18) is a scored miss vs v4: 26/200 = 13.0%** (−2.0 pp / −4 vs Run 15; +1.5 pp / +3 vs Run 17; ties Run 16). Recipe ([PR #38](https://github.com/sskarz/Thesis/pull/38)): v4 empty/recovery + only `drop_state_inconsistent`; turn-0 underspec/synth **off**. Model key [PR #39](https://github.com/sskarz/Thesis/pull/39). Empty still 1 (`_43`); residual state_mismatch 118 + under_call 55; avg api_calls **13.34** (fixed v5’s 40.3 churn). **Do not present 13.0% as a method win.** **No BoN on this ckpt; no more epochs on the v6 mix.** v4 IFFC 15% remains best SFT recovery. **SFT path locked 2026-09-23** as the main path (inference wrappers secondary). Next training lever: SFT **v7** (planned, **not scored**). Inference-time baselines after Run 46: BoN 36.5% best **on base Gemma**; iffc-ooda-card **35.5%** still best non-BoN; IFFC+STATE 34.0%; Run 30 32.5%; IFFC 30.5%; Run 46 iffc-ptc-a **13.5%** scored **FAIL** (closed). Full audit: `THESIS.md`.

### Decision: SFT path locked (2026-09-23)

**TRAINING (SFT) is the locked main path.** Inference wrappers (BoN / STATE / GST / CD) are secondary, not the primary next bet. Best measured inference remains BoN **36.5%** (Run 11, base Gemma). Best measured SFT remains v4 IFFC **15.0%** (Run 15). v5 / v6 are **closed** as below v4 (Run 17 **11.5%** / Run 18 **13.0%**). xLAM Runs 19–20 are **negative external baselines** (NFC **0.0%**; `--fc-mode xlam` **3.0%**) — not Tier-1 wins.

**Lit review:** [`sskarz/thesis-vault` `📝 Notes/2026-09-23-SFT-training-lit-review.md`](https://github.com/sskarz/thesis-vault/blob/main/%F0%9F%93%9D%20Notes/2026-09-23-SFT-training-lit-review.md) (12-source synthesis for v7 planning; vault owns the deep note — may land in a parallel vault PR before it is on `main`). Do **not** paste the full review here.

**Tightened stop-repeat PASSES** (Run 30 **65/200 = 32.5%** vs IFFC **30.5%**). Run 42 `--fc-mode iffc-ooda-card` remains **scored** at **71/200 = 35.5%** (best non-BoN harness Acc) — see [IFFC OODA card](#experiment-iffc-ooda-card---fc-mode-iffc-ooda-card). Run 46 `--fc-mode iffc-ptc-a` is **scored FAIL** at **27/200 = 13.5%** (Acc regression vs Run 42 / 30 / 5; **bank and close**) — see [IFFC PTC-A](#experiment-iffc-ptc-a---fc-mode-iffc-ptc-a). Do **not** overwrite Run 30 / Run 42. **PTC-B HOLD.** SFT v9.2 10-ID smoke-gate **FAIL** (Run 36) → **HOLD / park** — see [SFT v9.2 smoke-gate FAIL](#sft-v9--v91--v92--lever-2-action-quotas-10-id-smoke-gate-fail--hold). Codex **`HOLD_sft_quota_remix`**. Next path = **DPO** (β0.2 smoke100 10-ID Run 49 **PASS** health gate — **not** a BFCL %; DPO-full **scored** Run 51 **53/200 = 26.5%** — **not** a method win), **not** more PTC/OODA remix and **not** another quota remix. [PR #57](https://github.com/sskarz/Thesis/pull/57) stays **draft** @`cde93a3` — **do not merge**. See [DPO β0.2 smoke100 PASS](#dpo-β02-smoke100-10-id-pass-run-49--health-gate-not-a-bfcl-). **No BoN, no v7 resume, no full LoRA on failed mixes, no invented v9 / Run 49 BFCL %.** Paper BFCL % ≠ our harness.

**Next intended method — SFT v7 (planned, not scored; do not invent results):**

1. MT-native verified trajs core (**≥50%**) — APIGen-MT-5k → IFFC JSON-in-text, assistant-only loss
2. Rebalance stop taxonomy — tool / clarify / final / irrelevance `[]` at **~12–15%** non-tool
3. Student-aware mid-difficulty filter (ToolACE-R-lite)
4. Optional post-SFT: Hammer masking + RPO/mDPO **only after** a healthy (non-collapsed, not below-IFFC) SFT

**Related options (logged 2026-09-24, both unscored; do not invent scores):**

- **Data Turnstile** ([§24](#24-data-turnstile-open-function-calling-data-generation)) — optional **open** data-gen complement to Hub-gated APIGen-MT for v7+ mixes (API specs → DAG-role multi-turn FC SFT). Paper BFCL is **single-turn**; τ² Telecom ≠ our `multi_turn_base`.
- **Critical-State RL** ([§25](#25-critical-state-rl-diagnosing-trainable-multi-turn-calls)) — optional **post-healthy-SFT** occurrence-local RL (nested sampling on the trainable turn). Distinct from blanket GRPO. **Do not** unlock GRPO as the main path.
- **MCP-GRANITE** ([§3a](#3a-tool-interface-granularity-mcp-granite)) — optional composites if the tool list is huge. **Unscored.** Cross-link only; see Post–Run27 ranking item 7.

**Related options (logged 2026-09-28, all unscored; paper numbers are theirs; do not invent `multi_turn_base` scores):**

- **OODA-Tool** ([§26](#26-ooda-tool-typed-observe-orient-decide-act)) — Guo et al. typed Observe→Orient→Decide→Act paper method remains **unscored** (prefer Joint **one-pass** over Specialized ~2.4×). A **separate** one-pass `[OODA]` card harness (`--fc-mode iffc-ooda-card`) is **scored** Run 42 **71/200 = 35.5%** — **not** 4-pass Specialized; **not** iffc-state. Cross-link [IFFC OODA card](#experiment-iffc-ooda-card---fc-mode-iffc-ooda-card).
- **PTC-Decoder** ([§27](#27-ptc-decoder-plan-to-act-plus-tool-name-dfa)) — Yu et al. paper remains **theirs**. Our live-name DFA harness (`--fc-mode iffc-ptc-a`) is **scored FAIL** Run 46 **27/200 = 13.5%** — **bank and close**. PTC-B **HOLD**. Cross-link [IFFC PTC-A](#experiment-iffc-ptc-a---fc-mode-iffc-ptc-a).
- **Looped LMs** ([§28](#28-looped-language-models-for-compositional-tool-calling)) — mostly **inference / test-time compute**. Needs a looped backbone we do not run.
- **PARL-MT / PROGRA** ([§29](#29-parl-mt-progress-awareness-for-multi-turn-function-calling)) — **training-time** progress-aware PAG-RL. Logged research option after healthy SFT; **do not** unlock GRPO as the main path.
- **HyperTool** ([§30](#30-hypertool-code-block-execution-granularity)) — **train + infer** code-block execution granularity. Complements MCP-GRANITE §3a. Not MCP-Universe ≡ BFCL.
- **RC-GRPO** ([§31](#31-rc-grpo-reward-conditioned-grpo)) — **training-time** reward-conditioned GRPO. RL-stabilization if we ever GRPO; **do not** unlock GRPO as the main path.

Planning with Sanskar next. Scaffolding (unscored): `sft/format_apigen_mt.py`, `sft/build_mix_v7.py`, `sft/mix_stats_v7.py`, registry key `gemma4-e2b-sft-iffc-v7`. Hub Accept on `Salesforce/APIGen-MT-5k` (CC-BY-NC) is required before download. No thin-filter-only ToolACE+xLAM remakes. No BoN on collapsed/below-IFFC SFT. No more epochs on failed mixes.

**v1 (Run 12):** Mac `benchmark.db` `run_id=12`: **1/200 = 0.5%** (`gemma4-e2b-sft-iffc`, `--fc-mode iffc`, fused Q4_K_M best val @ iter 1000, `val=0.206`, mlx-lm 0.32). Only win: `multi_turn_base_50`. 200 responses, 199 `eval_errors`, 200 `multi_turn_timings`. Started `2026-09-20T12:19:15Z`, finished `2026-09-20T12:33:00Z`. All 200 `flags` empty — **zero** `iffc_unparseable`.

**v2 filtered mix (Run 13):** Mac `benchmark.db` `run_id=13`: **1/200 = 0.5%** — **same collapse** as Run 12. `model_display` Gemma-4-E2B-it SFT-IFFC-v2 filtered-[] (Q4_K_M); `model_key` `gemma4-e2b-sft-iffc-v2`; notes `SFT-IFFC-v2 filtered nonterminal-[] LoRA best@2000 val0.211; mlx-lm 0.32; Q4_K_M; fc_mode=iffc categories=multi_turn_base vs Run12 0.5% / BoN 36.5%`. Only win: `multi_turn_base_50` (correct-set overlap identical with Run 12). 200 responses, 199 `eval_errors`, 200 `multi_turn_timings`; ~14 min wall. All flags empty — **zero** `iffc_unparseable`. [PR #24](https://github.com/sskarz/Thesis/pull/24) dropped 334 mid-trajectory empty-after-exec rows (non-terminal 334→0) and still kept 552 terminal empty-after-exec. The filter is **insufficient**. Figures: `benchmark/results/Gemma-4-E2B-it-SFT-IFFC-v2-Q4_K_M/figures`.

**DIAGNOSTIC strip trailing `[]` (Run 14):** Mac `benchmark.db` `run_id=14`: **4/200 = 2.0%** (correct=4, accuracy=0.02) — **same model as Run 13** (`gemma4-e2b-sft-iffc-v2`), `--fc-mode iffc --iffc-strip-empty-assistant`. Notes: `iffc_strip_empty_assistant=1` + DIAGNOSTIC phrase. Wins: `multi_turn_base_50`, `_161`, `_192`, `_193` (shared with Run 13: only `_50`; +3 new). vs Run 13 1/200 = 0.5%; vs BoN 36.5%; vs IFFC 30.5%. 200 responses, 196 `eval_errors`, 200 `multi_turn_timings`; ~10 min wall. All flags empty — **zero** `iffc_unparseable`. Primary buckets: empty_turn **160**, state_mismatch **28**, under_call **8**. Empty-turn-0 error rows **153** (unchanged vs Run 13 `first_empty` idx 0); any-empty failed entries ~160. Stripping trailing assistant `[]` from `chat_history` does **NOT** rescue the SFT v2 never-start collapse. Root cause remains the data/harness recipe (post-tool `[]` conditioning / turn-0 refusal), not merely keeping those tokens in history. **Do not present 2% as a method win.** Label as DIAGNOSTIC / history bookkeeping ablation. Figures: `benchmark/results/Gemma-4-E2B-it-SFT-IFFC-v2-Q4_K_M/iffc-strip-empty-assistant/figures`.

**v4 recovery mix (Run 15):** Mac `benchmark.db` `run_id=15`: **30/200 = 15.0%** — **scored recovery** from the 0.5–2% collapse, still **below IFFC and NFC**. `model_display` Gemma-4-E2B-it SFT-IFFC-v4 (Q4_K_M); `model_key` `gemma4-e2b-sft-iffc-v4`; fused GGUF `sft/fused/gemma-4-e2b-it-iffc-v4-best1600-Q4_K_M.gguf`; LoRA best@1600 (`val=0.196`); `--fc-mode iffc`. Notes: `fc_mode=iffc categories=multi_turn_base`. Started `2026-09-21T16:52:59Z`; **`finished_at` NULL**; **`inference_metrics=0`** (do not invent tok/s). 200 responses, 170 `eval_errors`, 200 `multi_turn_timings`. Recipe ([PR #31](https://github.com/sskarz/Thesis/pull/31)): v3 empty-after-exec drop + recovery, **plus** drop first-assistant-empty and clarifying empty-after-user `[]` targets. Deltas: **−12.0 pp / −24 vs NFC**; **−15.5 pp / −31 vs IFFC**; **−21.5 pp / −43 vs BoN**; **+14.5 pp / +29 vs SFT v2**. Overlap vs IFFC: both 23, only15 7 (`_64,_149,_154,_156,_159,_163,_166`), only5 38. Primary buckets: empty_turn **2** (1.2%; residual `_43` turn2, `_199` turn4), state_mismatch **109** (64%; VehicleControl 27 / Travel 26 / TradingBot 20 / GFS 19; example `_51`), under_call **59** (35%; turn0=35). `hit_step_limit:turn_0` on **57** timing rows. Empty-`[]` largely fixed; residual is state/under-call. **Do not present 15% as beating IFFC.**

**v4 + IFFC+STATE (Run 16, scored negative/null overlay):** Mac `benchmark.db` `run_id=16`: **26/200 = 13.0%** on the **same** SFT-v4 GGUF as Run 15 (`gemma4-e2b-sft-iffc-v4`, `sft/fused/gemma-4-e2b-it-iffc-v4-best1600-Q4_K_M.gguf`), `--fc-mode iffc-state`. Started `2026-09-21T18:39:35Z`, finished `2026-09-21T20:52:58Z` (~133.4 min wall). 200 responses, 174 `eval_errors`, 200 `multi_turn_timings`; **`inference_metrics` all zeros** (do not invent tok/s). Deltas: **−2.0 pp / −4 vs Run 15**; **−21.0 pp vs base STATE**; **−17.5 pp vs IFFC**; **−23.5 pp vs BoN**. Overlap vs Run 15: both 17, only16 9 (`_42,_47,_102,_104,_140,_158,_169,_179,_192`), lost 13 (`_52,_78,_107,_116,_133,_135,_149,_156,_159,_166,_182,_184,_190`). Primary buckets: empty_turn **3** (2%; `_175` turn0, `_185` turn0, `_108` turn4), state_mismatch **112** (64%; VehicleControl 25 / GFS 23 / Travel 21 / TradingBot 20 / Twitter 10 / Message 9 / Ticket 4), under_call **59** (34%; turn0=31). Sum `predicted_ms` ≈ 15.9 gen-min; avg turns 3.67; avg `api_calls` 14.83. STATE helps **base** Gemma, **not** this SFT-v4 checkpoint. **Do not propose BoN as the next win on this ckpt.** Recipe v5 (Run 17) is a **scored regression**. Recipe v6 (Run 18) is a **scored miss vs v4**.

**v5 mix (Run 17, scored regression vs v4):** Mac `benchmark.db` `run_id=17`: **23/200 = 11.5%**. `model_display` Gemma-4-E2B-it SFT-IFFC-v5 (Q4_K_M); `model_key` `gemma4-e2b-sft-iffc-v5`; fused GGUF `sft/fused/gemma-4-e2b-it-iffc-v5-best1800-Q4_K_M.gguf`; LoRA best@1800 (`val=0.147`); `--fc-mode iffc`. Recipe ([PR #35](https://github.com/sskarz/Thesis/pull/35)): keep v4 empty-drops + recovery-after-`[]`; drop underspecified turn-0; synthetic complete first-episode; drop state-inconsistent continue-after-exec. Train ~29850 after prepare. Started `2026-09-21T23:00:31Z`; **`finished_at` NULL**; **`inference_metrics=0`** (do not invent tok/s). 200 responses, 177 `eval_errors`, 200 `multi_turn_timings`. Deltas: **−3.5 pp / −7 vs Run 15**; **−1.5 pp vs Run 16**; **−15.5 pp vs NFC**; **−19.0 pp vs IFFC**; **−25.0 pp vs BoN**. Overlap vs Run 15: both 18, only17 5 (`_12,_21,_42,_161,_169`), lost 12 (`_105,_106,_114,_116,_129,_135,_136,_142,_156,_159,_190,_191`). Primary buckets: empty_turn **0** (0%), state_mismatch **133** (~75%; Travel 32 / TradingBot 30 / VehicleControl 28 / GFS 23 / Twitter 8 / Message 8 / Ticket 4), under_call **44** (~25%; turn0=27). Sum `predicted_ms` ≈ 51.9 gen-min; avg turns 3.67; avg `api_calls` 40.27. Empty-`[]` fully suppressed; residual is state/under-call. Lost corrects heavily TradingBot. Hypothesis: **synthetic complete turn-0** taught parallel call dumps that desync observe→act APIs; underspecified-turn0 drop helped only modestly; **state-inconsistent drop is the one keep.** **Do not present 11.5% as a method win.** **No BoN on this ckpt; no more epochs on the v5 mix.** Code: `sft/curriculum_v5.py`, `sft/filter_empty_after_exec.py`. See `sft/README.md`.

**v6 mix (Run 18, scored miss vs v4):** Mac `benchmark.db` `run_id=18`: **26/200 = 13.0%**. `model_display` Gemma-4-E2B-it SFT-IFFC-v6 (Q4_K_M); `model_key` `gemma4-e2b-sft-iffc-v6`; fused GGUF `sft/fused/gemma-4-e2b-it-iffc-v6-best1800-Q4_K_M.gguf`; LoRA best@1800 (`val=0.142`, `adapters-v6-recovery`); `--fc-mode iffc`. Recipe ([PR #38](https://github.com/sskarz/Thesis/pull/38)): v4 empty/recovery + only `drop_state_inconsistent`; turn-0 underspec drop and synthetic complete turn-0 **off**. Model key [PR #39](https://github.com/sskarz/Thesis/pull/39). Started `2026-09-22T12:50:47Z`, finished `2026-09-22T14:01:37Z` (~**70.8 min** wall). 200 responses, 174 `eval_errors`, 200 `multi_turn_timings`; inference_metrics present (gen_tps_mean ≈ **103.1**; prompt_tps_mean ≈ 741.4; peak_rss ≈ 9170 MB). Deltas: **−2.0 pp / −4 vs Run 15**; **+1.5 pp / +3 vs Run 17**; **0.0 pp vs Run 16**; **−14.0 pp vs NFC**; **−17.5 pp vs IFFC**; **−23.5 pp vs BoN**. Overlap vs Run 15: both 18, only18 8 (`_42,_47,_102,_144,_161,_169,_186,_192`), lost 12 (`_52,_64,_78,_105,_106,_107,_109,_114,_129,_136,_159,_166` — mostly TradingBot/VehicleControl state_mismatch; turn-0 under_call `_105,_114,_159,_166`). Overlap vs Run 17: both 15, only18 11, only17 8. Primary buckets: empty_turn **1** (`_43` turn 2), state_mismatch **118** (~68%; VehicleControl 31 / TradingBot 26 / GFS 22 / Travel 19 / Twitter 10 / Ticket 5 / Message 5), under_call **55** (~32%; turn0=38). Avg turns 3.67; avg `api_calls` **13.34** (vs Run 15 17.3, Run 17 40.3). **Do not present 13.0% as a method win.** **No BoN on this ckpt; no more epochs on the v6 mix.** Thin filter ablation on the same family is likely exhausted. Code: `sft/filter_empty_after_exec.py` (`RECIPE_V6` / `recipe_v5_flags`). See `sft/README.md`. SFT v9.2 10-ID smoke-gate is a **FAIL → HOLD** (Run 36; **not** a v6 follow-on `multi_turn_base` score) — see [SFT v9.2 smoke-gate FAIL](#sft-v9--v91--v92--lever-2-action-quotas-10-id-smoke-gate-fail--hold).

**Chat format:** the same JSON-in-text IFFC shape as `--fc-mode iffc` (`benchmark/iffc.py` Appendix B). Assistant turns are a JSON list of `{"function_name", "parameters"}` (or `[]`). Tool/observation turns are user-role execution messages. Native `tool_calls` are **not** mixed into this adapter dataset.

**Mix (v1, locked for Run 12):**

- **All** of [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) (~11,300 multi-turn conversations)
- Stratified **25,000** rows from [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k), oversampling parallel / multi-answer rows (`n_tools_in_gold ≥ 2` taken first)

**Mix (v2, Run 13):** same conversation-level splits as v1; unroll drops mid-trajectory empty-after-exec `[]` ([PR #24](https://github.com/sskarz/Thesis/pull/24): non-terminal 334→0; 552 terminal empty-after-exec kept). **Scored negative** at the same 0.5% — filter insufficient.

**Mix (v4, Run 15):** same conversation-level splits; unroll is v3 (drop **all** empty-after-exec including terminal + recovery curriculum) **plus** drop first-assistant-empty and clarifying empty-after-user `[]` targets ([PR #31](https://github.com/sskarz/Thesis/pull/31)). LoRA best@1600 (`val=0.196`). **Scored recovery** at 15.0% — still below IFFC/NFC. Run 16 overlay on this mix is **13.0%** (null/negative).

**Mix (v5, Run 17):** same conversation-level splits; unroll is v4 **plus** turn-0 call-set completeness and env-state consistency (`sft/curriculum_v5.py`). ~29850 train rows after prepare. LoRA best@1800 (`val=0.147`). **Scored regression** at 11.5% vs v4 15.0%. **Do not present as a method win.**

**Mix (v6, Run 18):** same conversation-level splits; unroll is v4 **plus only** `drop_state_inconsistent` (no underspecified drop, no complete-turn0 synth). LoRA best@1800 (`val=0.142`). **Scored miss vs v4** at 13.0% (Run 15 15.0%). **Do not present as a method win.** Not a v4 relabel.

**Skipped for v1:** APIGen-MT-5k (NC), Glaive, Hermes (Hermes remains an optional backup in docs only if a Hub source disappears).

**Splits:** by conversation/example id, never by turn; `seed=42`; stratify on source, `is_multi_turn`, `n_tools_in_gold` (1 vs ≥2); ~90% / ~5% / ~5% → about Train 32.7k / Val 1.8k / Corpus-test 1.8k. **Thesis test is BFCL v4 `multi_turn_base` only** — BFCL prompts and past scored responses are never in train/val/corpus_test.

**Base model:** Gemma-4-E2B-it (`google/gemma-4-e2b-it`), the instruct checkpoint behind `benchmark/config.py` `gemma4-e2b` / `bartowski/google_gemma-4-E2B-it-GGUF`. There is no mlx-lm id in the registry; convert on Apple Silicon with `mlx_lm.convert` (see `sft/README.md`). Do not LoRA-train the mlx-vlm `mlx-community/gemma-4-e2b-it-4bit` blob.

**Code:** `sft/` (download → format → split → `mlx_lm.lora` with `--mask-prompt`, LoRA rank 16 default / 16–32 band). Conversations are unrolled so every assistant turn receives loss. Artifacts under `sft/data/` are gitignored. CI does not run overnight training.

### Run 12 / Run 13 results (both scored negative); Run 14 DIAGNOSTIC (not a method win); Run 15 scored recovery (still below IFFC); Run 16 STATE overlay (null/negative); Run 17 v5 (scored regression vs v4); Run 18 v6 (scored miss vs v4)

| Method | Run | Score |
|--------|-----|-------|
| NFC | 4 | 54/200 = 27.0% |
| IFFC (base) | 5 | 61/200 = 30.5% |
| CD | 8 | 21/200 = 10.5% |
| GST | 9 | 61/200 = 30.5% |
| IFFC+STATE (base) | 10 | 68/200 = 34.0% |
| BoN N=5 | 11 | 73/200 = 36.5% |
| SFT-IFFC LoRA best@1000 | 12 | 1/200 = 0.5% |
| SFT-IFFC-v2 filtered-[] LoRA best@2000 | 13 | 1/200 = 0.5% |
| SFT-IFFC-v2 + strip trailing `[]` (**DIAGNOSTIC**) | 14 | 4/200 = 2.0% |
| SFT-IFFC-v4 LoRA best@1600 | 15 | 30/200 = 15.0% |
| SFT-IFFC-v4 + IFFC+STATE | 16 | 26/200 = 13.0% |
| SFT-IFFC-v5 LoRA best@1800 | 17 | 23/200 = 11.5% |
| **SFT-IFFC-v6 LoRA best@1800** | **18** | **26/200 = 13.0%** |

Order: CD 10.5% < SFT-v5 IFFC 11.5% < **SFT-v6 IFFC 13.0%** = **SFT-v4+STATE 13.0%** < **SFT-v4 IFFC 15.0%** < NFC 27.0% < IFFC 30.5% < base STATE 34.0% < **BoN 36.5%**. **Do not present 15% as beating IFFC.** **Do not present 13% as a STATE win on this ckpt.** **Do not present 11.5% as a method win.** **Do not present v6 13.0% as a method win.**

**Primary BFCL bucket per failed entry** (n=199 for Runs 12–13; n=196 Run 14; n=170 Run 15; n=174 Run 16; n=177 Run 17; n=174 Run 18; do not conflate with the audit):

| Bucket | Run 12 | Run 13 | Run 14 | Run 15 | Run 16 | Run 17 | Run 18 |
|--------|--------|--------|--------|--------|--------|--------|--------|
| `empty_turn` | **101** | **167** | 160 | **2** | **3** | **0** | **1** |
| `state_mismatch` | **63** | **25** | 28 | **109** | **112** | **133** | **118** |
| `under_call` | **35** | **7** | 8 | **59** | **59** | **44** | **55** |

**Thesis Bot audit** (different metric — do not conflate with the primary buckets): empty turns 13.2% (Run 5 IFFC) → **69.9%** (Run 12) → still **~70%** on Run 13 (522/734 error rows). Early-stop trajectories 56/200 (IFFC) → **158/200** (Run 12) → **67/200** (Run 13). Early-stop fell 158→67 but empty-labeled primary fails rose 101→167 — more empty turns overall as first-fail labels, **not a win**. Both runs support **early abstain dominates**.

**Audit probes** (Thesis Bot — not primary buckets; do not claim the filter fixed multi-turn):

1. **Misleading short probe vs BFCL IFFC shape.** A short live multi-turn that returned `view_messages_sent` on v2 bundled the next ask into the tool-result user message. Under true BFCL IFFC shape (exec results → model emits `[]` to end the within-turn loop → a separate next user turn), the same v2 GGUF returns `[]` on turn 2 for `multi_turn_base_101` (matches scored Run 13). Only win remains `multi_turn_base_50`.
2. **Self-conditioning on `[]`.** Controlled probes, 3-tool IFFC system prompt: (A) BFCL-shaped history with intermediate `[]` → turn 2 `[]`; (B) next-ask bundled into results, no `[]` in history → `view_messages_sent`; (C) no `[]` assistant turns → works; (D) prior `[]` in history even if last user bundles the ask → still `[]`. Once SFT emits `[]`, later turns stay empty.
3. **Pattern shift** (C = call, E = empty `[]`). Run 5 mostly `CCC`/`CC`/`CCCC`; Run 12 `CE`/`CEE`/`CEEE` (start then stop); Run 13 `EE`/`EEEE`/`EEE` dominant (turn-0 refusal). Filter cut mid-trajectory `[]` targets but worsened leading empties (`first_empty` idx 0: **40→153**).
4. **Not tool-catalog size alone.** Short vs full tools both fail BFCL-shaped turn 2; turn-0-empty vs turn-0-call tool counts similar (~27–30).

### Root cause: ToolACE `[]` pathology, not parser

Live single-step IFFC probe on the served GGUF emits valid `[{"function_name","parameters"}]` — format + chat template OK. Zero `iffc_unparseable` on all 200 Run 12 responses, all 200 Run 13 responses, and all 200 Run 14 responses.

The mlx train split has **5520** assistant `[]` targets, **all from ToolACE** (32.5% of ToolACE assistant turns; xLAM 0%). After a `"Function execution results"` user message, **78.1%** of assistant targets are `[]` (2121 empty vs 596 continue). That teaches **stop-after-tools**. BFCL multi-turn needs continued calls on later user turns. Live probe: after stock→send, `"show messages sent"` → `[]`. Same entries IFFC wins (e.g. `multi_turn_base_101`) SFT gets turn0+turn1 right then `[]` on turn2 (`view_messages_sent` missing). Secondary: when it does call, some bad order (e.g. `mv` before `mkdir` on `multi_turn_base_0`) then empties.

PR #24 dropped 334 of those mid-trajectory empty-after-exec rows (non-terminal 334→0) and kept 552 terminal empty-after-exec. Run 13 scored the same **1/200 = 0.5%**. The filter is **insufficient**.

### DIAGNOSTIC: `--iffc-strip-empty-assistant` (Run 14)

Harness-only history bookkeeping on the Run 13 GGUF: omit trailing assistant `[]` from `chat_history` before the next BFCL user turn. **Not** a training method; **not** a new `fc-mode`. Mac `run_id=14`: **4/200 = 2.0%**. Primary buckets empty_turn **160** / state_mismatch **28** / under_call **8**. Empty-turn-0 **153** unchanged vs Run 13. The never-start collapse remains. **Do not present 2% as a method win.** Full writeup: `THESIS.md` Run 14 subsection.

### Run 15: SFT-IFFC-v4 best@1600 (scored recovery, still below IFFC)

Mac `run_id=15`, `--model gemma4-e2b-sft-iffc-v4 --fc-mode iffc`. Recipe is [PR #31](https://github.com/sskarz/Thesis/pull/31) v4 unroll (v3 empty-after-exec + recovery, plus drop first-assistant-empty and clarifying `[]`); LoRA best@1600 `val=0.196`; fused `sft/fused/gemma-4-e2b-it-iffc-v4-best1600-Q4_K_M.gguf`. **30/200 = 15.0%**. Empty_turn **2** / state_mismatch **109** / under_call **59**. Residual empties `_43` turn2 and `_199` turn4. VehicleControl state example `_51`. Overlap vs IFFC: both 23 / only15 7 / only5 38. `hit_step_limit:turn_0` on 57 rows. `finished_at` NULL; `inference_metrics=0` — do **not** invent tok/s. **Scored recovery from collapse, still below IFFC 30.5% and NFC 27.0%.** Full writeup: `THESIS.md` Run 15 subsection.

### Run 16: SFT-IFFC-v4 + IFFC+STATE (scored negative/null overlay)

Mac `run_id=16`, `--model gemma4-e2b-sft-iffc-v4 --fc-mode iffc-state`. **Same GGUF as Run 15.** **26/200 = 13.0%** (−4 / −2.0 pp vs Run 15). Empty_turn **3** / state_mismatch **112** / under_call **59**. Residual empties `_175` turn0, `_185` turn0, `_108` turn4. Overlap vs Run 15: both 17 / only16 9 / lost 13. Wall ~133.4 min; `inference_metrics` all zeros — do **not** invent tok/s. STATE helped base Gemma (Run 10 +3.5 pp) and is **null/negative on SFT-v4**. **Do not propose BoN as the next win on this ckpt.** Recipe v5 (Run 17) is a **scored regression** at 11.5%. Recipe v6 (Run 18) is a **scored miss vs v4** at 13.0%. Full writeup: `THESIS.md` Run 16 subsection.

### Run 17: SFT-IFFC-v5 best@1800 (scored regression vs v4)

Mac `run_id=17`, `--model gemma4-e2b-sft-iffc-v5 --fc-mode iffc`. Recipe is [PR #35](https://github.com/sskarz/Thesis/pull/35) v5 unroll (keep v4 empty-recovery; drop underspecified turn-0; synthetic complete first-episode; drop state-inconsistent continue-after-exec); LoRA best@1800 `val=0.147`; fused `sft/fused/gemma-4-e2b-it-iffc-v5-best1800-Q4_K_M.gguf`; ~29850 train rows. **23/200 = 11.5%** (−7 / −3.5 pp vs Run 15). Empty_turn **0** / state_mismatch **133** / under_call **44**. Overlap vs Run 15: both 18 / only17 5 (`_12,_21,_42,_161,_169`) / lost 12 (`_105,_106,_114,_116,_129,_135,_136,_142,_156,_159,_190,_191`). Sum `predicted_ms` ≈ 51.9 gen-min; avg `api_calls` 40.27. `finished_at` NULL; `inference_metrics=0` — do **not** invent tok/s. Lost corrects heavily TradingBot. Hypothesis: v5 **synthetic complete turn-0** taught parallel call dumps that desync observe→act APIs; **state-inconsistent drop is the one keep.** **Scored regression vs v4 recovery. Do not present as a method win. No BoN / no more epochs on this mix.** Recipe v6 (Run 18) is a **scored miss vs v4** at 13.0%. Full writeup: `THESIS.md` Run 17 subsection.

### Run 18: SFT-IFFC-v6 best@1800 (scored miss vs v4)

Mac `run_id=18`, `--model gemma4-e2b-sft-iffc-v6 --fc-mode iffc`. Recipe is [PR #38](https://github.com/sskarz/Thesis/pull/38) v6 unroll (v4 empty/recovery + only `drop_state_inconsistent`; turn-0 underspec/synth **off**); LoRA best@1800 `val=0.142`; fused `sft/fused/gemma-4-e2b-it-iffc-v6-best1800-Q4_K_M.gguf`; model key [PR #39](https://github.com/sskarz/Thesis/pull/39). **26/200 = 13.0%** (−4 / −2.0 pp vs Run 15; +3 / +1.5 pp vs Run 17). Empty_turn **1** (`_43` turn 2) / state_mismatch **118** / under_call **55** (turn0=38). Overlap vs Run 15: both 18 / only18 8 (`_42,_47,_102,_144,_161,_169,_186,_192`) / lost 12 (`_52,_64,_78,_105,_106,_107,_109,_114,_129,_136,_159,_166`). Overlap vs Run 17: both 15 / only18 11 / only17 8. Wall ~70.8 min; avg `api_calls` **13.34** (fixed v5’s 40.3); gen_tps_mean ≈ 103.1. Lost vs v4 mostly TradingBot/VehicleControl state_mismatch plus turn-0 under_call `_105,_114,_159,_166`. **Scored miss vs v4 recovery. Do not present as a method win. No BoN / no more epochs on this mix.** Next SFT lever is **v7** (planned, **not scored**; not another thin filter on the same family). Full writeup: `THESIS.md` Run 18 subsection.

### SFT v9 / v9.1 / v9.2 — lever-2 action quotas (10-ID smoke-gate FAIL → HOLD)

**Codex CONFIRM: `HOLD_sft_quota_remix`.** Formal **plain-IFFC 10-ID** smoke-gate **FAIL** on SFT v9.2 (Mac `run_id=36`). Full LoRA **never started**. **HOLD / park.** This is **not** a BFCL `multi_turn_base` accuracy. **Do not invent** a v9 / v9.1 / v9.2 BFCL %. Thursday banked stop-repeat remains **Run 30** BASE iffc-stop-repeat **65/200 = 32.5%** (harness, [PR #54](https://github.com/sskarz/Thesis/pull/54) @`7ad7ff3`) — already logged; do **not** overwrite; do **not** reframe as SFT. Run 42 `--fc-mode iffc-ooda-card` is a **later** scored harness row (**71/200 = 35.5%**; still below BoN; `state_mismatch` gate **MISS**) — also **not** SFT.

Worktree: Thesis-sft-v9 [PR #57](https://github.com/sskarz/Thesis/pull/57) **draft** @`cde93a3` (branch `cursor/sft-v9-lever2-action-quotas`). **Do not merge PR #57.** Tweak vs v9: shrink `V9_ACTION_MT_TOOL_CHAIN_ADVANCE` band `(0.40,0.50,0.35,0.55)` → `(0.30,0.38,0.28,0.42)`. Kept from v9.1: `V9_ACTION_FINAL_AFTER_OBS` `(0.20,0.24,0.18,0.26)`. Model key (smoke only): `gemma4-e2b-sft-iffc-v9-smoke92`.

**Smoke@100 (healthy; not a score):** val **1.303 → 0.645**; train@100 0.774; finite / no NaN; exit 0. Healthy loss does **not** pass the formal 10-ID gate.

**Formal 10-ID gate (Run 36) — FAIL** vs gate `avg api ≤8.3`, `hit_step_limit ≤2/10`, `iffc_unparseable ≤1/10`, `turn0_tool ≥6/10`:

| Metric | Gate | **Run 36 (v9.2)** | Run 33 (v9.1) | Run 31 (v9) |
|--------|------|-------------------|---------------|-------------|
| avg `api_calls` | ≤8.3 | **25.7** | 17.6 | 22.5 |
| `hit_step_limit` | ≤2/10 | **6/10** | 5/10 | 5/10 |
| `iffc_unparseable` | ≤1/10 | 0/10 | 0/10 | 1/10 |
| `turn0_tool` | ≥6/10 | 10/10 | 10/10 | 10/10 |

Worse than Run 33 v9.1 (**17.6** / **5/10**) and Run 31 v9 (**22.5** / **5/10**). Loopers (calls): `_4=60`, `_82=43`, `_84=40`, `_30=40`, `_3=23`, `_163=22`. Two sequential quota remixes failed the step-limit gate; the second **regressed** avg and hits.

**Do not cite Run 35** — **INVALID** (server died mid-probe). VC sidecar **Run 37** is diagnostic only (same-call `_52/_74/_83`); **not** folded into the /10 denominator.

**Diagnosis (Codex):** primary is identical-call / over-call looping after observations (**not** under-call). Advance-band shrink renormalized toward start/ask/refuse/final but did **not** teach stop; fewer advance golds may have weakened useful multi-step progress.

**Next:** no third remix, no rebuild/resume/fuse/probe another quota variant. Next path = **DPO** — β0.2 smoke100 10-ID (Run 49) is a **PASS** health gate (**not** a BFCL %); DPO-full **scored** Run 51 **53/200 = 26.5%** on plain IFFC (**not** a method win) — **not** another quota remix, **not** more PTC/OODA remix. Hard bans: no BoN, no v7 resume, no full LoRA on failed mixes, no invented v9 / Run 49 BFCL %, no iffc-stop-repeat as an SFT fix, no kitchen-sink multi-tweak, no PTC-B, no BoN on PTC-A. Scoreboard BFCL N=200 winners **unchanged**: BoN **36.5%** / Run 42 iffc-ooda-card **35.5%** / STATE **34.0%** / Run 30 **32.5%** / IFFC **30.5%** / SFT-v4 **15.0%**. Later scored: Run 51 DPO-full **26.5%** (not a win); Run 61 Decision+OODA **26.0% FAIL**; Run 58 Decision **25.0% FAIL**; Run 46 iffc-ptc-a **13.5%** (FAIL / closed). Run 30 / Run 42 **untouched**. Acc ranking of winners **unchanged**. See [DPO β0.2 smoke100 PASS](#dpo-β02-smoke100-10-id-pass-run-49--health-gate-not-a-bfcl-).

### DPO β0.2 smoke100 10-ID PASS (Run 49) — health gate, not a BFCL %

**PASS** health gate. Formal **plain-IFFC 10-ID** on DPO smoke100 β=0.2 (Mac `run_id=49`). This is **not** a BFCL `multi_turn_base` accuracy. **Do not invent** a Run 49 / 47 / 48 BFCL %. A later **full DPO** checkpoint is **scored** as Run 51 **53/200 = 26.5%** on plain IFFC — **below IFFC 30.5% and NFC 27.0%**; **not** a method win; **not** “full DPO still in flight.” Other Run 51 fields (tok/s, gates) **not verified**. PTC-A Run 46 FAIL **13.5%** is **banked and closed**; PTC-B **HOLD**. [PR #57](https://github.com/sskarz/Thesis/pull/57) DPO pair-audit stays **draft** — **do not merge** harness drafts as wins. **No BoN.**

Identity (verified 2026-09-29, live Mac `benchmark.db`):

- `run_id` **49**; `model_key` `gemma4-e2b-dpo-smoke100-beta02`
- display: Gemma-4-E2B-it DPO smoke100 beta0.2 (Q4_K_M)
- `fc_mode`: plain **iffc** (not PTC / OODA)
- n: **10 IDs** (same set as Run 47/48): `multi_turn_base_4,38,46,64,84,95,30,82,3,163`
- harness tip: `d344123fe`
- gguf: `sft/fused/gemma-4-e2b-it-dpo-smoke100-beta02-Q4_K_M.gguf`

**Train recipe (smoke only; not a full DPO train):** 100-update smoke from **untouched** `gemma-4-e2b-it-mlx`; β=**0.2**; LR 1e-6; LoRA r16/s32; `adapters-dpo-smoke100-beta02` → fused Q4. Probe wall ~225s. Do **not** invent a smoke / Run 49 BFCL %. Full DPO is **scored** as Run 51 **53/200 = 26.5%** (plain IFFC) — **not** a method win.

**Formal 10-ID gate (Run 49) — PASS** vs gate `avg api ≥4 and ≤8.3`, `hit_step_limit ≤2/10`, `iffc_unparseable ≤1/10`, `turn0_tool ≥6/10`:

| Metric | Gate | **Run 49 (β0.2)** | Run 47 (β0.1) | Run 48 (BASE) |
|--------|------|-------------------|---------------|---------------|
| avg `api_calls` | ≥4 and ≤8.3 | **4.4** | 4.50 | 6.80 |
| `hit_step_limit` | ≤2/10 | **0/10** | — | — |
| `iffc_unparseable` | ≤1/10 | **0/10** | — | — |
| `turn0_tool` | ≥6/10 | **7/10** | 6/10 | 10/10 |

All four gates **MET → PASS**. β0.2 improves turn0 vs β0.1 (7 vs 6) while staying below BASE turn0 (10) and roughly matching β0.1 API load. Run 47/48 comparators are the **same 10 IDs, plain iffc** — **not** `multi_turn_base` N=200 scores. **Do not invent** BFCL % for Runs 47–49.

**Do not** treat this PASS as a method win on the N=200 ladder. Scoreboard BFCL N=200 **unchanged**: BoN **36.5%** / Run 42 iffc-ooda-card **35.5%** / STATE **34.0%** / Run 30 **32.5%** / IFFC **30.5%** / SFT-v4 **15.0%** / Run 46 iffc-ptc-a **13.5%** (FAIL / closed). Run 30 / Run 42 / BoN / IFFC / SFT-v4 / Run 46 **untouched**.

**Next:** full DPO is **scored** as Run 51 **53/200 = 26.5%** (plain IFFC) — **not** a method win. Do **not** present the 10-ID smoke as N=200 Acc. Do **not** BoN this smoke ckpt. Do **not** merge PR #57. Do **not** more PTC/OODA remix. Do **not** unpark SFT quota remix. See Run 58 Decision **25.0% FAIL** and Run 61 Decision+OODA **26.0% FAIL** — park Decision(+OODA) Acc.

### Anti-recommendations

Do **not**: more epochs on the v1 mix **or** the v2 filtered mix **or** the v4 mix **or** the v5 mix **or** the v6 mix; BoN / IFFC+STATE on the Run 12, Run 13, Run 15, Run 17, or Run 18 checkpoint; fuse `final@2000` as the first fix; claim the `[]` filter fixed SFT or multi-turn; present Run 14’s 2.0% as a method win or a training-time recovery; present Run 15’s 15.0% as beating IFFC or NFC; present Run 16’s 13.0% as a STATE win on SFT-v4; present Run 17’s 11.5% as a method win; present Run 18’s 13.0% as a method win; invent tok/s for Run 15, Run 16, or Run 17 (`inference_metrics=0`); **propose BoN as the next win on the SFT-v4, SFT-v5, or SFT-v6 checkpoint**; invent a v9 / v9.1 / v9.2 BFCL `multi_turn_base` %; cite Run 35 (INVALID — server died); fold Run 37 into the 10-ID denominator; start full LoRA on the failed v9.2 mix; merge [PR #57](https://github.com/sskarz/Thesis/pull/57); treat iffc-stop-repeat as an SFT fix; run a third quota remix; invent a Run 49 / 47 / 48 BFCL `multi_turn_base` %; present the Run 49 10-ID **PASS** as a N=200 score or a method win; present Run 51 **26.5%** as beating IFFC/NFC; BoN the DPO smoke100 ckpt.

### Next (Sanskar decides)

1. **SFT v7 (planned, not scored).** Path locked 2026-09-23: **TRAINING (SFT)** is the main path; inference wrappers secondary. **Tightened stop-repeat PASSES** (Run 30 **32.5%** vs IFFC **30.5%**). Run 42 `--fc-mode iffc-ooda-card` **scored** **71/200 = 35.5%** (still best non-BoN; SFT still **HOLD**). Run 46 `--fc-mode iffc-ptc-a` **scored FAIL** **27/200 = 13.5%** — **bank and close**; PTC-B **HOLD**. **SFT v9.2 10-ID smoke-gate FAIL** (Run 36 avg `api_calls` **25.7**, `hit_step_limit` **6/10**) → **`HOLD_sft_quota_remix` / park** — see [SFT v9.2 smoke-gate FAIL](#sft-v9--v91--v92--lever-2-action-quotas-10-id-smoke-gate-fail--hold). **Not** a `multi_turn_base` %; **do not invent** a v9 BFCL %. **DPO β0.2 smoke100 10-ID PASS** (Run 49 avg API **4.4**, `hit_step_limit` **0/10**, `iffc_unparseable` **0/10**, `turn0_tool` **7/10**) — **health gate, not a BFCL %**; DPO-full **scored** Run 51 **53/200 = 26.5%** (plain IFFC; **not** a method win) — see [DPO β0.2 smoke100 PASS](#dpo-β02-smoke100-10-id-pass-run-49--health-gate-not-a-bfcl-). Next path = **DPO**, **not** more PTC/OODA remix, **not** another quota remix. [PR #57](https://github.com/sskarz/Thesis/pull/57) stays **draft** @`cde93a3` — **do not merge**. **No BoN, no v7 resume, no full LoRA on failed mixes.** Historical ranking in [Post–Run27 consensus (2026-09-28)](#post-run27-consensus-2026-09-28) still stands as a later SFT path (APIGen-MT-5k + ToolACE/ToolACE-MT-style MT + ST FC; Magnet/Critical-State then mDPO; **not** decode bans) — action-class **quota remix is HOLD**. Optional open data-gen complement: [Data Turnstile §24](#24-data-turnstile-open-function-calling-data-generation) (unscored; not a substitute for the APIGen-MT core). Optional post-healthy-SFT local RL: [Critical-State RL §25](#25-critical-state-rl-diagnosing-trainable-multi-turn-calls) (unscored; **not** blanket GRPO; **do not** unlock GRPO as the main path). 2026-09-28 methods sweep ([§26–§31](#26-ooda-tool-typed-observe-orient-decide-act)): Guo et al. paper **unscored**; one-pass `[OODA]` card **scored** Run 42; `--fc-mode iffc-ptc-a` **scored FAIL** Run 46 (**bank and close**; PTC-B **HOLD**); remaining sweep paper numbers are **theirs**. **Do not invent a v7 score.** Not more epochs on v4/v5/v6; not BoN on those ckpts; no thin-filter-only ToolACE+xLAM remakes. Run 17 scored v5 as a **regression** (11.5% vs v4 15.0%). Run 18 scored v6 as a **miss vs v4** (13.0% vs 15.0%). Thin filter ablation on the same family is exhausted. STATE-on-v4 is **measured negative** (Run 16 13.0% vs Run 15 15.0%) — closed. v4 IFFC 15% remains best SFT recovery. Lit review: [`sskarz/thesis-vault` `📝 Notes/2026-09-23-SFT-training-lit-review.md`](https://github.com/sskarz/thesis-vault/blob/main/%F0%9F%93%9D%20Notes/2026-09-23-SFT-training-lit-review.md).
2. ToolACE-only vs xLAM-only ablation at small scale.
3. Investigate whether the Gemma-4 chat-template / fuse path still hurts multi-turn state tracking despite valid IFFC format.
4. Inference wrappers stay on the **base** Gemma ladder (BoN **36.5%** is still the best measured inference result) and are **secondary**. Do **not** BoN the SFT-v4, SFT-v5, or SFT-v6 checkpoint.

`--fc-mode iffc --iffc-strip-empty-assistant` on `--model gemma4-e2b-sft-iffc-v2` is **scored** as Run 14: **4/200 = 2.0%** (**DIAGNOSTIC**; did **not** rescue). Do **not** re-run to overwrite that `runs` row. Default remains **off** for standard IFFC (Run 5–18 bookkeeping unchanged unless the diagnostic flag is set).

**Eval identity:** v1 fuse best val @ iter 1000 → Q4_K_M GGUF → `llama-server -m sft/fused/gemma-4-e2b-it-iffc-best1000-Q4_K_M.gguf`, then `uv run python -m benchmark.run_all --model gemma4-e2b-sft-iffc --only-multi-turn --categories multi_turn_base --fc-mode iffc`. v2: `llama-server -m sft/fused/gemma-4-e2b-it-iffc-v2-best2000-Q4_K_M.gguf` then `--model gemma4-e2b-sft-iffc-v2` (distinct `runs` row; figures `benchmark/results/Gemma-4-E2B-it-SFT-IFFC-v2-Q4_K_M/figures`). Diagnostic strip-`[]` eval (Run 14, already scored **4/200 = 2.0%** — do not overwrite) uses the same `--model gemma4-e2b-sft-iffc-v2` plus `--iffc-strip-empty-assistant` (distinct `runs` row; figures `.../iffc-strip-empty-assistant/figures`). v4: `llama-server -m sft/fused/gemma-4-e2b-it-iffc-v4-best1600-Q4_K_M.gguf` then `--model gemma4-e2b-sft-iffc-v4` (Run 15 already scored **30/200 = 15.0%** under `--fc-mode iffc` — do not overwrite; **scored recovery, still below IFFC/NFC**). Run 16 already scored **26/200 = 13.0%** under `--fc-mode iffc-state` on that same GGUF — **null/negative vs Run 15**; do not overwrite; do **not** BoN this ckpt. v5: `llama-server -m sft/fused/gemma-4-e2b-it-iffc-v5-best1800-Q4_K_M.gguf` then `--model gemma4-e2b-sft-iffc-v5` (Run 17 already scored **23/200 = 11.5%** under `--fc-mode iffc` — **scored regression vs Run 15 15.0%**; do not overwrite; **do not present as a method win**; **no BoN / no more epochs on this mix**; LoRA best@1800 val 0.147, PR #35). v6: `llama-server -m sft/fused/gemma-4-e2b-it-iffc-v6-best1800-Q4_K_M.gguf` then `--model gemma4-e2b-sft-iffc-v6` (Run 18 already scored **26/200 = 13.0%** under `--fc-mode iffc` — **scored miss vs Run 15 15.0%**; do not overwrite; **do not present as a method win**; **no BoN / no more epochs on this mix**; LoRA best@1800 val 0.142, PR #38 recipe, PR #39 model key). Do **not** pass `--model gemma4-e2b`. See `sft/README.md`. Do not overwrite Run 12, Run 13, Run 14, Run 15, Run 16, Run 17, or Run 18.

### Sources

- Liu, T., et al. "ToolACE: Winning the Points of LLM Function Calling." ICLR 2025. arXiv:2409.00920. https://arxiv.org/abs/2409.00920
- Zeng et al. ToolACE-R. arXiv:2504.01400. (v7 student-aware filter; **not scored**)
- Zeng et al. "ToolACE-MT." ICLR 2026. arXiv:2508.12685. https://arxiv.org/abs/2508.12685 (paper BFCL-v3 MT 9.25%→40.25%; **not** our harness)
- Prabhakar et al. "APIGen-MT." NeurIPS 2025 D&B. arXiv:2504.03601. https://arxiv.org/abs/2504.03601 (v7 planned mix core; **not scored**; paper MT 1B/3B/8B 43/56/69% — **not** our harness)
- Ramakrishnan, G. & Sharma, M. (Amazon AGI). Data Turnstile. arXiv:2607.29250. https://arxiv.org/abs/2607.29250 (optional open data-gen complement for v7+; **not scored**; same paper as [§24](#24-data-turnstile-open-function-calling-data-generation))
- Chen et al. (Salesforce). Critical-State RL. arXiv:2609.24985. https://arxiv.org/abs/2609.24985 (post-healthy-SFT occurrence-local RL; **not scored**; not blanket GRPO)
- Guo et al. OODA-Tool. arXiv:2608.24368. https://arxiv.org/abs/2608.24368 ([§26](#26-ooda-tool-typed-observe-orient-decide-act); paper method **unscored**; one-pass `[OODA]` card harness **scored** Run 42 — prefer one-pass)
- Yu, Mu, & Wu. PTC-Decoder. arXiv:2609.30836. https://arxiv.org/abs/2609.30836 ([§27](#27-ptc-decoder-plan-to-act-plus-tool-name-dfa); paper method **theirs**; harness `--fc-mode iffc-ptc-a` **scored FAIL** Run 46)
- Popescu et al. Looped LMs. arXiv:2608.18171. https://arxiv.org/abs/2608.18171 ([§28](#28-looped-language-models-for-compositional-tool-calling); **not scored**)
- Chai et al. PARL-MT. arXiv:2509.23206. https://arxiv.org/abs/2509.23206 ([§29](#29-parl-mt-progress-awareness-for-multi-turn-function-calling); **not scored**; not a GRPO unlock)
- Du, Zhou, et al. HyperTool. arXiv:2606.13663. https://arxiv.org/abs/2606.13663 ([§30](#30-hypertool-code-block-execution-granularity); **not scored**)
- Zhong et al. RC-GRPO. arXiv:2602.03025. https://arxiv.org/abs/2602.03025 ([§31](#31-rc-grpo-reward-conditioned-grpo); **not scored**; not a GRPO unlock)
- Zhao et al. "Calibration is the Bottleneck" (GAR). arXiv:2609.00949. https://arxiv.org/abs/2609.00949 (Acc ≤ GAR; SRI/CRI can hurt Acc — frames Run 27 ungated miss; Run 30 gated recover past IFFC)
- Zhang, J., et al. "xLAM: A Family of Large Action Models to Empower AI Agent Systems." arXiv:2409.03215, 2024.
- mlx-lm LoRA: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md
- IFFC Appendix B: arXiv:2608.22472
- Vault lit review: [`sskarz/thesis-vault` `📝 Notes/2026-09-23-SFT-training-lit-review.md`](https://github.com/sskarz/thesis-vault/blob/main/%F0%9F%93%9D%20Notes/2026-09-23-SFT-training-lit-review.md)

---

## Prioritized Recommendations

Ranked by expected impact × feasibility for the current setup (Qwen2.5-3B, Llama-3.2-3B, Qwen3.5-4B on llama.cpp with BFCL v4):

| Priority | Technique | Target Problem | Expected Impact | Effort |
|----------|-----------|----------------|-----------------|--------|
| 1 | **`tool_choice=required` + `parallel_tool_calls`** | Parallel calling (0%) | Potentially unlocks parallel | Low |
| 2 | **Guided-Structured Templates (`--fc-mode gst`)** | Gemma `multi_turn_base` vs NFC/IFFC/CD | **Measured** Run 9: 61/200 = 30.5% (ties IFFC, does not beat it) | Done (prompt-only) |
| 3 | **IFFC + harness STATE (`--fc-mode iffc-state`)** | Gemma `multi_turn_base` vs IFFC/GST 30.5% | **Measured** Run 10 (base): 68/200 = 34.0% (beats IFFC; first above 30.5%). **Measured** Run 16 (SFT-v4): 26/200 = 13.0% (**null/negative vs Run 15 15.0%**). Empty 41→31 on base; state 60→64 (did not improve). On SFT-v4 residual stays state 112 / under-call 59. **Measured** Run 17 (SFT-v5 IFFC): 23/200 = 11.5% (**regression vs Run 15**). **Measured** Run 18 (SFT-v6 IFFC): 26/200 = 13.0% (**miss vs Run 15**). Next training path is SFT **v7** (planned, **not scored**), **not** BoN-on-v4/v5/v6. | Done (harness + scores) |
| 4 | **Few-shot format demonstrations** | Parallel call format, cross-language | +5–20pp for SLMs | Medium |
| 5 | **Best-of-N majority voting (`--fc-mode iffc-state-bon`)** | Gemma `multi_turn_base` vs IFFC+STATE 34.0% | **Measured** Run 11: 73/200 = 36.5% (beats STATE by +5 / +2.5 pp). State 64→53; empty 31→36; under-call flat. ~7× wall / ~8.8× decode — modest lift, **not free**. | Done (harness + live score) |
| 5a | **IFFC stop-on-repeat (`--fc-mode iffc-stop-repeat`)** | Collapsed over-call vs healthy-base IFFC | **Measured** Run 27 ungated on **BASE** Gemma: 54/200 = 27.0% (`scores.id=20`; **200/200 responses**). **Miss vs Run 5 IFFC 30.5%** (−3.5 pp; −9.5 pp vs BoN). Same accuracy level as Run 4 NFC 27.0% — **not** an identical method. Stop flags **37/200**, all failures (**0 stopped and correct**). VehicleControlAPI **22→33** vs Run 5. **Measured** Run 30 tighten MIN=6 @ `78c22a0` (PR #54): **65/200 = 0.325** (`scores.id=21`; `multi_turn_base` only; **200/200 responses + timings**; `entry_id` identical to Run 5/Run 27; bartowski Q4_K_M). **First scored win vs Run 5 IFFC** (**+2.0 percentage points; 65−61 correct** — **not** “+4pp”); **+5.5 pp vs Run 27**. Still **below STATE 34.0% and BoN 36.5%**. Stop flags **37→3**, all FAIL. VC state **33→22** (recovered to Run 5). `state_mismatch` **60** (=Run 5). **Celebrate / park SFT-v9: YES.** **Caveats:** harness/decode guardrail on BASE — **not** an SFT/method win; `finished_at` NULL (metrics 4/5 soft bookkeeping; accuracy locked in `scores`); status JSON stale — trust DB+log; `--resume-run-id 30` after orphan; N=200 complete. **Codex CONFIRM_PASS.** Do **not** invent tok/s. Do **not** overwrite this row / BoN / IFFC / SFT-v4. Run 42 iffc-ooda-card is now **higher Acc** (35.5%) with a `state_mismatch` gate **MISS** — do **not** overwrite Run 30. Harness [PR #54](https://github.com/sskarz/Thesis/pull/54). | Done (harness + live scores; ungated miss; tighten **PASSES**; Run 42 is a **separate** mode) |
| 5b | **IFFC OODA card (`--fc-mode iffc-ooda-card`)** | One-pass `[OODA]` working-memory card on stop-repeat MIN=6 | **Measured** Run 42 on **BASE** Gemma: **71/200 = 35.5%** (`multi_turn_base` only; **200/200 responses**; finished `2026-09-29T05:51:02.787870+00:00`). Dedicated `--fc-mode` — **not** a flag on stop-repeat; **not** iffc-state; **not** 4-pass OODA. **+6 / +3.0 pp vs Run 30**; still **below BoN 36.5% (−1.0 pp)**; **above STATE 34.0% (+1.5 pp)**. `eval_errors`: state_mismatch **71**, empty **17**, missing_gt **41**, empty+missing **58**, total **129**. Gates: ≥65/200 **MET**; state_mismatch <60 **MISS**; empty+missing ≤75 **MET**; TTFT 517.8 vs Run 30 630.5 ms (≤1.25× **MET**). **Do not claim a clean gate sweep.** Calib Run 40 10-ID **PASS** (not a BFCL %). Run 41 orphan 0 responses — ignore. Do **not** overwrite Run 30. SFT parked. No BoN. Do **not** invent tok/s. Harness [PR #61](https://github.com/sskarz/Thesis/pull/61) draft @ `1dea341`. Still **best non-BoN harness Acc** after Run 46. | Done (docs for scored run; harness PR #61 separate) |
| 5c | **IFFC PTC-A (`--fc-mode iffc-ptc-a`)** | Live tool-name DFA, args free (PTC-Decoder-inspired) | **Measured** Run 46 on **BASE** Gemma: **27/200 = 13.5%** (`multi_turn_base` only; **200/200 responses**; finished `2026-09-29T08:15:24.061478+00:00` = 2026-09-29 04:15 ET). Dedicated `--fc-mode` — live tool-name DFA, args free; **not** Yu et al. paper F1; **not** CD; **not** OODA-card; **not** STATE; **not** BoN. **NEW run** (never resume Run 30/42). **−22.0 pp vs Run 42**; **−19.0 pp vs Run 30**; **−17.0 pp vs IFFC**. `eval_errors`: state_mismatch **102**, empty **17**, missing_gt **54**, empty+missing **71**, total **173**. Gates: ≥71/200 **MISS** (27); state_mismatch ≤60 **MISS** (102); empty+missing ≤75 **MET** (71); wall ~42.1 min / ratio vs Run 30 **0.234485** (≤1.25× **MET**). **Scored FAIL / Acc regression; bank and close.** Calib Run 45 10-ID **PASS** (avg API **5.4**, turn0_tool **10/10**; exit 0; **not** a BFCL %). PTC-B **HOLD**. Next = **DPO** (β0.2 smoke100 10-ID Run 49 **PASS** health gate — **not** a BFCL %; full train **HOLD**; [PR #57](https://github.com/sskarz/Thesis/pull/57) stays **draft**), **not** more PTC/OODA remix. Do **not** overwrite Run 30/42. SFT parked. No BoN. Do **not** invent tok/s. Harness [PR #63](https://github.com/sskarz/Thesis/pull/63) draft @ `3569311`. | Done (docs for scored FAIL; harness PR #63 separate; closed) |
| 6 | **LLGuidance build** | Type enforcement (Llama issue) | Stricter schema at grammar level. Complements [PTC-Decoder §27](#27-ptc-decoder-plan-to-act-plus-tool-name-dfa) (paper F1 **theirs**; harness `--fc-mode iffc-ptc-a` **scored FAIL** Run 46 **13.5%** — closed). | Medium |
| 7 | **RAG tool retrieval** | — | **OUT OF SCOPE** for this thesis paper (earlier title framing; not deferred; not a Dec 2026 roadmap item). Literature: Gao et al. RAG-MCP. Complements in METHODS remain **unscored** literature ([MCP-GRANITE §3a](#3a-tool-interface-granularity-mcp-granite); [HyperTool §30](#30-hypertool-code-block-execution-granularity)). Do not score RAG. | — |
| 8 | **Schema-based verification loop** | Argument errors across categories | +3–10% with retries | Medium |
| 9 | **DSPy prompt optimization** | Model-specific prompt tuning | Auto-discovered improvements | High |
| 10 | **Looped LMs (logged, unscored)** | Compositional / nested tool calling | [§28](#28-looped-language-models-for-compositional-tool-calling) recurrent latent depth. Paper BFCL is static single-turn; **not** our `multi_turn_base`. Needs a looped backbone we do not run. | High (architecture) |

The first four are directly measurable on BFCL v4 without changes to the benchmark format, making them strong thesis contributions. The key narrative: **inference-time techniques can close the gap between SLMs and cloud-scale models without fine-tuning.**

**Post–Run42 (2026-09-29):** `--fc-mode iffc-ooda-card` is **scored** (Run 42 **71/200 = 35.5%**; +3.0 pp vs Run 30; still below BoN 36.5%; `state_mismatch` **71** gate **MISS**). Run 30 **untouched**. Ungated stop-on-repeat remains a **scored miss** on healthy base ([PR #55](https://github.com/sskarz/Thesis/pull/55); GAR-aligned Acc regress). Tightened stop-repeat still **PASSES** (Run 30 **65/200 = 32.5%** vs IFFC **30.5%**). **SFT v9.2 10-ID smoke-gate FAIL** (Run 36 avg **25.7**, hit **6/10**) → **`HOLD_sft_quota_remix` / park**. Do **not** treat ungated decode bans as the next method win. Do **not** unlock GRPO as the main path. Do **not** claim a clean Run 42 gate sweep.

**Post–Run46 (2026-09-29):** `--fc-mode iffc-ptc-a` is **scored FAIL** (Run 46 **27/200 = 13.5%**; **−22.0 pp vs Run 42**; **−19.0 pp vs Run 30**; **−17.0 pp vs IFFC**; [PR #63](https://github.com/sskarz/Thesis/pull/63) @ `3569311`). Live tool-name DFA, args free. **200/200 responses**; finished `2026-09-29T08:15:24.061478+00:00`. Gates: ≥71/200 **MISS** (27); `state_mismatch` ≤60 **MISS** (102); empty+missing ≤75 **MET** (71); latency ≤1.25× Run 30 **MET** (wall ~42.1 min; wall ratio **0.234485**). **Bank and close.** PTC-B **HOLD**. Next path = **DPO** (β0.2 smoke100 10-ID Run 49 **PASS** health gate — **not** a BFCL %; DPO-full **scored** Run 51 **53/200 = 26.5%** — **not** a method win; [PR #57](https://github.com/sskarz/Thesis/pull/57) stays **draft**), **not** more PTC/OODA remix. Calib Run 45 10-ID PASS ≠ BFCL %. Do **not** overwrite Run 30 / Run 42. SFT parked. No BoN. Do **not** invent tok/s. Acc ranking of winners **unchanged**: BoN **36.5%** / Run 42 **35.5%** / STATE **34.0%** / Run 30 **32.5%** / IFFC **30.5%**. Best SFT still v4 **15%**. See [DPO β0.2 smoke100 PASS](#dpo-β02-smoke100-10-id-pass-run-49--health-gate-not-a-bfcl-).

---

## Additional Papers and Resources

### Error Taxonomies and Analysis

- **APIGen** (Liu et al., 2024): Automated dataset generation for tool calling. Error taxonomy for <7B models: 32% type mismatch, 24% missing required arguments, 18% wrong function selection, 15% hallucinated arguments. Confirms that type coercion addresses the #1 error category. arXiv:2406.18518. https://arxiv.org/abs/2406.18518
- **ToolSandbox** (Lu et al., 2024): Stateful multi-turn tool evaluation. Shows state management is the primary bottleneck for multi-turn, not individual tool call accuracy. Explicit state injection improves multi-turn success by 15–25%. https://arxiv.org/abs/2408.04682
- **BFCL** (Patil et al., ICML 2025): Berkeley Function Calling Leaderboard. https://proceedings.mlr.press/v267/patil25a.html. Gorilla: arXiv:2305.15334. https://arxiv.org/abs/2305.15334. Leaderboard: https://gorilla.cs.berkeley.edu/leaderboard.html

### Model-Specific Tool Calling Techniques

- **Granite Function Calling** (IBM, 2024): Decomposes function calling into granular subtasks: function name detection, parameter extraction, value extraction. Even without fine-tuning, prompting for subtasks sequentially improves SLM accuracy. arXiv:2407.00121. https://arxiv.org/abs/2407.00121
- **NexusRaven-V2** (Nexusflow, 2024): Tool description formatting matters more than model size. Adding usage examples within tool descriptions significantly boosts accuracy.
- **Octopus v2** (Chen et al., 2024): 2B on-device function calling model using functional tokens — a single special token per tool, reducing generation length to 1 token for tool selection. arXiv:2404.01744. https://arxiv.org/abs/2404.01744
- **ToolACE** (Liu et al., ICLR 2025): Including parameter constraints (min/max values, enum lists) in tool descriptions improves accuracy significantly. arXiv:2409.00920. https://arxiv.org/abs/2409.00920

### Structured Decoding Libraries

- **Outlines** (dottxt-ai): Python library for structured generation using finite-state machines compiled from regex/JSON Schema. Targets HuggingFace/vLLM ecosystem. https://github.com/dottxt-ai/outlines
- **SGLang** (Stanford, 2024): Structured generation language with compressed FSM for efficient constrained decoding. NeurIPS 2024. arXiv:2312.07104. https://arxiv.org/abs/2312.07104
- **Guidance** (Microsoft Research): Grammar-constrained generation library; LLGuidance is the Rust backend integrated into llama.cpp. https://github.com/guidance-ai/guidance

### Inference-Time Scaling Surveys

- Awesome Inference-Time Scaling: Curated paper list. https://github.com/ThreeSR/Awesome-Inference-Time-Scaling
- Microsoft Research. "Inference-Time Scaling for Complex Tasks: Where We Stand and What Lies Ahead." 2025. https://www.microsoft.com/en-us/research/wp-content/uploads/2025/03/Inference-Time-Scaling-for-Complex-Tasks-Where-We-Stand-and-What-Lies-Ahead-2.pdf

---

## References

1. Brown, B., et al. "Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models." ICLR 2025. arXiv:2408.00724.
2. Chen, W., et al. "Octopus v2: On-device Language Model for Super Agent." arXiv:2404.01744, 2024.
3. Gao, Y., et al. "RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation." arXiv:2505.03275, 2025.
4. Hao, S., et al. "ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings." NeurIPS 2023. arXiv:2305.11554.
5. He, S. "Achieving Tool Calling Functionality in LLMs Using Only Prompt Engineering Without Fine-Tuning." arXiv:2407.04997, 2024.
6. He, Y., et al. "Improving Large Language Models Function Calling and Interpretability via Guided-Structured Templates." arXiv:2509.18076, 2025.
7. Huang, J., et al. "Small Language Models Need Strong Verifiers to Self-Correct Reasoning." arXiv:2404.17140, 2024.
8. Jain, N., et al. "Lost in Space: Optimizing Tokens for Grammar-Constrained Decoding." arXiv:2502.14969, 2025.
9. Khattab, O., et al. "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." ICLR 2024. https://dspy.ai/.
10. Liu, J., et al. "APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets." arXiv:2406.18518, 2024.
11. Liu, T., et al. "ToolACE: Winning the Points of LLM Function Calling." ICLR 2025. arXiv:2409.00920. https://arxiv.org/abs/2409.00920
12. Lu, J., et al. "ToolSandbox: A Stateful, Conversational, Interactive Evaluation Framework for LLM Tool Use Capabilities." arXiv:2408.04682, 2024.
13. Madaan, A., et al. "Self-Refine: Iterative Refinement with Self-Feedback." NeurIPS 2023. arXiv:2303.17651.
14. Patil, S., et al. "Gorilla: Large Language Model Connected with Massive APIs." arXiv:2305.15334, 2023. https://arxiv.org/abs/2305.15334
14a. Patil et al. "Berkeley Function Calling Leaderboard." ICML 2025. https://proceedings.mlr.press/v267/patil25a.html. Leaderboard: https://gorilla.cs.berkeley.edu/leaderboard.html
15. Qin, Y., et al. "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs." ICLR 2024. arXiv:2307.16789.
16. Qu, C., et al. "The Art of Scaling Test-Time Compute for Large Language Models." arXiv:2512.02008, 2025.
17. Rubin, O., et al. "Learning To Retrieve Prompts for In-Context Learning." NAACL 2022. arXiv:2112.08633.
18. Shi, X., et al. "SimpleTool: Parallel Decoding for Real-Time LLM Function Calling." arXiv:2603.00030, 2026.
19. Shinn, N., et al. "Reflexion: Language Agents with Verbal Reinforcement Learning." NeurIPS 2023. arXiv:2303.11366.
20. Snell, C., et al. "Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters." arXiv:2408.03314, 2024.
21. Tam, Z., et al. "Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models." arXiv:2408.02442, 2024.
22. Wang, X., et al. "Self-Consistency Improves Chain of Thought Reasoning in Language Models." ICLR 2023. arXiv:2203.11171.
23. Wei, J., et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS 2022. arXiv:2201.11903.
24. Willard, B. T. & Louf, R. "Efficient Guided Generation for Large Language Models." arXiv:2307.09702, 2023.
25. Xu, B., et al. "ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models." arXiv:2305.18323, 2023.
26. Kang et al. "ACON: Optimizing Context Compression for Long-horizon LLM Agents." arXiv:2510.00615, 2025.
27. Yan, F., et al. "Berkeley Function Calling Leaderboard." arXiv:2402.15491, 2024.
28. Yao, S., et al. "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR 2023. arXiv:2210.03629.
29. Yuan, S., et al. "CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets." arXiv:2309.17428, 2024.
29a. Paschalides, D., et al. "MCP-GRANITE Benchmark: GRANularity Interface TEsting for MCP-Based LLM Agents." arXiv:2609.24161, 2026.
29b. Yu, M., Mu, K., & Wu, G. "PTC-Decoder: Towards Intelligent SLMs on Offline Resource-Constrained Edge Devices." arXiv:2609.30836, 2026. https://arxiv.org/abs/2609.30836
29c. Popescu, A. C., Sáez de Ocáriz Borde, H., & Liò, P. "Looped Language Models Improve Compositional Tool Calling." arXiv:2608.18171, 2026. https://arxiv.org/abs/2608.18171

---

# Part II: Training-Time Methods

Each method in this section involves modifying model weights — via supervised fine-tuning (SFT), reinforcement learning (RL), knowledge distillation, or structured pruning. These techniques produce specialized tool-calling models that can then be served via llama.cpp in the same way as the base models evaluated in Part I.

## Table of Contents (Part II)

- [12. Supervised Fine-Tuning with Synthetic Data](#12-supervised-fine-tuning-with-synthetic-data)
- [13. ToolACE: Automated Training Data Generation](#13-toolace-automated-training-data-generation)
- [24. Data Turnstile: Open Function-Calling Data Generation](#24-data-turnstile-open-function-calling-data-generation)
- [14. Hammer: Function Masking for Robust Tool Calling](#14-hammer-function-masking-for-robust-tool-calling)
- [15. xLAM: Large Action Models](#15-xlam-large-action-models)
- [16. FunctionGemma: Specialist Fine-Tuning from Base Models](#16-functiongemma-specialist-fine-tuning-from-base-models)
- [17. Structured Pruning + Fine-Tuning for Edge Deployment](#17-structured-pruning--fine-tuning-for-edge-deployment)
- [18. Parameter-Efficient Fine-Tuning (LoRA / QLoRA)](#18-parameter-efficient-fine-tuning-lora--qlora)
- [19. Reinforcement Learning with Verifiable Rewards (GRPO)](#19-reinforcement-learning-with-verifiable-rewards-grpo)
- [25. Critical-State RL: Diagnosing Trainable Multi-Turn Calls](#25-critical-state-rl-diagnosing-trainable-multi-turn-calls)
- [26. OODA-Tool: Typed Observe-Orient-Decide-Act](#26-ooda-tool-typed-observe-orient-decide-act)
- [29. PARL-MT: Progress Awareness for Multi-Turn Function Calling](#29-parl-mt-progress-awareness-for-multi-turn-function-calling)
- [30. HyperTool: Code-Block Execution Granularity](#30-hypertool-code-block-execution-granularity)
- [31. RC-GRPO: Reward-Conditioned GRPO](#31-rc-grpo-reward-conditioned-grpo)
- [20. End-to-End Agent RL (Agent-R1 / VerlTool)](#20-end-to-end-agent-rl-agent-r1--verltool)
- [21. DiaTool-DPO: Preference Optimization for Multi-Turn Tool Calling](#21-diatool-dpo-preference-optimization-for-multi-turn-tool-calling)
- [22. Knowledge Distillation for Tool Calling](#22-knowledge-distillation-for-tool-calling) (survey; Choice/router KD **path B first** ranking **PASS**; Acc harness IMPL **CONFIRM_PASS_WITH_FLAGS** — **separate Acc track**; ranking ≠ Acc; Acc still **NO-GO**)
- [23. Curriculum Learning for Progressive Tool Mastery](#23-curriculum-learning-for-progressive-tool-mastery)
- [Training-Time Prioritized Recommendations](#training-time-prioritized-recommendations)
- [Post–Run27 consensus (2026-09-28)](#post-run27-consensus-2026-09-28)
- [Part II References](#part-ii-references)

---

## 12. Supervised Fine-Tuning with Synthetic Data

### Overview

The most direct approach: fine-tune a small model on a dataset of (query, tool_call) pairs using standard next-token prediction loss. Recent work shows that even a 350M-parameter model can outperform GPT-3.5 and 7B generalist models on tool calling when trained on task-specific data.

### Mechanism

1. **Data preparation:** Collect or generate (user_query, available_tools, correct_tool_call) triples
2. **Format as chat completions:** Convert each example into the model's chat template format with system prompt, user message, and assistant response containing the tool call
3. **Train with cross-entropy loss:** Standard SFT on the formatted examples, typically for 1–3 epochs

### Evidence

A study fine-tuning facebook/opt-350m on the ToolBench dataset achieved a **77.55% pass rate** — substantially outperforming much larger models:

| Model | Parameters | Pass Rate |
|-------|-----------|-----------|
| **OPT-350M (fine-tuned)** | **350M** | **77.55%** |
| ToolLLaMA-DFS | 7B | 30.18% |
| ChatGPT-CoT | 175B | 26.00% |
| Claude-CoT | 52B | 2.73% |

Training details: single epoch, learning rate 5×10⁻⁵, batch size 32 via gradient accumulation, FP16 mixed precision, AdamW optimizer. Dataset: 187,542 examples from ToolBench.

### Key Finding

**"Task-specific optimization fundamentally outperforms scale-based approaches"** because general-purpose models suffer from parameter dilution where most capacity addresses irrelevant tasks. A 350M specialist can outperform a 175B generalist by a factor of 2.98x on tool calling.

### Applicability

- **BFCL relevance:** High. Fine-tuning on BFCL-format data (or synthetic data in the same format) could dramatically improve scores across all categories.
- **Hardware requirement:** Full SFT of a 3B model requires ~24GB VRAM (fits on RTX 4090). For 350M–1B models, 16GB is sufficient.
- **Dataset requirement:** Need high-quality tool-calling training data (see ToolACE, xLAM datasets below).
- **GGUF deployment:** After fine-tuning, convert to GGUF and serve via llama.cpp identically to base models.

### Source

Nicolas. "Fine Tuning SLMs on Agentic Tool Calling: An Experiment." Medium, 2024. https://medium.com/@dataenthusiast.io/fine-tuning-slms-on-agentic-tool-calling-an-experiment-ccbef62ac5c7

---

## 13. ToolACE: Automated Training Data Generation

### Overview

ToolACE is an automated agentic pipeline that generates accurate, complex, and diverse tool-calling training data. It achieved #1 on BFCL v1 by fine-tuning LLaMA-3.1-8B on its synthetic dataset, outperforming GPT-4 and Claude-3.5-Sonnet.

### Mechanism

The pipeline has three stages:

**1. Tool Self-Evolution Synthesis (TSS):**
- Extracts an API context tree from pretraining data, establishing hierarchical domains
- Generates 26,507 distinct APIs spanning 30 primary domains and 390 sub-domains
- APIs include diverse parameter types, nested structures, and multiple calling modes (single, parallel, dependent)

**2. Multi-Agent Interactive Dialog Generation (MAI):**
- Three LLM-powered agents collaborate: a user agent (generates queries), an assistant agent (determines tool calls), and a tool agent (simulates execution)
- Employs "formalized thinking" for reasoning and self-consistency via majority voting across multiple generations
- Generates both single-turn and multi-turn tool-calling dialogs

**3. Dual-Layer Verification (DLV):**
- Rule-based layer: validates syntactic correctness (API names, parameter formats, schema compliance)
- Model-based layer: detects hallucinations, validates consistency, and verifies tool responses using expert LLM agents

### Evidence

ToolACE-8B (LLaMA-3.1-8B fine-tuned with LoRA rank 16, alpha 32) on BFCL:

| Benchmark | ToolACE-8B | GPT-4 | Claude-3.5-Sonnet |
|-----------|-----------|-------|-------------------|
| BFCL v1 Overall | **91.41%** | 88.53% | 90.53% |
| BFCL v1 AST | 89.09% | — | — |
| BFCL v1 Exec | 98.24% | — | — |
| BFCL v2 Overall | 85.77% | — | — |

### Applicability

- **For this thesis:** The ToolACE pipeline could generate synthetic training data in the BFCL v4 format to fine-tune the existing 3B models (Qwen2.5-3B, Llama-3.2-3B). The dual-layer verification ensures data quality without human annotation.
- **Dataset availability:** The xlam-function-calling-60k dataset (derived from ToolBench, used by both ToolACE and xLAM) is publicly available on HuggingFace.
- **Complexity:** High for building the full pipeline; medium if using the existing public datasets.
- **Related (unscored):** [§24 Data Turnstile](#24-data-turnstile-open-function-calling-data-generation) is an **open** API-spec → multi-turn FC SFT generator (DAG roles + schema validation). Frame as an alternative / complement to Hub-gated APIGen-MT for **v7+ mixes**, not a scored thesis run.
- **Related (paper suite, not our harness):** **ToolACE-MT** (Zeng et al. 2025, ICLR 2026, [arXiv:2508.12685](https://arxiv.org/abs/2508.12685)) — non-AR skeleton → mask-fill → offline verify. Paper: Llama-3.1-8B MT **9.25%→40.25%** on **BFCL-v3** (vs MAS 31.38%). Do **not** write 40.25% as a thesis `multi_turn_base` score. Cross-link: [Post–Run27 consensus](#post-run27-consensus-2026-09-28).

### Sources

- Liu, T., et al. "ToolACE: Winning the Points of LLM Function Calling." ICLR 2025. arXiv:2409.00920. https://arxiv.org/abs/2409.00920
- Zeng et al. "ToolACE-MT." ICLR 2026. arXiv:2508.12685. https://arxiv.org/abs/2508.12685 (paper BFCL-v3 MT; **not** our harness)
- Salesforce. "xlam-function-calling-60k." HuggingFace Datasets. https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k

---

## 24. Data Turnstile: Open Function-Calling Data Generation

### Overview

Data Turnstile is an **open** framework that turns user-defined API specs into multi-turn function-calling SFT data. It is a training-time data generator (Amazon Science, Jul 2026), not an inference-time `--fc-mode`. Frame as an **alternative / complement to Hub-gated APIGen-MT** for **v7+ mixes** — **unscored** in this thesis. Do **not** invent a `multi_turn_base` number.

### Mechanism

The pipeline generates trajectories as a **DAG of typed roles**: USER / THINKING / API_CALL / API_OBS / ASSISTANT. Per-role generation uses **schema validation** and **error-feedback retries** (validate-before-generate; early abort). Diversity is template-controlled, with dynamic perturbations (execution failures, incomplete information / clarify). Generation is **execution-free** (unlike APIGen backends). The paper used an open-weight teacher (Qwen2.5-32B). Code and a public synthetic dump exist (`amazon-science/data-turnstile`; HuggingFace `amazon/Turnstile-Synthetic-Domains`, ~100K interactions, 1K+ APIs).

### Evidence

Paper results (Ramakrishnan & Sharma, 2026; cite as **theirs**, not ours):

- **BFCL single-turn:** Qwen3-0.6B Turnstile SFT **no-think** → **75.9%** overall (Turnstile-OOD+ID) vs base 0.6B **think** 67.4% / **no-think** 58.2%. Approaches Qwen3-1.7B think **78.4%** and 4B think **79.9%**.
- **Methodology ablation (same APIs):** Turnstile-OS **70.4%** vs Raw-OS **55.1%** (**+15.3 pp**).
- **τ²-bench Telecom (multi-turn, pass^1)** with Turnstile + tool-call-weighted SFT: Qwen3-1.7B **31.1%** (base 6.6%, ~4.7×); Qwen3-0.6B **24.6%** (base 3.5%, ~7×). At 1.7B, the paper table beats Qwen2.5-32B-Instruct **27.4%**.
- **CoT finding (their ablations):** chain-of-thought helps multi-turn agentic work and can **hurt** single-turn.

**Metric mismatch:** paper BFCL **single-turn** overall and τ²-bench Telecom ≠ this repo's BFCL v4 `multi_turn_base` execution score. Do **not** equate 75.9% / 31.1% / 24.6% to a thesis harness result.

### Applicability

- **Thesis role:** Optional **open data-gen complement** to Hub-gated APIGen-MT for SFT **v7+** mixes. Logged 2026-09-24; **not scored**. Does not unlock a new mix or change the locked SFT path (v4 IFFC **15.0%** remains best SFT recovery; v7 remains **unscored**).
- **Why it is interesting here:** Generates multi-turn FC SFT from **custom API specs** with schema-validated roles — a way to source MT-native trajs without relying only on Hub Accept datasets. Complements [§13 ToolACE](#13-toolace-automated-training-data-generation) (closed pipeline / public dumps) rather than replacing the v7 APIGen-MT core.
- **BFCL categories:** Paper evidence is single-turn BFCL + τ² Telecom. Relevance to `multi_turn_base` / `miss_func` / `miss_param` is **hypothesis only** until scored. Empty-`[]` / stop-taxonomy control still matters (v1–v6 lesson).
- **Complexity:** Medium–high to run their generator; low to *cite* as a v7-related option.
- **Limitation:** Open teacher + synthetic domains are not BFCL GorillaFileSystem / VehicleControl transcripts. Schema-valid calls can still be state-wrong.

### Sources

- Ramakrishnan, G. & Sharma, M. (Amazon AGI). "Data Turnstile: A Scalable Open Framework for Function-Calling Data Generation." arXiv:2607.29250, 2026. https://arxiv.org/abs/2607.29250
- Code: https://github.com/amazon-science/data-turnstile
- Dataset: https://huggingface.co/datasets/amazon/Turnstile-Synthetic-Domains

---

## 14. Hammer: Function Masking for Robust Tool Calling

### Overview

Hammer introduces **function masking** — a training-time augmentation that replaces function names, parameter names, and default values with random strings during fine-tuning. This forces the model to understand tool purpose from descriptions rather than memorizing naming patterns, yielding robust generalization to unseen APIs.

### Mechanism

During training, for a random subset of examples:
1. Replace function identifiers with random strings (e.g., `get_weather` → `fn_x7k2m`)
2. Replace parameter names with random tokens (e.g., `location` → `param_a3`)
3. Randomize default parameter values
4. Update training labels to match the masked names

This prevents the model from learning shallow correlations between function names and their purpose, forcing it to rely on the natural language descriptions instead.

### Irrelevance Detection Augmentation

Hammer identified an **inverse relationship** between function-calling accuracy and irrelevance detection: models that get better at calling tools get worse at knowing when *not* to call them. Solution: augment the training set with 7,500 examples where correct functions are deliberately excluded and labels are empty lists. Optimal ratio: ~10% irrelevance-augmented data in training batches.

### Evidence

Hammer models on BFCL (trained on xLAM-function-calling-60k + 7,500 irrelevance examples):

| Model | Base | Simple | Multiple | Parallel | Par. Mult. | Irrelevance |
|-------|------|--------|----------|----------|------------|-------------|
| Hammer-7B | Qwen2-7B | 69.31% | 82.52% | 78.88% | 84.08% | 72.87% |
| Hammer-4B | Qwen1.5-4B | — | — | — | — | — |
| Hammer-1.5B | Qwen2-1.5B | — | — | — | — | — |

Hammer-7B ranked #2 overall on BFCL, outperforming larger proprietary models on executable function evaluation. Critically, **Hammer achieved non-zero parallel scores** (78.88% parallel, 84.08% parallel_multiple) — demonstrating that fine-tuning can unlock parallel calling in small models, which all base models in this thesis score 0% on.

### Applicability

- **Directly addresses the 0% parallel problem:** Hammer demonstrates that the parallel calling failure is a training gap, not an architectural limitation. Fine-tuning with parallel examples could unlock this category for the 3B models.
- **Function masking is transferable:** The masking augmentation can be applied to any fine-tuning dataset, not just the Hammer pipeline. It could be combined with ToolACE or xLAM data.
- **Irrelevance augmentation:** Directly relevant to improving Qwen2.5-3B's 58.75% irrelevance score.
- **Model sizes:** Hammer-1.5B and Hammer-4B bracket the model sizes in this thesis, making the results directly comparable.

### Sources

- Ye, L., et al. "Hammer: Robust Function-Calling for On-Device Language Models via Function Masking." ICLR 2025. arXiv:2410.04587. https://arxiv.org/abs/2410.04587

---

## 15. xLAM: Large Action Models

### Overview

xLAM is Salesforce's family of Large Action Models, ranging from 1B to 8x22B parameters, fine-tuned specifically for function calling and agentic tasks. The 1B model is the smallest model on the BFCL leaderboard, with a **published** overall accuracy of 78.94% — outperforming GPT-3.5-Turbo. **That published overall figure is a different metric / suite aggregation than this repo's BFCL v4 `multi_turn_base` scores.** Run 19 NFC on **this** harness is **0/200 = 0.0%** (format-channel collapse). Run 20 `--fc-mode xlam` is the **format-fixed fair-channel** score: **6/200 = 3.0%** — still **very low**. Neither must be equated with 78.94%.

### Evidence

| Model | Parameters | BFCL Overall | Notes |
|-------|-----------|-------------|-------|
| xLAM-2-fc-r | — | **#1 on BFCL** (as of April 2025) | Latest generation |
| xLAM-7b-fc-r | 7B | 88.24% | #3 on leaderboard |
| xLAM-1b-fc-r | **1B** | **78.94%** (published overall; **not** this pipeline) | Smallest model on leaderboard; outperforms GPT-3.5-Turbo |

On τ-bench (agentic evaluation), xLAM-2-70b-fc-r achieves 56.2% success rate, outperforming GPT-4o (52.9%).

### Applicability

- **xLAM-1b-fc-r** is directly relevant to edge deployment — at 1B parameters it fits comfortably on all target hardware. GGUF versions are published on HuggingFace (`Salesforce/xLAM-1b-fc-r-gguf`).
- **As a benchmark model:** Registry key `xlam-1b` is in `benchmark/config.py` (`Salesforce/xLAM-1b-fc-r-gguf`, Q4_K_M; distinct `api_name` / `slug` so results do not overwrite other models). Serve with `-hf Salesforce/xLAM-1b-fc-r-gguf:Q4_K_M`. **Run 19 already scored `--fc-mode nfc` on BFCL v4 `multi_turn_base`: 0/200 = 0.0%** (Mac `run_id=19`; wall **13.55 min**; finished `2026-09-22T14:26:42Z`). This is a **channel/format collapse on our multi_turn NFC harness** (NFC never parses `content`). **All 200** eval errors: `Model response list is empty for turn 0`. Avg `api_calls` **3.63**; gen_tps_mean ≈ **196.7**. **Run 20 already scored `--fc-mode xlam` on the same GGUF / slice: 6/200 = 3.0%** (Mac `run_id=20`; wall **~13.28 h**; `2026-09-22T15:28:35Z` → `2026-09-23T04:45:30Z`). Adapter [PR #42](https://github.com/sskarz/Thesis/pull/42): no native tools payload; Salesforce-style JSON-in-text prompt; parse `content`. Smoke extracted calls (5/5). Full score is still **very low**. 194/200 eval errors; `state_mismatch` **128**; under_call/missing exec **46**; empty_turn **20**. Flags: `hit_step_limit` **171/200**; `xlam_unparseable` **111/200**; `context_overflow` **42/200**. Avg `api_calls` **50.47** (min 2, max 101) — tools actually ran vs Run 19 empty channel. gen_tps_mean ≈ **91.6** (n=50). Correct: `multi_turn_base_{50,119,133,139,144,161}`. Same-harness contrast: Gemma NFC Run 4 **27.0%**; CD **10.5%**. **Do not** claim xLAM scores 0 on BFCL overall. **Do not** equate 0.0% or 3.0% to published overall ~78.94%. **Do not** present 3.0% as a method win. Registry: [PR #41](https://github.com/sskarz/Thesis/pull/41). Full writeup: `THESIS.md`.
- **Training data:** The xlam-function-calling-60k dataset used for training is publicly available. **APIGen-MT → xLAM-2-fc-r** (Prabhakar et al. 2025, NeurIPS 2025 D&B, [arXiv:2504.03601](https://arxiv.org/abs/2504.03601); already the v7 mix core; **unscored** here): blueprint ground truth; **mask user/tool, train assistant only**; open [APIGen-MT-5k](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k). Paper MT (**theirs**, not this pipeline): **1B 43% / 3B 56% / 8B 69%**. Do **not** equate those % to Run 19/20 or Gemma `multi_turn_base`. Cross-link: [Post–Run27 consensus](#post-run27-consensus-2026-09-28).
- **No new method.** Run 19 is stock `--fc-mode nfc` on a published GGUF (format-channel collapse). Run 20 is the **already-merged** `--fc-mode xlam` format adapter ([PR #42](https://github.com/sskarz/Thesis/pull/42)) on that same GGUF — a **measured fair-channel baseline at 3.0%**, not a new training or inference-time method. **Do not claim published ~78.94% overall as a score from this pipeline.**

### Sources

- Zhang, J., et al. "xLAM: A Family of Large Action Models to Empower AI Agent Systems." arXiv:2409.03215, 2024. https://arxiv.org/abs/2409.03215
- Prabhakar et al. "APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay." NeurIPS 2025 D&B. arXiv:2504.03601. https://arxiv.org/abs/2504.03601
- Salesforce. "xLAM-1b-fc-r." HuggingFace. https://huggingface.co/Salesforce/xLAM-1b-fc-r

---

## 16. FunctionGemma: Specialist Fine-Tuning from Base Models

### Overview

FunctionGemma is Google's purpose-built function-calling model, fine-tuned from Gemma 3 270M. At 270M parameters (~512MB GGUF), it is the smallest dedicated tool-calling model available. Google provides a public fine-tuning recipe for users to specialize it further for domain-specific tools.

### Training Methodology

- **Base model:** Gemma 3 270M (256K vocabulary, 32K context, transformer architecture)
- **Training data:** 6T tokens including public tool definitions and tool-use interaction sequences (prompts, function calls, function responses, natural language summaries)
- **Infrastructure:** TPUv4p, TPUv5p, TPUv5e
- **Control tokens:** Strict chat template with dedicated tokens for function declarations (`<tool_call>`), function calls, and function responses
- **Knowledge cutoff:** August 2024

### Evidence

| Configuration | Mobile Actions Accuracy |
|--------------|------------------------|
| Base FunctionGemma | 58% |
| **+ Domain fine-tuning** | **85%** |

The +27pp gain from domain fine-tuning demonstrates that the base model provides a strong foundation that can be specialized further with relatively small datasets.

### Applicability

- **Already in the benchmark:** FunctionGemma-270M is one of the four models tested in this thesis (39.90% weighted BFCL accuracy). Domain-specific fine-tuning on BFCL-format data could significantly improve this.
- **Fine-tuning recipe:** Google's public cookbook provides step-by-step instructions for fine-tuning with custom tools, making this directly reproducible.
- **Edge deployment advantage:** At 512MB, even the fine-tuned model fits on Raspberry Pi 5, Jetson Orin Nano, and smartphones.
- **Non-standard output format:** FunctionGemma uses `<start_function_call>call:name{args}<end_function_call>` rather than OpenAI-style tool_calls. The existing `parse_functiongemma.py` handles this, but fine-tuning could also target the standard format.

### Sources

- Google. "FunctionGemma: New Gemma model for function calling." Google Blog, 2024. https://blog.google/technology/developers/functiongemma/
- Google. "A Guide to Fine-Tuning FunctionGemma." Google Developers Blog. https://developers.googleblog.com/en/a-guide-to-fine-tuning-functiongemma/
- Google. "FunctionGemma model card." https://ai.google.dev/gemma/docs/functiongemma/model_card

---

## 17. Structured Pruning + Fine-Tuning for Edge Deployment

### Overview

A combined approach that first reduces model size through structured pruning (removing layers and dimensions), then recovers capability through targeted fine-tuning. Demonstrated for in-vehicle function calling by pruning Phi-3 mini from 3.8B to 1.8B parameters while maintaining function-calling accuracy.

### Mechanism

**Pruning pipeline:**

1. **Depth-wise pruning:** Calculate angular distance between hidden states at each layer. Remove contiguous blocks of layers with highest similarity (they contribute least unique information). Example: removing layers 21–29 from Phi-3 mini (3.8B → 2.8B).
2. **Width-wise pruning:** Reduce hidden dimension (3072 → 2688), MLP dimension (8192 → 5120), and attention heads (32 → 28). Applied on top of depth-wise pruning (2.8B → 1.8B).

**Healing pipeline:**

3. **Initial healing:** QLoRA training for 5,000 steps on general web data to restore basic language capabilities
4. **Extended healing:** 45,000 steps (15B tokens) on educational web data to recover factual knowledge
5. **Instruction tuning:** 1 epoch on OpenHermes-2.5 for response quality
6. **Function-calling fine-tuning:** LoRA (rank 96, alpha 16) or full fine-tuning on 25,000 synthetic function-calling examples + 500 negative examples

### Evidence

| Model | Params | FC Accuracy | Gen. Speed (CPU) | GGUF Size |
|-------|--------|-------------|-------------------|-----------|
| Phi3-3.8B + LoRA | 3.8B | 0.86 | 6.76 tok/s | — |
| Phi3-2.8B + LoRA | 2.8B | **0.88** | 9.44 tok/s | — |
| Phi3-1.8B + LoRA | 1.8B | 0.86 | 11.02 tok/s | <2GB |
| Production baseline | — | 0.75 | — | — |

The pruned 1.8B model matches the full 3.8B model on function-calling accuracy (0.86 vs 0.86) while being 63% faster on CPU and fitting under 2GB with 4-bit quantization. Deployed via llama.cpp with GGUF conversion.

### Applicability

- **Directly applicable to thesis models:** The pruning + healing pipeline could be applied to Qwen2.5-3B or Llama-3.2-3B to create smaller, faster variants (e.g., 1.5B–2B) that maintain tool-calling capability.
- **Edge deployment narrative:** Demonstrates a path from 3B → sub-2B for Jetson Orin Nano Super (8GB) while preserving accuracy.
- **llama.cpp compatible:** The output GGUF models serve identically to base models.
- **Complexity:** High (multi-stage pipeline: prune → heal → instruct-tune → FC-tune).

### Source

Dhakal, S., et al. "Optimizing Small Language Models for In-Vehicle Function-Calling." arXiv:2501.02342, 2025. https://arxiv.org/abs/2501.02342

---

## 18. Parameter-Efficient Fine-Tuning (LoRA / QLoRA)

### Overview

LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) enable fine-tuning billion-parameter models on consumer hardware by training only a small number of additional parameters (rank-decomposed weight matrices) while keeping the base model frozen.

### Mechanism

**LoRA:** For each target weight matrix W ∈ ℝ^(d×k), add a low-rank decomposition BA where B ∈ ℝ^(d×r) and A ∈ ℝ^(r×k), with rank r << min(d, k). Only B and A are trained; W remains frozen. At inference, the adapter merges back: W' = W + BA, adding zero latency.

**QLoRA:** Extends LoRA by quantizing the frozen base model to 4-bit NormalFloat (NF4) precision and training LoRA adapters in FP16/BF16 on top. Key innovations:
- **4-bit NormalFloat:** Information-theoretically optimal for normally distributed weights
- **Double quantization:** Quantizes the quantization constants, saving ~0.37 bits/parameter
- **Paged optimizers:** Uses CPU memory for optimizer state during gradient spikes

### Evidence for Tool Calling

- **Hammer models:** Fine-tuned with standard SFT + LoRA, achieving BFCL #2 ranking at 7B (see Section 14)
- **ToolACE-8B:** LoRA rank 16, alpha 32 on LLaMA-3.1-8B, achieving BFCL #1 (see Section 13)
- **In-vehicle FC:** LoRA rank 96 on pruned Phi-3 variants, matching full fine-tuning accuracy (see Section 17)
- **QLoRA for 3B models:** A 3B model can be QLoRA-tuned on a single GPU with 16GB VRAM, making the RTX 4090 (24GB) more than sufficient

### Practical Details

| Hyperparameter | Typical Range | Notes |
|----------------|--------------|-------|
| Rank (r) | 8–96 | Higher for tool calling than general tasks |
| Alpha | 16–32 | Scaling factor; often 2× rank |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | All linear layers for best FC results |
| Learning rate | 1e-5 to 5e-5 | Lower than full fine-tuning |
| Epochs | 1–3 | Tool calling converges fast |
| Batch size | 4–32 | With gradient accumulation |

### LoRA Adapter Modularity

A key advantage: LoRA adapters are modular. Multiple adapters (e.g., one for Python tool calling, one for Java, one for irrelevance detection) can be trained independently and swapped at inference time. This enables dynamic customization without maintaining multiple full model copies.

### Applicability

- **Directly applicable:** QLoRA fine-tuning of Qwen2.5-3B or Llama-3.2-3B on tool-calling data fits within the RTX 4090's 24GB VRAM.
- **GGUF compatibility:** After training, merge the adapter into the base model and convert to GGUF for llama.cpp serving. The resulting model is identical in size and serving characteristics to the base model.
- **Frameworks:** HuggingFace TRL (SFTTrainer), Unsloth (2x faster LoRA training), axolotl.
- **Complexity:** Low-Medium. Standard tooling is mature and well-documented.

### Sources

- Dettmers, T., et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023. arXiv:2305.14314. https://arxiv.org/abs/2305.14314
- Hu, E., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. arXiv:2106.09685. https://arxiv.org/abs/2106.09685
- Microsoft. "Fine-Tuning Small Language Models for Function-Calling: A Comprehensive Guide." 2024. https://techcommunity.microsoft.com/blog/machinelearningblog/fine-tuning-small-language-models-for-function-calling-a-comprehensive-guide/4362539

---

## 19. Reinforcement Learning with Verifiable Rewards (GRPO)

### Overview

Group Relative Policy Optimization (GRPO) is a critic-free variant of PPO that trains models using verifiable rewards — binary signals from deterministic checkers (e.g., "did the tool call produce the correct output?") rather than learned reward models. GRPO has emerged as the dominant RL algorithm for improving tool-calling capability after initial SFT.

### Mechanism

1. **Sample N responses** for each prompt from the current policy
2. **Evaluate each response** with a verifiable reward function (e.g., execute the tool call and check the result)
3. **Compute group-relative advantages:** Each response's advantage is computed relative to the group mean, eliminating the need for a separate value network (critic)
4. **Update the policy** to increase probability of high-advantage responses

For tool calling, the reward function checks:
- Did the model call the correct function? (+1 / -1)
- Were the arguments correct? (partial credit possible)
- Did the model correctly decline to call a tool when none was needed?

### Evidence

- **DeepSeek-R1-Zero:** Trained purely with GRPO + verifiable rewards, exhibited emergent reasoning capabilities including chain-of-thought without explicit CoT training.
- **ToolRL (2025):** Explored reward design for tool-integrated GRPO training, showing that tool-use reward design is critical for boosting both tool calling accuracy and generalization.
- **ToRL (2025):** Integrated Python code interpreter into GRPO training for math tasks, surpassing RL baselines without tool integration on multiple datasets.
- **ReTool (2025):** Integrated tool calling in PPO training, jumping from 40% to 67% accuracy on AIME-2025 while needing 60% fewer gradient steps than text-only RL.

### Applicability

- **Post-SFT refinement:** GRPO is typically applied after initial SFT to further improve tool-calling accuracy. The BFCL AST checker provides a natural verifiable reward function.
- **Hardware requirement:** GRPO training requires sampling N responses per prompt, making it more compute-intensive than SFT. For 3B models with N=4, the RTX 4090 (24GB) is sufficient.
- **Frameworks:** OpenRLHF, VeRL, TRL all support GRPO training.
- **Complexity:** High. Requires setting up the RL training loop, reward function, and careful hyperparameter tuning.
- **Not the main training path:** Blanket GRPO on full trajectories is **not** unlocked as the next bet. SFT remains locked (2026-09-23). See [§25 Critical-State RL](#25-critical-state-rl-diagnosing-trainable-multi-turn-calls) for occurrence-local RL on a diagnosed turn only — an **unscored** research option **after healthy SFT**, distinct from this section. Logged 2026-09-28 GRPO-family options (still **not** a main-path unlock): [§29 PARL-MT](#29-parl-mt-progress-awareness-for-multi-turn-function-calling) (progress-aware PAG-RL) and [§31 RC-GRPO](#31-rc-grpo-reward-conditioned-grpo) (reward-token variance restoration).

### Sources

- Shao, Z., et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300, 2024. https://arxiv.org/abs/2402.03300
- Qian, C., et al. "ToolRL: Reward is All You Need for Tool Learning." 2025.
- Li, B., et al. "ToRL: Scaling Tool-Integrated RL." 2025.

---

## 25. Critical-State RL: Diagnosing Trainable Multi-Turn Calls

### Overview

Critical-State RL (Salesforce, Sep 2026) diagnoses **which** multi-turn tool call is actually trainable, then runs **occurrence-local** RL on that turn only. Trajectory-level reward variation is **not** the same as trainability. Distinct from [§19 blanket GRPO](#19-reinforcement-learning-with-verifiable-rewards-grpo) / always-decision / always-recovery fixed-role rules.

**Not a scored thesis experiment.** Do not invent a `multi_turn_base` number. Do **not** unlock GRPO as the main path. SFT remains locked (2026-09-23); this is an **unscored** research option **after healthy SFT**.

### Mechanism

1. **Task-defined candidate calls** plus local rewards.
2. **Nested sampling** at a fixed prefix: sample actions, hold an action fixed, resample continuations — separates **action-dependent** reward variance from continuation noise.
3. **Occurrence-local RL** (contextual bandit) **only** on the selected call. Surrounding turns provide context and reward but receive **no gradient**.

Relevant to this harness's **empty-`[]` / recovery** failure modes and to BFCL multi-turn `miss_func` / `miss_param` structure (tool appears later; argument arrives later).

### Evidence

Paper BFCL v4 results (Gemma **four-cell study in the paper**, cite as **theirs**, not our Runs 4–18):

- **`miss_func`:** diagnostic selects the **recovery** turn (after the tool becomes available): **0.14 → 0.283 ± 0.015** ≈ **+14.3 pp**. Training the **decision** alternative instead: 0.14 → 0.095 (**−4.5 pp**).
- **`miss_param`:** diagnostic selects the **decision** turn (before the missing argument arrives): **0.435 → 0.473 ± 0.010** ≈ **+3.8 pp**. Recovery alternative ≈ flat (**+1 pp**).
- **Nemotron `miss_func` application:** trains **decision** *before* tool availability (**+4.4 pt**) — location is **model/task-specific**.

Do **not** write these as thesis harness scores on `multi_turn_base`. Our measured Gemma `multi_turn_base` ladder is unchanged (BoN **36.5%** best inference; SFT-v4 **15.0%** best SFT recovery).

### Applicability

- **Thesis role:** Post-healthy-SFT **local RL** research option. Logged 2026-09-24; **unscored**. Not a substitute for SFT v7. Not BoN-on-collapsed-SFT. Not more epochs on v5/v6.
- **Why it is interesting here:** v4 restored calling (empty-`[]` largely fixed) but residual is state mismatch / under-call / never-start recovery. Nested sampling asks *which turn* to train rather than GRPO-ing the whole trajectory.
- **BFCL categories:** Paper cells are `miss_func` and `miss_param` (train splits in their study). Our primary thesis slice remains `multi_turn_base`; `miss_func` / `miss_param` stay secondary stress tests until someone scores them.
- **Complexity:** High (nested sampling + local RL loop). Higher than SFT mix work; not the next default after v7 scaffolding.
- **Limitation:** Requires a **healthy** SFT checkpoint. Applying local RL to v1/v2 collapse or below-IFFC v4–v6 adapters is out of scope.

### Source

Chen, Z., Zhao, W., et al. (Salesforce). "Critical-State RL: Diagnosing Trainable States for Multi-Turn Tool Use." arXiv:2609.24985, 2026. https://arxiv.org/abs/2609.24985

---

## 26. OODA-Tool: Typed Observe-Orient-Decide-Act

### Overview

OODA-Tool (Guo et al., 2026) is a typed Observe→Orient→Decide→Act controller with central-controller checks. It targets **state–action competition**: the pressure to emit the next call overwrites accumulated state (`state_mismatch`-like failures). Biggest paper gains are on **Qwen3 0.6B–4B**.

**Train vs infer: both** — stage LoRAs (Joint / Shared / Specialized) plus optional multi-pass inference. **Guo et al. paper method is not a scored thesis experiment.** Do not paste their ToolDial pp as a `multi_turn_base` number. A **separate** one-pass `[OODA]` card harness (`--fc-mode iffc-ooda-card`; **not** 4-pass Specialized; **not** iffc-state) is **scored** Run 42 **71/200 = 35.5%** — see [IFFC OODA card](#experiment-iffc-ooda-card---fc-mode-iffc-ooda-card). Cross-link [Post–Run27 consensus](#post-run27-consensus-2026-09-28) (do not duplicate a second ranking). Prefer **Joint typed CoT / one-pass** over Specialized ~2.4× multi-pass at edge.

### Mechanism

Inspired by Boyd’s Observe–Orient–Decide–Act cycle:

1. **Observe** reconstructs a provenance-aware task state.
2. **Orient** gates whether execution is warranted.
3. **Decide** forms an admissible action structure.
4. **Act** realizes a schema-valid tool call or authorized user-facing response.

A central controller checks each handoff. Training variants: Joint (typed CoT in one pass), Shared, Specialized (four-call loop). Specialized is the paper’s strongest Task Success variant and the expensive one.

### Evidence

Paper ToolDial Task Success, Specialized OODA vs Direct-LoRA (Guo et al., 2026; cite as **theirs**, not ours): **+6.86 / +6.79 / +6.99 / +5.94 / +4.48 pp** at 0.6B / 1.7B / 4B / 8B / 14B. Gains concentrate on state-intensive slices; smaller on parallel calls. Transfer: FAIL-TaLMs, MTU-Bench, BFCL (margin shrinks with scale; smaller on BFCL Multi-turn). Specialized ≈ **2.36×** latency vs Direct-LoRA at 1.7B in their ToolDial study.

Do **not** paste these as thesis `multi_turn_base` scores. Our measured ladder after Run 46: BoN **36.5%** best inference; Run 42 iffc-ooda-card **35.5%** still best non-BoN (`state_mismatch` gate **MISS**); STATE **34.0%**; Run 30 tighten **32.5%** scored win vs IFFC; IFFC Run 5 **30.5%** baseline; SFT-v4 **15.0%** best SFT; Run 46 iffc-ptc-a **13.5%** scored **FAIL** (closed); Run 27 ungated **27.0%** scored miss.

### Applicability

- **Thesis role:** Logged 2026-09-28 as a full section (was a light Post–Run27 cite). Guo et al. paper method remains **unscored**. Complements Post–Run27 “lightweight state prefix in **one pass**” — prefer Joint typed CoT over Specialized 4-call loop at edge. Relevant to VehicleControl / `state_mismatch`. The one-pass `[OODA]` **card** (`--fc-mode iffc-ooda-card`) is **scored** Run 42 **71/200 = 35.5%** (accuracy win vs Run 30; still below BoN; `state_mismatch` **71** gate **MISS**). That card is **not** this paper’s 4-pass Specialized loop.
- **BFCL categories:** Paper transfer includes BFCL Multi-turn (their suite). Our one-pass card on `multi_turn_base` is **measured** (Run 42); Guo et al. Specialized / Joint LoRA transfer is **not**.
- **Complexity:** Medium for Joint one-pass typed CoT; high for Specialized multi-pass + stage LoRAs.
- **Limitation:** Specialized latency (~2.4×) is a poor edge default. Direct-LoRA remains cheaper when one-pass latency is the constraint.

### Source

Guo, R., Huang, Y., Wu, Y., Zhong, M., Chen, Y., Tang, M., Long, T., & Hu, V. T. "From State to Action: OODA-Tool for Reliable Multi-Turn Tool Use." arXiv:2608.24368, 2026. https://arxiv.org/abs/2608.24368

---

## 29. PARL-MT: Progress Awareness for Multi-Turn Function Calling

### Overview

PARL-MT (Chai et al., ACL 2026 Findings; also styled **PROGRA** in notes) trains **progress awareness** = history summary + future plan. **PAG** synthesizes awareness data; **PAG-RL** (progress-awareness–guided GRPO) conditions rollouts on awareness to cut context redundancy and fight loops / lost global task. Eval: BFCL multi-turn + τ-bench; includes **xLAM-2-3B / 8B**.

**Train vs infer: training-time** (SFT warm-up on awareness + PAG-RL / GRPO). **Not a scored thesis experiment.** Do **not** unlock GRPO as the main path. SFT remains locked (2026-09-23). Analogous to [§25 Critical-State](#25-critical-state-rl-diagnosing-trainable-multi-turn-calls) framing — a logged research option, not the next default bet.

### Mechanism

1. **PAG:** synthesize (history summary, future plan) awareness pairs.
2. **SFT warm-up** on that awareness data.
3. **PAG-RL:** GRPO rollouts conditioned on awareness so local actions stay aligned with the global task and redundant context is compressed.

### Evidence

Paper Table 1 (Chai et al., 2025; cite as **theirs**, not ours) — e.g. Qwen2.5-7B BFCL Overall **8.25→10.33**; xLAM-2-3B τ-Bench Overall **25.45→31.52**; xLAM-2-8B **46.70→51.85**. Ablations: both PAG and PAG-RL needed.

**Metric mismatch:** their BFCL multi-turn executable accuracy / τ-bench ≠ our `multi_turn_base`-only sheet. Do **not** equate 10.33 / 31.52 / 51.85 to a thesis harness result.

### Applicability

- **Thesis role:** Logged 2026-09-28; **unscored**. Aligns with Post–Run27 anti-loop / progress theme and Magnet / Critical-State contrast. Optional **post-healthy-SFT** progress-aware RL — **not** blanket GRPO, **not** a substitute for SFT v7.
- **BFCL categories:** Paper is BFCL multi-turn overall + τ-bench. Hypothesis for our loop / `state_mismatch` residuals only until scored.
- **Complexity:** High (synthetic PAG + GRPO loop). Extra inference cost per action during RL rollout (paper limitation).
- **Limitation:** Does not unlock GRPO as the main path. Requires a healthy SFT checkpoint first.

### Source

Chai, H., Cao, Z., Ran, M., Yang, Y., Lin, J., Peng, X., Wang, H., Ding, R., Wan, Z., Wen, M., Liu, W., Zhang, W., Huang, F., & Wen, Y. "PARL-MT: Learning to Call Functions in Multi-Turn Conversation with Progress Awareness." ACL 2026 Findings. arXiv:2509.23206, 2025. https://arxiv.org/abs/2509.23206

---

## 30. HyperTool: Code-Block Execution Granularity

### Overview

HyperTool (Du*, Zhou*, et al., 2026) is an MCP-style **code-block** interface that folds deterministic multi-tool chains into **one outer call** (local vars / filter / aggregate inside the block). It attacks consecutive-call **context pressure** and reasoning fragmentation from atomic step-wise MCP. SFT on verified HyperTool trajectories in real MCP environments.

**Train vs infer: both** — new executable interface at inference + SFT on HyperTool-format trajectories. **Not a scored thesis experiment.** Complements [§3a MCP-GRANITE](#3a-tool-interface-granularity-mcp-granite) (schema cardinality) with **execution granularity** (what the model sees per turn).

### Mechanism

The model emits a code block that can call existing tools through their original schemas, store intermediates locally, and return only the task-relevant result. Deterministic subroutines stay out of the main trace. Supervision is SFT on execution-verified HyperTool trajectories, not atomic ReAct traces.

### Evidence

Paper MCP-Universe Avg Acc (Du, Zhou, et al., 2026; cite as **theirs**, not ours): Qwen3-8B **9.93%→33.33%**; Qwen3-32B **15.69%→35.29%** (their HyperTool-SFT). Beats same-backbone ReAct-SFT / CodeAct / AgentFold etc.

**Not** BFCL `multi_turn_base`. Do **not** equate 33.33% / 35.29% to a thesis harness score. Needs a sandbox for code execution (security limitation).

### Applicability

- **Thesis role:** Logged 2026-09-28; **unscored**. Relevant if consecutive multi-call context bloat dominates. BFCL harness is **not** MCP-Universe.
- **BFCL categories:** Hypothesis only for long multi-call traces. No `--fc-mode`.
- **Complexity:** High (sandbox + trajectory SFT in a new action space).
- **Limitation:** Code execution sandbox / security. Paper env is MCP-Universe, not GorillaFileSystem / VehicleControl.

### Source

Du, Y., Zhou, Y., Ge, Y., Wang, J., Pang, X., Tang, S., Zheng, T., Dai, B., Yang, J., & Chen, S. "HyperTool: Beyond Step-Wise Tool Calls for Tool-Augmented Agents." arXiv:2606.13663, 2026. https://arxiv.org/abs/2606.13663

---

## 31. RC-GRPO: Reward-Conditioned GRPO

### Overview

RC-GRPO (Zhong et al., 2026) addresses a GRPO pathology after strong SFT: groups can be all-0 or all-1 → vanishing group-normalized advantage. **RCTP** SFT on mixed-quality trajectories with discrete reward tokens (`<|high_reward|>` / `<|low_reward|>`); **RC-GRPO** samples diverse reward tokens inside each GRPO group to restore within-group variance. Evaluated on BFCLv4 multi-turn tool agents.

**Train vs infer: training-time** (RCTP-FT then RC-GRPO). Inference conditions on the high-reward token. **Not a scored thesis experiment.** Do **not** unlock GRPO as the main path. Log next to [§19 GRPO](#19-reinforcement-learning-with-verifiable-rewards-grpo) / [§25 Critical-State](#25-critical-state-rl-diagnosing-trainable-multi-turn-calls) as an RL-stabilization technique for multi-turn binary rewards.

### Mechanism

1. **RCTP-FT:** SFT on mixed-quality trajectories with a discrete reward token in the prompt so the policy can emit distinct quality on demand.
2. **RC-GRPO:** inside each GRPO group, sample diverse reward tokens and condition rollouts on them — restoring advantage signal when vanilla GRPO groups collapse to all-success or all-fail.
3. **Inference:** condition on `<|high_reward|>`.

### Evidence

Paper BFCLv4 multi-turn validation (Zhong et al., 2026; cite as **theirs**, not ours): Qwen2.5-7B RCTP-FT+RC-GRPO **85.00%** overall vs SFT+GRPO **48.75%** / RCTP+GRPO **73.75%**; LLaMA-3.1-8B **48.75%**. Their Qwen result exceeds listed closed API baselines in their table.

**Metric mismatch:** their BFCLv4 multi-turn validation split / categories ≠ our `multi_turn_base` 200-entry Mac harness. Do **not** write 85% as a thesis score. Our measured highs stay BoN **36.5%** / Run 42 iffc-ooda-card **35.5%** / STATE **34.0%** / Run 30 tighten **32.5%** / IFFC **30.5%** / SFT-v4 **15.0%** / Run 27 ungated **27.0%**.

### Applicability

- **Thesis role:** Logged 2026-09-28; **unscored**. Addresses a real pathology **if** we ever do GRPO post-SFT. Still **do not unlock GRPO as the main path**.
- **BFCL categories:** Paper is BFCLv4 multi-turn validation overall — not our 200-entry `multi_turn_base` sheet.
- **Complexity:** High (mixed-quality RCTP + GRPO). Higher than SFT mix work.
- **Limitation:** Requires strong SFT first. Not a substitute for v7. Not BoN-on-collapsed-SFT.

### Source

Zhong, H., Zhai, J., Song, L., Bian, J., Liu, Q., & Tan, T. "RC-GRPO: Reward-Conditioned Group Relative Policy Optimization for Multi-Turn Tool Calling Agents." arXiv:2602.03025, 2026. https://arxiv.org/abs/2602.03025

---

## 20. End-to-End Agent RL (Agent-R1 / VerlTool)

### Overview

Agent-R1 and VerlTool extend reinforcement learning to **multi-turn agentic interactions** where the model must make sequences of tool calls with environmental feedback between steps. Unlike single-turn RL, these frameworks handle credit assignment across multiple turns and tool invocations.

### Agent-R1 Mechanism

Extends the MDP framework to multi-turn agent interactions:
- **Action Mask:** Distinguishes agent-generated tokens from environmental feedback, enabling precise credit assignment
- **Process rewards:** Intermediate rewards for successful tool invocations (not just final outcome), providing denser learning signals
- **Loss masks:** Focus gradients exclusively on agent-generated tokens
- **Advantage masks:** Align credit assignment with actual agent decisions

### Agent-R1 Results

Trained on **Qwen2.5-3B-Instruct** (same base model as in this thesis) with a Wikipedia search tool:

| Algorithm | HotpotQA | 2WikiMultihop | Musique (OOD) | Average EM |
|-----------|----------|---------------|---------------|------------|
| GRPO | **0.4405** | **0.5741** | 0.1485 | **0.3877** |
| PPO | 0.4136 | 0.5468 | **0.1552** | 0.3719 |
| RLOO | 0.4089 | 0.5641 | 0.1419 | 0.3716 |
| Naive RAG (baseline) | — | — | — | 0.1328 |
| Base Tool Call | — | — | — | 0.0847 |

All RL methods provide 3–4x improvement over baselines. GRPO performed best overall; PPO showed strongest out-of-domain generalization. Disabling the advantage mask reduced PPO performance by 5.8pp, confirming its criticality for multi-turn credit assignment.

### VerlTool Framework

A unified framework for multi-turn agentic RL supporting diverse tool modalities:

- **Tool types:** Code execution, web search, SQL databases, vision processing, bash terminals
- **Asynchronous rollout:** Trajectories interact with tools independently, achieving ~2x speedup over synchronous execution
- **Reward design:** Task-specific; e.g., accuracy + tool usage incentive for search tasks, execution accuracy for SQL, multi-component rewards for visual reasoning
- **Results on Qwen2.5-7B:** 62.2% on math (AIME/AMC/MATH), 45.9% on search QA (vs. 35.0% baseline), 83.9% on SQL

### Applicability

- **Directly relevant to multi-turn BFCL:** Agent-R1's results on Qwen2.5-3B show that multi-turn RL training can dramatically improve agentic tool use for the exact base model used in this thesis.
- **BFCL as reward:** The BFCL execution-based evaluation for multi-turn categories provides a natural reward function for RL training.
- **Complexity:** Very high. Requires implementing the full RL training loop with tool execution environments, advantage masking, and multi-turn trajectory management.
- **Frameworks:** VeRL/VerlTool is open-source and provides the infrastructure.

### Sources

- Wang, K., et al. "Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning." arXiv:2511.14460, 2025. https://arxiv.org/abs/2511.14460
- Sun, Y., et al. "VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use." arXiv:2509.01055, 2025. https://arxiv.org/abs/2509.01055

---

## 21. DiaTool-DPO: Preference Optimization for Multi-Turn Tool Calling

### Overview

DiaTool-DPO applies Direct Preference Optimization to multi-turn tool-calling dialogues, treating the entire dialogue trajectory as the unit of optimization rather than individual model outputs. It models tool-augmented interactions as a Markov Decision Process with 5 dialogue states and 3 query types.

### Mechanism

**Five dialogue states:**

1. **Initial:** No history; system either rejects out-of-scope requests or awaits input
2. **Tool selected, incomplete slots:** Tool identified but missing required parameters; slot-filling Q&A occurs
3. **Tool selected, complete slots:** All arguments available for execution
4. **Waiting for tool response:** Tool execution in progress
5. **Complete:** Results received; generate completion message

**Three query types (by trajectory):**

- **Type 1:** User provides all parameters upfront (1→3→4→5)
- **Type 2:** Parameters missing, requires clarification (1→2→3→4→5)
- **Type 3:** Out-of-scope request, requires rejection (1→1)

**Preference pair construction:**
- Start with correct Type 1 trajectories from glaive-2.0 dataset
- Use ChatGPT to generate Type 2 variants (incomplete queries) and Type 3 examples (out-of-scope requests)
- Pair chosen (correct dialogue flow) against rejected (incorrect flow) trajectories
- 16,794 total paired trajectories (8,357 easy + 8,437 hard)

### Evidence

DiaTool-DPO on LLaMA3-8B-Instruct + proprietary 3.1B model:

| Metric | SFT-only | DiaTool-DPO | vs. GPT-4o |
|--------|----------|-------------|------------|
| Slot-filling | ~50% | **94.8%** | 94.8% of GPT-4o |
| Tool rejection | ~81% | **91.3%** | 91.3% of GPT-4o |
| Call accuracy | 0.857 | 0.857 | Stable |
| Macro average | — | **90.4%** | — |

**Key finding:** SFT alone cannot learn conversational tool-calling skills. Applying SFT to chosen trajectories only yielded 45% performance drops in tool-calling accuracy. DPO's contrastive learning is essential — the model needs to see both correct and incorrect dialogue flows.

### Applicability

- **Multi-turn BFCL categories:** DiaTool-DPO directly addresses the multi-turn tool calling evaluated in BFCL v4 (`multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`).
- **Irrelevance / miss_func:** The Type 3 (rejection) training maps directly to BFCL's irrelevance category and the `miss_func` multi-turn category where tools are initially withheld.
- **Tested at 3.1B:** The proprietary 3.1B variant validates applicability at the model sizes used in this thesis.
- **Complexity:** Medium-High. Requires constructing preference pairs and running DPO training (simpler than RL, harder than SFT).
- **Our smoke (health gate, not paper / not BFCL %):** DPO β0.2 smoke100 10-ID (Run 49) is a **PASS** on the same Acc-style gates used for SFT v9.2 — **not** a `multi_turn_base` N=200 score and **not** DiaTool-DPO paper numbers. Full DPO train remains **HOLD**. See [DPO β0.2 smoke100 PASS](#dpo-β02-smoke100-10-id-pass-run-49--health-gate-not-a-bfcl-).

### Source

Jung, S., et al. "DiaTool-DPO: Multi-Turn Direct Preference Optimization for Tool-Augmented Large Language Models." SIGdial 2025. arXiv:2504.02882. https://arxiv.org/abs/2504.02882

---

## 22. Knowledge Distillation for Tool Calling

### Overview

Transfer tool-calling capabilities from a large teacher model (e.g., GPT-4, Claude) to a small student model by training on the teacher's outputs rather than (or in addition to) ground-truth labels.

### Mechanism

Two primary approaches:

**1. Response-based distillation (most common for tool calling):**
- Run the teacher model on a large set of user queries with tool definitions
- Collect the teacher's tool-call outputs as training labels
- Fine-tune the student model to reproduce the teacher's outputs
- This is the approach used by most tool-calling datasets (ToolACE, xLAM, glaive)

**2. Logit-based distillation:**
- Train the student to match the teacher's output probability distribution (soft labels) rather than just the argmax output
- Requires access to teacher logits (not available for proprietary models)
- Preserves richer information about the teacher's uncertainty over tool choices

### Evidence

- **GPT-4o mini:** OpenAI's distilled version of GPT-4o maintains strong tool-calling performance at a fraction of the size and cost. Demonstrates commercial viability.
- **ACON distillation:** Kang et al. (2025, arXiv:2510.00615) distilled context compression capabilities from GPT-4.1 to Qwen3-8B and Phi-4, "preserving over 95% of the teacher's accuracy." **Author correction:** earlier METHODS drafts credited Xu; arXiv lists **Kang et al.** ReWOO remains Xu B. (arXiv:2305.18323); FunReason-MT remains Xu et al. (arXiv:2510.24645).
- **Multi-teacher distillation:** Recent work (2024–2025) leverages knowledge from multiple teacher models to create more robust student models that generalize better across tool-calling domains.

### Applicability

- **Dataset generation:** The most practical form of distillation for this thesis is response-based: use a strong model (GPT-4, Claude) to generate correct tool calls for BFCL-format queries, then fine-tune the SLMs on these outputs. **Choice/router KD plan below:** Claude is **paused** — **no Claude-as-teacher**.
- **Complementary to SFT:** Distillation produces the training data; SFT/LoRA consumes it. Works with all the fine-tuning methods above.
- **Cost consideration:** Generating distillation data from API models has a monetary cost. For BFCL's ~1,390 single-turn + 800 multi-turn test cases, this is modest.
- **Complexity:** Low (if using existing distillation datasets) to Medium (if generating custom data from teacher models).

### Planned: Choice/router KD (path B first) — ranking PASS; Acc harness IMPL banked; Acc still NO-GO

**Survey/plan plus executed ranking trail.** Ranking probes **ran** (tiny → richer → soft-menu **HOLD**; MLX LoRA Choice-SLM **PASS**). Acc harness IMPL **CONFIRM_PASS_WITH_FLAGS** (dual-server smoke; harness/impl readiness, **not** a BFCL Acc score). Acc still **NO-GO** (no ≤10-ID / no full-200). Plan source (summarized, not dumped): Codex `codex_KD_TOOLCHOICE_RESEARCH.md`, `codex_KD_TOOLCHOICE_PLAN.md`, `codex_KD_TOOLCHOICE_FEASIBILITY.md`, `codex_KD_TOOLCHOICE_DOCS_STATUS.md` (2026-10-01). Ranking STATUS: `codex_KD_PATH_B_TRAIN_RANK_STATUS.md` (tiny HOLD), `codex_KD_PATH_B_RICHER_PIVOT_STATUS.md` (recipe lock), `codex_KD_PATH_B_RICHER_TRAIN_RANK_STATUS.md` (+ SCORE.json), `codex_KD_PATH_B_SOFT_MENU_STATUS.md` (+ SCORE.json; 2026-10-01 ~12:16 ET), `codex_KD_PATH_B_MLX_LORA_CHOICE_TRAIN_RANK_STATUS.md` (+ SCORE.json; 2026-10-02 10:36 ET; ranking **PASS**, Acc **NO-GO**). Do not invent a separate ranking Codex CONFIRM. Acc harness: `codex_KD_PATH_B_CHOICE_SLM_ACC_HARNESS_IMPL_STATUS.md` (**PASS_WITH_FLAGS**, 2026-10-02 11:31 ET), independent verify `codex_KD_PATH_B_CHOICE_SLM_ACC_HARNESS_IMPL_CONFIRM.md` (**CONFIRM_PASS_WITH_FLAGS**, 2026-10-02 11:38 ET), smoke `choice_slm_acc_harness_smoke.md` (H1–H12).

**Motivation (sourced Acc only — not a forecast):** Run 61 Decision+OODA **52/200 = 26.0%** and Run 58 Decision **50/200 = 25.0%** did **not** fix `MODEL_WEAK_RANKING` (R58 WHY: tev1 turn-0 gold top-1 ≈ **38.8%** when Choice returned). Both sit well below IFFC **30.5%** / Run 30 **32.5%** / Run 42 **35.5%**. **Do not forecast post-KD Acc.** Decision Acc stays parked for tev1 @ these levers.

**Separate tracks:** Choice/ranking KD is **not** the Acc ladder. Ranking metrics are **not** BFCL `%`. Acc only after explicit Acc GOs, on **new** `run_id`s. **Never resume 30/42/51–61.** **No Claude-as-teacher** (Claude paused).

**ELI5:** practical tool-calling KD is mostly “copy the teacher’s next-tool answers” (response SFT). Soft logits help when a local open teacher can emit menu probabilities. Multi-turn is brittle: one wrong tool poisons later labels (error cascade, state/history, noop/irrelevance, vocab mismatch for closed APIs).

**What transfers (lit, not Acc):**

| Family | Mechanism | Thesis Choice fit |
|--------|-----------|-------------------|
| Response-based KD | Teacher tool-call / Choice strings → student SFT/LoRA | **Primary** — ToolACE / xLAM / glaive pattern; closed teachers OK |
| Logit / soft KD | Student matches teacher distribution over tools | **Optional** — local open teacher only; unavailable for closed API |
| Preference | gold ≻ wrong Choice / trajectory DPO | DiaTool-DPO; Magnet (SFT + multi-turn DPO) |

Also logged (paper suites **theirs**, not our harness): ODIA (response-based FC KD + router; arXiv:2507.08877); Agent Distillation (Kang et al., trajectory imitation into 0.5B–3B; arXiv:2505.17612); ACON compressor distill (Kang et al., arXiv:2510.00615) as evidence that distilling an **auxiliary head** can work — ACON itself is context compression, **not** next-tool Choice.

**Prefer path B first** (Choice/router student: next_tool ∈ `active ∪ {noop}`):

| Path | Student | Role |
|------|---------|------|
| **(B)** | Choice / router (tev1 successor or Choice-schema SLM) | **FIRST** — direct `MODEL_WEAK_RANKING` lever |
| **(A)** | Gemma-4-E2B-it Q4 IFFC | Secondary Acc-track SFT lane — **not** the primary Decision fix |
| **(C)** | Both | Ranking gate now **PASS**; Acc still **NO-GO** — C **not auto** |

**Teachers (when a later implement GO exists):** (1) BFCL gold next-tool / checker-aligned trajectories (preferred; no API); (2) local larger open mid-size for diversity / optional menu soft ranks; (3) API non-Claude only with explicit spend GO. **Forbidden:** Claude. **Forbidden as primary:** tev1 argmax (copies `MODEL_WEAK`; hard-negative / rejected side only). Optional preference mine from read-only R58/R61 `DECISION_LOG` where gold was on menu but tev1 top-1 ≠ gold.

**Gates (ordered — ranking trail ends PASS; Acc still later):**

1. **Smoke** — Choice schema parse / menu clamp / noop legal; sidecar HTTP healthy. Ops, **not** Acc. **Executed:** Acc harness dual-server smoke H1–H12 **PASS_WITH_FLAGS** (H7 **24/24**; FLAG `greedy_only`).
2. **Ranking pre-gate (NOT Acc)** — held-out Decision states: gold **top-1 / top-3 / mean-rank** (same family as R58 WHY). Suggest beating frozen tev1 on the **same** probe IDs (R58-era turn-0 top-1 ≈ 38.8% is a WHY baseline, not a BFCL %). Exact margin is a GO-time contract. **Executed:** tiny → richer → soft-menu **HOLD**, then MLX LoRA Choice-SLM **PASS** (trail below). Acc still **NO-GO**.
3. **≤10-ID Acc** — only on **explicit Acc GO**; new `run_id`; llama.cpp + BFCL v4 `multi_turn_base`.
4. **Full-200 Acc** — only on a **separate Acc GO**; new `run_id`; **never resume 30/42/51–61**.

Leakage: hold out `entry_id`s used for Acc claims; ranking probe set ≠ Acc claim set (or disclose overlap and do not claim Acc). Label/train offline; never co-schedule with Acc llama.cpp + Ollama. Ranking lift ≠ trajectory PASS (args, STATE, empty/`[]`, OODA/stop-repeat still matter). Acc only beats R58/R61 (25–26%) → **HOLD** (same class as weak Decision). **Do not invent a predicted Acc.** Ranking metrics **are not** BFCL `%`.

**Executed ranking trail (path B; ranking ≠ Acc):** CPU numpy students (Metal-safe) for tiny/richer/soft-menu, then MLX LoRA Choice-SLM. Tiny/richer: llama-server not contended; teacher = **BFCL gold** next-tool. Soft-menu: Gemma-4-E2B Q4 via llama-server `:8080` menu softmax T=2 on **train only** (426/426; teacher top-1=gold **280/426≈65.7%**); hard CE + BT still gold/train prefs. Frozen tev1 = ranking baseline / rejected side in prefs **only** — **not** teacher. Fit: gold **train** 426 rows / 117 `entry_id`s. Leakage **OK**: train ⊥ rank_holdout ⊥ acc_holdout by `entry_id`; acc_holdout unused for train and ranking; soft labels train-only. Prior-tool bag uses **gold previous tools** (oracle history — Acc would use executed). **MLX LoRA Choice-SLM** (`gemma4_e2b_choice_slm_lora_v1`): Stage A **hard CE only** (λ_soft=0; no prefs) on gold train **426/117**; best@**1500** val **0.731**; adapter `kd_choice/adapters-choice-slm-v1/best@1500/`; fused `kd_choice/fused/gemma-4-e2b-it-choice-v1/`; GGUF `kd_choice/fused/gemma-4-e2b-it-choice-v1-Q4_K_M.gguf`. Gate method: MLX menu-constrained tool-name LL with Gemma-4 chat template (matches train). `:8080` stopped for train, **restored** after. R61 Decision Acc remains parked at **52/200 = 26.0%** (unchanged; do not re-score). **Do not invent Acc.** Ranking metrics **are not** BFCL `%`.

| Student | Probe | n | student top-1 / top-3 / mean_rank | frozen tev1 | Gate |
|---------|-------|---|-----------------------------------|-------------|------|
| Tiny `small_classifier_tool_vocab_hard_CE` | R58-aligned | 115 | **23.5%** (0.2348) / 49.6% (0.4957) / 5.43 | **73.9%** (0.7391) / 87.8% / 1.90 | **HOLD** (FAIL ranking) |
| Richer `rich_classifier_v2_ce_plus_pref` | R58-aligned (primary) | 115 | **67.0%** (0.6696) / **86.1%** (0.8609) / **2.04** | **73.9%** / **87.8%** / **1.90** | **HOLD** (~7 pp top-1) |
| Same richer | R61-aligned (secondary) | 141 | **65.3%** / **83.7%** / **2.31** | **73.8%** / **87.9%** / **1.99** | **HOLD** |
| Soft-menu `rich_classifier_v2_soft_menu` | R58-aligned (primary) | 115 | **66.09%** (0.6609) / **85.22%** (0.8522) / **2.13** (2.1304) | **73.91%** / **87.83%** / **1.90** | **HOLD** (~7.8 pp vs tev1; **no lift** vs richer) |
| Same soft-menu | R61-aligned (secondary) | 141 | **65.96%** / **82.27%** / **2.33** | **73.76%** / **87.94%** / **1.99** | **HOLD** |
| MLX LoRA Choice-SLM `gemma4_e2b_choice_slm_lora_v1` | R58-aligned (primary) | 115 | **80.0%** (0.8000) / **96.5%** (0.9652) / **1.72** (1.7217) | **73.9%** (0.7391) / **87.8%** (0.8783) / **1.90** (1.8957) | **PASS** (Δ top-1 **+6.09 pp**; clear_top1=true; pairwise 21/9/85) |
| Same MLX LoRA Choice-SLM | R61 intersection (secondary; not full 141) | 115 | **80.0%** vs tev1 **73.0%** | **73.0%** | **PASS** on intersection; disclose n=115 |

Tiny: hard CE only (prefs unused); ask hash dim 64. Richer recipe lock: ask uni+bi hash **512**; tool↔ask overlap; CE on train gold (426/117) then Bradley-Terry on **190** train prefs (gold≻tev1-wrong); model `kd_choice/data/choice_rich_v2_train.json`. Big lift vs tiny (**+43.5 pp** top-1 on R58) but **does not clearly beat tev1** → ranking gate **HOLD**. Soft-menu: same rich v2 features; Soft-KD 30 ep → hard CE 15 → BT prefs 10; lr 0.06/0.04/0.03; model `kd_choice/data/choice_rich_v2_soft_menu.json`. R58 pairwise rank wins (student better / tev1 better / tie): **21 / 32 / 62**. Soft-menu **no lift** vs prior richer (**66.09%** vs **66.96%** top-1) and still behind frozen tev1 → ranking gate **HOLD**. Soft-menu CPU path exhausted without PASS (prior HOLD). **MLX LoRA Choice-SLM ranking PASS** (STATUS/SCORE 2026-10-02 10:36 ET; clear_top1=true; Stage A hard CE only; best@1500). Ranking ≠ Acc. Acc still **NO-GO** despite ranking PASS.

**Preferred next:** ranking **PASS** (MLX LoRA Choice-SLM **80.0%** vs tev1 **73.9%**). Acc harness IMPL **CONFIRM_PASS_WITH_FLAGS** (H7 dual stampede **24/24**; FLAG `probs_source=greedy_only`). Acc still **NO-GO**. Acc harness ready for **propose-only** Acc GO (≤10-ID first, new `run_id`s, exact `entry_id` list) — **not auto**. Still **NO-GO** until human Acc GO + exact `entry_id` list. Never resume 30/42/51–61. No Claude-as-teacher. Optional later: prefs Stage C / soft mix λ≤0.5 under new GO. **Do not Acc** until separate Acc GO + new `run_id`s.

**This docs pass:** bank Acc harness IMPL **CONFIRM_PASS_WITH_FLAGS** (dual-server smoke H7 **24/24**; FLAG `greedy_only` ≠ ranking menu-LL). Ranking **PASS** already on main ([PR #73](https://github.com/sskarz/Thesis/pull/73); MLX LoRA Choice-SLM **80.0%** vs tev1 **73.9%**) — keep that; do not invent Acc. Acc still **NO-GO**. Acc GO is **propose-only**, not auto — still **NO-GO** until human Acc GO + exact `entry_id` list. **NO-GO:** ≤10-ID Acc, full-200 Acc, Claude teacher, resume 30/42/51–61, auto Acc.

**Acc harness IMPL (CONFIRM_PASS_WITH_FLAGS; not Acc):** Choice-SLM Acc harness is implemented + dual-server smoke confirmed (2026-10-02). This is **harness/impl readiness**, **not** a BFCL Acc score. Independent verify **CONFIRM_PASS_WITH_FLAGS** (`codex_KD_PATH_B_CHOICE_SLM_ACC_HARNESS_IMPL_CONFIRM.md`, 11:38 ET) after impl **PASS_WITH_FLAGS** (`codex_KD_PATH_B_CHOICE_SLM_ACC_HARNESS_IMPL_STATUS.md`, 11:31 ET) and smoke H1–H12 (`choice_slm_acc_harness_smoke.md`).

- **Backend:** `DECISION_BACKEND=choice_slm_chat` factory in decision-probe sidecar; Thesis env-only (no second Choice client in Thesis).
- **Prompt parity:** Acc path uses exact shared `build_router_prompt` / `build_router_user_content` from `kd_choice/router_prompt.py`; no silent Decision state-card.
- **Ports:** `:8080` Acc Gemma Q4 (unchanged); `:8081` Choice Q4 fused `kd_choice/fused/gemma-4-e2b-it-choice-v1-Q4_K_M.gguf` (`serve_path=llama_cpp_8081_q4`).
- **H7 dual stampede:** n=24, n_ok=24, n_timeout=0, both ports up; verify light re-probe N=8 all OK.
- **ID refuse:** `FORBIDDEN_RUN_IDS` includes 30/42/51–61; refuse path confirmed.
- **FLAG:** `probs_source=greedy_only` (conf=1.0) — do **not** equate to ranking menu-LL top-1. Prefer `CHOICE_SLM_PROBS=menu_ll` on ≤10-ID Acc GO when latency allows. Menu-LL implemented but **not** Acc claim metric this pass.

Acc still **NO-GO** (no ≤10-ID, no full-200, no Acc launch). Acc GO remains **propose-only** (≤10-ID first, new `run_id`s) — **not auto**. Never resume 30/42/51–61. No Claude-as-teacher.

### Sources

- Gu, Y., et al. "MiniLLM: Knowledge Distillation of Large Language Models." ICLR 2024. arXiv:2306.08543. https://arxiv.org/abs/2306.08543
- Microsoft. "Distillation: Turning Smaller Models into High-Performance, Cost-Effective Solutions." 2024. https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/distillation-turning-smaller-models-into-high-performance-cost-effective-solutio/4355029
- Hinton, G., Vinyals, O., & Dean, J. "Distilling the Knowledge in a Neural Network." arXiv:1503.02531, 2015.
- Kang et al. "Acon: Optimizing Context Compression for Long-horizon LLM Agents." arXiv:2510.00615, 2025. https://arxiv.org/abs/2510.00615
- Kang et al. "Distilling LLM Agent into Small Models with Retrieval and Code Tools." NeurIPS 2025. arXiv:2505.17612.
- Zhang, Yang, Li, He, Gong. "ODIA." arXiv:2507.08877, 2025.
- Yin et al. "Magnet: Multi-turn Tool-use Data Synthesis and Distillation via Graph Translation." ACL 2025. arXiv:2503.07826.
- Jung et al. "DiaTool-DPO." SIGdial 2025. arXiv:2504.02882 (see [§21](#21-diatool-dpo-preference-optimization-for-multi-turn-tool-calling)).
- Codex notes (plan source, 2026-10-01): `codex_KD_TOOLCHOICE_RESEARCH.md`, `codex_KD_TOOLCHOICE_PLAN.md`, `codex_KD_TOOLCHOICE_FEASIBILITY.md`, `codex_KD_TOOLCHOICE_DOCS_STATUS.md`. Ranking trail: `codex_KD_PATH_B_TRAIN_RANK_STATUS.md` (tiny HOLD), `codex_KD_PATH_B_RICHER_PIVOT_STATUS.md`, `codex_KD_PATH_B_RICHER_TRAIN_RANK_STATUS.md` (+ SCORE.json), `codex_KD_PATH_B_SOFT_MENU_STATUS.md` (+ SCORE.json), `codex_KD_PATH_B_MLX_LORA_CHOICE_TRAIN_RANK_STATUS.md` (+ SCORE.json; 2026-10-02 10:36 ET; ranking **PASS**, Acc **NO-GO**). Acc harness IMPL (2026-10-02): `codex_KD_PATH_B_CHOICE_SLM_ACC_HARNESS_IMPL_STATUS.md` (**PASS_WITH_FLAGS**, 11:31 ET), `codex_KD_PATH_B_CHOICE_SLM_ACC_HARNESS_IMPL_CONFIRM.md` (**CONFIRM_PASS_WITH_FLAGS**, 11:38 ET), `choice_slm_acc_harness_smoke.md` (H1–H12; H7 **24/24**); FLAG `greedy_only`; Acc still **NO-GO**.

---

## 23. Curriculum Learning for Progressive Tool Mastery

### Overview

Train the model on tool-calling tasks in order of increasing difficulty, starting with simple single-function calls and progressing to parallel calls, multi-turn sequences, and irrelevance detection. This prevents the model from being overwhelmed by complex examples early in training.

### Mechanism

A typical curriculum for tool-calling fine-tuning:

1. **Stage 1 — Single function, all parameters provided:** Learn basic function call format and argument extraction
2. **Stage 2 — Multiple functions available, select correct one:** Learn tool discrimination
3. **Stage 3 — Missing parameters, require slot-filling:** Learn to identify and request missing information
4. **Stage 4 — Parallel function calls:** Learn to emit multiple calls in a single response
5. **Stage 5 — Irrelevance detection:** Learn when *not* to call any tool
6. **Stage 6 — Multi-turn sequences:** Learn to chain tool calls across turns with state management

### Evidence

- **Confucius framework:** Uses iterative learning with introspection feedback and an easy-to-difficult curriculum for tool calling.
- **Kimi K1.5 and DeepSeek-Prover-V2:** Both use curriculum learning in pretraining, demonstrating effectiveness at scale.
- **Key finding:** Training on samples far beyond the model's current capabilities degrades performance. Progressive difficulty prevents this by ensuring the model has foundational skills before attempting complex tasks.
- **Guided-Structured Templates (Section 1)** can be viewed as curriculum learning applied at inference time — the template enforces the same progressive reasoning stages that curriculum training encodes into weights.

### Applicability

- **Natural fit for BFCL categories:** BFCL's category structure (simple → multiple → parallel → parallel_multiple → irrelevance) forms a natural difficulty curriculum.
- **Addresses specific failure modes:** Training `simple` first ensures correct format; training `irrelevance` last avoids the accuracy/rejection tradeoff identified by Hammer.
- **Implementation:** Order training data by BFCL category difficulty, potentially with data replay (revisiting earlier stages periodically to prevent catastrophic forgetting).
- **Complexity:** Medium. Requires organizing training data into stages and potentially adjusting learning rates per stage.

### Sources

- Qin, Y., et al. "Confucius: Iterative Tool Learning from Introspection Feedback." arXiv:2308.14034, 2024.
- Raschka, S. "The State of Reinforcement Learning for LLM Reasoning." 2025. https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training

---

## Training-Time Prioritized Recommendations

Ranked by expected impact × feasibility for the current setup (Qwen2.5-3B, Llama-3.2-3B, Qwen3.5-4B, RTX 4090 with 24GB VRAM). **Post–Run42 (2026-09-29):** `--fc-mode iffc-ooda-card` **scored** (Run 42 **35.5%**; `state_mismatch` gate **MISS**; still below BoN). Tightened stop-repeat still **PASSES** (Run 30 **32.5%** vs IFFC **30.5%**; **do not overwrite**). Ranked next-SFT list in [Post–Run27 consensus](#post-run27-consensus-2026-09-28) is a **later** path, not the next immediate pivot — not a new `multi_turn_base` score, not GRPO-as-main-path, not SFT v9. **SFT still HOLD.**

| Priority | Technique | Target Problem | Expected Impact | Effort |
|----------|-----------|----------------|-----------------|--------|
| 1 | **SFT v7 (planned, not scored; recipe v6 is closed)** | Gemma `multi_turn_base` after Run 18 scored v6 as a **miss vs v4** (26/200 = 13.0% vs Run 15 15.0%, −2.0 pp / −4). Modest +1.5 pp vs Run 17 11.5%. Empty 1; state 118 / under-call 55; avg api_calls 13.34. **SFT path locked 2026-09-23.** Run 27 ungated stop-on-repeat is a **scored miss** on base (27.0% vs IFFC 30.5%). Run 30 tighten **PASSES** at **32.5%** vs IFFC 30.5%. | **Later** path (not the next immediate pivot after Run 30): [Post–Run27 ranking](#post-run27-consensus-2026-09-28) — APIGen-MT-5k + ToolACE/ToolACE-MT-style MT + ST FC; mask user/tool, train assistant only; action-class quotas + GAR; Magnet/Critical-State contrast then mDPO. Optional open data-gen complement: [Data Turnstile §24](#24-data-turnstile-open-function-calling-data-generation) (**unscored**). Optional post-healthy-SFT local RL: [Critical-State RL §25](#25-critical-state-rl-diagnosing-trainable-multi-turn-calls) (**unscored**; not blanket GRPO). **Do not** invent a v7 score. **Do not** pivot to SFT v9 ([PR #57](https://github.com/sskarz/Thesis/pull/57) stays draft/parked). **No BoN / no train / no v7 resume** as the reaction to Run 30. **Do not** BoN v6; **do not** more-epochs the v6 mix. Thin filter on this family is exhausted. v4 IFFC 15% remains best SFT recovery. Do **not** present 13.0% as a method win. | Planned (unscored; later path; v9 parked) |
| 2 | **Optional ablation: ToolACE-only vs xLAM-only (small scale)** | Isolate 5520 ToolACE `[]` targets (xLAM 0%) | Diagnose mix vs format | Medium |
| 3 | **Function masking augmentation (Hammer)** | Generalization, irrelevance | Robust to unseen APIs; +irrelevance | Medium |
| 4 | **Add xLAM-1b-fc-r as benchmark baseline** | Comparison point. Run 19 NFC **0/200 = 0.0%** was format-channel collapse (JSON in `content`, not `tool_calls`). Fair path `--fc-mode xlam` is **scored**. | **Scored NFC:** 0.0% format-channel collapse. **Scored `--fc-mode xlam`:** Run 20 **6/200 = 3.0%** (fair channel; still **very low**; tools actually ran, avg `api_calls` **50.47**). **Not** published overall ~78.94%. **Not** a method win. Still **below CD 10.5%**. Registry [PR #41](https://github.com/sskarz/Thesis/pull/41); adapter [PR #42](https://github.com/sskarz/Thesis/pull/42). Do **not** overwrite Run 19 or 20. | **Done** (NFC + fair-channel scored) |
| 5 | **GRPO post-SFT with BFCL checker as reward** | All categories, esp. multi-turn | **Not the main path.** Logged alternatives after healthy SFT: [Critical-State RL §25](#25-critical-state-rl-diagnosing-trainable-multi-turn-calls) (occurrence-local; **unscored**); [PARL-MT §29](#29-parl-mt-progress-awareness-for-multi-turn-function-calling) (progress-aware PAG-RL; **unscored**); [RC-GRPO §31](#31-rc-grpo-reward-conditioned-grpo) (reward-token variance; **unscored**). Distinct from blanket GRPO on the full trajectory. Do **not** unlock GRPO as the next bet. Paper-style +5–15pp / 85% claims are **not** a thesis harness score. | High |
| 6 | **DiaTool-DPO for multi-turn** | Multi-turn categories | Better dialogue flow + rejection | High |
| 7 | **FunctionGemma domain fine-tuning** | FunctionGemma-270M scores | 58% → 85% on domain tasks | Medium |
| 8 | **Structured pruning (Phi-3 style)** | Edge size reduction | 3B → 1.8B with maintained accuracy | Very High |
| 9 | **Curriculum-ordered training** | Training stability | Prevents catastrophic forgetting | Medium |

### Recommended Training Pipeline

For a complete training-time experiment within this thesis:

```
Base model: Gemma-4-E2B-it (google/gemma-4-e2b-it)
    ↓ v1 mix (ToolACE-all + 25k xLAM, IFFC JSON-in-text) is SCORED NEGATIVE
      Run 12: 1/200 = 0.5% — stop-after-tools from ToolACE [] targets
    ↓ v2 mix: drop non-terminal [] after tool results (PR #24; 334→0 non-terminal;
      552 terminal empty-after-exec kept) is SCORED NEGATIVE
      Run 13: 1/200 = 0.5% — same collapse; filter insufficient
    ↓ DIAGNOSTIC (harness only, not training): strip trailing assistant []
      from chat_history on the same v2 GGUF (Run 14): 4/200 = 2.0% —
      did NOT rescue never-start collapse; do not present 2% as a method win
    ↓ v4 mix (PR #31): v3 empty-after-exec + drop first-assistant-empty and
      clarifying [] ; LoRA best@1600 val 0.196 is SCORED RECOVERY
      Run 15: 30/200 = 15.0% — empty [] largely fixed (2/170); still below
      IFFC 30.5% and NFC 27.0%; do not present as beating IFFC
    ↓ IFFC+STATE on this same v4 GGUF is SCORED NEGATIVE / NULL
      Run 16: 26/200 = 13.0% (−4 / −2.0 pp vs Run 15). STATE helps base Gemma,
      not this ckpt. Do NOT BoN this checkpoint as the next win.
    ↓ v5 mix (PR #35): keep v4 empty-recovery + turn-0 completeness +
      env-state consistency; LoRA best@1800 val 0.147 is SCORED REGRESSION
      Run 17: 23/200 = 11.5% (−7 / −3.5 pp vs Run 15). Empty still 0;
      state 133 / under-call 44; avg api_calls/entry 17.3→40.3.
      Hypothesis: synthetic complete turn-0. Do NOT present as a method win.
      Do NOT BoN this ckpt; do NOT more-epochs the v5 mix.
    ↓ v6 mix (PR #38): v4 empty/recovery + only drop_state_inconsistent;
      turn-0 rewrites off; LoRA best@1800 val 0.142 is SCORED MISS vs v4
      Run 18: 26/200 = 13.0% (−4 / −2.0 pp vs Run 15; +3 / +1.5 pp vs Run 17).
      Empty still 1 (_43); state 118 / under-call 55; avg api_calls 13.34
      (fixed v5's 40.3 churn). Do NOT present as a method win.
      Do NOT BoN this ckpt; do NOT more-epochs the v6 mix.
    ↓ Tightened stop-repeat PASSES (Run 30 65/200=0.325 vs IFFC 30.5%;
      +2.0 percentage points vs Run 5 (65-61 correct), NOT "+4pp";
      +5.5 pp vs Run 27 ungated 27.0%). Thursday: BASE + iffc-stop-repeat
      (observe_same gated after 6 tool-turns) → 32.5% vs IFFC 30.5% /
      ungated 27%. First scored win vs plain IFFC on BASE. Celebrate /
      park SFT-v9: YES. Still below STATE 34.0% / BoN 36.5%. Guardrail
      on BASE — not an SFT/method win. finished_at NULL (metrics 4/5
      soft bookkeeping; accuracy locked in scores); status JSON stale —
      trust DB+log; --resume-run-id 30 after orphan; N=200 complete.
      Do NOT overwrite BoN / IFFC / SFT-v4 / this row. Codex CONFIRM_PASS.
    ↓ IFFC OODA-card SCORED (Run 42 71/200=0.355 vs Run 30 32.5%;
      +6 / +3.0 pp). Dedicated --fc-mode iffc-ooda-card (PR #61 @ 1dea341);
      NOT a flag on stop-repeat; NOT iffc-state; NOT 4-pass OODA.
      NEW run — never resume Run 30. 200/200 responses. Finished
      2026-09-29T05:51:02.787870+00:00. Still below BoN 36.5% (−1.0 pp);
      above STATE 34.0% (+1.5 pp). state_mismatch 71 GATE MISS vs <60;
      empty+missing 58 MET; TTFT 517.8 vs Run 30 630.5 ms MET.
      Do NOT claim a clean gate sweep. Calib Run 40 ≠ BFCL %.
      Run 41 orphan ignore. SFT still parked. No BoN. Do NOT invent tok/s.
    ↓ SFT v9.2 10-ID smoke-gate FAIL (Run 36; 2026-09-28): avg api 25.7
      (gate ≤8.3), hit_step_limit 6/10 (gate ≤2). Smoke@100 healthy
      (val→0.645); full LoRA never started. Worse than Run 33 v9.1
      (17.6 / 5/10) and Run 31 v9 (22.5 / 5/10). NOT a multi_turn_base
      accuracy — do NOT invent a v9 BFCL %. Do NOT cite Run 35 (INVALID).
      Run 37 VC sidecar diagnostic only — not in /10. PR #57 stays draft
      @cde93a3 (advance-band shrink; kept v9.1 final_after_obs). Codex
      HOLD_sft_quota_remix. No BoN / no v7 resume / no full LoRA on
      failed mixes. Next after Thursday advisor: Stage B advance_vs_loop
      pref pairs (stub → audited builder), NOT another quota remix.
    ↓ Later path: Post–Run27 ranked SFT (2026-09-28; NOT scored) —
      APIGen-MT-5k + ToolACE/ToolACE-MT-style MT + ST FC; Gemma template;
      mask user/tool, train assistant only; action-class quota remix HOLD
      (v9.2 10-ID FAIL); next after advisor = Stage B advance_vs_loop
      pref pairs then Magnet/Critical-State mDPO after SFT.
      Optional open data-gen complement: Data Turnstile (§24; unscored).
      Optional post-healthy-SFT local RL: Critical-State RL (§25; unscored;
      not blanket GRPO). Do NOT unlock GRPO as the main path.
      2026-09-28 methods sweep (§26–§31; paper OODA unscored; one-pass
      [OODA] card SCORED Run 42): OODA one-pass preferred;
      PTC / Looped LMs inference; PARL-MT / RC-GRPO training-time (not a
      GRPO unlock); HyperTool execution granularity.
      SFT path locked 2026-09-23 as the main path; inference wrappers secondary.
      Not another thin filter on the same family. Not decode bans as the win.
      Not a third quota remix. Not invented v9 BFCL %.
      BoN 36.5% still best on base Gemma. Run 42 35.5% is second inference
      (state_mismatch gate MISS). v4 IFFC 15.0% remains best SFT recovery.
    ↓ Frozen split seed=42 (~90/5/5 train/val/corpus_test); BFCL multi_turn_base held out
    ↓ mlx-lm LoRA on Apple Silicon (--mask-prompt, rank 16–32)
    ↓ Fuse adapter, convert to GGUF (Q4_K_M), serve via llama-server
    ↓ Evaluate on BFCL v4 multi_turn_base vs NFC 27.0% / IFFC 30.5% / SFT v1 0.5% / SFT v2 0.5%
      / Run 14 DIAGNOSTIC strip-[] 2.0% (not a method win) / Run 15 SFT-v4 15.0%
      (scored recovery, still below IFFC/NFC) / Run 16 v4+STATE 13.0% (null/negative)
      / Run 17 SFT-v5 11.5% (scored regression vs v4)
      / Run 18 SFT-v6 13.0% (scored miss vs v4)
    ↓ Do not more epochs on v1 or v2 or v4 or v5 or v6, BoN/STATE on the Run 12 / 13 / 17 / 18 GGUF,
      fuse final@2000 first, claim the [] filter fixed SFT, present Run 14 2% as a win,
      present Run 15 15% as beating IFFC, present Run 16 13% as a STATE win,
      present Run 17 11.5% as a method win, present Run 18 13.0% as a method win,
      or BoN the SFT-v4 or SFT-v5 or SFT-v6 checkpoint
```

This pipeline answers the central thesis question: **does fine-tuning or inference-time optimization close the SLM-to-cloud-model gap more effectively for tool calling on edge devices?**

---

## Post-Run27 consensus (2026-09-28)

Dated fold of small-LLM (≤5–8B) multi-turn tool-calling literature after Run 27. **Does not invent a `multi_turn_base` score.** Paper BFCL percentages below are **their** suites (often BFCL-v3 overall / paper MT) — **not** this repo’s BFCL v4 `multi_turn_base` execution checker. Do **not** claim paper % on our harness. Do **not** unlock GRPO as the main path.

### Our measured context (authoritative)

- Run 5 BASE IFFC: **61/200 = 30.5%** (baseline; do **not** overwrite)
- Run 27 BASE `--fc-mode iffc-stop-repeat` **ungated**: **54/200 = 27.0%** (−3.5 pp / −7). **200/200 responses.** Stop flags **37/200**, all among failures (**0 stopped and correct**). VehicleControlAPI **22→33** vs Run 5. Same accuracy level as Run 4 NFC 27.0% — **not** an identical method. Evidence: [PR #55](https://github.com/sskarz/Thesis/pull/55) / [IFFC stop-on-repeat experiment](#experiment-iffc-stop-on-repeat---fc-mode-iffc-stop-repeat). **Scored miss. Ungated decode heuristic is not a free win on healthy base.**
- Run 30 BASE `--fc-mode iffc-stop-repeat` **tighten MIN=6** @ `78c22a0` (PR #54): **65/200 = 0.325** (`scores.id=21`; `multi_turn_base` only; **200/200 responses + timings**; `entry_id` identical to Run 5/Run 27; bartowski Q4_K_M). **+2.0 percentage points vs Run 5 (65−61 correct)** — **not** “+4pp”; **+5.5 pp vs Run 27**. Still **below STATE 34.0% and BoN 36.5%**. Stop flags **37→3**, all FAIL. VC state **22** (recovered from Run 27’s 33). `state_mismatch` **60** (=Run 5). **First scored win of tightened stop-repeat over plain IFFC on BASE.** **Celebrate / park SFT-v9: YES.** **Caveats:** harness/decode guardrail on BASE — **not** an SFT/method win; `finished_at` still **NULL** (metrics 4/5 soft bookkeeping; accuracy locked in `scores`); status JSON **stale** — trust DB+log; `--resume-run-id 30` after orphan; **N=200 complete**. **Codex CONFIRM_PASS.** Do **not** overwrite this row.
- Run 42 BASE `--fc-mode iffc-ooda-card` ([PR #61](https://github.com/sskarz/Thesis/pull/61) draft @ `1dea341`): **71/200 = 0.355**. Dedicated mode — **not** a flag on stop-repeat; **not** iffc-state; **not** 4-pass OODA. **NEW run** (never resume Run 30). **200/200 responses.** Finished `2026-09-29T05:51:02.787870+00:00`. **+6 / +3.0 pp vs Run 30**. Still **below BoN 36.5% (−1.0 pp)**; **above STATE 34.0% (+1.5 pp)**. `state_mismatch` **71** gate **MISS** vs <60; empty+missing **58** MET; TTFT **517.8 ms** vs Run 30 **630.5 ms** (≤1.25× **MET**). **Do not claim a clean gate sweep.** Calib Run 40 10-ID **PASS** (not a BFCL %). Run 41 orphan 0 responses — ignore. SFT parked. No BoN. Do **not** invent tok/s. Details: [IFFC OODA card](#experiment-iffc-ooda-card---fc-mode-iffc-ooda-card).
- Best inference BoN **36.5%** (Run 11). Second inference Run 42 **35.5%**. Best SFT v4 IFFC **15.0%** (Run 15). Do **not** overwrite BoN / IFFC / SFT-v4 / Run 30.
- Teacher loop-filters v8/v8.1 failed; ungated stop-on-repeat helped collapsed smoke but hurt healthy base; **gated recover** past IFFC.
- **SFT v9.2 10-ID smoke-gate FAIL** (Run 36): avg `api_calls` **25.7** (gate ≤8.3), `hit_step_limit` **6/10** (gate ≤2). Smoke@100 healthy (val→**0.645**); full LoRA never started. Worse than Run 33 v9.1 (17.6 / 5/10) and Run 31 v9 (22.5 / 5/10). **Not** a `multi_turn_base` %; **do not invent** a v9 BFCL %. **Do not cite Run 35** (INVALID — server died). Run 37 VC sidecar diagnostic only — not in /10. Codex **`HOLD_sft_quota_remix`**. Details: [SFT v9.2 smoke-gate FAIL](#sft-v9--v91--v92--lever-2-action-quotas-10-id-smoke-gate-fail--hold).
- **DPO β0.2 smoke100 10-ID PASS** (Run 49): avg API **4.4**, `hit_step_limit` **0/10**, `iffc_unparseable` **0/10**, `turn0_tool` **7/10** — all gates **MET**. Same 10 IDs as Run 47 (β0.1 turn0 **6/10**, avg **4.50**) / Run 48 (BASE turn0 **10/10**, avg **6.80**). **Health gate, not a BFCL %.** **Do not invent** a Run 49 / 47 / 48 `multi_turn_base` %. Full DPO train still **HOLD** pending Sanskar GO + Codex verify. Details: [DPO β0.2 smoke100 PASS](#dpo-β02-smoke100-10-id-pass-run-49--health-gate-not-a-bfcl-).

**Tightened stop-repeat PASSES** (Run 30 **32.5%**). Run 42 iffc-ooda-card **scored 35.5%** (`state_mismatch` gate **MISS**; still below BoN). SFT v9.2 smoke-gate **FAIL → park**. Quota remix **HOLD**. DPO β0.2 smoke100 10-ID **PASS**; DPO-full Run 51 **53/200 = 26.5%** scored (**not** a method win); Run 61 Decision+OODA **26.0% FAIL**; Run 58 Decision **25.0% FAIL**. Choice/router KD **path B first** ranking **PASS** (tiny ~**23.5%** / richer ~**67.0%** / soft-menu ~**66.1%** HOLD vs tev1 ~**73.9%**; MLX LoRA Choice-SLM **80.0%** vs tev1 **73.9%**, Δ **+6.09 pp**, clear_top1; ranking ≠ Acc; Acc harness IMPL **CONFIRM_PASS_WITH_FLAGS**, H7 **24/24**, FLAG `greedy_only`; Acc still **NO-GO**; Acc GO propose-only, not auto) — [§22](#22-knowledge-distillation-for-tool-calling). Next after Thursday advisor: Stage B **advance_vs_loop** pref pairs (stub → audited same-prefix builder), **not** another quota remix. [PR #57](https://github.com/sskarz/Thesis/pull/57) stays **draft** @`cde93a3` — **do not merge**. **No BoN, no v7 resume, no full LoRA on failed mixes, no invented v9 / Run 49 BFCL %.**

### Theme

**Verified multi-turn trajectories + action-class balance beat ungated decode heuristics.** GAR explains the Run 27 Acc regress: Acc ≤ GAR; SRI/CRI-like decode can **hurt Acc** even when it looks calibrated (Zhao et al., *Calibration is the Bottleneck*, [arXiv:2609.00949](https://arxiv.org/abs/2609.00949), 2026). Ungated force-stop on repeat is CRI-adjacent; on a healthy IFFC channel it cut Acc **30.5% → 27.0%** while VehicleControl state mismatch rose. **Gated recover:** MIN=6 `observe_same` recovered past IFFC (**27.0% → 32.5%**; VehicleControl **33→22**). Do **not** invent a new paper score.

### Cite map (paper suite ≠ our harness)

| Work | Cite | Notes (theirs, not ours) |
|------|------|--------------------------|
| ToolACE | Liu et al. 2025, ICLR 2025, [arXiv:2409.00920](https://arxiv.org/abs/2409.00920) — [§13](#13-toolace-automated-training-data-generation) | ~26.5k APIs; non-tool ask/decline |
| ToolACE-MT | Zeng et al. 2025, ICLR 2026, [arXiv:2508.12685](https://arxiv.org/abs/2508.12685) | Non-AR skeleton → mask-fill → offline verify. Paper: Llama-3.1-8B MT **9.25%→40.25%** BFCL-v3 (vs MAS 31.38%) |
| APIGen-MT → xLAM-2-fc-r | Prabhakar et al. 2025, NeurIPS 2025 D&B, [arXiv:2504.03601](https://arxiv.org/abs/2504.03601); [§15](#15-xlam-large-action-models) | Blueprint GT; **mask user/tool, train assistant only**; open APIGen-MT-5k. Paper MT **1B 43% / 3B 56% / 8B 69%** |
| FunReason-MT | Xu et al. 2025, [arXiv:2510.24645](https://arxiv.org/abs/2510.24645) | Qwen3-4B SFT MT **15.75→46.90**; **+RL →57.75** BFCL-v3. Cite as paper; **not** a GRPO unlock here |
| Magnet (mDPO) | Yin et al. 2025, ACL 2025, [arXiv:2503.07826](https://arxiv.org/abs/2503.07826) | Qwen2.5-Coder-14B; 34k SFT + ~4.5k pos/neg → mDPO |
| GAR / Calibration is the Bottleneck | Zhao et al. 2026, [arXiv:2609.00949](https://arxiv.org/abs/2609.00949) | 4-class tool_call / ask / refuse / confirm; Acc ≤ GAR; SRI/CRI can hurt Acc — aligns with stop-repeat |
| Critical-State RL | Chen et al. (Salesforce) 2026, [arXiv:2609.24985](https://arxiv.org/abs/2609.24985) — [§25](#25-critical-state-rl-diagnosing-trainable-multi-turn-calls) | Already logged; **unscored**. Occurrence-local RL after healthy SFT. **Not** blanket GRPO |
| OODA-Tool | Guo et al. 2026, [arXiv:2608.24368](https://arxiv.org/abs/2608.24368) — [§26](#26-ooda-tool-typed-observe-orient-decide-act) | Full section 2026-09-28; **paper method unscored**. Qwen3 0.6–14B; typed OODA; largest gains on smallest (~+7 pp ≤4B); Specialized ≈ 2.36× — prefer Joint typed CoT **one-pass** at edge. **Separate** one-pass `[OODA]` card harness **scored** Run 42 **71/200 = 35.5%** ([experiment](#experiment-iffc-ooda-card---fc-mode-iffc-ooda-card); **not** 4-pass Specialized; `state_mismatch` gate **MISS**) |
| PTC-Decoder | Yu, Mu, & Wu 2026, [arXiv:2609.30836](https://arxiv.org/abs/2609.30836) — [§27](#27-ptc-decoder-plan-to-act-plus-tool-name-dfa) | Paper method **theirs** (Jetson/satellite LLM-judge; **not** our `multi_turn_base`). **Separate** live-name DFA harness `--fc-mode iffc-ptc-a` **scored FAIL** Run 46 **27/200 = 13.5%** ([experiment](#experiment-iffc-ptc-a---fc-mode-iffc-ptc-a); **bank and close**; PTC-B **HOLD**) |
| Looped LMs | Popescu et al. 2026, [arXiv:2608.18171](https://arxiv.org/abs/2608.18171) — [§28](#28-looped-language-models-for-compositional-tool-calling) | Logged 2026-09-28; **unscored**. Mostly test-time compute. Paper BFCL static single-turn; live/MT left future |
| PARL-MT / PROGRA | Chai et al. ACL 2026 Findings, [arXiv:2509.23206](https://arxiv.org/abs/2509.23206) — [§29](#29-parl-mt-progress-awareness-for-multi-turn-function-calling) | Logged 2026-09-28; **unscored**. **Training-time** PAG + PAG-RL. **Not** a GRPO unlock |
| HyperTool | Du*, Zhou*, et al. 2026, [arXiv:2606.13663](https://arxiv.org/abs/2606.13663) — [§30](#30-hypertool-code-block-execution-granularity) | Logged 2026-09-28; **unscored**. **Train + infer** code-block granularity. MCP-Universe ≠ BFCL |
| RC-GRPO | Zhong et al. 2026, [arXiv:2602.03025](https://arxiv.org/abs/2602.03025) — [§31](#31-rc-grpo-reward-conditioned-grpo) | Logged 2026-09-28; **unscored**. **Training-time** reward-conditioned GRPO. Paper BFCLv4 val ≠ our 200-entry sheet. **Not** a GRPO unlock |
| MCP-GRANITE | Paschalides et al. 2026, [arXiv:2609.24161](https://arxiv.org/abs/2609.24161) — [§3a](#3a-tool-interface-granularity-mcp-granite) | Already logged; **unscored**. Optional if the tool list is huge |
| Data Turnstile | Ramakrishnan & Sharma (Amazon AGI) 2026, [arXiv:2607.29250](https://arxiv.org/abs/2607.29250) — [§24](#24-data-turnstile-open-function-calling-data-generation) | Already logged (same paper as §24); **unscored**. Harness ID/mask note, not the primary SFT-loop fix |
| When2Call | Ross et al. 2025, NAACL 2025, [arXiv:2504.18851](https://arxiv.org/abs/2504.18851) | Light pointer: 4-way when-to-call |
| MetaTool | Huang et al. 2024, ICLR 2024, [arXiv:2310.03128](https://arxiv.org/abs/2310.03128) | Light pointer: 4-way when-to-call |
| BFCL / Gorilla multi-turn | Patil et al. 2025, [ICML 2025](https://proceedings.mlr.press/v267/patil25a.html); Gorilla [arXiv:2305.15334](https://arxiv.org/abs/2305.15334); [leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) | Light pointer: state + response checkers; empty gold turns matter |
| BUTTON | Chen et al., ICLR 2025, [arXiv:2410.12952](https://arxiv.org/abs/2410.12952) | Light pointer only; do **not** duplicate a full section |

### Ranked next SFT (later path; quota remix HOLD; PTC-A closed FAIL; DPO smoke PASS / full train HOLD)

1. **APIGen-MT-5k + ToolACE / ToolACE-MT-style MT + single-turn FC** (~30–50% MT). Gemma template. **Mask user/tool; train assistant only.**
2. **Action-class quotas HOLD** — v9 / v9.1 / v9.2 10-ID smoke-gate **FAIL** (Run 36 avg **25.7**, hit **6/10**). Do **not** third-remix. Next path = **DPO** (β0.2 smoke100 10-ID Run 49 **PASS** health gate — **not** a BFCL %; DPO-full **scored** Run 51 **53/200 = 26.5%** — **not** a method win; [PR #57](https://github.com/sskarz/Thesis/pull/57) stays **draft**). Historical GAR-per-class measurement still applies; remix itself is **parked**.
3. **Anti-loop contrast (next lever):** Magnet-style pos/neg or Critical-State labels; **DPO** (smoke100 β0.2 10-ID **PASS**; full train **HOLD**) then mDPO / small preference **after** SFT ≫ decode bans ≫ another quota remix ≫ more PTC/OODA remix.
4. **Lightweight state prefix** (goal / slots / last fact / next mode) in **one pass** — **scored** as `--fc-mode iffc-ooda-card` Run 42 **71/200 = 35.5%** ([experiment](#experiment-iffc-ooda-card---fc-mode-iffc-ooda-card); stacked on stop-repeat MIN=6; `next` advisory). Still **best non-BoN harness Acc**; still **below BoN 36.5%**; `state_mismatch` gate **MISS**. Guo et al. Specialized ~2.4× multi-pass remains **not** the edge default ([OODA-Tool §26](#26-ooda-tool-typed-observe-orient-decide-act)). **PTC-A** (`--fc-mode iffc-ptc-a`) is a **later scored FAIL** (Run 46 **13.5%**) — **bank and close**; do **not** remix as the next Acc bet.
5. **Hold decode stops on the scored base.** Optional soft max-N identical successful signatures, then force answer.
6. **Occurrence-local RL** only if SFT plateaus — [§25](#25-critical-state-rl-diagnosing-trainable-multi-turn-calls), **not** [§19 blanket GRPO](#19-reinforcement-learning-with-verifiable-rewards-grpo). [PARL-MT §29](#29-parl-mt-progress-awareness-for-multi-turn-function-calling) / [RC-GRPO §31](#31-rc-grpo-reward-conditioned-grpo) remain logged research options; **do not** unlock GRPO as the main path.
7. **Optional MCP-GRANITE composites** if the tool list is huge — [§3a](#3a-tool-interface-granularity-mcp-granite).

### Anti-patterns

Teacher loop-filter-only; stop-on-repeat / CRI-like decode; recovery-targeted SFT without refusal / final negatives; Acc-only while GAR(ask/refuse) collapses; quota remix after two sequential 10-ID FAILs (v9 / v9.1 / v9.2); inventing a v9 BFCL `multi_turn_base` %; inventing a Run 49 / 47 / 48 BFCL %; presenting the DPO smoke100 10-ID PASS as a N=200 score; claiming full DPO train started; full LoRA on a failed mix; more PTC/OODA remix as a reaction to Run 46 FAIL; BoN on PTC-A; overwriting Run 30 / Run 42.

### Already-logged options (cross-link, do not duplicate)

[Critical-State RL §25](#25-critical-state-rl-diagnosing-trainable-multi-turn-calls), [OODA-Tool §26](#26-ooda-tool-typed-observe-orient-decide-act), [PTC-Decoder §27](#27-ptc-decoder-plan-to-act-plus-tool-name-dfa), [Looped LMs §28](#28-looped-language-models-for-compositional-tool-calling), [PARL-MT §29](#29-parl-mt-progress-awareness-for-multi-turn-function-calling), [HyperTool §30](#30-hypertool-code-block-execution-granularity), [RC-GRPO §31](#31-rc-grpo-reward-conditioned-grpo), [MCP-GRANITE §3a](#3a-tool-interface-granularity-mcp-granite), [Data Turnstile §24](#24-data-turnstile-open-function-calling-data-generation), [APIGen-MT / xLAM §15](#15-xlam-large-action-models), [ToolACE §13](#13-toolace-automated-training-data-generation). Sweep entries remain **unscored** research options except the one-pass `[OODA]` **card** harness (Run 42 **scored**; not Guo et al. Specialized) and the live-name DFA `--fc-mode iffc-ptc-a` harness (Run 46 **scored FAIL 13.5%**; bank and close; PTC-B HOLD). Paper numbers are **theirs**.

---

## Part II References

30. Dettmers, T., et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023. arXiv:2305.14314.
31. Dhakal, S., et al. "Optimizing Small Language Models for In-Vehicle Function-Calling." arXiv:2501.02342, 2025.
32. Google. "FunctionGemma: New Gemma model for function calling." 2024. https://blog.google/technology/developers/functiongemma/.
33. Gu, Y., et al. "MiniLLM: Knowledge Distillation of Large Language Models." ICLR 2024. arXiv:2306.08543.
34. Hu, E., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. arXiv:2106.09685.
35. Jung, S., et al. "DiaTool-DPO: Multi-Turn Direct Preference Optimization for Tool-Augmented Large Language Models." SIGdial 2025. arXiv:2504.02882.
36. Liu, T., et al. "ToolACE: Winning the Points of LLM Function Calling." ICLR 2025. arXiv:2409.00920. https://arxiv.org/abs/2409.00920
37. Qin, Y., et al. "Confucius: Iterative Tool Learning from Introspection Feedback." arXiv:2308.14034, 2024.
38. Shao, Z., et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300, 2024.
39. Sun, Y., et al. "VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use." arXiv:2509.01055, 2025.
40. Wang, K., et al. "Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning." arXiv:2511.14460, 2025.
41. Ye, L., et al. "Hammer: Robust Function-Calling for On-Device Language Models via Function Masking." ICLR 2025. arXiv:2410.04587.
42. Zhang, J., et al. "xLAM: A Family of Large Action Models to Empower AI Agent Systems." arXiv:2409.03215, 2024.
43. Ramakrishnan, G. & Sharma, M. (Amazon AGI). "Data Turnstile: A Scalable Open Framework for Function-Calling Data Generation." arXiv:2607.29250, 2026. https://arxiv.org/abs/2607.29250
44. Chen, Z., Zhao, W., et al. (Salesforce). "Critical-State RL: Diagnosing Trainable States for Multi-Turn Tool Use." arXiv:2609.24985, 2026. https://arxiv.org/abs/2609.24985
45. Zeng et al. "ToolACE-MT." ICLR 2026. arXiv:2508.12685. https://arxiv.org/abs/2508.12685
46. Prabhakar et al. "APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay." NeurIPS 2025 D&B. arXiv:2504.03601. https://arxiv.org/abs/2504.03601
47. Xu et al. "FunReason-MT." arXiv:2510.24645, 2025. https://arxiv.org/abs/2510.24645 (paper BFCL-v3; **not** our harness)
48. Yin et al. "Magnet" (mDPO). ACL 2025. arXiv:2503.07826. https://arxiv.org/abs/2503.07826
49. Zhao et al. "Calibration is the Bottleneck" (GAR). arXiv:2609.00949, 2026. https://arxiv.org/abs/2609.00949
50. Guo, R., Huang, Y., Wu, Y., Zhong, M., Chen, Y., Tang, M., Long, T., & Hu, V. T. "From State to Action: OODA-Tool for Reliable Multi-Turn Tool Use." arXiv:2608.24368, 2026. https://arxiv.org/abs/2608.24368
51. Paschalides, D., et al. "MCP-GRANITE Benchmark: GRANularity Interface TEsting for MCP-Based LLM Agents." arXiv:2609.24161, 2026. https://arxiv.org/abs/2609.24161
52. Ross et al. "When2Call." NAACL 2025. arXiv:2504.18851. https://arxiv.org/abs/2504.18851
53. Huang et al. "MetaTool." ICLR 2024. arXiv:2310.03128. https://arxiv.org/abs/2310.03128
54. Patil et al. "Berkeley Function Calling Leaderboard." ICML 2025. https://proceedings.mlr.press/v267/patil25a.html
55. Chen et al. "BUTTON." ICLR 2025. arXiv:2410.12952. https://arxiv.org/abs/2410.12952
56. Yu, M., Mu, K., & Wu, G. "PTC-Decoder: Towards Intelligent SLMs on Offline Resource-Constrained Edge Devices." arXiv:2609.30836, 2026. https://arxiv.org/abs/2609.30836
57. Popescu, A. C., Sáez de Ocáriz Borde, H., & Liò, P. "Looped Language Models Improve Compositional Tool Calling." arXiv:2608.18171, 2026. https://arxiv.org/abs/2608.18171
58. Chai, H., et al. "PARL-MT: Learning to Call Functions in Multi-Turn Conversation with Progress Awareness." ACL 2026 Findings. arXiv:2509.23206, 2025. https://arxiv.org/abs/2509.23206
59. Du, Y., Zhou, Y., et al. "HyperTool: Beyond Step-Wise Tool Calls for Tool-Augmented Agents." arXiv:2606.13663, 2026. https://arxiv.org/abs/2606.13663
60. Zhong, H., Zhai, J., Song, L., Bian, J., Liu, Q., & Tan, T. "RC-GRPO: Reward-Conditioned Group Relative Policy Optimization for Multi-Turn Tool Calling Agents." arXiv:2602.03025, 2026. https://arxiv.org/abs/2602.03025
61. Kang et al. "Acon: Optimizing Context Compression for Long-horizon LLM Agents." arXiv:2510.00615, 2025. https://arxiv.org/abs/2510.00615 (**not** Xu; ReWOO remains Xu B.)
62. Kang et al. "Distilling LLM Agent into Small Models with Retrieval and Code Tools." NeurIPS 2025. arXiv:2505.17612.
63. Zhang, Yang, Li, He, Gong. "ODIA." arXiv:2507.08877, 2025.
