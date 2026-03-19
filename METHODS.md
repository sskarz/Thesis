# Methods for Improving SLM Tool Calling

A survey of techniques to improve small language model (SLM, <5B parameter) performance on agentic multi-step tool calling — specifically on the BFCL v4 benchmark. Part I covers inference-time methods (prompt engineering, structured decoding, retrieval augmentation, context management, compute scaling). Part II covers training-time methods (supervised fine-tuning, reinforcement learning, knowledge distillation, synthetic data generation).

# Part I: Inference-Time Methods

Each method in this section operates at inference time without modifying model weights.

## Table of Contents

- [1. Guided-Structured Templates](#1-guided-structured-templates)
- [2. Constrained Decoding and Grammar-Guided Generation](#2-constrained-decoding-and-grammar-guided-generation)
- [3. RAG-Based Tool Retrieval](#3-rag-based-tool-retrieval)
- [4. Inference-Time Compute Scaling](#4-inference-time-compute-scaling)
- [5. Context Compression for Multi-Turn](#5-context-compression-for-multi-turn)
- [6. ReAct and ReWOO Planning Patterns](#6-react-and-rewoo-planning-patterns)
- [7. Prompt-Only Tool Calling](#7-prompt-only-tool-calling)
- [8. Few-Shot Example Retrieval](#8-few-shot-example-retrieval)
- [9. Verification and Self-Correction Loops](#9-verification-and-self-correction-loops)
- [10. DSPy Automatic Prompt Optimization](#10-dspy-automatic-prompt-optimization)
- [11. Parallel Function Calling via SimpleTool](#11-parallel-function-calling-via-simpletool)
- [Key Finding: llama.cpp Already Uses Grammar-Constrained Decoding](#key-finding-llamacpp-already-uses-grammar-constrained-decoding)
- [Prioritized Recommendations](#prioritized-recommendations)
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

- **BFCL categories:** `multiple`, `irrelevance`, `parallel` (reasoning about which tools and how many)
- **Implementation:** Replace the current `TOOL_CALLING_SYSTEM_PROMPT` in `generate_responses.py` with a structured template; pure prompt-side change
- **Complexity:** Low
- **Limitation:** Increases prompt token count; template must fit within the 8192-token context window alongside tool definitions

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
- **Implementation:** Use the `grammar` or `response_format` parameter in the llama-server API; or test LLGuidance for stricter type enforcement
- **Complexity:** Medium (must coordinate with existing `tools` parameter and chat template)
- **Limitation:** Guarantees syntactic validity only, not semantic correctness. The model could produce perfectly formatted JSON with wrong function names or argument values.

### Sources

- Willard, B. T. & Louf, R. "Efficient Guided Generation for Large Language Models." arXiv:2307.09702, 2023. https://arxiv.org/abs/2307.09702
- Tam, Z., et al. "Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models." arXiv:2408.02442, 2024. https://arxiv.org/abs/2408.02442
- Jain, N., et al. "Lost in Space: Optimizing Tokens for Grammar-Constrained Decoding." arXiv:2502.14969, 2025. https://arxiv.org/abs/2502.14969
- llama.cpp GBNF documentation: https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
- llama.cpp LLGuidance documentation: https://github.com/ggml-org/llama.cpp/blob/master/docs/llguidance.md

---

## 3. RAG-Based Tool Retrieval

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
- **Thesis contribution:** Demonstrate the crossover point — at what tool count does RAG start providing benefit for 3B models? Create an extended benchmark by padding BFCL tool sets with distractor tools (20, 50, 100) and measuring accuracy with and without retrieval.
- **Implementation:** Embed BFCL function descriptions with a small embedding model (e.g., `all-MiniLM-L6-v2`, ~22M params, <50MB, runs on CPU in <5ms per query). Build a FAISS index. Add a retrieval step in `_load_func_docs()` before calling llama-server.
- **Complexity:** Medium

### Sources

- Gao, Y., et al. "RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation." arXiv:2505.03275, 2025. https://arxiv.org/abs/2505.03275
- Patil, S., et al. "Gorilla: Large Language Model Connected with Massive APIs." arXiv:2305.15334, 2023. https://arxiv.org/abs/2305.15334
- Qin, Y., et al. "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs." ICLR 2024, arXiv:2307.16789. https://arxiv.org/abs/2307.16789
- Yuan, S., et al. "CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets." arXiv:2309.17428, 2024. https://arxiv.org/abs/2309.17428
- Hao, S., et al. "ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings." NeurIPS 2023, arXiv:2305.11554. https://arxiv.org/abs/2305.11554

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
- **Implementation:** Wrapper around `generate_for_category` that samples N times at `temperature=0.7`, parses all results, and selects by majority vote on `(function_name, arguments)` tuples. For multi-turn: vote at each turn independently.
- **Complexity:** Medium
- **Tradeoff:** Multiplies inference time by N. With FunctionGemma's 16ms TTFT, 5 samples take ~80ms. With Qwen3.5's 656ms TTFT, 5 samples take ~3.3 seconds.

### Sources

- Snell, C., et al. "Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters." arXiv:2408.03314, 2024. https://arxiv.org/abs/2408.03314
- Qu, C., et al. "The Art of Scaling Test-Time Compute for Large Language Models." arXiv:2512.02008, 2025. https://arxiv.org/abs/2512.02008
- Wang, X., et al. "Self-Consistency Improves Chain of Thought Reasoning in Language Models." ICLR 2023, arXiv:2203.11171. https://arxiv.org/abs/2203.11171
- Brown, B., et al. "Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models." ICLR 2025, arXiv:2408.00724. https://arxiv.org/abs/2408.00724

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

ACON results (Xu et al., 2025):

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

- **BFCL categories:** `multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multi_turn_long_context`. The current `generate_multi_turn.py` appends every tool call and result to `chat_history` indefinitely. By turn 4–5 with multiple tool calls per turn, this can exceed 8192 tokens (triggering the `context_overflow` flag).
- **Implementation:** After each turn, summarize tool results from turns older than N-2 into a condensed state summary message. Keep the last 2 turns verbatim. A simpler alternative is a sliding window that retains only the system prompt, tool definitions, and the most recent K turns.
- **Complexity:** Medium (sliding window: Low)
- **Caveat:** Summarization itself requires an LLM call. For SLMs, the summarization quality may be poor. A simpler structured state tracker (explicit key-value record of tool results) avoids this.

### Sources

- Xu, Z., et al. "ACON: Optimizing Context Compression for Long-horizon LLM Agents." arXiv:2510.00615, 2025. https://arxiv.org/abs/2510.00615
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

### Source

Local source code analysis: `llama.cpp/common/chat.cpp`, `llama.cpp/common/json-schema-to-grammar.cpp`, `llama.cpp/grammars/README.md`

---

## Prioritized Recommendations

Ranked by expected impact × feasibility for the current setup (Qwen2.5-3B, Llama-3.2-3B, Qwen3.5-4B on llama.cpp with BFCL v4):

| Priority | Technique | Target Problem | Expected Impact | Effort |
|----------|-----------|----------------|-----------------|--------|
| 1 | **`tool_choice=required` + `parallel_tool_calls`** | Parallel calling (0%) | Potentially unlocks parallel | Low |
| 2 | **Guided-Structured Templates** | multiple, irrelevance accuracy | +3–12% on BFCL | Low |
| 3 | **Few-shot format demonstrations** | Parallel call format, cross-language | +5–20pp for SLMs | Medium |
| 4 | **Best-of-N majority voting** | All categories | +5–15% with N=5 | Medium |
| 5 | **Context compression** | Multi-turn overflow and accuracy | -30–50% tokens, +accuracy | Medium |
| 6 | **LLGuidance build** | Type enforcement (Llama issue) | Stricter schema at grammar level | Medium |
| 7 | **RAG tool retrieval** | Large tool sets, real-world demo | 3x accuracy at scale | Medium |
| 8 | **Schema-based verification loop** | Argument errors across categories | +3–10% with retries | Medium |
| 9 | **DSPy prompt optimization** | Model-specific prompt tuning | Auto-discovered improvements | High |

The first four are directly measurable on BFCL v4 without changes to the benchmark format, making them strong thesis contributions. The key narrative: **inference-time techniques can close the gap between SLMs and cloud-scale models without fine-tuning.**

---

## Additional Papers and Resources

### Error Taxonomies and Analysis

- **APIGen** (Liu et al., 2024): Automated dataset generation for tool calling. Error taxonomy for <7B models: 32% type mismatch, 24% missing required arguments, 18% wrong function selection, 15% hallucinated arguments. Confirms that type coercion addresses the #1 error category. arXiv:2406.18518. https://arxiv.org/abs/2406.18518
- **ToolSandbox** (Lu et al., 2024): Stateful multi-turn tool evaluation. Shows state management is the primary bottleneck for multi-turn, not individual tool call accuracy. Explicit state injection improves multi-turn success by 15–25%. https://arxiv.org/abs/2408.04682
- **BFCL** (Yan et al., 2024): Berkeley Function Calling Leaderboard. Identified structured output formatting as the primary failure mode for open-source models. arXiv:2402.15491. https://arxiv.org/abs/2402.15491

### Model-Specific Tool Calling Techniques

- **Granite Function Calling** (IBM, 2024): Decomposes function calling into granular subtasks: function name detection, parameter extraction, value extraction. Even without fine-tuning, prompting for subtasks sequentially improves SLM accuracy. arXiv:2407.00121. https://arxiv.org/abs/2407.00121
- **NexusRaven-V2** (Nexusflow, 2024): Tool description formatting matters more than model size. Adding usage examples within tool descriptions significantly boosts accuracy.
- **Octopus v2** (Chen et al., 2024): 2B on-device function calling model using functional tokens — a single special token per tool, reducing generation length to 1 token for tool selection. arXiv:2404.01744. https://arxiv.org/abs/2404.01744
- **ToolACE** (Liu et al., 2024): Including parameter constraints (min/max values, enum lists) in tool descriptions improves accuracy significantly. arXiv:2409.00920. https://arxiv.org/abs/2409.00920

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
11. Liu, T., et al. "ToolACE: Winning the Points of LLM Function Calling." arXiv:2409.00920, 2024.
12. Lu, J., et al. "ToolSandbox: A Stateful, Conversational, Interactive Evaluation Framework for LLM Tool Use Capabilities." arXiv:2408.04682, 2024.
13. Madaan, A., et al. "Self-Refine: Iterative Refinement with Self-Feedback." NeurIPS 2023. arXiv:2303.17651.
14. Patil, S., et al. "Gorilla: Large Language Model Connected with Massive APIs." arXiv:2305.15334, 2023.
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
26. Xu, Z., et al. "ACON: Optimizing Context Compression for Long-horizon LLM Agents." arXiv:2510.00615, 2025.
27. Yan, F., et al. "Berkeley Function Calling Leaderboard." arXiv:2402.15491, 2024.
28. Yao, S., et al. "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR 2023. arXiv:2210.03629.
29. Yuan, S., et al. "CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets." arXiv:2309.17428, 2024.

---

# Part II: Training-Time Methods

Each method in this section involves modifying model weights — via supervised fine-tuning (SFT), reinforcement learning (RL), knowledge distillation, or structured pruning. These techniques produce specialized tool-calling models that can then be served via llama.cpp in the same way as the base models evaluated in Part I.

## Table of Contents (Part II)

- [12. Supervised Fine-Tuning with Synthetic Data](#12-supervised-fine-tuning-with-synthetic-data)
- [13. ToolACE: Automated Training Data Generation](#13-toolace-automated-training-data-generation)
- [14. Hammer: Function Masking for Robust Tool Calling](#14-hammer-function-masking-for-robust-tool-calling)
- [15. xLAM: Large Action Models](#15-xlam-large-action-models)
- [16. FunctionGemma: Specialist Fine-Tuning from Base Models](#16-functiongemma-specialist-fine-tuning-from-base-models)
- [17. Structured Pruning + Fine-Tuning for Edge Deployment](#17-structured-pruning--fine-tuning-for-edge-deployment)
- [18. Parameter-Efficient Fine-Tuning (LoRA / QLoRA)](#18-parameter-efficient-fine-tuning-lora--qlora)
- [19. Reinforcement Learning with Verifiable Rewards (GRPO)](#19-reinforcement-learning-with-verifiable-rewards-grpo)
- [20. End-to-End Agent RL (Agent-R1 / VerlTool)](#20-end-to-end-agent-rl-agent-r1--verltool)
- [21. DiaTool-DPO: Preference Optimization for Multi-Turn Tool Calling](#21-diatool-dpo-preference-optimization-for-multi-turn-tool-calling)
- [22. Knowledge Distillation for Tool Calling](#22-knowledge-distillation-for-tool-calling)
- [23. Curriculum Learning for Progressive Tool Mastery](#23-curriculum-learning-for-progressive-tool-mastery)
- [Training-Time Prioritized Recommendations](#training-time-prioritized-recommendations)
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

### Sources

- Liu, T., et al. "ToolACE: Winning the Points of LLM Function Calling." ICLR 2025. arXiv:2409.00920. https://arxiv.org/abs/2409.00920
- Salesforce. "xlam-function-calling-60k." HuggingFace Datasets. https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k

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

xLAM is Salesforce's family of Large Action Models, ranging from 1B to 8x22B parameters, fine-tuned specifically for function calling and agentic tasks. The 1B model is the smallest model on the BFCL leaderboard, achieving 78.94% overall accuracy — outperforming GPT-3.5-Turbo.

### Evidence

| Model | Parameters | BFCL Overall | Notes |
|-------|-----------|-------------|-------|
| xLAM-2-fc-r | — | **#1 on BFCL** (as of April 2025) | Latest generation |
| xLAM-7b-fc-r | 7B | 88.24% | #3 on leaderboard |
| xLAM-1b-fc-r | **1B** | **78.94%** | Smallest model on leaderboard; outperforms GPT-3.5-Turbo |

On τ-bench (agentic evaluation), xLAM-2-70b-fc-r achieves 56.2% success rate, outperforming GPT-4o (52.9%).

### Applicability

- **xLAM-1b-fc-r** is directly relevant to edge deployment — at 1B parameters it fits comfortably on all target hardware. GGUF versions are published on HuggingFace (`Salesforce/xLAM-1b-fc-r-gguf`).
- **As a benchmark model:** Running xLAM-1b-fc-r through the existing BFCL v4 pipeline would provide a fine-tuned baseline to compare against the inference-time techniques applied to base models.
- **Training data:** The xlam-function-calling-60k dataset used for training is publicly available.

### Sources

- Zhang, J., et al. "xLAM: A Family of Large Action Models to Empower AI Agent Systems." arXiv:2409.03215, 2024. https://arxiv.org/abs/2409.03215
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

### Sources

- Shao, Z., et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300, 2024. https://arxiv.org/abs/2402.03300
- Qian, C., et al. "ToolRL: Reward is All You Need for Tool Learning." 2025.
- Li, B., et al. "ToRL: Scaling Tool-Integrated RL." 2025.

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
- **ACON distillation:** Xu et al. (2025) distilled context compression capabilities from GPT-4.1 to Qwen3-8B and Phi-4, "preserving over 95% of the teacher's accuracy."
- **Multi-teacher distillation:** Recent work (2024–2025) leverages knowledge from multiple teacher models to create more robust student models that generalize better across tool-calling domains.

### Applicability

- **Dataset generation:** The most practical form of distillation for this thesis is response-based: use a strong model (GPT-4, Claude) to generate correct tool calls for BFCL-format queries, then fine-tune the SLMs on these outputs.
- **Complementary to SFT:** Distillation produces the training data; SFT/LoRA consumes it. Works with all the fine-tuning methods above.
- **Cost consideration:** Generating distillation data from API models has a monetary cost. For BFCL's ~1,390 single-turn + 800 multi-turn test cases, this is modest.
- **Complexity:** Low (if using existing distillation datasets) to Medium (if generating custom data from teacher models).

### Sources

- Gu, Y., et al. "MiniLLM: Knowledge Distillation of Large Language Models." ICLR 2024. arXiv:2306.08543. https://arxiv.org/abs/2306.08543
- Microsoft. "Distillation: Turning Smaller Models into High-Performance, Cost-Effective Solutions." 2024. https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/distillation-turning-smaller-models-into-high-performance-cost-effective-solutio/4355029

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

Ranked by expected impact × feasibility for the current setup (Qwen2.5-3B, Llama-3.2-3B, Qwen3.5-4B, RTX 4090 with 24GB VRAM):

| Priority | Technique | Target Problem | Expected Impact | Effort |
|----------|-----------|----------------|-----------------|--------|
| 1 | **QLoRA fine-tuning on xlam-function-calling-60k** | All categories | +20–40pp overall; unlocks parallel | Medium |
| 2 | **Function masking augmentation (Hammer)** | Generalization, irrelevance | Robust to unseen APIs; +irrelevance | Medium |
| 3 | **Add xLAM-1b-fc-r as benchmark baseline** | Comparison point | Shows fine-tuned vs. base SLM gap | Low |
| 4 | **GRPO post-SFT with BFCL checker as reward** | All categories, esp. multi-turn | +5–15pp on top of SFT | High |
| 5 | **DiaTool-DPO for multi-turn** | Multi-turn categories | Better dialogue flow + rejection | High |
| 6 | **FunctionGemma domain fine-tuning** | FunctionGemma-270M scores | 58% → 85% on domain tasks | Medium |
| 7 | **Structured pruning (Phi-3 style)** | Edge size reduction | 3B → 1.8B with maintained accuracy | Very High |
| 8 | **Curriculum-ordered training** | Training stability | Prevents catastrophic forgetting | Medium |

### Recommended Training Pipeline

For a complete training-time experiment within this thesis:

```
Base model (e.g., Qwen2.5-3B-Instruct)
    ↓ QLoRA fine-tuning on xlam-60k + Hammer function masking
    ↓ Optional: GRPO refinement with BFCL AST checker as reward
    ↓ Merge LoRA adapter into base model
    ↓ Convert to GGUF (Q4_K_M)
    ↓ Serve via llama-server, evaluate on BFCL v4
    ↓ Compare against base model + inference-time methods from Part I
```

This pipeline answers the central thesis question: **does fine-tuning or inference-time optimization close the SLM-to-cloud-model gap more effectively for tool calling on edge devices?**

---

## Part II References

30. Dettmers, T., et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023. arXiv:2305.14314.
31. Dhakal, S., et al. "Optimizing Small Language Models for In-Vehicle Function-Calling." arXiv:2501.02342, 2025.
32. Google. "FunctionGemma: New Gemma model for function calling." 2024. https://blog.google/technology/developers/functiongemma/.
33. Gu, Y., et al. "MiniLLM: Knowledge Distillation of Large Language Models." ICLR 2024. arXiv:2306.08543.
34. Hu, E., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. arXiv:2106.09685.
35. Jung, S., et al. "DiaTool-DPO: Multi-Turn Direct Preference Optimization for Tool-Augmented Large Language Models." SIGdial 2025. arXiv:2504.02882.
36. Liu, T., et al. "ToolACE: Winning the Points of LLM Function Calling." ICLR 2025. arXiv:2409.00920.
37. Qin, Y., et al. "Confucius: Iterative Tool Learning from Introspection Feedback." arXiv:2308.14034, 2024.
38. Shao, Z., et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300, 2024.
39. Sun, Y., et al. "VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use." arXiv:2509.01055, 2025.
40. Wang, K., et al. "Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning." arXiv:2511.14460, 2025.
41. Ye, L., et al. "Hammer: Robust Function-Calling for On-Device Language Models via Function Masking." ICLR 2025. arXiv:2410.04587.
42. Zhang, J., et al. "xLAM: A Family of Large Action Models to Empower AI Agent Systems." arXiv:2409.03215, 2024.
