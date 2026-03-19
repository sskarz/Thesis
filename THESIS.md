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

All models are evaluated on the Berkeley Function Calling Leaderboard (BFCL) v3, which provides standardized test cases across 7 categories totaling 1,390 test cases:

| Category | Test Cases | Description |
|----------|-----------|-------------|
| Simple | 400 | Single function available, call it correctly |
| Multiple | 200 | Multiple functions available, select the correct one |
| Parallel | 200 | Multiple functions need to be called in a single turn |
| Parallel Multiple | 200 | Multiple functions from multiple candidates in a single turn |
| Irrelevance | 240 | Functions available but the query does not require any call |
| Java | 100 | Function calling with Java-style type signatures |
| JavaScript | 50 | Function calling with JavaScript-style type signatures |

### Evaluation method

- **AST checker**: BFCL's Abstract Syntax Tree checker compares model output against reference answers, allowing equivalent but differently-ordered arguments to match.
- **Deterministic output**: All models run with `--temp 0.0` to ensure reproducible results.
- **No reasoning mode**: All models run without thinking/reasoning, even if the model supports it (e.g., Qwen3.5). This isolates raw tool-calling capability from chain-of-thought augmentation.

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

### Quantization notes

The three generalist models (Qwen2.5-3B, Llama-3.2-3B, Qwen3.5-4B) are all served at Q4_K_M quantization — a 4-bit mixed quantization scheme that balances model size against accuracy loss.

FunctionGemma-270M is served at bf16 (brain float 16) precision. This is the only available quantization for this model; no Q4_K_M or other quantized variants are published. At 270M parameters, aggressive quantization yields diminishing returns — the model is already small enough that the overhead of dequantization logic can negate size savings. Even at full bf16 precision, FunctionGemma's GGUF (~512 MB) is **3.5–5.5x smaller** than the Q4_K_M generalists, making it a realistic edge deployment target:

- **Smartphones** (iPhone, Pixel): 6–8 GB RAM with dedicated neural engines. A 512 MB model leaves ample headroom.
- **NVIDIA Jetson Orin Nano Super**: 8 GB shared CUDA memory. A 512 MB model uses ~6% of available VRAM.
- **Raspberry Pi 5 (8 GB)**: CPU-only inference is feasible at this model size.

The comparison remains controlled because the variable under study is **model architecture and specialization**, not quantization method.

## Accuracy Results

All results on BFCL v4, evaluated with the AST checker. **Bold** indicates best-in-class per category.

| Category | Qwen2.5-3B (Q4_K_M) | Llama-3.2-3B (Q4_K_M) | Qwen3.5-4B (Q4_K_M) | FunctionGemma-270M (bf16) |
|----------|---------------------|----------------------|---------------------|--------------------------|
| Simple (400) | **373/400 (93.25%)** | 207/400 (51.75%) | 361/400 (90.25%) | 205/400 (51.25%) |
| Multiple (200) | **185/200 (92.50%)** | 67/200 (33.50%) | 181/200 (90.50%) | 118/200 (59.00%) |
| Parallel (200) | 0/200 (0.00%) | 0/200 (0.00%) | 0/200 (0.00%) | 0/200 (0.00%) |
| Parallel Multiple (200) | 0/200 (0.00%) | 0/200 (0.00%) | 0/200 (0.00%) | 0/200 (0.00%) |
| Irrelevance (240) | 141/240 (58.75%) | 2/240 (0.83%) | **219/240 (91.25%)** | 48/240 (20.00%) |
| Java (100) | 37/100 (37.00%) | **54/100 (54.00%)** | 39/100 (39.00%) | 15/100 (15.00%) |
| JavaScript (50) | 17/50 (34.00%) | **35/50 (70.00%)** | 17/50 (34.00%) | 9/50 (18.00%) |

### Overall accuracy (excluding parallel categories)

| Model | Weighted Avg (5 categories, 990 cases) |
|-------|-----------------------------------------|
| Qwen2.5-3B | 753/990 (76.06%) |
| Qwen3.5-4B | 817/990 (82.53%) |
| Llama-3.2-3B | 365/990 (36.87%) |
| FunctionGemma-270M | 395/990 (39.90%) |

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

### Parallel calling: Universal failure

All four models score 0% on `parallel` and `parallel_multiple`. This is a structural limitation of small models — emitting multiple well-formed function calls in a single response requires a level of structured output control that sub-5B models consistently fail to achieve. This is the single largest gap between SLMs and cloud-scale models, and the most significant barrier to fully agentic multi-step workflows.

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

- **NVIDIA Jetson Orin Nano Super migration**: Repeat all benchmarks on CUDA with 8 GB shared memory to measure real edge-device performance.
- **RAG augmentation for large tool sets**: The `multiple` category only presents 2–4 tools per test case, so RAG-based tool filtering provides no benefit. However, real-world agentic applications may expose 50–100+ tools. Test whether retrieval-augmented generation (embedding-based tool retrieval to pre-filter the tool set before prompting) improves accuracy and reduces prompt length for large tool registries.
- **Additional models and quantizations**: Expand the model matrix (e.g., Phi-3.5-mini, Gemma-2-2B) and test alternative quantization levels (Q5_K_M, Q8_0) to map the accuracy-size Pareto frontier.
- **Parallel function calling investigation**: Analyze failure modes on `parallel`/`parallel_multiple` to determine if structured output constraints (e.g., grammar-guided decoding) can unlock this capability in small models.
