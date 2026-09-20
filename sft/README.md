# SFT v1 — MLX LoRA on Gemma-4-E2B (IFFC JSON-in-text)

Frozen data mix and split for supervised fine-tuning of **Gemma-4-E2B-it** on Apple Silicon with [mlx-lm](https://github.com/ml-explore/mlx-lm) LoRA. Training targets match the existing `--fc-mode iffc` harness (`benchmark/iffc.py`): assistant turns are a JSON list of `{"function_name", "parameters"}`; tool results are user-role observations. Native `tool_calls` are **not** mixed into this adapter dataset.

This folder prepares data and prints/runs the mlx-lm command. **Run 12 scored the v1 mix as a negative: 1/200 = 0.5%** on BFCL v4 `multi_turn_base` (`gemma4-e2b-sft-iffc`, fused Q4_K_M best@1000). Root cause is ToolACE stop-after-tools `[]` targets (5520 empty assistant turns; 78.1% of post-tool-result targets are `[]`), **not** the IFFC parser (zero `iffc_unparseable`). **Run 13 scored the v2 filtered non-terminal-`[]` mix as the same collapse: 1/200 = 0.5%** (`gemma4-e2b-sft-iffc-v2`, LoRA best@2000 `val=0.211`; only `multi_turn_base_50`; correct-set overlap identical with Run 12). PR #24 dropped 334 mid-trajectory empty-after-exec (non-terminal 334→0) and still kept 552 terminal; the filter did **not** recover. Do **not** train more epochs on v1 or this filtered mix, run BoN/STATE on these checkpoints, or claim the `[]` filter fixed SFT. Next (Sanskar decides — not started): drop **all** empty-after-exec including terminal + stronger continuation curriculum; ToolACE-only vs xLAM-only ablation at small scale; investigate Gemma-4 chat-template / fuse path; or pause SFT (BoN 36.5% still best). Details in `THESIS.md` / `METHODS.md`. The thesis test remains BFCL v4 `multi_turn_base` and is **never** in train / val / corpus_test.

## Locked mix (do not reopen)

| Source | Role | Size |
|---|---|---|
| [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | **all** rows (~11,300 multi-turn) | full train split |
| [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | stratified **25,000** rows | oversample parallel / multi-answer (`n_tools_in_gold ≥ 2` taken first, then fill from single-call) |

**Skipped for v1:** APIGen-MT-5k (NC license), Glaive, Hermes (Hermes is an optional backup only if a source disappears — not in this pipeline).

**Splits** (by conversation / example id, never by turn): `seed=42`; stratify on `source`, `is_multi_turn`, `n_tools_in_gold` (1 vs ≥2); ~90% train / ~5% val / ~5% corpus_test → about **Train 32.7k / Val 1.8k / Corpus-test 1.8k**.

**Thesis test:** BFCL v4 `multi_turn_base` only. Do not add BFCL prompts or past scored responses to these JSONL files.

## Base model

`benchmark/config.py` registers Gemma as GGUF:

- key: `gemma4-e2b` — base instruct GGUF (`bartowski/google_gemma-4-E2B-it-GGUF`, `Q4_K_M`)
- key: `gemma4-e2b-sft-iffc` — Mac-local MLX LoRA SFT v1 (IFFC mix, best val @ iter 1000). Run 12 scored **1/200 = 0.5%**. Distinct `api_name` / `slug` so BFCL results do **not** overwrite base Gemma. Serve with `llama-server -m <local.gguf>` (not `-hf`).
- key: `gemma4-e2b-sft-iffc-v2` — filtered-mix SFT v2 (drop non-terminal `[]` after tool-exec, LoRA best@2000 val 0.211, PR #24). Distinct `api_name` / `slug` so Run 13 does **not** overwrite Run 12. Run 13 scored **1/200 = 0.5%** (same collapse). Figures: `benchmark/results/Gemma-4-E2B-it-SFT-IFFC-v2-Q4_K_M/figures`. Serve with `llama-server -m <local.gguf>` (not `-hf`).

Instruct weights: **`google/gemma-4-e2b-it`** (gated; HuggingFace token required).

There is **no** mlx-lm id in the model registry. Do **not** LoRA-train `mlx-community/gemma-4-e2b-it-4bit` with mlx-lm — that repo is an **mlx-vlm** conversion and is missing text-only PLE weights. Convert on the Mac:

```bash
# Full-precision MLX (LoRA)
python -m mlx_lm.convert \
  --hf-path google/gemma-4-e2b-it \
  --mlx-path sft/models/gemma-4-e2b-it-mlx

# 4-bit QLoRA (lower memory)
python -m mlx_lm.convert \
  --hf-path google/gemma-4-e2b-it \
  --mlx-path sft/models/gemma-4-e2b-it-mlx-4bit \
  -q --q-bits 4
```

Needs a recent `mlx-lm` with Gemma 4 text support (mlx-lm ≥ the Gemma 4 PRs). If `convert` fails, upgrade: `pip install -U mlx-lm`.

## Mac copy-paste

Apple Silicon (M4 Pro or similar). Python deps for download/format/split live in this repo; **mlx-lm is Mac-only** and is not a CI dependency.

```bash
# 0. Repo + extras (datasets / huggingface_hub)
cd /path/to/Thesis
uv sync --extra sft
huggingface-cli login   # or: export HF_TOKEN=hf_...
# Accept licenses: google/gemma-4-e2b-it and Salesforce/xlam-function-calling-60k

# 1. Download raw JSONL → sft/data/raw/  (xLAM is gated)
uv run python -m sft.download_datasets

# 2. Convert both sources to IFFC chat JSONL → sft/data/formatted/
uv run python -m sft.format_iffc

# 3. Frozen splits + MANIFEST.json → sft/data/splits/
uv run python -m sft.split_data
# writes train.jsonl, valid.jsonl, corpus_test.jsonl, MANIFEST.json

# 4. Install mlx-lm (not in uv lock; Apple Silicon)
pip install -U mlx-lm
# or: uv pip install mlx-lm   # into the uv env, on macOS only

# 5. Convert Gemma-4-E2B instruct → local MLX weights
python -m mlx_lm.convert \
  --hf-path google/gemma-4-e2b-it \
  --mlx-path sft/models/gemma-4-e2b-it-mlx

# 6. Smoke: unroll a tiny slice, 100 iters, mask-prompt (should finish in minutes)
chmod +x sft/train_mlx.sh
./sft/train_mlx.sh --smoke --run
# equivalent:
# uv run python -m sft.train_mlx --smoke --run \
#   --model sft/models/gemma-4-e2b-it-mlx

# 7. Full train (iters from sft/lora_config.yaml, default 2000; override --iters)
./sft/train_mlx.sh --full --run --iters 2000
# adapters → sft/adapters/

# 8. Fuse LoRA into the base weights
python -m mlx_lm.fuse \
  --model sft/models/gemma-4-e2b-it-mlx \
  --adapter-path sft/adapters \
  --save-path sft/fused/gemma-4-e2b-it-iffc
# or: uv run python -m sft.train_mlx --full --fuse --run

# 9. GGUF (for llama-server). Eval uses the best-val checkpoint @ iter 1000 —
#    see [Eval](#eval-bfcl-v4-multi_turn_base) (separate MODELS key, not gemma4-e2b).
```

`--mask-prompt` only backprops the **last** chat message. `sft.train_mlx` unrolls each conversation so every assistant turn is a row ending on that turn (split membership stays at conversation id), then applies the [empty-after-exec filter](#unroll-filter-drop-non-terminal--after-tool-results).

Dry-run (Linux/CI): `uv run python -m sft.train_mlx --smoke` prints the command and writes unrolled JSONL without calling mlx-lm.

## Unroll filter (drop non-terminal `[]` after tool results)

SFT v1 trained on every unrolled assistant turn. ToolACE maps natural-language “done” replies to IFFC `[]`. After a user message that is an IFFC tool-result observation (`Function execution results:…` from `format_iffc_execution_user_message`), **78%** of those targets were `[]` (2121 empty vs 596 continue). All ~5,520 empty `[]` targets in the mlx train set came from ToolACE; xLAM had 0. That taught early stop: BFCL Run 12 empty-turn rate was ~70% vs ~13% on base IFFC, and `multi_turn_base` collapsed to 0.5% even though the live probe still emitted valid IFFC JSON. Run 13 scored the filtered mix at the **same 0.5%** (primary empty_turn 167 vs Run 12 101; audit empty turns still ~70%, 522/734 error rows).

**v2 policy (drop, not downweight):** when building unrolled mlx JSONL, drop a row iff

1. the assistant target is exactly `[]`,
2. the previous user message looks like an IFFC tool-result observation (contains `Function execution results` or the `format_iffc_execution_user_message` footer), **and**
3. the parent conversation still has a later assistant turn with a non-empty tool call (this `[]` was mid-trajectory, not a true terminal stop).

Keep true terminal `[]` (no later tool-calling assistant turn). Keep every non-empty tool-call target, including continuation after exec (`user` observation → `[call]`). Keep ordinary user → `[]` clarifying / irrelevance turns. xLAM rows are unchanged unless they somehow gain this pattern.

mlx-lm chat JSONL has no sample-weight field, so v2 **drops** the bad rows. Conversation-level `split_data` files are **not** rewritten (split unit stays conversation id). Ablation: `--keep-nonterminal-empty-after-exec` restores the v1 unroll.

Unroll logs before/after counts of total rows, empty `[]` targets, empty-after-exec targets (and the non-terminal subset), by source (`toolace` vs `xlam`). On the mix that produced Run 13, the filter dropped **334** non-terminal empty-after-exec rows (334→0) and **kept 552** terminal empty-after-exec. That v2 mix scored **1/200 = 0.5%** — filter insufficient.

```bash
# Already have formatted ToolACE + xLAM and frozen splits? Re-unroll mlx train/val:
uv run python -m sft.train_mlx --prepare-only
# equivalent audit (prints stats; does not write mlx/ unless --out):
uv run python -m sft.filter_empty_after_exec --splits-dir sft/data/splits

# Full regenerate from formatted sources (splits stay conversation-level; mlx changes):
uv run python -m sft.format_iffc          # only if raw → formatted is stale
uv run python -m sft.split_data           # optional; seed=42, same conversation ids
uv run python -m sft.train_mlx --prepare-only
# then train as usual: ./sft/train_mlx.sh --full --run
```

## Eval (BFCL v4 `multi_turn_base`)

Thesis test is **`multi_turn_base` only**. Best val checkpoint for v1 is **iter 1000**. Fuse that adapter (not necessarily the last iter), convert to Q4_K_M GGUF, serve with `-m`, then score under `--model gemma4-e2b-sft-iffc` so results land in a new folder / DB key. **Run 12 already scored this v1 GGUF: 1/200 = 0.5%** (scored negative; do not overwrite that `runs` row). **Run 13 already scored the v2 filtered-[] GGUF: 1/200 = 0.5%** (`gemma4-e2b-sft-iffc-v2`, LoRA best@2000; do not overwrite). Do **not** `--fc-mode iffc-state` or BoN on these checkpoints. Do **not** claim the `[]` filter fixed SFT. Paths may vary slightly.

```bash
# Fuse best@1000 (mlx-lm adapter filename is often 0001000_adapters.safetensors;
# point --adapter-path at a dir that contains that checkpoint as adapters.safetensors
# if fuse would otherwise pick a later iter)
python -m mlx_lm.fuse \
  --model sft/models/gemma-4-e2b-it-mlx \
  --adapter-path sft/adapters \
  --save-path sft/fused/gemma-4-e2b-it-iffc-best1000

# Fuse writes MLX/HF-style weights. Convert + quantize to match other thesis GGUFs:
python llama.cpp/convert_hf_to_gguf.py sft/fused/gemma-4-e2b-it-iffc-best1000 \
  --outfile sft/fused/gemma-4-e2b-it-iffc-best1000.gguf
./llama.cpp/build/bin/llama-quantize \
  sft/fused/gemma-4-e2b-it-iffc-best1000.gguf \
  sft/fused/gemma-4-e2b-it-iffc-best1000-Q4_K_M.gguf Q4_K_M

# Serve the local GGUF (not -hf). llama-server loads -m; --model is scoring identity.
./llama.cpp/build/bin/llama-server --jinja -fa on \
  -m sft/fused/gemma-4-e2b-it-iffc-best1000-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 -ngl 99 -c 65536 --temp 0.0 \
  -np 4 -cb -ub 1024

uv run python -m benchmark.run_all --model gemma4-e2b-sft-iffc --only-multi-turn \
  --categories multi_turn_base --fc-mode iffc
# Do not --fc-mode iffc-state / iffc-state-bon on this v1 checkpoint (Run 12 0.5%).
# Run 13 already scored the v2 filtered-[] GGUF at 1/200 = 0.5% (do not overwrite;
# do not BoN/STATE on v2; do not claim the [] filter fixed SFT).
# Diagnostic-only history bookkeeping (not a training method; new runs row):
# uv run python -m benchmark.run_all --model gemma4-e2b-sft-iffc-v2 --only-multi-turn \
#   --categories multi_turn_base --fc-mode iffc --iffc-strip-empty-assistant \
#   --notes "DIAGNOSTIC: strip trailing IFFC assistant [] vs Run 13 0.5%"
```

If `convert_hf_to_gguf.py` rejects the fused layout, export/dequantize via mlx_lm.fuse docs for your mlx-lm version, then convert. Results: `benchmark/results/Gemma-4-E2B-it-SFT-IFFC-Q4_K_M/` (distinct from `Gemma-4-E2B-it-Q4_K_M/`).

## Manifest schema (`sft/data/splits/MANIFEST.json`)

Produced by `sft.split_data`. JSON object:

| Field | Meaning |
|---|---|
| `experiment` | `"sft-v1"` |
| `seed` | `42` |
| `train_frac` / `val_frac` / `corpus_test_frac` | `0.90` / `0.05` / `0.05` |
| `chat_format` | `"iffc"` |
| `thesis_test` | BFCL v4 `multi_turn_base` reminder |
| `xlam_sample_size` | `25000` |
| `xlam_oversample` | `"all_parallel_then_fill_single"` |
| `xlam_pool_formatted` / `xlam_sampled` / `toolace_n` | row counts before/after sample |
| `skipped_datasets_v1` | APIGen-MT, Glaive, Hermes |
| `split_unit` | conversation / example id |
| `stratify_on` | source, is_multi_turn, n_tools_in_gold (1 vs ≥2) |
| `base_model` | HF instruct id, GGUF repo, convert commands |
| `counts.{train,valid,corpus_test}` | `n`, `by_source`, `by_is_multi_turn`, `by_n_tools_bin`, `by_stratum` |
| `checksums_sha256` | sha256 of `train.jsonl`, `valid.jsonl`, `corpus_test.jsonl` |
| `files` | absolute paths written |

Re-running `split_data` with seed 42 on the same formatted inputs is deterministic (same ids per split). Checksums change only if the JSONL bytes change.

Each split JSONL line:

```json
{
  "id": "toolace-0",
  "source": "toolace",
  "is_multi_turn": true,
  "n_tools_in_gold": 2,
  "n_tools_bin": "ge2",
  "chat_format": "iffc",
  "messages": [
    {"role": "system", "content": "You are an expert in function calling. ..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "[{\"function_name\": \"...\", \"parameters\": {...}}]"}
  ]
}
```

mlx-lm chat format only requires `messages`. Extra keys are metadata for the manifest / tests.

## Layout

```
sft/
├── README.md                 # this file
├── config.py                 # seed, paths, model ids
├── download_datasets.py
├── format_iffc.py            # reuses benchmark.iffc
├── filter_empty_after_exec.py  # drop mid-trajectory [] after tool results
├── split_data.py
├── train_mlx.py              # unroll + filter + mlx_lm.lora wrapper
├── train_mlx.sh
├── lora_config.yaml          # rank 16, scale 32, mask_prompt
└── data/                     # gitignored artifacts
    ├── raw/
    ├── formatted/
    ├── splits/               # train.jsonl, valid.jsonl, corpus_test.jsonl, MANIFEST.json
    └── mlx/                  # unrolled train/valid for mlx_lm.lora
```

## Tests

```bash
uv run pytest -q tests/sft
uv run pytest -q          # full hermetic suite, including SFT
```

No overnight training in CI. mlx-lm is not imported by unit tests.
