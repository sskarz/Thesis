# SFT v5 default / v6 selectable — MLX LoRA on Gemma-4-E2B (IFFC JSON-in-text)

Frozen data mix and split for supervised fine-tuning of **Gemma-4-E2B-it** on Apple Silicon with [mlx-lm](https://github.com/ml-explore/mlx-lm) LoRA. Training targets match the existing `--fc-mode iffc` harness (`benchmark/iffc.py`): assistant turns are a JSON list of `{"function_name", "parameters"}`; tool results are user-role observations. Native `tool_calls` are **not** mixed into this adapter dataset.

This folder prepares data and prints/runs the mlx-lm command. **Run 12 scored the v1 mix as a negative: 1/200 = 0.5%** on BFCL v4 `multi_turn_base` (`gemma4-e2b-sft-iffc`, fused Q4_K_M best@1000). Root cause is ToolACE stop-after-tools `[]` targets (5520 empty assistant turns; 78.1% of post-tool-result targets are `[]`), **not** the IFFC parser (zero `iffc_unparseable`). **Run 13 scored the v2 filtered non-terminal-`[]` mix as the same collapse: 1/200 = 0.5%** (`gemma4-e2b-sft-iffc-v2`, LoRA best@2000 `val=0.211`; only `multi_turn_base_50`; correct-set overlap identical with Run 12). PR #24 dropped 334 mid-trajectory empty-after-exec (non-terminal 334→0) and still kept 552 terminal; the filter did **not** recover. **Run 14** (`--iffc-strip-empty-assistant` on the same v2 GGUF) scored **4/200 = 2.0%** as a **DIAGNOSTIC** history-bookkeeping ablation — did **not** rescue the never-start collapse; **do not present 2% as a method win**. **Run 15 scored SFT-IFFC-v4 LoRA best@1600 as a recovery: 30/200 = 15.0%** (`gemma4-e2b-sft-iffc-v4`, `val=0.196`) — empty-`[]` largely fixed (2/170 empty errors), still **below IFFC 30.5% and below NFC 27.0%**. **Do not present 15% as beating IFFC.** **Run 16 scored IFFC+STATE on that same v4 GGUF as a negative/null overlay: 26/200 = 13.0%** (−2.0 pp vs Run 15). STATE helps base Gemma, not this checkpoint. **Do not propose BoN as the next win on this ckpt.** **Run 17 scored SFT-IFFC-v5 LoRA best@1800 as a regression: 23/200 = 11.5%** (`gemma4-e2b-sft-iffc-v5`, `val=0.147`) — **−3.5 pp / −7 vs Run 15**, also below Run 16 13.0% (state_mismatch 109→133, under_call 59→44, empty 2→0, avg api_calls/entry 17.3→40.3; lost corrects heavily TradingBot). Hypothesis: v5 **synthetic complete turn-0** taught parallel call dumps that desync observe→act APIs. **Do not present 11.5% as a method win.** **No BoN on this ckpt; no more epochs on the v5 mix.** **Run 18 scored SFT-IFFC-v6 LoRA best@1800 as a miss vs v4: 26/200 = 13.0%** (`gemma4-e2b-sft-iffc-v6`, `val=0.142`) — **−2.0 pp / −4 vs Run 15**, modest +1.5 pp vs Run 17, ties Run 16 (empty still 1 on `_43`; state_mismatch 118 / under_call 55; avg api_calls/entry **13.34**, fixing v5’s 40.3 churn). **Do not present 13.0% as a method win.** **No BoN on this ckpt; no more epochs on the v6 mix.** v4 IFFC 15% remains best SFT recovery. Next SFT lever TBD (likely not another thin filter on the same family). Do **not** train more epochs on v1/v2/v4/v5/v6, run BoN/STATE on those checkpoints, or claim the v2 `[]` filter fixed SFT.

**SFT v3** (PR #29) dropped **all** empty-after-exec `[]` targets including terminal and kept / synthesized **recovery_after_empty**. Prepare: empty-after-exec targets = 0; `recovery_after_empty` ≈ 397. Recovery-after-`[]` already PASSes. **SFT v4** (PR #31, Run 15 **30/200 = 15.0%**) extends that curriculum without weakening it: leftover ToolACE empties after ordinary (non-exec) user messages (~2471 clarifying / chitchat `[]`) and first-assistant-in-prefix `[]` (~1859 rows, ~5.7%) still taught turn-0 never-start. v4 drops those first-assistant-empty and clarifying empty-after-user **targets**. Empty-`[]` self-conditioning is largely fixed (2 empty-turn primaries); residual on Run 15 is **state_mismatch ~64%** and **under_calling ~35% (mostly turn 0)**. **SFT v5** (default `sft.train_mlx --prepare-only`, PR #35, Run 17 **23/200 = 11.5%**) keeps every v4 empty-target drop and the recovery-after-`[]` curriculum, then adds **turn-0 call-set completeness** (underspecified xLAM / incomplete first-episode ToolACE, including **synthetic complete turn-0**) and **env-state consistency**. **Scored regression vs v4.** **SFT v6** (`--recipe v6`, Run 18 **26/200 = 13.0%**) is **not** a v4 relabel: same v4 empty/recovery filters, **plus only** `drop_state_inconsistent=True`. v5’s harmful turn-0 rewrites stay **off** (`drop_underspecified_turn0=False`, `synthesize_complete_turn0=False`). Conversation-level ToolACE+xLAM splits stay frozen; BFCL `multi_turn_base` stays held out. **Scored miss vs v4 15.0%.** **Do not present as a method win.** Details in `THESIS.md` / `METHODS.md`.

## Locked mix (do not reopen)

| Source | Role | Size |
|---|---|---|
| [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | **all** rows (~11,300 multi-turn) | full train split |
| [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | stratified **25,000** rows | oversample parallel / multi-answer (`n_tools_in_gold ≥ 2` taken first, then fill from single-call) |

**Skipped for v1:** APIGen-MT-5k (NC license), Glaive, Hermes (Hermes is an optional backup only if a source disappears — not in this pipeline).

## SFT v7 (APIGen-MT core — scaffolding, **unscored**)

New experiment id `sft-v7`. **Do not invent a v7 BFCL score.** Distinct from the frozen v1 ToolACE+xLAM splits (`sft/data/splits/`, seed 42) and from v4–v6 adapters.

**License / Hub gate:** [Salesforce/APIGen-MT-5k](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k) is **CC-BY-NC-4.0** and Hub-gated. Cite Prabhakar et al., *APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay*, arXiv:2504.03601, 2025. Thesis academic use only — not for competing with OpenAI. **Sanskar must Accept the Hub license before download.** Do not commit raw/formatted dataset files (already gitignored under `sft/data/`).

```bash
# 0) Hub Accept + token (required before any APIGen-MT download)
huggingface-cli login
# Open https://huggingface.co/datasets/Salesforce/APIGen-MT-5k and click Agree

# 1) Ingest + format (does not run in CI)
uv run python -m sft.download_datasets --apigen-mt-only
uv run python -m sft.format_iffc --apigen-mt
# optional filler refresh: uv run python -m sft.format_iffc --all-v7

# 2) Mix → sft/data/splits-v7/ (seed 47; refuses to overwrite sft/data/splits/)
#    Conversation-level gates only (pct_apigen ≥ 0.5, irrelevance caps).
#    Full APIGen trajs are stored; do not expect unrolled pct_nontool here.
uv run python -m sft.build_mix_v7 --out sft/data/splits-v7
uv run python -m sft.mix_stats_v7 --splits sft/data/splits-v7 --gate

# 3) Unroll (recipe v7: v4 empty/recovery on ToolACE/xLAM filler;
#    light empty-after-exec on apigen_mt; NO v5 synthetic complete-turn0;
#    then cap non-tool unrolls so mlx-v7 pct_nontool is ~15%, seed 47).
#    Splits-v7 conversations stay full. Never map NL finals to [].
uv run python -m sft.train_mlx --prepare-only --recipe v7 \
  --splits-dir sft/data/splits-v7 --mlx-dir sft/data/mlx-v7
uv run python -m sft.mix_stats_v7 --splits sft/data/splits-v7 \
  --mlx-dir sft/data/mlx-v7 --gate
./sft/train_mlx.sh --smoke --run --recipe v7 \
  --mlx-dir sft/data/mlx-v7 --adapter-path sft/adapters-v7-smoke
```

Converter (`sft/format_apigen_mt.py`): ShareGPT `human`→user, `function_call`→IFFC JSON, `observation`→`format_iffc_execution_user_message`, `gpt`→assistant **natural language**. **Never** auto-map NL done to `[]`. Unrolled non-tool share is capped at prepare (`sft/cap_nontool_v7.py`, seed 47): keep every `tool_call` row; downsample clarify / final / irrelevance so mlx-v7 `pct_nontool` is in `[0.10, 0.20]` (target ~0.15). Student-aware filter is a stub (`python -m sft.student_filter_v7 --help`). Registry key: `gemma4-e2b-sft-iffc-v7` (GGUF: `sft/fused/gemma-4-e2b-it-iffc-v7-best1800-Q4_K_M.gguf`, LoRA best@1800).

**Splits** (by conversation / example id, never by turn): `seed=42`; stratify on `source`, `is_multi_turn`, `n_tools_in_gold` (1 vs ≥2); ~90% train / ~5% val / ~5% corpus_test → about **Train 32.7k / Val 1.8k / Corpus-test 1.8k**.

**Thesis test:** BFCL v4 `multi_turn_base` only. Do not add BFCL prompts or past scored responses to these JSONL files.

## Base model

`benchmark/config.py` registers Gemma as GGUF:

- key: `gemma4-e2b` — base instruct GGUF (`bartowski/google_gemma-4-E2B-it-GGUF`, `Q4_K_M`)
- key: `gemma4-e2b-sft-iffc` — Mac-local MLX LoRA SFT v1 (IFFC mix, best val @ iter 1000). Run 12 scored **1/200 = 0.5%**. Distinct `api_name` / `slug` so BFCL results do **not** overwrite base Gemma. Serve with `llama-server -m <local.gguf>` (not `-hf`).
- key: `gemma4-e2b-sft-iffc-v2` — filtered-mix SFT v2 (drop non-terminal `[]` after tool-exec, LoRA best@2000 val 0.211, PR #24). Distinct `api_name` / `slug` so Run 13 does **not** overwrite Run 12. Run 13 scored **1/200 = 0.5%** (same collapse). Figures: `benchmark/results/Gemma-4-E2B-it-SFT-IFFC-v2-Q4_K_M/figures`. Serve with `llama-server -m <local.gguf>` (not `-hf`).
- key: `gemma4-e2b-sft-iffc-v4` — SFT v4 (v3 empty-after-exec + drop first-assistant-empty and clarifying `[]`, LoRA best@1600 val 0.196, PR #31). Distinct `api_name` / `slug` so Run 15 does **not** overwrite Runs 12–14. Run 15 scored **30/200 = 15.0%** (scored recovery, still below IFFC/NFC). Run 16 scored **26/200 = 13.0%** under `--fc-mode iffc-state` on the same GGUF (null/negative vs Run 15). Serve with `llama-server -m sft/fused/gemma-4-e2b-it-iffc-v4-best1600-Q4_K_M.gguf` (not `-hf`).
- key: `gemma4-e2b-sft-iffc-v5` — SFT v5 (PR #35: v4 empty-recovery + turn-0 completeness + env-state consistency; LoRA best@1800 val 0.147). Distinct `api_name` / `slug` so Run 17 does **not** overwrite Runs 12–16. Serve with `llama-server -m sft/fused/gemma-4-e2b-it-iffc-v5-best1800-Q4_K_M.gguf` (not `-hf`). Run 17 scored **23/200 = 11.5%** under `--fc-mode iffc` (**scored regression vs Run 15 15.0%**). **Do not present as a method win.** Do **not** overwrite. **No BoN / no more epochs on this mix.**
- key: `gemma4-e2b-sft-iffc-v6` — SFT v6 (PR #38: v4 empty/recovery + `drop_state_inconsistent` only; turn-0 rewrites off; LoRA best@1800 val 0.142). Distinct `api_name` / `slug` so Run 18 does **not** overwrite Runs 12–17. Serve with `llama-server -m sft/fused/gemma-4-e2b-it-iffc-v6-best1800-Q4_K_M.gguf` (not `-hf`). Run 18 scored **26/200 = 13.0%** under `--fc-mode iffc` (**scored miss vs Run 15 15.0%**). **Do not present as a method win.** Do **not** overwrite. **No BoN / no more epochs on this mix.**
- key: `gemma4-e2b-sft-iffc-v7` — SFT v7 (APIGen-MT-5k core; IFFC JSON-in-text; LoRA best@1800 val 0.197, fuse locked with Codex). Distinct `api_name` / `slug`. Serve with `llama-server -m sft/fused/gemma-4-e2b-it-iffc-v7-best1800-Q4_K_M.gguf` (not `-hf`). **Unscored. Do not invent a v7 BFCL number.**

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

`--mask-prompt` only backprops the **last** chat message. `sft.train_mlx` unrolls each conversation so every assistant turn is a row ending on that turn (split membership stays at conversation id), then applies the [SFT v5 unroll recipe](#unroll-filter-sft-v5-turn-0-completeness--env-state-consistency) by default, or `--recipe v6` for [v4 + state-inconsistent only](#unroll-filter-sft-v6-v4--state-inconsistent-only).

Dry-run (Linux/CI): `uv run python -m sft.train_mlx --smoke` prints the command and writes unrolled JSONL without calling mlx-lm.

## Unroll filter (SFT v3: drop all empty-after-exec + recovery curriculum)

SFT v1 trained on every unrolled assistant turn. ToolACE maps natural-language “done” replies to IFFC `[]`. After a user message that is an IFFC tool-result observation (`Function execution results:…` from `format_iffc_execution_user_message`), **78%** of those targets were `[]` (2121 empty vs 596 continue). All ~5,520 empty `[]` targets in the mlx train set came from ToolACE; xLAM had 0. That taught early stop: BFCL Run 12 empty-turn rate was ~70% vs ~13% on base IFFC, and `multi_turn_base` collapsed to 0.5% even though the live probe still emitted valid IFFC JSON. Run 13 scored the v2 filtered mix at the **same 0.5%** (primary empty_turn 167 vs Run 12 101; audit empty turns still ~70%, 522/734 error rows). **Run 14** (`--iffc-strip-empty-assistant` on the same v2 GGUF) scored **4/200 = 2.0%** as a **DIAGNOSTIC** history-bookkeeping ablation — it did **not** rescue the never-start collapse; **do not present 2% as a method win**.

BFCL `multi_turn_base` is call → results → `[]` → **new user turn**. Training that over-weights assistant `[]` after tool results makes the SFT model stop and then fail to start again. v3 attacks that in the **unrolled mlx JSONL**, not in the live harness.

**v3 policy (ablation, drop not downweight):** when building unrolled mlx JSONL:

1. Drop a row iff the assistant target is exactly `[]` **and** the previous user message looks like an IFFC tool-result observation (`Function execution results` or the `format_iffc_execution_user_message` footer). This includes **terminal** empty-after-exec (stricter than v2’s non-terminal-only filter).
2. Keep every non-empty tool-call target, including continuation after exec (`user` observation → `[call]`).
3. Keep ordinary user → `[]` clarifying / irrelevance turns (previous user is not an execution observation).
4. **Recovery curriculum:** keep rows whose history already contains a prior assistant `[]` (formatter NL→`[]`, or a clarifying stop), then a **new user question**, with a nonempty tool-call gold. These are counted as `recovery_after_empty`.
5. **Synthetic stop in history only:** ToolACE multi-turn often has no explicit `[]` between tool results and the next human turn (NL is either collapsed into a `[]` *target* — which v3 drops — or the transcript jumps from the execution observation to the next user question). For the *next* user→call unroll row, if history is `… exec results, new user, [call]` with nothing in between, insert assistant `[]` **only in that row’s history prefix**. The gold target stays the nonempty call. This does **not** add a `[]` training row and must not be mistaken for putting BFCL test prompts into train — it is bookkeeping structure only.

xLAM rows are unchanged unless they somehow gain this pattern. Conversation-level `split_data` files are **not** rewritten (split unit stays conversation id). mlx-lm chat JSONL has no sample-weight field, so v3 **drops** empty-after-exec rows.

**v3 ablation flags** (no longer the default; pass both v4 keep-flags):

| Flag | Recipe | Behavior |
|---|---|---|
| `--keep-first-assistant-empty` **and** `--keep-clarifying-empty-after-user` | **v3** | Drop all empty-after-exec (incl. terminal) + synthetic recovery history |
| `--keep-terminal-empty-after-exec` | v2 | Drop non-terminal empty-after-exec only; keep terminal `[]`; no synthetic recovery |
| `--keep-nonterminal-empty-after-exec` | v1 | Keep every assistant turn (Run 12 mix) |
| `--no-synthetic-recovery` | v3 minus synth | Still drop all empty-after-exec; do not insert history-only `[]` |

Unroll logs before/after counts of total rows, empty `[]` targets, empty-after-exec (non-terminal + terminal), `recovery_after_empty`, and synthetic insertions, by source (`toolace` vs `xlam`). On the mix that produced Run 13, v2 dropped **334** non-terminal empty-after-exec (334→0) and **kept 552** terminal — that mix scored **1/200 = 0.5%**. v3 should take empty-after-exec **targets** to ~0 and show a nonzero `recovery_after_empty` count.

```bash
# v3 unroll (same frozen splits; no longer the default):
uv run python -m sft.train_mlx --prepare-only \
  --keep-first-assistant-empty --keep-clarifying-empty-after-user
```

Confirm a v3 prepare log: `recipe=v3`, empty-after-exec after ≈ 0, `recovery_after_empty` after > 0. Do **not** add BFCL `multi_turn_base` ids or prompts to these JSONL files.

## Unroll filter (SFT v4: drop first-assistant-empty + clarifying empty-after-user)

v3 zeroed empty-after-exec **targets** and kept recovery. Remaining empties on that mix: **~2471** assistant `[]` after ordinary (non-exec) user messages (all ToolACE clarifying / chitchat) and **~1859** unrolled rows whose first assistant in the prefix is `[]` (~5.7%). Those first-assistant `[]` targets are the turn-0 never-start teacher: v3 smoke (`mlx_lm` + `sft/adapters-v3-smoke` @200 and @400) still emits `[]` on GFS `multi_turn_base_1` turn-0, while base IFFC Run 5 starts that entry with `ls(a=True)`. Recovery-after-`[]` already PASSes — v4 does **not** weaken the v3 recovery curriculum.

**v4 policy (ablation, drop not downweight):** all of v3, plus:

1. **Drop first-assistant-empty targets:** when the unrolled row’s gold is exactly `[]` **and** that assistant is the first assistant message in the messages prefix, drop the row.
2. **Drop clarifying empty-after-user:** when gold is `[]`, the immediately previous message is `user`, and it is **not** an IFFC tool-result observation, drop the row. (Empty-after-exec stays a v3 drop; do not treat tool-result observations as clarifying.)
3. Keep nonempty tool-call targets, including continuation after exec and **recovery_after_empty** (prior `[]` in history, then a new user question, nonempty call gold). The dropped `[]` remains in later rows’ history so recovery still trains.
4. Stats log `first_turn_empty` / `clarifying_empty` before → after and dropped counts, in addition to empty-after-exec / `recovery_after_empty`.

A first-assistant `[]` after a normal user message matches **both** v4 rules. Either keep-flag alone is not v3; pass **both** to ablate.

**Ablation flags** (pass through `sft.train_mlx` / `./sft/train_mlx.sh`):

| Flag | Recipe | Behavior |
|---|---|---|
| `--recipe v4` | **v4** | v3 + drop first-assistant-empty + drop clarifying empty-after-user |
| `--keep-first-assistant-empty` **and** `--keep-clarifying-empty-after-user` | v3 | Empty-after-exec + recovery only (v3 smoke mix) |
| `--keep-first-assistant-empty` | v4 minus first-turn | Still drops clarifying empty-after-user (overlap still going) |
| `--keep-clarifying-empty-after-user` | v4 minus clarifying | Still drops first-assistant-empty (overlap still going) |
| `--keep-terminal-empty-after-exec` | v2 | Drop non-terminal empty-after-exec only; keep terminal `[]`; no synthetic recovery |
| `--keep-nonterminal-empty-after-exec` | v1 | Keep every assistant turn (Run 12 mix) |
| `--no-synthetic-recovery` | v4/v3 minus synth | Still apply the active empty-target drops; do not insert history-only `[]` |

```bash
# v4 unroll (same frozen splits; no longer the default):
uv run python -m sft.train_mlx --prepare-only --recipe v4
# equivalent audit:
uv run python -m sft.filter_empty_after_exec --splits-dir sft/data/splits --recipe v4
```

Confirm a v4 prepare log: `recipe=v4`, `first_turn_empty` after ≈ 0, `clarifying_empty` after ≈ 0, empty-after-exec after ≈ 0, `recovery_after_empty` after > 0. Do **not** add BFCL `multi_turn_base` ids or prompts to these JSONL files.

## Unroll filter (SFT v5: turn-0 completeness + env-state consistency)

v4 recovered calling (Run 15 15.0%) by dropping empty-`[]` teachers. Residual on that checkpoint: **state_mismatch ~64%** (VehicleControl / Travel / TradingBot / GFS) and **under_calling ~35%**, **mostly turn 0**. IFFC+STATE on the same GGUF did not help (Run 16 **26/200 = 13.0%**, scored negative/null vs Run 15). Recipe v5 (Run 17) is a **scored regression** at **11.5%**. Do **not** add epochs on the v4 or v5 or v6 mix; do **not** design BoN as the win for a still-below-IFFC ckpt. Recipe v6 is **scored** (Run 18 13.0%, miss vs v4).

Verified data gaps (inspect `sft/format_iffc.py`, not a guessed root cause):

* **xLAM is single-turn** (`format_xlam_row`: query → one assistant JSON list). A query that names two catalog tools with only one gold call is a turn-0 under-call teacher. xLAM has **no** exec observations, so it cannot teach env-state sequential calls.
* **ToolACE is multi-turn with IFFC exec observations**, but a first-turn gold that omits later first-episode calls whose arguments were already in the opening user message is an under-specified parallel set. Continue-after-exec that `cd(docs)` after a listing that never contained `docs` teaches state mismatch.

**v5 policy (default, drop / rewrite not downweight):** all of v4, plus:

1. **Drop underspecified turn-0:** first nonempty gold has **one** call, the opening user names ≥2 catalog tools (distinctive name tokens). Typical xLAM under-call.
2. **Complete first-episode gold:** independently-knowable later continue-after-exec calls (every nontrivial string param already in the opening user **and** the exec dump confirmed the entity) are **merged into the first nonempty gold**. History prefix is unchanged. This is not a `[]` training row and is not a BFCL prompt.
3. **Drop absorbed follow-ups:** later first-episode continues whose whole gold was merged into turn 0 are dropped so we do not teach calling them twice.
4. **Drop state-inconsistent continue-after-exec:** a nontrivial string param appears in a prior non-exec user message but **not** in the exec observation or the previous assistant call (user named an entity the env never confirmed). State-grounded sequential continues are kept.
5. Recovery-after-`[]` is unchanged (nonempty gold after a prior stop then a **new** user question is not continue-after-exec). Empty-after-exec / first-assistant-empty / clarifying drops stay on.

**Ablation flags** (pass through `sft.train_mlx` / `./sft/train_mlx.sh`):

| Flag | Recipe | Behavior |
|---|---|---|
| *(default)* / `--recipe v5` | **v5** | v4 + underspecified drop + complete-turn0 synth + state-inconsistent drop. **Run 17 scored 11.5%** (regression vs v4). |
| `--recipe v6` | **v6** | v4 empty/recovery + **only** `drop_state_inconsistent`. Turn-0 rewrites **off**. Distinct from v4. **Run 18 scored 13.0%** (miss vs v4). |
| `--recipe v7` | **v7** | New mix (APIGen-MT core). v4 empty/recovery on ToolACE/xLAM filler; light empty-after-exec on `apigen_mt`; **no** v5 complete-turn0; then cap non-tool unrolls (~15%). **Unscored.** |
| `--recipe v4` | v4 | Empty-target curriculum only (Run 15 mix, 15.0%) |
| `--keep-underspecified-turn0` **and** `--keep-state-inconsistent` **and** `--no-synthetic-complete-turn0` | v4 | Same as `--recipe v4` |
| `--keep-underspecified-turn0` **and** `--no-synthetic-complete-turn0` | v6 | Same as `--recipe v6` |
| `--no-synthetic-complete-turn0` | v5 minus synth | Incomplete first-turns are **dropped** instead of rewritten |
| `--keep-underspecified-turn0` | v5 minus xLAM underspec | Still merges ToolACE first-episode extras / drops state-inconsistent |
| `--keep-state-inconsistent` | v5 minus state drop | Still drops underspecified and merges complete turn-0 |
| `--recipe v3` / both v4 keep-flags | v3 | Empty-after-exec + recovery only |
| `--keep-terminal-empty-after-exec` | v2 | Drop non-terminal empty-after-exec only |
| `--keep-nonterminal-empty-after-exec` | v1 | Keep every assistant turn |
| `--no-synthetic-recovery` | v5/v6/v4/v3 minus synth `[]` | Still apply the active drops; do not insert history-only `[]` |

```bash
# Already have formatted ToolACE + xLAM and frozen splits? Re-unroll mlx train/val (v5 default):
uv run python -m sft.train_mlx --prepare-only
# print v5 vs v4 row deltas on the same frozen splits (no second write):
uv run python -m sft.train_mlx --prepare-only --compare-recipe v4
# equivalent audit (prints stats; does not write mlx/ unless --out):
uv run python -m sft.filter_empty_after_exec --splits-dir sft/data/splits --compare-recipe v4

# Next Mac prepare (v6 = v4 + state-inconsistent only; expect train size
# ≈ v4 minus state-inconsistent continues, closer to ~31232 than v5’s ~29850):
uv run python -m sft.train_mlx --prepare-only --recipe v6 --compare-recipe v4
uv run python -m sft.filter_empty_after_exec --splits-dir sft/data/splits --recipe v6 --compare-recipe v4

# v4 / v3 / v2 / v1 ablations (same frozen splits, different mlx unroll):
uv run python -m sft.train_mlx --prepare-only --recipe v4
uv run python -m sft.train_mlx --prepare-only --recipe v3
uv run python -m sft.train_mlx --prepare-only --keep-terminal-empty-after-exec
uv run python -m sft.train_mlx --prepare-only --keep-nonterminal-empty-after-exec

# Full regenerate from formatted sources (splits stay conversation-level; mlx changes):
uv run python -m sft.format_iffc          # only if raw → formatted is stale
uv run python -m sft.split_data           # optional; seed=42, same conversation ids
uv run python -m sft.train_mlx --prepare-only --recipe v6 --compare-recipe v4
# then train as usual: ./sft/train_mlx.sh --full --run --recipe v6
```

**Expected prepare stats** (exact N is the Mac `--compare-recipe v4` log; this cloud checkout has no frozen splits):

| Signal | v4 (Mac mix, PR #31) | v5 (Run 17) | v6 (this recipe) |
|---|---|---|---|
| empty-after-exec after | 0 | 0 (unchanged) | 0 (unchanged) |
| first_turn_empty after | 0 | 0 (unchanged) | 0 (unchanged) |
| clarifying_empty after | 0 | 0 (unchanged) | 0 (unchanged) |
| recovery_after_empty after | ≈ 397 | ≈ same (not weakened) | ≈ same (not weakened) |
| underspecified_turn0 | kept | dropped (mostly xLAM) | **kept** (v5 drop off) |
| incomplete first-episode | kept as under-call teacher | rewritten to the full independent call set | **kept sequential** (synth off) |
| state_inconsistent continue-after-exec | kept | dropped | **dropped** (the one keep from v5) |
| synthetic_complete_turn0 | 0 | > 0 (Run 17 hypothesis: harmful) | **0** |
| train rows (Mac, approx.) | > ~31232 | ~29850 | **~31232** (v4 minus state-inconsistent continues) |

Confirm a v5 prepare log: `recipe=v5`, `first_turn_empty` after ≈ 0, `clarifying_empty` after ≈ 0, empty-after-exec after ≈ 0, `recovery_after_empty` after > 0, plus the v5 lines (`underspecified_turn0`, `incomplete_turn0`, `state_inconsistent`). Confirm a v6 prepare log: `recipe=v6`, `drop_underspecified_turn0=False`, `synthetic_complete_turn0=False`, `drop_state_inconsistent=True`, empty-after-exec / first_turn_empty / clarifying after ≈ 0, `recovery_after_empty` after > 0. Do **not** add BFCL `multi_turn_base` ids or prompts to these JSONL files. **Run 17 already scored the v5 mix at 11.5% (regression vs v4). Run 18 already scored the v6 mix at 13.0% (miss vs v4). Do not claim v5 or v6 beat IFFC or v4.**

## Unroll filter (SFT v6: v4 + state-inconsistent only)

**Run 17 scored v5 IFFC 23/200 = 11.5%** vs Run 15 v4 IFFC **15.0%**. Diagnosis: state_mismatch 109→133, under_call 59→44, empty 2→0, avg api_calls/entry **17.3→40.3**. Lost corrects heavily TradingBot. v5’s **synthetic complete turn-0** is the suspected harm (parallel call dumps that desync observe→act APIs). The underspecified-turn0 drop helped only modestly. **`drop_state_inconsistent` is the one keep.**

**v6 policy (selectable `--recipe v6`, drop not rewrite):**

1. All of **v4**: empty-after-exec (terminal + non-terminal), first-assistant-empty, clarifying empty-after-user, recovery-after-`[]` teaching (including synthetic history-only `[]`).
2. **Drop state-inconsistent continue-after-exec** (same detector as v5 / `sft.curriculum_v5.is_state_inconsistent_continue`).
3. **Off:** `drop_underspecified_turn0=False`, `synthesize_complete_turn0=False` (must be off), and therefore no absorbed-followup drop.

Distinct recipe id `v6` — **do not** re-label identical v4. No BoN, no more-epoch trainers, no BFCL harness changes. Rebuild mlx JSONL with `--recipe v6 --compare-recipe v4`. Expected train size ≈ v4 minus state-inconsistent continues (closer to **~31232** than v5’s **~29850**). **Run 18 already scored 26/200 = 13.0% (miss vs v4 15.0%). Do not present as a method win.** See `METHODS.md`.

## Eval (BFCL v4 `multi_turn_base`)

Thesis test is **`multi_turn_base` only**. Best val checkpoint for v1 is **iter 1000**. Fuse that adapter (not necessarily the last iter), convert to Q4_K_M GGUF, serve with `-m`, then score under `--model gemma4-e2b-sft-iffc` so results land in a new folder / DB key. **Run 12 already scored this v1 GGUF: 1/200 = 0.5%** (scored negative; do not overwrite that `runs` row). **Run 13 already scored the v2 filtered-[] GGUF: 1/200 = 0.5%** (`gemma4-e2b-sft-iffc-v2`, LoRA best@2000; do not overwrite). **Run 15 already scored the v4 GGUF: 30/200 = 15.0%** (`gemma4-e2b-sft-iffc-v4`, LoRA best@1600; scored recovery, still below IFFC/NFC — do not overwrite). **Run 16 already scored `--fc-mode iffc-state` on that same v4 GGUF: 26/200 = 13.0%** (null/negative vs Run 15 — do not overwrite; do **not** BoN this ckpt). **Run 17 already scored the v5 GGUF: 23/200 = 11.5%** (`gemma4-e2b-sft-iffc-v5`, LoRA best@1800 val 0.147, PR #35; **scored regression vs Run 15** — do not overwrite; do **not** present as a method win; do **not** BoN / more-epochs this mix). v5 fused GGUF is `sft/fused/gemma-4-e2b-it-iffc-v5-best1800-Q4_K_M.gguf`. v6 fused GGUF is `sft/fused/gemma-4-e2b-it-iffc-v6-best1800-Q4_K_M.gguf` (LoRA best@1800 val 0.142, PR #38); score under `--model gemma4-e2b-sft-iffc-v6`. **Run 18 already scored this v6 GGUF: 26/200 = 13.0%** (**scored miss vs Run 15 15.0%** — do not overwrite; do **not** present as a method win; do **not** BoN / more-epochs this mix). Frame against v4 15.0% / v5 11.5% / base IFFC 30.5%. Do **not** `--fc-mode iffc-state` or BoN on the collapsed v1/v2 checkpoints. Do **not** claim the v2 `[]` filter fixed SFT. Paths may vary slightly.

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
# Run 14 already scored --iffc-strip-empty-assistant on that same v2 GGUF at
# 4/200 = 2.0% (DIAGNOSTIC history bookkeeping — not a training method; do not
# overwrite; do not present 2% as a method win).
# Run 15 already scored SFT-IFFC-v4 best@1600 at 30/200 = 15.0%
# (scored recovery, still below IFFC 30.5% and NFC 27.0%; do not overwrite;
# do not present 15% as beating IFFC).
# Run 16 already scored --fc-mode iffc-state on that same v4 GGUF at
# 26/200 = 13.0% (null/negative vs Run 15; do not overwrite; do not BoN this ckpt).
# Run 17 already scored SFT-IFFC-v5 best@1800 at 23/200 = 11.5%
# (scored regression vs Run 15 15.0%; do not overwrite; do not present as a
# method win; do not BoN this ckpt; do not more-epochs the v5 mix).
# Run 18 already scored SFT-IFFC-v6 best@1800 at 26/200 = 13.0%
# (scored miss vs Run 15 15.0%; do not overwrite; do not present as a
# method win; do not BoN this ckpt; do not more-epochs the v6 mix).
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
├── format_apigen_mt.py       # ShareGPT function_call/observation/gpt → IFFC
├── filter_empty_after_exec.py  # v5 default; --recipe v7 = APIGen-MT mix unroll
├── curriculum_v5.py            # turn-0 completeness + env-state detectors (v6 reuses state drop)
├── cap_nontool_v7.py         # v7: downsample non-tool unrolls to ~15% (mlx-v7)
├── mix_stats_v7.py           # v7 taxonomy / source / empty-after-exec CLI
├── build_mix_v7.py           # assemble splits-v7 (seed 47; NC license in MANIFEST)
├── student_filter_v7.py      # Phase 3 stub (MLX Jaccard loop not in this PR)
├── split_data.py
├── train_mlx.py              # unroll + filter + mlx_lm.lora wrapper
├── train_mlx.sh
├── lora_config.yaml          # rank 16, scale 32, mask_prompt
└── data/                     # gitignored artifacts
    ├── raw/
    ├── formatted/
    ├── splits/               # v1 train.jsonl, valid.jsonl, corpus_test.jsonl, MANIFEST.json
    ├── splits-v7/            # v7 mix (do not overwrite splits/)
    ├── mlx/                  # unrolled train/valid for mlx_lm.lora
    └── mlx-v7/
```

## Tests

```bash
uv run pytest -q tests/sft
uv run pytest -q          # full hermetic suite, including SFT
```

No overnight training in CI. mlx-lm is not imported by unit tests.
