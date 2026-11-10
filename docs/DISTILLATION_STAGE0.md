# Distillation Stage 0 report (setup, no spend)

Written 2026-11-06 on the Mac. Nothing here called a model, the DeepSeek API, or
the benchmark DB for new numbers. Branch `thesis/distill-stage0`, local only.

## 1. Earlier simulators and checker

| What | Where (Mac, untracked) | Covers BFCL backends? |
|---|---|---|
| v11 sims: file system, social, ticket, math; `synth_v11.py` task builder | `sft/v11/sims_v11.py`, `synth_v11.py` | No. Own tool names, "NO BFCL tool names" (module docstring). |
| v13-lk lookup sims: trading, travel, vehicle, messaging, fx; `tasks_lookup.py`, `checker.py`, `gen_lookup.py`, `leak_lookup.py`, entity blocklist | `sft/v13lk/` | No. Same own-name design; never imports `bfcl_eval`. |
| Self-distillation core (`gen.py`, `sdcore.py`, `build_sd.py`) behind Runs 74-75 | none | **Lost** (inventory table rows 8-10, `sft/v13/codex_LOOKUP_SIMS_INVENTORY.md`). Rewritable from `probe.tgz` in `~/Documents/sft_logs_mirror/selfdistill_probe/`. |

The surviving checker rules (end state per turn, response check, tracer on
successful calls only, gate counting only successful calls) are in
`sft/v13/codex_SFT_SELFDISTILL_PLAN.md` section P2. So: **nothing found can make fresh tasks over the real BFCL
backends.** A new generator is needed.

BFCL side: 8 backends (Travel 51, FileSystem 50, Vehicle 50, Trading 50,
Twitter 38, Ticket 28, Message 23, Math 20 of the 200 tasks). A task is
`{question turns, initial_config, involved_classes}` plus gold call strings, and
the stock `multi_turn_checker` already scores any such entry by end state and
response check, so verified rows need no new checker.

## 2. Built in this stage

* `benchmark/distill_exclusions.py`: exclusion list (800 IDs across base,
  miss_func, miss_param, long_context; 1,153 unique user-turn texts; mid20,
  soft10 and fresh-bank groups, all inside the 200) and `check_task` /
  `scan_rows` leak checks. Thresholds are in `THRESHOLDS` and written to every
  report. `python -m benchmark.distill_exclusions build` writes
  `sft/data/distill/exclusion.json` (gitignored).
* `benchmark/teacher_api.py`: DeepSeek adapter. Key only from
  `DEEPSEEK_API_KEY`; `thinking: {"type": "disabled"}` (thinking is on by
  default per DeepSeek's thinking-mode docs); `reasoning_content` stripped
  if it appears (`assistant_text` would otherwise prepend it); retries on
  408/429/5xx/timeouts with backoff and `Retry-After`; concurrency cap and min
  request interval; per-call log (`calls.jsonl`) and per-episode log
  (`episodes.jsonl`) with the exact `model` string the API returns, token
  usage, date, `verified: null`.
* `generate_multi_turn.py`: `_post_chat_sync/_async` delegate to the adapter
  only when one is installed. Default behavior unchanged (22 existing tests and
  `smoke_iffc` pass).
* `smoke_teacher_api.py`, `smoke_distill_exclusions.py`: offline, mock server.

## 3. Open items

* Gate/fresh-bank IDs are copied from `sft/v11/leak_v11.py` and
  `sft/v13lk/fresh_bank_ids.txt`; mid20 matches the Run 73 result doc.
* Leak thresholds (8-gram share >= 1, turn similarity >= 0.85, literal
  overlap report-only) are my picks; Sans to confirm.
* Pricing and the account concurrency limit were not used. Cost comes from the
  Stage 1 pilot.
