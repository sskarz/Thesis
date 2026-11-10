# Teacher-student distillation: the plan, in plain words

For Sans. Written 2026-11-06. Every number is from the repo docs or the pilot
logs and is marked with where it came from. Things I believe but have not tested
are marked **(hypothesis)**.

---

## 1. The goal

Get a fine-tuned **Gemma-4-E2B-it** that does *better than, or at least no worse
than,* the untuned model on BFCL v4 `multi_turn_base`, so Chapter 6 can say
"training helped" instead of "training hurt".

A negative result with a clean test is still a valid thesis result. The point of
this work is to give training a fair, well-designed shot, and to report it
honestly either way.

## 2. Where we are and why we're here

Every fine-tune so far lost to the untuned model (200 tasks, `docs/PAPER_WRITING_PROCESS.md`):

| Model | Score |
|---|---|
| Untuned, JSON-text calling (Run 5) | 61/200 = 30.5% |
| Untuned + the full harness (Run 78) | 77/200 = 38.5% |
| DPO (Run 51) | 53/200 = 26.5% |
| Best SFT (SFT-v4, Run 15) | 30/200 = 15.0% |

On the 20-task screening gate ("mid20", bar = 17/20, Run 69):

| Attempt | Gate result |
|---|---|
| v11 mix, 30,000 rows, lowest validation loss yet | 2/20 |
| Self-distillation, on the model's own checker-approved answers (Runs 74, 75, 76) | 8, 14, 12 / 20 |
| Null test, 39 updates on 156 rows (Run 73) | 15/20 |

Two things stand out:
1. **Low validation loss did not predict success.** v11 fit its data very well and
   still collapsed. So the problem is *what* the data teaches, not how well the
   trainer fits it.
2. **Even a tiny, gentle training run (the null test) slightly hurt** (15 vs 17).
   Any new data has to clear a high bar just to do no harm.

Why the models fail (from the paper's failure labels, Run 78): empty turns,
under-calling (stopping before finishing), and **state mismatch** (the end state
of the tools doesn't match the answer). Public datasets use *other tools and
other states*, which is exactly where the model slips **(hypothesis)**.

## 3. The idea

**Use a stronger model as a teacher, in the exact same world the student is
tested in, and keep only the teacher's answers that are checker-accepted.**

Pipeline:

```
 new task          teacher solves it       real tools run,          keep only if
 (user turns +  -> blind, in the same -> real results fed back -> end state matches
 gold calls)       JSON-text loop          to the teacher           the gold
      |                                                                |
  leak-checked                                              teacher's step-by-step
  vs all BFCL tasks                                         decisions = training rows
```

Key choices and why:
- **Same world as the test.** New tasks, but over the *real BFCL tool backends*
  (file system, trading, travel, vehicle, tickets, messages, Twitter, math). The
  student learns state tracking on the tools it will actually face.
- **Teacher decisions in real states.** Each training row is "given this
  conversation and these real tool results, the teacher's next move was X". That
  is exactly what the student must do at test time.
- **Only verified episodes.** The stock BFCL checker scores end state against
  gold. The checker is the quality filter, but an accepted episode can still contain a poor step (see the Codex review notes).
- **Text only.** The teacher is DeepSeek V4.1 Flash. Different tokenizer, so we
  copy its *text outputs* only (black-box distillation). Gemma stays the only
  model we evaluate.
- **No reasoning in rows.** The student format has no reasoning text, so the
  teacher runs with reasoning off.

## 4. A real example (from the pilot)

Task `distill_pilot_0002`, TradingBot, written by the teacher, validated by
running the gold in the real backend, then solved blind by the teacher. Result:
**verified**.

| User says | Gold calls | Teacher did (step by step) |
|---|---|---|
| "Log me in as stagehand_jo with password footlights99..." | `trading_login(...)` | login, then `[]` (stop) |
| "Top up our account with 2500 and tell me the balance" | `fund_account(2500)`, `get_account_info()` | both calls, then `[]` |
| "Pull up our order history, find the pending one, cancel it" | `get_order_history()`, `get_order_details(12446)`, `cancel_order(12446)` | history, then details of **both** orders (an extra look-up), then cancel, then `[]` |

Notes: each step also sees the harness's status card (`[OODA]` goal, slots, last
call, next). The extra look-up is fine, the end state still matches. This is the
kind of row (conversation so far -> next call or stop) that would be a training
example.

A **bad** example, why we need a fix: task `distill_pilot_0020` (travel). The gold
uses `access_token='482910'` and a booking id `BK-5501`. The user never said
them and no earlier tool result shows them. The teacher can't know them, so the
task *fails verification through no fault of the teacher*.

## 5. Why generate data instead of mixing public sets

- It was tried: v11 *was* the careful public-data mix (APIGen-MT, ToolACE, xLAM,
  synthetic) and scored 2/20.
- You asked for a really good dataset, not another filter on old mixes.
- It's cheap. The pilot used ~$0.07 of the $20 key. Our first training run is only
  39 updates, so we need a few hundred good rows, not 30,000.

## 6. Why data first, not a fancier trainer

- The trainer wasn't the thing that varied wrongly: low loss, bad behavior.
- Changing trainer and data together makes the first test unreadable. We want one
  clean comparison against the 15/20 null test.
- `mlx-lm-lora` (DPO / ORPO / GRPO) stays a **stage-5 idea**: preference pairs of
  teacher-verified answers vs Gemma's failures, or the checker as an RL reward.
  Unchecked: Gemma-4 support, and multi-turn RL on one Mac is probably too slow
  for the deadlines.

## 7. What the pilot found

Teacher `deepseek/deepseek-v4.1-flash-1017` via OpenRouter, pinned to DeepInfra
(DeepSeek's own endpoint is blocked by your account's privacy setting). Source:
`docs/DISTILLATION_PILOT.md` and the logs in `sft/data/distill/pilot/`.

- Authoring: 50 slots -> 22 valid tasks (21 after dedup), about $0.06, a few
  minutes. 0 leaked.
- Solving: **6 of 21 verified**, 36 s wall, $0.0137.
- Why so few: mostly bad tasks (gold uses values nobody was told). An offline
  rough check: tasks with all gold values grounded verified 3 of 4, tasks with
  ungrounded values verified 3 of 17. Tiny sample, imperfect test, so
  suggestive only.
- Skew: wins on Trading/Math/File system; 0 on Ticket, Travel, Vehicle, Message,
  Twitter. Real coverage needs a per-backend quota.

## 8. The plan from here (approval stop at every stage)

| Stage | What | Status |
|---|---|---|
| 0 | Setup: exclusion list, teacher adapter, smoke tests | **done** |
| 1 | Pilot (~20 episodes): does the pipeline work, what does it cost | **done** (results above) |
| 1b | Fix task quality: reject ungrounded gold at authoring time; backend quotas; second small pilot | **next, needs your OK** |
| 2 | Full generation: a few hundred verified episodes, leak check, decision-mix report (stop share, look-up share vs the null and self-distill data), hand-check examples | needs OK |
| 3 | Null-scale training: 39 updates, full chain (template, LoRA, fuse, GGUF Q4_K_M, serve) | needs OK + time estimate |
| 4 | Gate: 20-task gate + fresh 20-task bank, same setup as the bar | needs OK |
| 5 | Larger training, a 200-task run, or mlx-lm-lora ideas | only if stage 4 passes, new OK |

**Pass rule (stage 4):** gate at or above 17/20; fresh bank no worse than base
(base R42 scored 3/20 there); at most one regression versus base. Gate results are
tripwires, not accuracies. We never screen on validation loss. If the gate fails,
I stop and report. No tuning on the gate and no variants on my own.

## 9. Hard rules that stay on

- The 200 test tasks, the gate, the fresh bank and the miss_func / miss_param /
  long_context splits are never training data. Exclusion list: 800 task IDs, 1,153
  unique user-turn texts; checks for exact match, any shared 8-gram, and turn
  similarity >= 0.85 (my thresholds; tell me if you want others).
- Scope: Gemma-4-E2B-it only, `multi_turn_base`, one Mac.
- Every number comes from `benchmark.db` and logs. No invented numbers.
- No paid calls beyond the capped pilots, no runs, no pushes, no PRs without your
  explicit OK each time. Key only from env, never printed or committed.
- I don't edit `paper/main.tex`; I report results with their sources.

## 10. Honest expectations

- Nothing guarantees this works. A strong teacher's style may be hard for a 2B
  model to copy, and even the gentle null test slipped below the bar.
- The most likely failure is the same shape as before: the model learns to stop
  early. That's why the decision-mix report (stop share capped) comes before any
  training.
- A clean negative result ("teacher-verified, in-world data also did not beat the
  untuned model") is a real finding for Chapter 6.

## 11. Timing

Deadlines (yours, 2026-11-06): presentation Dec 1-14, preliminary review Dec 4,
complete draft Dec 11, final Dec 18. The bot suggested about **Nov 25** as the
last day a new run can finish and still be written up; that's a suggestion, not a
repo fact, and I need your answer. Measured so far: authoring ~2.5 min per 25
slots, solving ~36 s for 21 tasks at 4 workers. Training and gate times are not
measured yet; I'll give estimates before stage 3.

## 12. What I need from you

1. OK for stage 1b (grounding fix + backend quotas + second small pilot, same
   cap scale)?
2. Weight the backend mix toward the file system (where Gemma fails most)?
3. Confirm: old mixes excluded from the first run; last run day (Nov 25?).
4. Do you want a teacher-only run on the 200 tasks as a reference ceiling (never
   used for training)? Needs its own approval and estimate.
