#!/usr/bin/env python3
"""Request-coverage judge: does each task's gold do what each user turn asks? One teacher call per task (API, capped).

usage: yield_judge.py --tasks tasks.jsonl --out DIR --max-cost 0.60 --workers 4 --deadline-minutes 30
Writes DIR/verdicts.jsonl (one line per task: id, verdict FAITHFUL|AMBIGUOUS|WRONG, reason, per-turn detail), DIR/log/calls.jsonl (every call with cost),
DIR/guard_flags.json. Uses the same guarded client as the other paid steps: the first response without a reported cost, or a budget stop,
halts all requests and makes the script exit 3. A task with no parsable verdict is recorded as UNPARSED and counts as a drop.
Temperature 0. Model/provider pins are those of benchmark/teacher_api.py (deepseek-v4.1-flash-1017 on DeepInfra).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import teacher_api  # noqa: E402
from yield_guard import GuardedClient  # noqa: E402
import yield_render as R  # noqa: E402

SYSTEM = "You are a strict reviewer of test tasks for a tool-calling assistant. You output ONLY one JSON object, no prose, no code fences."

PROMPT = """Below is one synthetic task: a starting state, then user turns, each followed by the 'gold' tool calls that are supposed to carry out that turn, and what each call returned when run in the real backend.

Decide whether the gold really does what each user turn asks. Work turn by turn:
1. List every action or piece of information the user asks for in that turn (requested_actions).
2. For each, say whether the gold carries it out (covered true/false). An action that is only looked up but never performed (for example the user says to send, post, book or cancel and the gold only fetches an id) is NOT covered.
3. Check values. A value that REFERS TO something that already exists (an id, a name, a file, a folder, a card, an order, a ticket, a quoted description such as "the card ending in NNNN") must match the starting state or an earlier result; a reference that matches nothing, or that the gold satisfies only through an unrelated token, is an issue. A value the user SUPPLIES as new content or as a parameter of the action (text to post or send, an amount to pay or deposit, a title for a new object, credentials the user states) needs no support in the state, but the gold must reproduce it exactly as the user gave it.
4. Check meaning: the function and arguments must mean what the wording says, judged from the function's name and the shown result. Look at what each call actually did (did it add, set, replace or only read; which quantity did it compute; which unit or kind of measure). An unrequested state change, an unrequested destructive action, or a missing requested one is an issue.
5. Check results: if a result shows the call did not do what the user asked (empty where the state holds data, an id that reuses an existing object so it may overwrite it, an error-like or truncated output the gold ignores), that is an issue.

Verdict for the whole task:
- FAITHFUL: every requested action is covered, every reference to an existing object is grounded in the state or an earlier result, supplied content is reproduced exactly, meanings match, results support the request.
- AMBIGUOUS: a competent assistant could reasonably do something different, or the wording can be read two ways, or an unstated assumption is needed.
- WRONG: some requested action is missing or mismatched, a reference to an existing object matches nothing in the state or earlier results, supplied content is altered, or a result shows the request was not carried out.
When unsure between FAITHFUL and AMBIGUOUS, choose AMBIGUOUS. Judge only from what is shown.

Return exactly this JSON:
{{"turns": [{{"turn": 1, "requested_actions": ["..."], "covered": [true], "issues": ["..."]}}], "verdict": "FAITHFUL|AMBIGUOUS|WRONG", "reason": "<=30 words"}}

TASK
{task}"""


def parse_verdict(text: str):
    dec = json.JSONDecoder()
    for i in [k for k, ch in enumerate(text) if ch == "{"]:
        try:
            obj, _ = dec.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("verdict") in ("FAITHFUL", "AMBIGUOUS", "WRONG"):
            return obj
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-cost", type=float, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--deadline-minutes", type=float, required=True)
    a = ap.parse_args(argv)
    if (a.out / "verdicts.jsonl").exists():
        print("refusing to overwrite existing verdicts", file=sys.stderr)
        return 2
    a.out.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(l) for l in open(a.tasks, encoding="utf-8") if l.strip()]
    cfg = teacher_api.TeacherConfig.openrouter(log_dir=a.out / "log", max_cost_usd=a.max_cost, max_concurrency=a.workers,
                                               timeout_s=60.0, max_retries=2)
    client = GuardedClient(cfg)
    deadline = time.time() + a.deadline_minutes * 60
    lock = threading.Lock()

    def one(t):
        if time.time() > deadline:
            return {"id": t["id"], "verdict": "SKIPPED", "reason": "deadline"}
        try:
            text = R.render_task(t)
            with client.episode(f"judge_{t['id']}", kind="judge"):
                data = client.post_chat({"messages": [{"role": "system", "content": SYSTEM},
                                                      {"role": "user", "content": PROMPT.format(task=text)}],
                                         "temperature": 0.0, "max_tokens": 1800})
        except teacher_api.BudgetExceeded:
            return {"id": t["id"], "verdict": "SKIPPED", "reason": "budget_or_cost_guard"}
        except Exception as e:  # noqa: BLE001
            return {"id": t["id"], "verdict": "SKIPPED", "reason": f"api:{type(e).__name__}"}
        raw = (data["choices"][0]["message"].get("content") or "")
        v = parse_verdict(raw)
        if v is None:
            return {"id": t["id"], "verdict": "UNPARSED", "reason": raw[:200]}
        return {"id": t["id"], "verdict": v["verdict"], "reason": str(v.get("reason", ""))[:300], "turns": v.get("turns")}

    ex = ThreadPoolExecutor(max_workers=a.workers)
    futs = {t["id"]: ex.submit(one, t) for t in tasks}
    wait(list(futs.values()), timeout=max(1.0, deadline - time.time()))
    res = []
    for t in tasks:
        f = futs[t["id"]]
        res.append(f.result() if f.done() else {"id": t["id"], "verdict": "SKIPPED", "reason": "deadline_watchdog"})
    ex.shutdown(wait=False, cancel_futures=True)       # lingering requests end with the process (each has a 60 s timeout and 2 retries)
    with open(a.out / "verdicts.jsonl", "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    GuardedClient.write_flags(a.out / "guard_flags.json")
    counts = {}
    for r in res:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print(json.dumps({"tasks": len(tasks), "verdicts": counts, "spent_usd": round(client.stats["cost_usd"], 4), "calls": client.stats["calls"]}))
    sys.stdout.flush()
    if GuardedClient.MISSING_COST or GuardedClient.BUDGET_HIT or counts.get("SKIPPED") or counts.get("UNPARSED"):
        print("ERROR: cost-less response, budget hit, skipped or unparsed tasks; the run is incomplete", file=sys.stderr)
        sys.stderr.flush()
        os._exit(3)
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
