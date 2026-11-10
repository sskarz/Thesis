"""Shared helpers for the yield-check filters: execute a task's gold in the real backend and render it as text.

run_gold(task) -> [[(call, result_str), ...] per turn]. Each call runs under a fresh backend instance id, so nothing is cached between
tasks. render_task(task) is the text the accuracy reviewers (and the request-coverage judge) see: starting state (up to 6000 characters),
each user turn, each gold call and what it returned.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call  # noqa: E402

STATE_CHARS = 6000
RESULT_CHARS = 400


def run_gold(task: dict) -> list[list[tuple[str, str]]]:
    vid = f"{task['id']}_render_{time.time_ns()}"
    out = []
    for calls in task["ground_truth"]:
        res, _ = execute_multi_turn_func_call(func_call_list=calls, initial_config=task["initial_config"],
                                              involved_classes=task["involved_classes"], model_name="distill_render",
                                              test_entry_id=vid, long_context=False, is_evaL_run=False)
        out.append([(c, r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)) for c, r in zip(calls, res)])
    return out


def render_task(task: dict, gold: list | None = None) -> str:
    gold = gold if gold is not None else run_gold(task)
    ic = json.dumps(task["initial_config"], ensure_ascii=False)
    lines = [f"## {task['id']}  backends: {', '.join(task['involved_classes'])}",
             "initial_config" + ("" if len(ic) <= STATE_CHARS else f" (first {STATE_CHARS} of {len(ic)} chars)") + ": " + ic[:STATE_CHARS]]
    for i, (q, g) in enumerate(zip(task["question"], gold)):
        lines.append(f"turn {i + 1} user: {q[0]['content']}")
        for c, r in g:
            lines.append(f"    gold: {c}\n        result: {r[:RESULT_CHARS]}")
    return "\n".join(lines) + "\n"
