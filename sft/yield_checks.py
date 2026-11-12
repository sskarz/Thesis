#!/usr/bin/env python3
"""Deterministic, free state checks for authored tasks (no model, no network). Flags are reasons to drop a task.

usage: yield_checks.py --tasks tasks.jsonl --out flags.json
Checks (each found a real defect in the 40-task accuracy review of the first yield attempt, docs/DATA_YIELD_FACTORY_FIX.md):
  id_collision              a creation call (post_/place_/create_/open_) returns an id that already exists in the starting state or was
                            created earlier in the task, i.e. it overwrote or reused an object.
  info_seeded_comments_invisible  (information only, not a drop reason) TwitterAPI initial state seeds comments or retweets; the BFCL
                            backend converts tweet keys to int but not these, so they can never be read back. Harmless unless the gold reads
                            them, which empty_read_with_state_key catches.
  empty_read_with_state_key a read-only call returns an empty list/dict although one of its numeric arguments is a key of a non-empty list in
                            the starting state (the data is there but the call did not see it).
  quoted_suffix_not_in_state a user turn says a card "ending in NNNN" but no card_number value ends with those digits.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yield_render as R  # noqa: E402

ID_KEYS = ("id", "order_id", "ticket_id")
CREATE_RE = re.compile(r"^(post_|place_|create_|open_)")
READ_RE = re.compile(r"^(get_|list|ls$|cat$|find|grep|search|show|view|display|check|read|tail|head)")
SUFFIX_RE = re.compile(r"(?:ending|ends)\s*(?:in|with|at)?\s*(?:digits\s*)?(\d{4})", re.I)


def state_ids(obj, acc):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.isdigit():
                acc.add(k)
            if k in ID_KEYS and isinstance(v, (int, str)) and not isinstance(v, bool):
                acc.add(str(v))
            state_ids(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            state_ids(v, acc)
    return acc


def nonempty_list_keys(obj, acc):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.isdigit() and isinstance(v, list) and v:
                acc.add(k)
            nonempty_list_keys(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            nonempty_list_keys(v, acc)
    return acc


def card_numbers(obj, acc):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "card_number":
                acc.add(str(v))
            card_numbers(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            card_numbers(v, acc)
    return acc


def parse(res):
    if isinstance(res, str):
        try:
            return json.loads(res)
        except (json.JSONDecodeError, TypeError):
            try:
                return ast.literal_eval(res)
            except (ValueError, SyntaxError):
                return None
    return res


def call_parts(call):
    try:
        node = ast.parse(call, mode="eval").body
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        args = {k.arg: k.value.value for k in node.keywords if isinstance(k.value, ast.Constant)}
        return name, args
    except (SyntaxError, AttributeError):
        return "", {}


def check_task(task: dict, gold=None) -> list[str]:
    flags: list[str] = []
    ic = task["initial_config"]
    tw = (ic or {}).get("TwitterAPI") or {}
    if tw.get("comments") or tw.get("retweets"):
        flags.append("info_seeded_comments_invisible")
    seen_ids = state_ids(ic, set())
    list_keys = nonempty_list_keys(ic, set())
    gold = gold if gold is not None else R.run_gold(task)
    for turn in gold:
        for call, res in turn:
            name, args = call_parts(call)
            val = parse(res)
            if CREATE_RE.match(name) and isinstance(val, dict):
                arg_vals = {str(v) for v in args.values()}
                for k in ID_KEYS:
                    if k in val and not isinstance(val[k], bool):
                        v = str(val[k])
                        if v in seen_ids and v not in arg_vals:
                            flags.append(f"id_collision:{name}:{v}")
                        seen_ids.add(v)
            elif READ_RE.match(name) and (val == [] or val == {} or (isinstance(val, dict) and val.get("results") == [])):
                if any(str(v) in list_keys for v in args.values() if isinstance(v, (int, str)) and str(v).isdigit()):
                    flags.append(f"empty_read_with_state_key:{name}")
    cards = card_numbers(ic, set())
    if cards:
        for q in task["question"]:
            for m in SUFFIX_RE.finditer(" ".join(x["content"] for x in q)):
                if not any(c.endswith(m.group(1)) for c in cards):
                    flags.append(f"quoted_suffix_not_in_state:{m.group(1)}")
    return sorted(set(flags))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = [json.loads(l) for l in open(a.tasks, encoding="utf-8") if l.strip()]
    res = {t["id"]: check_task(t) for t in tasks}
    json.dump(res, open(a.out, "w"), indent=1)
    bad = sum(1 for v in res.values() if v)
    kinds = {}
    for v in res.values():
        for f in v:
            kinds[f.split(":")[0]] = kinds.get(f.split(":")[0], 0) + 1
    print(json.dumps({"tasks": len(tasks), "flagged": bad, "by_kind": kinds}))


if __name__ == "__main__":
    main()
