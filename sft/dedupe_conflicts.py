#!/usr/bin/env python3
"""Drop every row whose prompt (messages[:-1]) also appears with a different target (messages[-1]), across train and valid.

The row builder only removes rows that are identical in prompt AND target. When the teacher is sampled (k > 1) it can answer the same
state differently; both answers would then teach opposite things for one prompt (handoff pitfall 11: Round 5 removed such prompts).
Policy: drop ALL rows of a conflicting prompt (no choosing a winner), count them, write the rest unchanged.

usage: dedupe_conflicts.py --rows-dir rows --out-dir rows_clean      (reads train.jsonl, valid.jsonl; writes both + conflicts.json)
"""
import argparse
import collections
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    src, out = pathlib.Path(a.rows_dir), pathlib.Path(a.out_dir)
    if src.resolve() == out.resolve():
        raise SystemExit("refusing to write into the source dir")
    out.mkdir(parents=True, exist_ok=True)
    data = {s: [json.loads(l) for l in open(src / f"{s}.jsonl", encoding="utf-8") if l.strip()] for s in ("train", "valid")}
    targets = collections.defaultdict(set)
    key = lambda r: json.dumps(r["messages"][:-1], sort_keys=True, ensure_ascii=False)
    for rows in data.values():
        for r in rows:
            targets[key(r)].add(json.dumps(r["messages"][-1], sort_keys=True, ensure_ascii=False))
    bad = {k for k, v in targets.items() if len(v) > 1}
    rep = {"prompts_with_conflicting_targets": len(bad), "rows_in": {s: len(r) for s, r in data.items()}, "rows_dropped": {}, "rows_out": {}}
    for s, rows in data.items():
        keep = [r for r in rows if key(r) not in bad]
        rep["rows_dropped"][s] = len(rows) - len(keep)
        rep["rows_out"][s] = len(keep)
        with open(out / f"{s}.jsonl", "w", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep["tasks_with_dropped_rows"] = len({r["meta"]["task_id"] for rows in data.values() for r in rows if key(r) in bad})
    json.dump(rep, open(out / "conflicts.json", "w"), indent=1)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
