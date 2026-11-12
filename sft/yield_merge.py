#!/usr/bin/env python3
"""Merge the tasks_raw.jsonl files of parallel authoring processes and apply, across processes, the same duplicate rule author() applies inside one process:
two tasks are duplicates if their user turns share any 8-word n-gram (benchmark.distill_exclusions.ngrams). Order is fixed (files in the order given, then file order),
so the first process wins; every drop is listed. Also refuses duplicate task ids.
usage: yield_merge.py --inputs A.jsonl B.jsonl C.jsonl --existing P1.jsonl ... --out merged.jsonl --report merge_report.json
Task ids must carry distinct prefixes per process (e.g. distill_g2a_0001). No API, no GPU."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import distill_exclusions as X  # noqa: E402


def users(t):
    return [m["content"] for q in t["question"] for m in q]


def grams(t):
    u = users(t)
    return set().union(*(X.ngrams(x) for x in u)) if u else set()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, nargs="+", required=True); ap.add_argument("--existing", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        print("refusing to overwrite", a.out, file=sys.stderr); return 2
    seen = set(); keep, drops = [], []
    per_file = {}
    for p in a.existing:
        for l in open(p, encoding="utf-8"):
            if l.strip():
                seen |= grams(json.loads(l))
    ids = set()
    for p in a.inputs:
        n_in = n_keep = 0
        for l in open(p, encoding="utf-8"):
            if not l.strip():
                continue
            t = json.loads(l); n_in += 1
            if t["id"] in ids:
                raise SystemExit(f"duplicate task id across inputs: {t['id']} (use distinct prefixes per process)")
            ids.add(t["id"])
            g = grams(t)
            if g & seen:
                drops.append({"id": t["id"], "file": str(p), "reason": "dup_cross_process_or_existing"}); continue
            seen |= g; keep.append(t); n_keep += 1
        per_file[str(p)] = {"in": n_in, "kept": n_keep}
    with open(a.out, "w", encoding="utf-8") as f:
        for t in keep:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    rep = {"inputs": per_file, "existing_files": [str(p) for p in a.existing], "kept": len(keep), "dropped": len(drops), "drops": drops}
    json.dump(rep, open(a.report, "w"), indent=1); print(json.dumps({k: v for k, v in rep.items() if k != "drops"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
