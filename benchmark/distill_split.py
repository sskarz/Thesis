"""Family-level dev2 / pool2 split with a cross-split similarity audit (round 3).

Input: authored tasks (``tasks_audited``-style jsonl) and the ids that the teacher verified.
Families = identical gold function-name sequence per turn (the workflow skeleton); a whole family goes
to one side. dev2 is filled round-robin over backends until ``--dev-n``. Pool tasks whose user turns
resemble ANY dev2 / old dev / confirm task (difflib ratio >= 0.85 on a normalised turn) are dropped.

    python -m benchmark.distill_split --tasks T.jsonl --verified IDS.json --dev-n 80 --out DIR
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import random
import re
import sys
from pathlib import Path

from .distill_exclusions import norm


def signature(task: dict) -> tuple:
    return tuple(tuple(re.match(r"\s*([A-Za-z_][\w]*)", c).group(1) for c in turn)
                 for turn in task["ground_truth"])


def turns(task: dict) -> list[str]:
    out = []
    for t in task["question"]:
        out.append(norm(" ".join(m["content"] for m in t if m.get("role") == "user")))
    return out


def max_ratio(a: dict, b: dict) -> float:
    sm = difflib.SequenceMatcher(autojunk=False)
    best = 0.0
    tb = turns(b)
    for x in turns(a):
        sm.set_seq2(x)
        for y in tb:
            sm.set_seq1(y)
            if sm.real_quick_ratio() >= best and sm.quick_ratio() >= best:
                best = max(best, sm.ratio())
    return best


def split(tasks: list[dict], dev_n: int, seed: int, held: list[dict]) -> dict:
    fam: dict[tuple, list[dict]] = collections.defaultdict(list)
    for t in tasks:
        fam[signature(t)].append(t)
    # held-out families (old dev/confirm) also keep their skeleton out of the training side
    held_sigs = {signature(t) for t in held}
    rng = random.Random(seed)
    groups = list(fam.items())
    rng.shuffle(groups)
    by_backend: dict[str, list[tuple]] = collections.defaultdict(list)
    for sig, ts in groups:
        by_backend[min(ts[0]["involved_classes"])].append((sig, ts))
    dev: list[dict] = []
    dev_sigs: set[tuple] = set()
    order = sorted(by_backend)
    while len(dev) < dev_n and any(by_backend.values()):
        for b in order:
            if by_backend[b] and len(dev) < dev_n:
                sig, ts = by_backend[b].pop()
                dev.extend(ts)
                dev_sigs.add(sig)
    pool_cand = [t for sig, ts in groups if sig not in dev_sigs for t in ts]
    anchors = dev + held
    pool, dropped_family, dropped_sim = [], [], []
    for t in pool_cand:
        if signature(t) in held_sigs:
            dropped_family.append(t["id"]); continue
        worst = max((max_ratio(t, d) for d in anchors), default=0.0)
        if worst >= 0.85:
            dropped_sim.append((t["id"], round(worst, 3))); continue
        pool.append(t)
    return {"dev2": dev, "pool2": pool, "dropped_family_with_old_heldout": dropped_family,
            "dropped_similarity": dropped_sim, "families": len(fam),
            "dev2_backends": dict(collections.Counter(c for t in dev for c in t["involved_classes"]))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--verified", type=Path, required=True, help="json list of teacher-verified task ids")
    ap.add_argument("--held", type=Path, nargs="*", default=[], help="old dev/confirm banks")
    ap.add_argument("--dev-n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    ok = set(json.loads(a.verified.read_text()))
    tasks = [json.loads(l) for l in a.tasks.read_text().splitlines() if l.strip()]
    tasks = [t for t in tasks if t["id"] in ok]
    held = [json.loads(l) for f in a.held for l in f.read_text().splitlines() if l.strip()]
    res = split(tasks, a.dev_n, a.seed, held)
    a.out.mkdir(parents=True, exist_ok=True)
    for name in ("dev2", "pool2"):
        with open(a.out / f"{name}.jsonl", "w") as f:
            for t in res[name]:
                f.write(json.dumps(t) + "\n")
    rep = {k: v for k, v in res.items() if k not in ("dev2", "pool2")}
    rep.update({"verified_in": len(tasks), "dev2": len(res["dev2"]), "pool2": len(res["pool2"]), "seed": a.seed})
    (a.out / "split_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
