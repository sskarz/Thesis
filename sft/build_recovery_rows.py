"""Round 4 Stage B: candidate rows at teacher-labelled student states, split into

  R  = recovery candidates: the student's request directly follows a tool error and the labeller KEPT the teacher action
       (act_ok = teacher action replays cleanly, or stop_correct = turn complete),
  C  = control candidates: kept states that do NOT follow an error, drawn with the same action mix (execute / stop)
       and the same count as R (seeded), for the size-matched control arm C_match.

Both go through the same hint-free native sampling (benchmark.distill_rationalize --hinted 0) afterwards, so a row
survives only if the base student itself reproduces the teacher's action. Nothing here looks at any evaluation bank.

usage:
  python sft/build_recovery_rows.py --labels L1.jsonl L2.jsonl ... --drop-tasks drop.json --out-dir DIR [--seed 61]
writes DIR/R_json.jsonl, DIR/C_json.jsonl, DIR/candidates_report.json
"""
import argparse, collections, json, pathlib, random, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def after_error(messages):
    last = [m for m in messages if m["role"] == "user"][-1]["content"]
    return '-> {"error"' in str(last) or "-> Error during" in str(last)


def row_of(x, arm):
    tgt = "[]" if not x["label"] else json.dumps(x["label"], indent=2, ensure_ascii=False)
    return {"messages": [*x["messages"], {"role": "assistant", "content": tgt}],
            "meta": {"task_id": x["task_id"], "episode_id": x["episode_id"], "step": x["step"], "turn": x["turn"],
                     "action": "end_turn" if not x["label"] else "execute", "kind": x["kind"],
                     "same_as_student": bool(x.get("same_effect", x["same_as_student"])), "classes": x["classes"],
                     "exercised": [], "n_calls": len(x["label"] or []), "lookup_calls": 0, "arm": arm,
                     "after_error": after_error(x["messages"]), "student_stopped": x["student_calls"] == []}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--drop-tasks", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=61)
    a = ap.parse_args()
    drop = set(json.load(open(a.drop_tasks))) if a.drop_tasks else set()
    L = [json.loads(l) for p in a.labels for l in open(p) if l.strip()]
    seen, R, C = set(), [], []
    info = collections.Counter()
    for x in L:
        if x["decision"] != "keep":
            info["not_kept"] += 1
            continue
        if x["task_id"] in drop:
            info["dropped_task"] += 1
            continue
        ae = after_error(x["messages"])
        r = row_of(x, "R" if ae else "C")
        key = json.dumps(r["messages"], sort_keys=True)
        if key in seen:
            info["duplicate_prompt_target"] += 1
            continue
        seen.add(key)
        (R if ae else C).append(r)
    rng = random.Random(a.seed)
    # control: same action mix and same count as R (or all available if fewer)
    want = collections.Counter(r["meta"]["action"] for r in R)
    ctrl = []
    for act, n in want.items():
        pool = [r for r in C if r["meta"]["action"] == act]
        rng.shuffle(pool)
        ctrl += pool[:n]
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("R_json.jsonl", R), ("C_json.jsonl", ctrl)):
        with open(out / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = {"labels": a.labels, "kept_labels": sum(x["decision"] == "keep" for x in L), "info": dict(info),
           "R_candidates": len(R), "R_by_action": dict(want), "R_tasks": len({r["meta"]["task_id"] for r in R}),
           "R_same_as_student": sum(r["meta"]["same_as_student"] for r in R),
           "control_candidates_available": len(C), "control_selected": len(ctrl),
           "control_by_action": dict(collections.Counter(r["meta"]["action"] for r in ctrl)), "seed": a.seed}
    json.dump(rep, open(out / "candidates_report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
