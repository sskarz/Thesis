"""Build the fresh bank F from authored tasks (Round 4, Stage A). No model, no network.

usage: build_bank_f.py audit  <tasks_raw.jsonl> <out_audited.jsonl> <out_audit.json>
       build_bank_f.py filter <tasks_audited.jsonl> <teacher_solve_dir> <out_dir> <other_bank.jsonl> [more ...]
audit : re-run gold under a fresh id (distill_tasks.audit_task); keep tasks with no semantic issues.
filter: keep teacher-verified tasks (>=1 of the k teacher episodes valid, same rule as round 3), then drop any task whose
        gold-function-sequence signature equals, or whose user-turn similarity is >= 0.85 to, any task in the other banks
        (dev, confirm, training pools). Writes tasks_F.jsonl and a report; F text is never shown to an authoring prompt."""
import collections, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from benchmark import distill_tasks as dt, distill_split as ds


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def audit(raw, out, rep):
    ok, bad = [], {}
    for t in load(raw):
        iss = dt.audit_task(t)
        (bad.__setitem__(t["id"], iss) if iss else ok.append(t))
    with open(out, "w") as f:
        for t in ok:
            f.write(json.dumps(t) + "\n")
    json.dump(bad, open(rep, "w"), indent=1)
    print(json.dumps({"in": len(ok) + len(bad), "kept": len(ok), "issues": len(bad)}))


def filt(audited, solve_dir, out_dir, others):
    V = collections.defaultdict(list)
    skipped = 0
    for x in map(json.loads, open(pathlib.Path(solve_dir) / "verdicts.jsonl")):
        if "task_id" not in x:      # distill_solve writes budget/deadline-skipped verdicts without a task id
            skipped += 1
            continue
        V[x["task_id"]].append(bool(x["valid"]) and not x.get("api_error"))
    tasks = load(audited)
    verified = [t for t in tasks if any(V.get(t["id"], []))]
    other = [t for p in others for t in load(p)]
    osig = {}
    for t in other:
        try:
            osig.setdefault(ds.signature(t), t["id"])
        except Exception:
            pass
    keep, dropped = [], []
    for t in verified:
        why = None
        sg = ds.signature(t)
        if sg in osig:
            why = ("signature", osig[sg])
        else:
            for o in other:
                if o["id"] == t["id"]:
                    why = ("same_id", o["id"]); break
                r = ds.max_ratio(t, o)
                if r >= 0.85:
                    why = ("similarity", o["id"], round(r, 3)); break
        (dropped.append({"id": t["id"], "why": why}) if why else keep.append(t))
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "tasks_F.jsonl", "w") as f:
        for t in keep:
            f.write(json.dumps(t) + "\n")
    json.dump(sorted(t["id"] for t in verified), open(out / "verified_ids.json", "w"))
    rep = {"audited": len(tasks), "skipped_verdicts_without_task_id": skipped, "verified": len(verified), "dropped_vs_other_banks": dropped, "F": len(keep),
           "per_class": dict(collections.Counter(c for t in keep for c in t["involved_classes"])),
           "n_turns": dict(collections.Counter(len(t["question"]) for t in keep))}
    json.dump(rep, open(out / "F_report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "audit":
        audit(*sys.argv[2:5])
    else:
        filt(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
