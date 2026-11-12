#!/usr/bin/env python3
"""G4 documentation helper (CPU only, read-only). Four jobs:
 A. The paper's failure labeler (primary label per failed task from the checker message in eval_errors: "list is empty" -> empty turn,
    "does not contain all" -> under-call, "does not match the state" -> state mismatch; PAPER_WRITING_PROCESS.md, paper/main.tex "Error labels"),
    validated on the stored runs behind Table tab:failfull before it is used on G4.
 B. The labels per G4 run.
 C. Pair 1 (U1 vs T1): trained-only and untuned-only tasks split by backend (BFCL tool class).
 D. Mean tool calls, API calls and generated tokens per task for each G4 run (responses and multi_turn_timings tables).
usage: python3 g4_labels_calls.py ATTEMPT_DIR MAIN_BENCHMARK_DB   (both opened read-only)
"""
import collections, json, os, re, sqlite3, sys

att, maindb = sys.argv[1], sys.argv[2]
ro = lambda p: sqlite3.connect(f"file:{p}?mode=ro", uri=True)

def label(msg):
    if "list is empty" in msg: return "empty turn"
    if "does not contain all" in msg: return "under-call"
    if "does not match the state" in msg: return "state mismatch"
    return "other"

def labels(db, run_id=None):
    """one primary label per failed task: the first eval_errors row of that task (by row id); duplicated rows (Run 4) count once"""
    q = "select entry_id, error from eval_errors where category='multi_turn_base'" + (" and run_id=?" if run_id is not None else "") + " order by id"
    seen = {}
    for e, m in db.execute(q, (run_id,) if run_id is not None else ()):
        seen.setdefault(e, label(m))
    return seen

# ---- A. validation against the paper's Table tab:failfull (columns: run id in the stored database)
paper = {  # column: (run id, empty, under, state, failed)
    "NFC": (4, 59, 19, 68, 146), "IFFC": (5, 41, 38, 60, 139), "Stop": (30, 38, 37, 60, 135), "Ungated": (27, 40, 31, 75, 146),
    "Summary": (10, 31, 37, 64, 132), "Minimal": (42, 17, 41, 71, 129), "Full": (78, 20, 35, 68, 123), "Vote": (11, 36, 38, 53, 127),
    "CD": (8, 14, 70, 95, 179), "Autom.": (46, 17, 54, 102, 173), "SFT": (15, 2, 59, 109, 170)}
mdb = ro(maindb)
print("## A. Labeler check against the paper's failure-label table (stored runs, multi_turn_base)\n")
print("| column | run | empty (paper / mine) | under-call (paper / mine) | state mismatch (paper / mine) | failed (paper / mine) | equal |\n|---|---|---|---|---|---|---|")
allok = True
for col, (rid, pe, pu, ps, pf) in paper.items():
    lab = labels(mdb, rid); c = collections.Counter(lab.values())
    mine = (c["empty turn"], c["under-call"], c["state mismatch"], len(lab)); ok = mine == (pe, pu, ps, pf) and c["other"] == 0; allok &= ok
    print(f"| {col} | {rid} | {pe} / {mine[0]} | {pu} / {mine[1]} | {ps} / {mine[2]} | {pf} / {mine[3]} | {'yes' if ok else 'NO'} |")
print(f"\nLabeler reproduces every column of the paper's table: {allok}.\n")

# ---- B. G4 runs
runs = [r for r in ("U1", "T1", "T2", "U2") if os.path.exists(f"{att}/results_{r}/benchmark.db")]
db = {r: ro(f"{att}/results_{r}/benchmark.db") for r in runs}
lab = {r: labels(db[r]) for r in runs}
print("## B. Primary failure label per failed task, G4 runs (200 tasks each)\n")
print("| label | " + " | ".join(runs) + " |\n|---|" + "---|" * len(runs))
for k in ("empty turn", "under-call", "state mismatch", "other"):
    print(f"| {k} | " + " | ".join(str(sum(1 for v in lab[r].values() if v == k)) for r in runs) + " |")
print("| failed | " + " | ".join(str(len(lab[r])) for r in runs) + " |")
print("| passed | " + " | ".join(str(200 - len(lab[r])) for r in runs) + " |")

# ---- C. pair 1 by backend
import bfcl_eval
data = os.path.join(os.path.dirname(bfcl_eval.__file__), "data", "BFCL_v4_multi_turn_base.json")
cls = {}
for l in open(data):
    if l.strip():
        e = json.loads(l); cls[e["id"]] = sorted(e.get("involved_classes", []))
P = {r: set(json.load(open(f"{att}/RUN_{r}.json"))["passed_ids"]) for r in runs}
for pair, (u, t) in (("pair 1", ("U1", "T1")), ("pair 2", ("U2", "T2"))):
    if u not in P or t not in P: continue
    t_only, u_only = sorted(P[t] - P[u]), sorted(P[u] - P[t])
    print(f"\n## C. {pair} ({u} vs {t}): trained-only {len(t_only)}, untuned-only {len(u_only)}, by backend\n")
    allc = sorted({c for v in cls.values() for c in v})
    print("| backend (task counts under every class it involves) | tasks | trained only | untuned only | net |\n|---|---|---|---|---|")
    for c in allc:
        a = sum(1 for x in t_only if c in cls[x]); b = sum(1 for x in u_only if c in cls[x])
        print(f"| {c} | {sum(1 for v in cls.values() if c in v)} | {a} | {b} | {a - b:+d} |")
    print("\n| exact class set | tasks | trained only | untuned only |\n|---|---|---|---|")
    sets = collections.defaultdict(lambda: [0, 0, 0])
    for t_id, v in cls.items(): sets[" + ".join(v)][0] += 1
    for x in t_only: sets[" + ".join(cls[x])][1] += 1
    for x in u_only: sets[" + ".join(cls[x])][2] += 1
    for k, (n, a, b) in sorted(sets.items(), key=lambda kv: -(kv[1][1] + kv[1][2])):
        if a or b: print(f"| {k} | {n} | {a} | {b} |")

# ---- D. calls and tokens per task
print("\n## D. Per task means over the 200 tasks (tool calls = call strings in the stored result; API calls and generated tokens from multi_turn_timings)\n")
print("| run | mean tool calls | mean API calls | mean generated tokens | mean prompt tokens (summed over requests) | mean generated tokens, passed tasks | mean generated tokens, failed tasks |\n|---|---|---|---|---|---|---|")
for r in runs:
    calls = {}
    for e, res in db[r].execute("select entry_id, result from responses where category='multi_turn_base'"):
        calls[e] = sum(len(step) for turn in json.loads(res) for step in turn)
    tm = {e: (a, g, p) for e, a, g, p in db[r].execute("select entry_id, api_calls, total_predicted_n, total_prompt_n from multi_turn_timings where category='multi_turn_base'")}
    assert len(calls) == 200 and len(tm) == 200, (r, len(calls), len(tm))
    mean = lambda xs: sum(xs) / len(xs)
    ps = [tm[e][1] for e in tm if e in P[r]]; fs = [tm[e][1] for e in tm if e not in P[r]]
    print(f"| {r} | {mean(list(calls.values())):.2f} | {mean([v[0] for v in tm.values()]):.2f} | {mean([v[1] for v in tm.values()]):.0f} | {mean([v[2] for v in tm.values()]):.0f} | {mean(ps):.0f} ({len(ps)} tasks) | {mean(fs):.0f} ({len(fs)} tasks) |")
