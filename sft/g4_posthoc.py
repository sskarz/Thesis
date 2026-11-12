#!/usr/bin/env python3
"""G4 post-hoc reading (EXPLORATORY, written after the registered analysis; never feeds a model, retrain or data choice).
Reads a finished or partly finished G4 attempt folder READ-ONLY and prints:
  1. passes per run by task-id block (0-49, 50-99, 100-149, 150-199) and by BFCL tool class (a task counts under every class it involves);
  2. failure messages per run, normalised into: under-call (checker: execution results do not contain all ground truth), state mismatch (per class), other;
  3. the training rows' tool-domain coverage from the frozen rows' metadata only (no task text is read): rows and tasks per exercised class.
usage: python3 g4_posthoc.py ATTEMPT_DIR TRAIN_JSONL [TRAIN_JSONL ...] > out.md
"""
import collections, json, os, re, sqlite3, sys

att = sys.argv[1]; train_paths = sys.argv[2:]
import bfcl_eval
data = os.path.join(os.path.dirname(bfcl_eval.__file__), "data", "BFCL_v4_multi_turn_base.json")
tasks = {}
for l in open(data):
    if l.strip():
        e = json.loads(l); tasks[e["id"]] = e.get("involved_classes", [])
assert len(tasks) == 200, len(tasks)
idx = lambda t: int(t.rsplit("_", 1)[1])
runs = [r for r in ("U1", "T1", "T2", "U2") if os.path.exists(f"{att}/RUN_{r}.json")]
passed = {r: set(json.load(open(f"{att}/RUN_{r}.json"))["passed_ids"]) for r in runs}
cls_all = sorted({c for v in tasks.values() for c in v})

def table(title, groups):
    print(f"\n### {title}\n"); print("| group | tasks | " + " | ".join(runs) + " |"); print("|---|---|" + "---|" * len(runs))
    for name, ids in groups:
        print(f"| {name} | {len(ids)} | " + " | ".join(str(len(passed[r] & set(ids))) for r in runs) + " |")

print("# G4 post-hoc reading (exploratory; written after the registered analysis; decides nothing)")
print(f"\nRuns read: {', '.join(runs)} from `{os.path.basename(att)}`. Passes are the harness's own pass list (errored tasks already counted as fails).")
blocks = [(f"ids {a}-{a+49}", [t for t in tasks if a <= idx(t) < a + 50]) for a in (0, 50, 100, 150)]
table("Passes by task-id block", blocks)
table("Passes by BFCL tool class (a task counts under every class it involves)", [(c, [t for t, v in tasks.items() if c in v]) for c in cls_all])
combo = collections.defaultdict(list)
for t, v in tasks.items(): combo[" + ".join(sorted(v))].append(t)
table("Passes by exact class set", sorted(combo.items(), key=lambda kv: -len(kv[1])))

def norm(msg):
    if "does not contain all the ground truth" in msg: return "under-call (required call missing from the execution results)"
    m = re.match(r"Model instance for (\w+) does not match the state", msg)
    if m: return f"state mismatch: {m.group(1)}"
    if re.match(r"Model response list is empty", msg): return "empty turn (model response list is empty)"
    return "other: " + re.sub(r"\d+", "N", msg)[:70]
print("\n### Failure messages per run (checker message, normalised; one primary message per failed task)\n")
fails = {}
for r in runs:
    c = sqlite3.connect(f"file:{att}/results_{r}/benchmark.db?mode=ro", uri=True)
    cnt = collections.Counter(norm(m) for (m,) in c.execute("select error from eval_errors"))
    fails[r] = cnt
kinds = sorted({k for c in fails.values() for k in c}, key=lambda k: -sum(c[k] for c in fails.values()))
print("| failure message | " + " | ".join(runs) + " |"); print("|---|" + "---|" * len(runs))
for k in kinds: print(f"| {k} | " + " | ".join(str(fails[r][k]) for r in runs) + " |")
print("\nNote: the table counts checker messages, and a task can carry more than one; wrong-call, extra-call and format failures are not separable from these messages alone (the checker reports state or missing calls, not why). A finer breakdown would need the stored replies (task text), which this script does not read.")

print("\n### Training coverage (frozen rows' metadata only: `exercised` classes per row)\n")
rows = collections.Counter(); tks = collections.defaultdict(set)
for p in train_paths:
    for l in open(p):
        if not l.strip(): continue
        m = json.loads(l)["meta"]
        for c in m.get("exercised", []): rows[c] += 1; tks[c].add(m["task_id"])
print("| class | training rows | training tasks | BFCL 200 tasks involving it |"); print("|---|---|---|---|")
for c in sorted(set(rows) | set(cls_all)):
    print(f"| {c} | {rows[c]} | {len(tks[c])} | {sum(1 for v in tasks.values() if c in v)} |")
