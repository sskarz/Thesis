#!/usr/bin/env python3
"""G4 scorer pin: re-score the stored responses of Runs 82 to 85 with the BFCL checker that G4 will use, offline, read-only.

usage (cwd = a worktree root with benchmark/ importable): g4_rescore.py DB_PATH OUT.json RUN_ID
SCORE ONE RUN PER PROCESS: the BFCL checker keeps backend instance state keyed by model name and task id, so scoring several runs in one process leaks
state from the earlier runs into the later ones (verified 2026-11-11: runs 83 to 85 scored 67, 67 and 59 instead of 84, 88 and 72 when scored after run 82 in one process).
Reads `responses` (mode 'multi') and `eval_errors` from the benchmark database opened read-only; never writes to it.
A task passed in the original run iff it has no row in `eval_errors` for that run (the harness stores one row per failed task);
the stored score is read from `scores`. Output: per run the original pass count, the re-scored pass count, the number of tasks whose
verdict differs, and the SHA-256 of the checker files and the bfcl_eval package version.
"""
import hashlib, json, os, sqlite3, sys
sys.path.insert(0, os.getcwd())
import importlib.metadata as md
import bfcl_eval
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
from benchmark import config
from benchmark.evaluate_responses import load_jsonl

db_path, out = sys.argv[1], sys.argv[2]
runs = [int(x) for x in sys.argv[3:]] or [84, 85]
c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
pkg = os.path.dirname(bfcl_eval.__file__)
files = []
for root, _, fs in os.walk(os.path.join(pkg, "eval_checker", "multi_turn_eval")):
    for f in sorted(fs):
        if f.endswith(".py"):
            files.append(os.path.join(root, f))
sha = {os.path.relpath(f, pkg): hashlib.sha256(open(f, "rb").read()).hexdigest() for f in sorted(files)}
agg = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sha.items()).encode()).hexdigest()
info = config.MULTI_TURN_CATEGORIES["multi_turn_base"] if hasattr(config, "MULTI_TURN_CATEGORIES") else {"file": "BFCL_v4_multi_turn_base.json"}
tests = {e["id"]: e for e in load_jsonl(config.BFCL_DATA_DIR / info["file"])}
answers = {e["id"]: e for e in load_jsonl(config.BFCL_POSSIBLE_ANSWER_DIR / info["file"])}
rep = {"bfcl_eval_version": md.version("bfcl_eval") if "bfcl_eval" in {d.metadata["Name"].replace("-", "_") for d in md.distributions()} else getattr(bfcl_eval, "__version__", "unknown"),
       "checker_files": len(sha), "checker_sha256_of_file_hashes": agg, "runs": {}}
for rid in runs:
    rows = c.execute("select entry_id, result from responses where run_id=? and category='multi_turn_base' and mode='multi'", (rid,)).fetchall()
    failed_orig = {r[0] for r in c.execute("select entry_id from eval_errors where run_id=? and category='multi_turn_base'", (rid,))}
    stored = c.execute("select correct, total from scores where run_id=? and category='multi_turn_base'", (rid,)).fetchone()
    passed, diff, errors = set(), [], 0
    for eid, res in rows:
        try:
            res = json.loads(res) if isinstance(res, str) else res
            r = multi_turn_checker(multi_turn_model_result_list_decoded=res, multi_turn_ground_truth_list=answers[eid]["ground_truth"],
                                   test_entry=tests[eid], test_category="multi_turn_base", model_name=config.MODEL_NAME)
            ok = bool(r["valid"])
        except Exception:
            ok = False; errors += 1
        if ok: passed.add(eid)
        if ok == (eid in failed_orig): diff.append(eid)
    rep["runs"][str(rid)] = {"responses": len(rows), "stored_score": list(stored) if stored else None, "original_passes_from_eval_errors": len(rows) - len(failed_orig),
                             "rescored_passes": len(passed), "tasks_with_different_verdict": len(diff), "different_ids": diff[:20], "checker_exceptions": errors}
json.dump(rep, open(out, "w"), indent=1); print(json.dumps(rep))
