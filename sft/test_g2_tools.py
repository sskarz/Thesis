#!/usr/bin/env python3
"""Offline tests for g2_author.py + g2_tools.py with a fake author command (no API, no Mac-only imports)."""
import json, pathlib, subprocess, sys, tempfile, time
HERE = pathlib.Path(__file__).resolve().parent
FAKE = '''
import argparse, json, pathlib, sys
ap = argparse.ArgumentParser()
for k in ("--n","--out","--log-dir","--seed","--prefix","--max-cost","--workers","--deadline-minutes","--report-out","--variant","--weights-file"): ap.add_argument(k)
ap.add_argument("--existing", nargs="*", default=[])
a = ap.parse_args()
n = int(a.n); valid = n // 5
pathlib.Path(a.log_dir).mkdir(parents=True, exist_ok=True)
tasks = [{"id": f"{a.prefix}_{i:04d}", "question": [[{"role": "user", "content": f"task {a.prefix} {i}"}]]} for i in range(valid)]
pathlib.Path(a.out).write_text("".join(json.dumps(t) + "\\n" for t in tasks))
cost = 0.10 * n / 5
calls = [{"cost_usd": cost / 2, "ts": "x"}, {"cost_usd": cost / 2, "ts": "x"}]
pathlib.Path(a.log_dir, "calls.jsonl").write_text("".join(json.dumps(c) + "\\n" for c in calls))
pathlib.Path(a.log_dir, "guard_flags.json").write_text(json.dumps({"missing_cost_responses": 0, "budget_hit": False}))
rj = {"dup_in_batch": n // 5, "gold_error_t1": n // 5, "unparseable": n - valid - 2 * (n // 5)}
json.dump({"slots": n, "valid": valid, "reject_reasons": rj, "spent_usd": cost, "calls": 2}, open(a.report_out, "w"))
print(json.dumps({"existing_seen": len(a.existing)}))
'''
def run(*args, **kw):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, **kw)
with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d); (d / "fake.py").write_text(FAKE)
    # 1. normal: 2 processes, 3 chunks of 10 slots each (30 slots), cap generous
    for proc in "ab":
        r = run(HERE / "g2_author.py", "--proc", proc, "--seed-base", "100", "--slots", "30", "--chunk", "10", "--deadline-epoch", time.time() + 600, "--cap", "50",
                "--workers", "1", "--out-dir", d / "att/author" / proc, "--author-cmd", f"{sys.executable} {d / 'fake.py'}")
        assert r.returncode == 0, r.stderr[-400:]
        pr = json.load(open(d / "att/author" / proc / "proc_report.json"))
        assert pr["chunks_done"] == 3 and pr["stop_reason"] == "all_chunks_done" and abs(pr["spent_usd"] - 0.6) < 1e-9, pr
    prog = [json.loads(l) for l in open(d / "att/author/a/progress.jsonl")]
    assert [p["chunk"] for p in prog] == [0, 1, 2] and all(p["valid"] == 2 for p in prog)
    # later chunks must see earlier chunks as --existing (chunk 2 sees 2 earlier files)
    assert '"existing_seen":2' in open(d / "att/author/a/chunk_2/author.out").read().replace(" ", "")
    r = run(HERE / "g2_tools.py", "progress", "--author-dir", d / "att/author", "--slots-total", 60, "--target", 10, "--ratio", 0.392, "--floor", 5, "--write-halfway")
    assert r.returncode == 0 and "HALFWAY PROJECTION WRITTEN" in r.stdout, r.stdout + r.stderr
    hp = json.load(open(d / "att/author/halfway_projection.json"))
    assert hp["valid_tasks"] == 12 and abs(hp["survivors_estimate_so_far"] - 12 * 0.392) < 0.1 and abs(hp["projected_survivors_at_all_slots"] - 12 * 0.392) < 0.1, hp  # all 60 slots worked
    # 2. aggregate and spend: no double counting of the aggregated calls.jsonl
    r = run(HERE / "g2_tools.py", "aggregate", "--author-dir", d / "att/author", "--out-dir", d / "att/author")
    assert r.returncode == 0, r.stderr[-300:]
    rep = json.load(open(d / "att/author/report.json")); assert rep["slots"] == 60 and rep["valid"] == 12 and abs(rep["spent_usd"] - 1.2) < 1e-9, rep
    assert rep["proc_stop_reasons"] == {"a": "all_chunks_done", "b": "all_chunks_done"} and rep["slots_lost_to_deadline_or_stop"] == 0, rep
    # (aggregate wrote author/log/calls.jsonl INTO the author dir tree used above: move check to the attempt-level layout)
    r = run(HERE / "g2_tools.py", "spend", "--attempt", d / "att", "--root", d / "root_missing_ok" if False else d)  # root contains att; earlier=0
    out = json.loads(r.stdout.strip().splitlines()[-1]); assert abs(out["attempt_usd"] - 1.2) < 1e-6, out
    # max-attempt exceeded -> exit 5
    r = run(HERE / "g2_tools.py", "spend", "--attempt", d / "att", "--root", d, "--max-attempt", 1.0); assert r.returncode == 5
    # missing cost -> exit 6
    (d / "att/author/a/chunk_0/log/calls.jsonl").write_text(json.dumps({"cost_usd": None}) + "\n")
    r = run(HERE / "g2_tools.py", "spend", "--attempt", d / "att", "--root", d); assert r.returncode == 6
    # 3. process cap: cap 0.7 with 0.2/chunk? chunk cost = 0.1*n/5 = 0.2 for n=10: cap 0.35 -> chunk0 0.2 (rem .35), chunk1 0.2 -> spent .4 >= cap -> stop
    r = run(HERE / "g2_author.py", "--proc", "c", "--seed-base", "100", "--slots", "30", "--chunk", "10", "--deadline-epoch", time.time() + 600, "--cap", "0.35",
            "--workers", "1", "--out-dir", d / "capcase", "--author-cmd", f"{sys.executable} {d / 'fake.py'}")
    assert r.returncode == 5, r.returncode
    pr = json.load(open(d / "capcase/proc_report.json")); assert pr["stop_reason"] == "process_cap_reached" and pr["chunks_done"] == 2 and pr["slots_not_started"] == 10, pr
    # 4. deadline already passed
    r = run(HERE / "g2_author.py", "--proc", "e", "--seed-base", "100", "--slots", "30", "--chunk", "10", "--deadline-epoch", time.time() - 5, "--cap", "50",
            "--workers", "1", "--out-dir", d / "dl", "--author-cmd", f"{sys.executable} {d / 'fake.py'}")
    pr = json.load(open(d / "dl/proc_report.json")); assert pr["stop_reason"] == "deadline" and pr["chunks_done"] == 0 and pr["slots_not_started"] == 30, pr
    # 5. cost guard rc 3 stops the process
    (d / "fake3.py").write_text(FAKE.replace("print(json.dumps({\"existing_seen\"", "sys.exit(3)\nprint(json.dumps({\"existing_seen\""))
    r = run(HERE / "g2_author.py", "--proc", "g", "--seed-base", "100", "--slots", "30", "--chunk", "10", "--deadline-epoch", time.time() + 600, "--cap", "50",
            "--workers", "1", "--out-dir", d / "guard", "--author-cmd", f"{sys.executable} {d / 'fake3.py'}")
    assert r.returncode == 3 and json.load(open(d / "guard/proc_report.json"))["stop_reason"] == "cost_guard"
# 6. a Part 1 style attempt (author/log/calls.jsonl is the REAL log) must be counted in full
    p1 = d / "root2" / "p1"; (p1 / "author" / "log").mkdir(parents=True); (p1 / "solve").mkdir(); p1 = d / "root2" / "20261109_111111_part1.ZZZ"; (p1 / "author" / "log").mkdir(parents=True); (p1 / "solve").mkdir()
    (p1 / "author/log/calls.jsonl").write_text(json.dumps({"cost_usd": 1.5}) + "\n"); (p1 / "solve/calls.jsonl").write_text(json.dumps({"cost_usd": 0.25}) + "\n")
    (d / "root2" / "new").mkdir()
    (d / "root2" / "replay_r3" / "author" / "log").mkdir(parents=True); (d / "root2" / "replay_r3/author/log/calls.jsonl").write_text(json.dumps({"cost_usd": 9.0}) + "\n")
    (d / "root2" / "20261109_000000_x.AAA").mkdir(); (d / "root2" / "20261109_000000_x.AAA/calls.jsonl").write_text(json.dumps({"cost_usd": 0.5}) + "\n")
    r = run(HERE / "g2_tools.py", "spend", "--attempt", d / "root2" / "new", "--root", d / "root2")
    assert abs(json.loads(r.stdout.strip().splitlines()[-1])["earlier_usd"] - 2.25) < 1e-9, r.stdout
print("test_g2_tools: OK")
