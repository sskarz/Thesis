import json, pathlib, subprocess, sys, tempfile
HERE = pathlib.Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as t:
    t = pathlib.Path(t); att = t / "20261110_000000_x.a"; (att / "author" / "f" / "chunk_6" / "log").mkdir(parents=True)
    (att / "author" / "f" / "chunk_6" / "log" / "calls.jsonl").write_text(json.dumps({"cost_usd": 0.01}) + "\n")
    g = att / "author" / "f" / "chunk_6" / "log" / "guard_flags.json"
    def run(max_attempt=None):
        cmd = [sys.executable, str(HERE / "g2_tools.py"), "spend", "--attempt", str(att), "--root", str(t)] + (["--max-attempt", str(max_attempt)] if max_attempt else [])
        r = subprocess.run(cmd, capture_output=True, text=True); return r.returncode, json.loads(r.stdout.strip().splitlines()[-1])
    g.write_text(json.dumps({"missing_cost_responses": 0, "budget_hit": True}))
    rc, o = run(); assert rc == 0 and o["attempt_flags_bad"] == 0 and o["attempt_budget_hit_events"] == 1, (rc, o)      # a cap event is not a failure
    g.write_text(json.dumps({"missing_cost_responses": 2, "budget_hit": False}))
    rc, o = run(); assert rc == 6 and o["attempt_flags_bad"] == 1, (rc, o)                                             # a response without cost still is
    g.write_text(json.dumps({"missing_cost_responses": 0, "budget_hit": False}))
    rc, o = run(max_attempt=0.005); assert rc == 5, (rc, o)                                                            # the total limit still is rc 5
print("test_ledger_budget_hit: OK")
