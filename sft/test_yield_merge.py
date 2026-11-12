#!/usr/bin/env python3
import json, pathlib, subprocess, sys, tempfile
HERE = pathlib.Path(__file__).resolve().parent
def task(i, text): return {"id": i, "question": [[{"role": "user", "content": text}]]}
A = "please open the quarterly budget spreadsheet in the finance folder and tell me the total for march"
B = "open the quarterly budget spreadsheet in the finance folder and tell me the total for march please"   # shares 8-grams with A
C = "send a short thank you message to the new intern about the onboarding session this morning today"
with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    (d / "a.jsonl").write_text(json.dumps(task("distill_g2a_0001", A)) + "\n" + json.dumps(task("distill_g2a_0002", C)) + "\n")
    (d / "b.jsonl").write_text(json.dumps(task("distill_g2b_0001", B)) + "\n")
    (d / "ex.jsonl").write_text(json.dumps(task("distill_y_0007", "cancel the pending order for the watchlist symbol after checking the account balance first")) + "\n")
    r = subprocess.run([sys.executable, str(HERE / "yield_merge.py"), "--inputs", str(d / "a.jsonl"), str(d / "b.jsonl"), "--existing", str(d / "ex.jsonl"),
                        "--out", str(d / "m.jsonl"), "--report", str(d / "r.json")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    rep = json.load(open(d / "r.json"))
    kept = [json.loads(l)["id"] for l in open(d / "m.jsonl")]
    assert kept == ["distill_g2a_0001", "distill_g2a_0002"], kept            # first process wins; the near copy from process b is dropped
    assert rep["dropped"] == 1 and rep["drops"][0]["id"] == "distill_g2b_0001"
    # same id in two inputs must be refused
    (d / "c.jsonl").write_text(json.dumps(task("distill_g2a_0001", "something completely different about booking a flight to the coast next week")) + "\n")
    r2 = subprocess.run([sys.executable, str(HERE / "yield_merge.py"), "--inputs", str(d / "a.jsonl"), str(d / "c.jsonl"), "--out", str(d / "m2.jsonl"), "--report", str(d / "r2.json")], capture_output=True, text=True)
    assert r2.returncode != 0 and "duplicate task id" in (r2.stderr + r2.stdout)
print("test_yield_merge: OK")
