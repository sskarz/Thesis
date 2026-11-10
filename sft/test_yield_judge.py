#!/usr/bin/env python3
"""Offline tests for yield_judge.parse_verdict and yield_calibrate (no network, no backend)."""
import json, pathlib, subprocess, sys, tempfile
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import os; os.environ.setdefault("OPENROUTER_API_KEY", "test-not-a-real-key")
import yield_judge as J

assert J.parse_verdict('{"turns": [], "verdict": "FAITHFUL", "reason": "ok"}')["verdict"] == "FAITHFUL"
assert J.parse_verdict('noise {"a": 1} then {"verdict": "WRONG", "reason": "x"} tail')["verdict"] == "WRONG"
assert J.parse_verdict('{"verdict": "MAYBE"}') is None and J.parse_verdict("no json") is None

with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    ids = [f"distill_y_{i:04d}" for i in range(1, 41)]
    lab = {}
    lines = []
    for i, t in enumerate(ids):
        v = "WRONG" if i < 7 else ("AMBIGUOUS" if i < 13 else "FAITHFUL")
        lab[t] = v; lines.append(f"{t}: {v}: reason")
    (d / "codex.md").write_text("\n".join(lines) + "\nTotals: ...\n")
    judge = {t: ("FAITHFUL" if (lab[t] == "FAITHFUL" and i not in (20, 21)) else ("WRONG" if i == 0 else "AMBIGUOUS" if lab[t] != "FAITHFUL" else "AMBIGUOUS"))
             for i, t in enumerate(ids)}
    # judge misses WRONG tasks 1..6 except: make it catch tasks 0 (WRONG) and mark others WRONG-labelled as FAITHFUL
    for i in range(1, 7):
        judge[ids[i]] = "FAITHFUL"
    (d / "j.jsonl").write_text("".join(json.dumps({"id": t, "verdict": v}) + "\n" for t, v in judge.items()))
    det = {t: [] for t in ids}
    for i in (1, 2, 3, 4, 5):          # deterministic checks catch five of the six the judge missed
        det[ids[i]] = ["id_collision:post_tweet:3"]
    det[ids[30]] = ["info_seeded_comments_invisible"]       # info flag must not drop a FAITHFUL task
    (d / "det.json").write_text(json.dumps(det))
    (d / "tasks.jsonl").write_text("".join(json.dumps({"id": t}) + "\n" for t in ids))
    (d / "flags.json").write_text(json.dumps({"missing_cost_responses": 0, "budget_hit": False}))
    r = subprocess.run([sys.executable, str(HERE / "yield_calibrate.py"), "--codex", str(d / "codex.md"), "--judge", str(d / "j.jsonl"),
                        "--det", str(d / "det.json"), "--tasks", str(d / "tasks.jsonl"), "--out", str(d / "rep.json"),
                        "--guard-flags", str(d / "flags.json")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rep = json.load(open(d / "rep.json"))
    assert rep["union"]["WRONG"] == {"dropped": 6, "of": 7}, rep["union"]              # task 6 is missed
    assert rep["missed_wrong_ids"] == [ids[6]]
    assert rep["union"]["FAITHFUL"]["dropped"] == 2 and rep["false_drop_ids"] == [ids[20], ids[21]], rep["false_drop_ids"]
    assert rep["calibration_pass"] is True
    assert ids[30] not in rep["false_drop_ids"]
    assert rep["judge_only"]["WRONG"]["dropped"] == 1 and rep["det_only"]["WRONG"]["dropped"] == 5
# an incomplete judge run (SKIPPED / missing verdicts, or a guard flag) must never pass calibration
    broken = {t_: v_ for t_, v_ in judge.items()}
    broken[ids[1]] = "SKIPPED"; broken[ids[2]] = "UNPARSED"; del broken[ids[3]]
    (d / "j2.jsonl").write_text("".join(json.dumps({"id": t_, "verdict": v_}) + "\n" for t_, v_ in broken.items()))
    for fl in ({"missing_cost_responses": 0, "budget_hit": False}, {"missing_cost_responses": 1, "budget_hit": False}):
        (d / "flags2.json").write_text(json.dumps(fl))
        r = subprocess.run([sys.executable, str(HERE / "yield_calibrate.py"), "--codex", str(d / "codex.md"), "--judge", str(d / "j2.jsonl"), "--det", str(d / "det.json"),
                            "--tasks", str(d / "tasks.jsonl"), "--out", str(d / "rep2.json"), "--guard-flags", str(d / "flags2.json")], capture_output=True, text=True)
        rep2 = json.load(open(d / "rep2.json"))
        assert rep2["run_complete"] is False and rep2["calibration_pass"] is False, rep2["incomplete"]
        assert set(rep2["incomplete"]["missing_or_unparsed_or_skipped_ids"]) == {ids[1], ids[2], ids[3]}
    # a clean run with a guard flag also fails
    (d / "flags3.json").write_text(json.dumps({"missing_cost_responses": 1, "budget_hit": False}))
    subprocess.run([sys.executable, str(HERE / "yield_calibrate.py"), "--codex", str(d / "codex.md"), "--judge", str(d / "j.jsonl"), "--det", str(d / "det.json"),
                    "--tasks", str(d / "tasks.jsonl"), "--out", str(d / "rep3.json"), "--guard-flags", str(d / "flags3.json")], capture_output=True, text=True)
    assert json.load(open(d / "rep3.json"))["calibration_pass"] is False
print("test_yield_judge: OK")
