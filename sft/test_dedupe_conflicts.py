#!/usr/bin/env python3
import json, pathlib, subprocess, sys, tempfile
HERE = pathlib.Path(__file__).resolve().parent

def r(task, user, tgt, step=0):
    return {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": user}, {"role": "assistant", "content": tgt}],
            "meta": {"task_id": task, "episode_id": task + "__s0", "step": step}}

with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d); (d / "in").mkdir()
    train = [r("a", "u1", "[]"), r("a", "u2", "X"), r("a", "u2", "Y"), r("b", "u3", "Z"), r("b", "u3", "Z")]   # u2 conflicts; u3 identical (kept; duplicates are not conflicts)
    valid = [r("c", "u4", "W")]
    (d / "in" / "train.jsonl").write_text("".join(json.dumps(x) + "\n" for x in train))
    (d / "in" / "valid.jsonl").write_text("".join(json.dumps(x) + "\n" for x in valid))
    p = subprocess.run([sys.executable, str(HERE / "dedupe_conflicts.py"), "--rows-dir", str(d / "in"), "--out-dir", str(d / "out")], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    rep = json.load(open(d / "out" / "conflicts.json"))
    assert rep["prompts_with_conflicting_targets"] == 1 and rep["rows_dropped"] == {"train": 2, "valid": 0} and rep["rows_out"] == {"train": 3, "valid": 1}, rep
    left = [json.loads(l)["messages"][1]["content"] for l in open(d / "out" / "train.jsonl")]
    assert left == ["u1", "u3", "u3"], left
    assert subprocess.run([sys.executable, str(HERE / "dedupe_conflicts.py"), "--rows-dir", str(d / "in"), "--out-dir", str(d / "in")], capture_output=True).returncode != 0
print("test_dedupe_conflicts: OK")
