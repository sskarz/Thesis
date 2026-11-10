#!/usr/bin/env python3
"""Offline self-test for r6_tools.py."""
import json, pathlib, subprocess, sys, tempfile
import numpy as np
from safetensors.numpy import save_file

HERE = pathlib.Path(__file__).resolve().parent


def run(*args, ok=True):
    r = subprocess.run([sys.executable, str(HERE / "r6_tools.py"), *args], capture_output=True, text=True)
    assert (r.returncode == 0) == ok, r.stderr
    return r.stdout


def probe(tag, preds, trunc=(), err=(), n=70, sha="s"):
    res = []
    for i in range(n):
        x = {"set": "valid" if i < 35 else "train", "i": i % 35, "pred": preds(i), "target": ["a"], "rendered_sha": sha + str(i), "think_markers": 1,
             "finish_reason": "length" if i in trunc else "stop"}
        if i in err:
            x["error"] = "boom"
        res.append(x)
    return {"tag": tag, "lora_adapters": [], "model_path": "m", "results": res}


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        base = lambda i: ["a"] if i % 2 else ["b"]
        json.dump(probe("ref", base), open(d / "ref.json", "w"))
        _m = probe("ref", base)
        for x in _m["results"]: x["think_markers"] = 1
        json.dump(_m, open(d / "ref_m.json", "w"))
        json.dump(probe("x", lambda i: base(i) if i > 2 else ["z"]), open(d / "x.json", "w"))                # 3 differ
        json.dump(probe("trunc", base, trunc=(3,)), open(d / "t.json", "w"))                                  # invalid run
        json.dump(probe("sha", base, sha="other"), open(d / "s.json", "w"))                                   # different rendered prompts
        run("probes", "--ref", str(d / "ref.json"), "--runs", str(d / "x.json"), str(d / "t.json"), str(d / "s.json"), "--out", str(d / "rep.json"), ok=False)   # invalid runs -> rc 3
        run("probes", "--ref", str(d / "ref.json"), "--runs", str(d / "ref.json"), str(d / "x.json"), "--out", str(d / "rep_ok.json"))
        assert json.load(open(d / "rep_ok.json"))["all_valid"] is True
        bad = probe("badm", base)
        bad["results"][0]["think_markers"] = 2
        json.dump(bad, open(d / "badm.json", "w")); run("probe-check", "--probe", str(d / "badm.json"), ok=False)   # marker count != 1
        good = probe("good", base)
        for x in good["results"]: x["think_markers"] = 1
        json.dump(good, open(d / "good.json", "w")); run("probe-check", "--probe", str(d / "good.json"))
        run("probe-check", "--probe", str(d / "t.json"), ok=False)
        run("probes", "--ref", str(d / "good.json"), "--runs", str(d / "nonexistent.json"), "--out", str(d / "rep_missing.json"), ok=False)
        assert json.load(open(d / "rep_missing.json"))["all_valid"] is False
        r = json.load(open(d / "rep.json"))["runs"]
        assert r["x"]["comparison_valid"] and r["x"]["actions_differ_from_ref"]["all"] == 3, r["x"]
        assert r["trunc"]["valid"] is False and r["trunc"]["comparison_valid"] is False and r["trunc"]["actions_differ_from_ref"] is None
        assert r["sha"]["comparison_valid"] is False
        # invalid reference invalidates every comparison
        json.dump(probe("ref", base, trunc=(5,)), open(d / "ref_bad.json", "w"))
        run("probes", "--ref", str(d / "ref_bad.json"), "--runs", str(d / "x.json"), "--out", str(d / "rep2.json"), ok=False)
        rr = json.load(open(d / "rep2.json"))
        assert rr["ref_valid"] is False and rr["runs"]["x"]["comparison_valid"] is False
        # summarize
        att = d / "att"
        def verd(arm, bank, passes, n, dup=False):
            p = att / "evals" / f"{bank}_{arm}"; p.mkdir(parents=True)
            with open(p / "verdicts.jsonl", "w") as f:
                for i in range(n):
                    f.write(json.dumps({"task_id": f"t{0 if (dup and i == 1) else i}", "valid": i < passes}) + "\n")
        for arm, (pf, pk) in {"B3": (35, 33), "B3b": (40, 33), "S0-1.0c": (29, 30), "R-S0": (45, 40), "S0-0.5": (30, 30), "S0-0.25": (30, 30)}.items():
            verd(arm, "F", pf, 81); verd(arm, "K2", pk, 74)
        run("summarize", "--attempt", str(att), "--out", str(d / "sum.json"))
        s = json.load(open(d / "sum.json"))
        assert s["arms"]["B3"]["band"].startswith("D") and s["arms"]["S0-1.0c"]["band"].startswith("B")
        assert any("B3b" in f and "FOLLOW-UP" in f for f in s["flags"]) and any("SCREEN RUN 1 MET" in f and "R-S0" in f for f in s["flags"])
        assert not any("S0-1.0c" in f for f in s["flags"]) and not any("base arm B3 " in f for f in s["flags"])
        # S0 drift requires the extra base run
        att2 = d / "att2"; att = att2
        for arm, (pf, pk) in {"B3": (35, 33), "B3b": (35, 33), "S0-1.0c": (40, 30), "R-S0": (30, 30), "S0-0.5": (30, 30), "S0-0.25": (30, 30)}.items():
            verd(arm, "F", pf, 81); verd(arm, "K2", pk, 74)
        run("summarize", "--attempt", str(att2), "--out", str(d / "sum2.json"))
        assert any("contemporaneous S0" in f and "add one base run" in f for f in json.load(open(d / "sum2.json"))["flags"])
        # incomplete (missing arm) and duplicate ids must fail
        att3 = d / "att3"; att = att3
        verd("B3", "F", 35, 81); verd("B3", "K2", 33, 74)
        run("summarize", "--attempt", str(att3), "--out", str(d / "sum3.json"), ok=False)
        att4 = d / "att4"; att = att4
        for arm in ["B3", "S0-1.0c", "R-S0", "S0-0.5", "S0-0.25", "B3b"]:
            verd(arm, "F", 35, 81, dup=(arm == "R-S0")); verd(arm, "K2", 33, 74)
        run("summarize", "--attempt", str(att4), "--out", str(d / "sum4.json"), ok=False)
        # norms
        ad = d / "ad"; ad.mkdir()
        rng = np.random.default_rng(1)
        a = rng.standard_normal((6, 2)).astype(np.float32); b = rng.standard_normal((2, 5)).astype(np.float32)
        save_file({"language_model.model.layers.19.self_attn.q_proj.lora_a": a, "language_model.model.layers.19.self_attn.q_proj.lora_b": b}, str(ad / "adapters.safetensors"))
        (ad / "adapter_config.json").write_text(json.dumps({"lora_parameters": {"rank": 2, "scale": 2.0}}))
        run("norms", "--adapters", str(ad), "--out", str(d / "n.json"))
        n = json.load(open(d / "n.json"))[str(ad)]
        want = 2.0 * np.linalg.norm(a.astype(np.float64) @ b.astype(np.float64))
        assert abs(n["total_update_norm"] - want) < 1e-9 and n["pairs"] == 1
        # provenance rows
        rows = d / "rows"; rows.mkdir()
        row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "[]", "reasoning_content": "because X"}], "meta": {}}
        (rows / "train.jsonl").write_text(json.dumps(row) + "\n"); (rows / "valid.jsonl").write_text(json.dumps(row) + "\n")
        run("provenance-rows", "--rows-dir", str(rows), "--out-dir", str(d / "out"))
        o = json.loads(open(d / "out" / "train.jsonl").readline())
        assert "because X" in o["messages"][-1]["content"]
        assert json.loads(open(rows / "train.jsonl").readline())["messages"][-1]["content"] == "[]", "source rows modified"
    print("test_r6_tools: OK")


if __name__ == "__main__":
    main()
