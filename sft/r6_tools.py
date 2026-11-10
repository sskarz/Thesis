#!/usr/bin/env python3
"""Round 6 aggregate reporters and audits (counts only; no per-task text or per-task results are printed).

  probes     --ref probe_B3.json --runs probe_X.json ... --out report.json
  summarize  --attempt ATTEMPT_DIR --out report.json
  gguf-check --adapter-dir DIR --gguf FILE.gguf --src-dir DIR
  norms      --adapters dirA dirB ... --out report.json
  provenance-rows --rows-dir DIR --out-dir DIR2        (stage 2 only; see its own review)

probes: per probe run: n, errors, truncated (finish_reason == "length"), unparsable, exact target agreement, and the number of greedy
actions that differ from the reference run (an unparsable result counts as its own action). A comparison is VALID only if the reference
and the run each have 70 prompts, 0 errors and 0 truncations and identical prompt keys with identical rendered prompts; otherwise it is
reported as invalid and no difference count is given.
summarize: per arm in an attempt dir: composite passes (F + K2), per-bank passes, band label, and the registered follow-up flags
(base bracket outside 65 to 70; contemporaneous S0 outside 56 to 62; screen met). It prints what follow-up is REQUIRED; it schedules nothing.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

BANDS = [(-1, 55, "A (<=55, below S0-like)"), (56, 62, "B (56-62, S0-like)"), (63, 64, "C (63-64, ambiguous)"),
         (65, 70, "D (65-70, base-like)"), (71, 76, "E (71-76, above base, below screen)")]


def band(c):
    if c >= 77:
        return "F (>=77, screen run 1 met if also >68)"
    for lo, hi, name in BANDS:
        if lo <= c <= hi:
            return name
    return "?"


def _valid_probe(d):
    """Complete validity predicate, shared by probe-check and probes: 70 unique prompts, 0 errors, 0 truncations, one think marker each."""
    res = d["results"]
    n = len(res)
    err = sum("error" in x for x in res)
    trunc = sum(x.get("finish_reason") == "length" for x in res)
    keys = [(x["set"], x["i"]) for x in res]
    markers = all(x.get("think_markers") == 1 for x in res if "error" not in x)
    return (n == 70 and err == 0 and trunc == 0 and len(set(keys)) == 70 and markers), {"n": n, "errors": err, "truncated": trunc, "one_think_marker_each": markers}


def probe_check(a):
    d = json.load(open(a.probe))
    ok, info = _valid_probe(d)
    print(json.dumps({**info, "valid": ok}))
    if not ok:
        raise SystemExit(3)


def probes(a):
    def load(path):
        try:
            return json.load(open(path))
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    ref = load(a.ref)
    ref_ok, ref_info = _valid_probe(ref) if ref else (False, {"missing": True})
    ref = ref or {"results": []}
    key = lambda x: (x["set"], x["i"])
    refd = {key(x): x for x in ref["results"]}
    rep = {"ref": a.ref, "ref_valid": ref_ok, "ref_info": ref_info, "runs": {}}
    for p in a.runs:
        d = load(p)
        if d is None:
            rep["runs"][pathlib.Path(p).stem.replace("probe_", "")] = {"file": p, "valid": False, "comparison_valid": False, "n": 0, "errors": 0, "truncated": 0,
                                                                         "unparsable": 0, "exact_target": 0, "actions_differ_from_ref": None, "note": "missing or unreadable probe file"}
            continue
        res = d["results"]
        ok, info = _valid_probe(d)
        entry = {"file": p, **info, "valid": ok,
                 "unparsable": sum(x.get("pred") is None and "error" not in x for x in res),
                 "exact_target": sum(int(x.get("pred") is not None and x.get("pred") == x.get("target")) for x in res),
                 "lora_adapters": d.get("lora_adapters"), "model_path": d.get("model_path")}
        same_keys = {key(x) for x in res} == set(refd)
        same_prompts = same_keys and all(x.get("rendered_sha") and refd[key(x)].get("rendered_sha") == x.get("rendered_sha") for x in res)
        entry["comparison_valid"] = bool(ok and ref_ok and same_keys and same_prompts)
        if entry["comparison_valid"]:
            diff = {"all": 0, "valid": 0, "train": 0}
            for x in res:
                if x.get("pred") != refd[key(x)].get("pred"):
                    diff["all"] += 1
                    diff[x["set"]] += 1
            entry["actions_differ_from_ref"] = diff
        else:
            entry["actions_differ_from_ref"] = None
            entry["comparison_note"] = "reference or run invalid, or prompt keys / rendered prompts differ"
        rep["runs"][d["tag"]] = entry
    json.dump(rep, open(a.out, "w"), indent=1)
    all_valid = ref_ok and all(r["valid"] and r["comparison_valid"] for r in rep["runs"].values())
    rep["all_valid"] = all_valid
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps({"ref_valid": ref_ok, "all_valid": all_valid, **{t: {k: v[k] for k in ("n", "errors", "truncated", "unparsable", "exact_target", "valid", "comparison_valid", "actions_differ_from_ref")} for t, v in rep["runs"].items()}}))
    if not all_valid:
        raise SystemExit(3)


def _passes(verd_path):
    v = [json.loads(l) for l in open(verd_path)]
    ids = [x.get("task_id") for x in v]
    return {"passes": sum(bool(x.get("valid")) for x in v), "rows": len(v), "unique": len(set(ids)), "has_ids": all(i is not None for i in ids),
            "api_errors": sum(bool(x.get("api_error")) for x in v)}


EXPECTED_ARMS = ["B3", "S0-1.0c", "R-S0", "S0-0.5", "S0-0.25", "B3b"]


def summarize(a):
    att = pathlib.Path(a.attempt)
    rep, flags, bad = {}, [], []
    for arm in EXPECTED_ARMS:
        try:
            f = _passes(att / "evals" / f"F_{arm}" / "verdicts.jsonl")
            k = _passes(att / "evals" / f"K2_{arm}" / "verdicts.jsonl")
        except FileNotFoundError as e:
            bad.append(f"{arm}: missing {e.filename}")
            continue
        ok = (f["rows"] == 81 and f["unique"] == 81 and f["has_ids"] and k["rows"] == 74 and k["unique"] == 74 and k["has_ids"]
              and f["api_errors"] + k["api_errors"] == 0)
        c = f["passes"] + k["passes"]
        rep[arm] = {"F": f["passes"], "K2": k["passes"], "composite": c, "valid": ok, "band": band(c)}
        if not ok:
            bad.append(f"{arm}: invalid verdict file (rows/unique ids/api errors): F {f} K2 {k}")
    for arm, r in rep.items():
        if not r["valid"]:
            continue
        c = r["composite"]
        if arm.startswith("B") and not (65 <= c <= 70):
            flags.append(f"FOLLOW-UP REQUIRED: base arm {arm} composite {c} is outside 65 to 70; add one base run, list all attempts, mark the session with a drift flag")
        if arm == "S0-1.0c" and not (56 <= c <= 62):
            flags.append(f"FOLLOW-UP REQUIRED: contemporaneous S0 composite {c} is outside 56 to 62; add one base run, list all attempts, mark the session with a drift flag")
        if c >= 77 and c > 68:
            flags.append(f"SCREEN RUN 1 MET: {arm} composite {c}; the registered restarted repeat needs Sans's go; no statement before it")
    json.dump({"arms": rep, "flags": flags, "problems": bad}, open(a.out, "w"), indent=1)
    for arm, r in rep.items():
        print(arm, json.dumps(r))
    for x in flags + bad:
        print(x)
    if not flags and not bad:
        print("no follow-up flags")
    if bad:
        raise SystemExit("incomplete or invalid stage report")


# llama.cpp tensor names for every module the G1 wide adapter can contain (read from a real conversion of a wide adapter, 2026-11-09)
GGUF_MODULE = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v", "o_proj": "attn_output",
               "gate_proj": "ffn_gate", "up_proj": "ffn_up", "down_proj": "ffn_down"}


def gguf_check(a):
    """Audit a converted LoRA GGUF against its source MLX adapter. Orientation is fixed (no permutation search): the source update is
    P = A[in,r] @ B[r,out]; the GGUF holds lora_a [r,in] and lora_b [out,r] (as the converter writes them), so scale * P^T must equal
    (alpha / r) * (lora_b @ lora_a). Requires the exact tensor set, finite values and metadata, and reports the worst relative Frobenius
    error of the whole matrix per pair (fp16 storage error is expected to be about 1e-4 or below; tolerance 1e-2)."""
    sys.path.insert(0, "/Users/sanskarthapa/Documents/GitHub/Thesis/llama.cpp/gguf-py")
    from gguf import GGUFReader
    from safetensors import safe_open
    scale = float(json.load(open(pathlib.Path(a.adapter_dir) / "adapter_config.json"))["lora_parameters"]["scale"])
    rd = GGUFReader(a.gguf)
    ts = {t.name: np.array(t.data, dtype=np.float64) for t in rd.tensors}
    alphas = [float(f.parts[f.data[0]][0]) for k, f in rd.fields.items() if k.endswith("lora.alpha")]
    if len(alphas) != 1 or not np.isfinite(alphas[0]) or alphas[0] <= 0:
        raise SystemExit(f"bad or missing lora.alpha metadata: {alphas}")
    alpha = alphas[0]
    src = {}
    with safe_open(str(pathlib.Path(a.src_dir) / "adapters.safetensors"), "np") as f:
        for k in f.keys():
            src[k] = f.get_tensor(k).astype(np.float64)
    prefixes = sorted({k[: -len(".lora_a")] for k in src if k.endswith(".lora_a")})
    names = {}
    for p in prefixes:
        n = p.split(".layers.")[1].split(".")[0]
        mod = GGUF_MODULE[p.split(".")[-1]]
        names[p] = (f"blk.{n}.{mod}.weight.lora_a", f"blk.{n}.{mod}.weight.lora_b")
    expected = {x for pair in names.values() for x in pair}
    if set(ts) != expected:
        raise SystemExit(f"GGUF tensor set differs from the source: extra {sorted(set(ts) - expected)[:3]}, missing {sorted(expected - set(ts))[:3]}")
    worst = 0.0
    with np.errstate(all="ignore"):          # Accelerate on macOS raises spurious FP flags; finiteness is checked explicitly below
        for p, (na, nb) in names.items():
            ga, gb = ts[na], ts[nb]
            r = ga.shape[0]
            if not (np.isfinite(ga).all() and np.isfinite(gb).all()):
                raise SystemExit(f"non-finite GGUF tensor for {p}")
            if gb.shape[1] != r or ga.shape[1] != src[p + ".lora_a"].shape[0] or gb.shape[0] != src[p + ".lora_b"].shape[1]:
                raise SystemExit(f"unexpected GGUF shapes for {p}: {ga.shape} {gb.shape}")
            want = scale * (src[p + ".lora_a"] @ src[p + ".lora_b"]).T
            got = (alpha / r) * (gb @ ga)
            if not (np.isfinite(want).all() and np.isfinite(got).all()):
                raise SystemExit(f"non-finite product for {p}")
            nw = np.linalg.norm(want)
            err = float(np.linalg.norm(got - want) / nw) if nw > 0 else float(np.linalg.norm(got))
            if not np.isfinite(err):
                raise SystemExit(f"non-finite error for {p}")
            worst = max(worst, err)
    print(json.dumps({"pairs": len(prefixes), "alpha": alpha, "max_rel_matrix_err_gguf_vs_source": worst}))
    if worst > 1e-2:
        raise SystemExit(f"GGUF update differs from the source adapter by {worst}")


def norms(a):
    from safetensors import safe_open
    rep = {}
    for dpath in a.adapters:
        d = pathlib.Path(dpath)
        scale = float(json.load(open(d / "adapter_config.json"))["lora_parameters"]["scale"])
        t = {}
        with safe_open(str(d / "adapters.safetensors"), "np") as f:
            for k in f.keys():
                t[k] = f.get_tensor(k).astype(np.float64)
        pairs = sorted({k[: -len(".lora_a")] for k in t if k.endswith(".lora_a")})
        per = {p: float(np.linalg.norm(scale * (t[p + ".lora_a"] @ t[p + ".lora_b"]))) for p in pairs}
        rep[str(d)] = {"scale": scale, "pairs": len(pairs), "total_update_norm": float(np.sqrt(sum(v * v for v in per.values()))), "per_pair": per}
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps({k: {"pairs": v["pairs"], "total_update_norm": v["total_update_norm"]} for k, v in rep.items()}))


def provenance_rows(a):
    src, out = pathlib.Path(a.rows_dir), pathlib.Path(a.out_dir)
    if src.resolve() == out.resolve():
        raise SystemExit("refusing to write into the source dir")
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for split in ("train", "valid"):
        with open(out / f"{split}.jsonl", "w") as f:
            for line in open(src / f"{split}.jsonl"):
                if not line.strip():
                    continue
                r = json.loads(line)
                for m in r["messages"]:
                    if m["role"] == "assistant" and m.get("reasoning_content"):
                        m["content"] = (m.get("content") or "") + "\n" + m["reasoning_content"]
                f.write(json.dumps(r) + "\n")
                n += 1
    print(json.dumps({"rows_written": n}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="c", required=True)
    p = sub.add_parser("probes"); p.add_argument("--ref", required=True); p.add_argument("--runs", nargs="+", required=True); p.add_argument("--out", required=True)
    pc = sub.add_parser("probe-check"); pc.add_argument("--probe", required=True)
    s = sub.add_parser("summarize"); s.add_argument("--attempt", required=True); s.add_argument("--out", required=True)
    g = sub.add_parser("gguf-check"); g.add_argument("--adapter-dir", required=True); g.add_argument("--gguf", required=True); g.add_argument("--src-dir", required=True)
    n = sub.add_parser("norms"); n.add_argument("--adapters", nargs="+", required=True); n.add_argument("--out", required=True)
    r = sub.add_parser("provenance-rows"); r.add_argument("--rows-dir", required=True); r.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    {"probe-check": probe_check, "probes": probes, "summarize": summarize, "gguf-check": gguf_check, "norms": norms, "provenance-rows": provenance_rows}[a.c](a)


if __name__ == "__main__":
    main()
