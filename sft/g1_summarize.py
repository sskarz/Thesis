#!/usr/bin/env python3
"""G1 result: apply the registered pass rule to the two 40-task scoring reports.  usage: g1_summarize.py --base DIR --adapter DIR --out G1_REPORT.json
Rule (docs/DISTILLATION_G1_PREREGISTRATION.md): PASS iff adapter passes >= 24 of 40 AND adapter - base >= 8. Parse rate = 1 - unparsed/assistant outputs, reported next to each pass count.
Validity: 40 rows, 40 unique tasks, 0 API-error episodes in both runs; otherwise INVALID (never PASS or FAIL)."""
import argparse, json, pathlib, sys
ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--adapter", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
def load(d):
    d = pathlib.Path(d); v = [json.loads(l) for l in open(d / "verdicts.jsonl")]; rep = json.load(open(d / "report.json")); dm = rep["decision_mix"]
    n_out = dm["assistant_outputs"]
    return {"rows": len(v), "unique_tasks": len({x["task_id"] for x in v}), "api_error_episodes": sum(bool(x.get("api_error")) for x in v),
            "passes": sum(bool(x["valid"]) for x in v), "assistant_outputs": n_out, "unparsed_outputs": dm["unparsed_outputs"],
            "parse_rate": round(1 - dm["unparsed_outputs"] / n_out, 4) if n_out else None, "stop_share": dm["stop_share"], "lookup_share": dm["lookup_share"]}
b, ad = load(a.base), load(a.adapter)
valid = all(x["rows"] == 40 and x["unique_tasks"] == 40 and x["api_error_episodes"] == 0 for x in (b, ad))
res = {"base_thinking_off": b, "adapter_thinking_off": ad, "valid": valid, "rule": "adapter>=24 and adapter-base>=8"}
res["verdict"] = "INVALID" if not valid else ("PASS" if ad["passes"] >= 24 and ad["passes"] - b["passes"] >= 8 else "FAIL")
json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res)); sys.exit(0)
