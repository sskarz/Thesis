#!/bin/bash
# G3 base references (SANS APPROVED 2026-11-10 "Approve for all 3", item 1): untrained Gemma-4-E2B-it Q4_K_M bar GGUF on FK (F 81 + K2 74), compact-list mode
# iffc-state-ooda-slim (Thesis-wt-slim dfdbc2b), greedy, server restarted between runs. Runs: on1 and on2 (thinking on), off (thinking off). GPU only, $0.
# Threshold (registered before any candidate is scored): run 1 of a candidate needs >= ceil(mean(on1,on2) + 0.06 x 155) AND > max(on1,on2); a restarted repeat uses the same mean rule.
# usage: bash sft/g3_base.sh base
set -u
case "${1:-}" in base) ;; *) echo "usage: $0 base"; exit 2;; esac
cd /Users/sanskarthapa/Documents/GitHub/Thesis-wt-r6 || exit 2
source sft/g3lib.sh
init_attempt base
echo "$(date +%T) g3 base start, attempt $ATT"
preflight
chain_clock
arm G3on1 1.0 "" on
arm G3on2 1.0 "" on
arm G3off 1.0 "" off
$PY -B - $ATT <<"PYEOF"
import json, sys, pathlib, math
a = pathlib.Path(sys.argv[1]); out = {}; ok = True
for n in ("G3on1", "G3on2", "G3off"):
    r = {}
    for bank, cnt in (("F", 81), ("K2", 74)):
        d = a / "evals" / f"{bank}_{n}"; v = [json.loads(l) for l in open(d / "verdicts.jsonl")]
        rep = json.load(open(d / "report.json")); dm = rep["decision_mix"]
        valid = len(v) == cnt and len({x["task_id"] for x in v}) == cnt and not any(x.get("api_error") for x in v)
        ok &= valid
        r[bank] = {"passes": sum(bool(x["valid"]) for x in v), "rows": len(v), "valid": valid, "assistant_outputs": dm["assistant_outputs"], "unparsed": dm["unparsed_outputs"]}
    r["composite"] = r["F"]["passes"] + r["K2"]["passes"]
    n_out = r["F"]["assistant_outputs"] + r["K2"]["assistant_outputs"]; r["parse_rate"] = round(1 - (r["F"]["unparsed"] + r["K2"]["unparsed"]) / n_out, 4) if n_out else None
    out[n] = r
m = (out["G3on1"]["composite"] + out["G3on2"]["composite"]) / 2
out["threshold"] = {"mean_thinking_on": m, "run1_needs_at_least": max(math.ceil(m + 0.06 * 155), int(max(out["G3on1"]["composite"], out["G3on2"]["composite"])) + 1), "rule": "ceil(mean + 0.06*155) and > max(on1,on2)"}
out["valid"] = ok
json.dump(out, open(a / "G3_BASE_REPORT.json", "w"), indent=1); print(json.dumps(out)); sys.exit(0 if ok else 1)
PYEOF
RC=$?
[ $RC -eq 0 ] && say "G3 BASE COMPLETE: $ATT" || die "g3 base invalid (counts/ids/api errors)"
