#!/usr/bin/env python3
"""Gate between the 3-slot smoke run and the paid author run. Exit 0 only if the smoke calls prove the setup works.

Requires: at least 1 logged call; every call has a numeric cost > 0; exactly one API model string, equal to the pin; exactly one provider,
containing "deepinfra"; no call's finish_reason is missing. (Whether a task is authored in 3 slots is NOT required: valid rate is ~10%.)
usage: yield_smoke_check.py --calls smoke/log/calls.jsonl
"""
import argparse, json, sys

PIN = "deepseek/deepseek-v4.1-flash-1017"

def check(calls):
    errs = []
    if not calls:
        return ["no calls were logged"]
    if any(not isinstance(c.get("cost_usd"), (int, float)) or c["cost_usd"] <= 0 for c in calls):
        errs.append("a call has no positive cost")
    models = sorted({str(c.get("api_model")) for c in calls}); provs = sorted({str(c.get("api_provider")) for c in calls})
    if models != [PIN]:
        errs.append(f"api model strings {models} != [{PIN!r}]")
    if len(provs) != 1 or "deepinfra" not in provs[0].lower():
        errs.append(f"providers {provs}")
    if any(not c.get("finish_reason") for c in calls):
        errs.append("a call has no finish_reason")
    return errs

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--calls", required=True); a = ap.parse_args()
    try:
        calls = [json.loads(l) for l in open(a.calls) if l.strip()]
    except FileNotFoundError:
        calls = []
    e = check(calls)
    print(json.dumps({"calls": len(calls), "errors": e}))
    sys.exit(1 if e else 0)
