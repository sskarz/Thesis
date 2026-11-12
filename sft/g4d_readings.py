#!/usr/bin/env python3
"""Mechanical application of the registered reading rules for the G4 diagnostics (docs/DISTILLATION_G4_DIAGNOSTICS_PREREGISTRATION.md, section 5,
copied from the advisor's plan file). Exploratory. usage: g4d_readings.py ATTEMPT_DIR  (reads RUN_D1/D2/D3.json, writes DIAG_RESULT.json)"""
import json, sys

def read_d1(s):      # 75 or below / 82 or above / between
    if s <= 75: return "75 or below: answering without thinking explains most of the loss."
    if s >= 82: return "82 or above: the style costs little here and the training itself did the damage."
    return "Between (76 to 81): both contribute."

def read_d3(s):      # at least 6 above 69 / within 5 of 69
    if s >= 75: return "At least 6 above 69: the second pass over-fitted; train shorter and pick the stopping point with the new screen."
    if s >= 64: return "Within 5 of 69: over-fitting in the second pass is not the issue."
    return "More than 5 below 69: not covered by the registered readings; reported as a score only."

def read_d2(s):      # 82 or above / 76 to 81 / 75 or below
    if s >= 82: return "82 or above: reasoning-first training does no harm even with the gap; strong case for this plan in that format."
    if s >= 76: return "76 to 81: partial."
    return "75 or below: the data gap hurts in either format; this plan must fix coverage to have any chance, and the odds in section 9 drop."

def use_rule(d1, d2):  # "reasoning-first unless D2 is 75 or below and D1 is 82 or above, in which case I come back to you before anything is generated"
    return ("COME BACK TO SANS before anything is generated (D2 is 75 or below and D1 is 82 or above)." if d2 <= 75 and d1 >= 82
            else "Format stays reasoning-first (the exception, D2 at 75 or below together with D1 at 82 or above, did not occur).")

def main(att):
    runs = {k: json.load(open(f"{att}/RUN_{k}.json")) for k in ("D1", "D3", "D2")}
    sc = {k: len(v["passed_ids"]) for k, v in runs.items()}
    ref = {k: set(json.load(open(f"{att}/../../../r6g4/attempts/20261111_203247_g4run.JoHtbc/RUN_{k}.json"))["passed_ids"]) for k in ("U1", "T1", "U2")}
    out = {"exploratory": True, "scores": sc, "errored": {k: v["errored_tasks"] for k, v in runs.items()},
           "readings": {"D1": read_d1(sc["D1"]), "D3": read_d3(sc["D3"]), "D2": read_d2(sc["D2"])}, "use_rule": use_rule(sc["D1"], sc["D2"]),
           "descriptive_counts_vs_reference": {k: {r: {"this_only": len(set(v["passed_ids"]) - ref[r]), "reference_only": len(ref[r] - set(v["passed_ids"]))} for r in ref} for k, v in runs.items()},
           "note": "Readings are the registered wording and are hypotheses, not established causes: no diagnostic changes data coverage or isolates one training factor. Exploratory, post hoc, outside the G4 registration; U1 88 and U2 89 (thinking on) and T1 69 (arm A, thinking off) are from an earlier session."}
    json.dump(out, open(f"{att}/DIAG_RESULT.json", "w"), indent=1); print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main(sys.argv[1])
