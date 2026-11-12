#!/usr/bin/env python3
"""Yield-check report: funnel, spend, diversity and row-format conformance for one data-yield attempt directory.

usage (from the worktree root, Thesis venv):  python sft/yield_report.py --attempt DIR [--total-cap 5.0] [--out DIR/YIELD_REPORT.json]

Expected layout (written by sft/data_yield_chain.sh):
  DIR/author/{tasks_raw.jsonl, author.out, log/calls.jsonl}   DIR/audited.jsonl   DIR/semantic_audit.json
  DIR/solve/{verdicts.jsonl, pilot_report.json, calls.jsonl}   DIR/filtered/{tasks_F.jsonl, F_report.json}
  DIR/rows/{train.jsonl, valid.jsonl, build_log.json}          DIR/provenance.json   DIR/smoke/log/calls.jsonl (optional)

Everything is counted from files; nothing is estimated here. Ratios carry 95% Wilson intervals. Aggregate output only: no per-task
text is printed. Acceptance flags (A2..A6 of docs/DATA_YIELD_PLAN.md) are computed here; A1 (independent semantic review) is separate.
"""
from __future__ import annotations

import argparse
import os
import collections
import difflib
import json
import math
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def ratio(k, n):
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


def parse_author_out(text):
    i = text.find("{")
    rep, _ = json.JSONDecoder().raw_decode(text[i:])
    m = re.search(r"spent_usd\s+([0-9.]+)\s+calls\s+(\d+)", text)
    return rep, (float(m.group(1)) if m else None), (int(m.group(2)) if m else None)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2_tools import calls_files as g2_calls_files  # noqa: E402


def ledger(call_files):
    out = {"calls": 0, "cost_usd": 0.0, "calls_missing_cost": 0, "api_models": set(), "api_providers": set(),
           "prompt_tokens": 0, "completion_tokens": 0}
    for p in call_files:
        if not pathlib.Path(p).exists():
            continue
        for r in load(p):
            out["calls"] += 1
            c = r.get("cost_usd")
            if c is None:
                out["calls_missing_cost"] += 1
            else:
                out["cost_usd"] += float(c)
            out["api_models"].add(str(r.get("api_model")))
            out["api_providers"].add(str(r.get("api_provider")))
            u = r.get("usage") or {}
            out["prompt_tokens"] += u.get("prompt_tokens") or 0
            out["completion_tokens"] += u.get("completion_tokens") or 0
    out["cost_usd"] = round(out["cost_usd"], 4)
    out["api_models"] = sorted(out["api_models"])
    out["api_providers"] = sorted(out["api_providers"])
    return out


# --------------------------------------------------------------------------- diversity
def signature(task):
    return tuple(tuple(re.match(r"\s*([A-Za-z_][\w]*)", c).group(1) for c in turn) for turn in task["ground_truth"])


def user_turns(task):
    return [m["content"] for q in task["question"] for m in q if m.get("role") == "user"]


def diversity(tasks, ngrams, norm):
    sigs = collections.Counter(signature(t) for t in tasks)
    first = collections.Counter(" ".join(re.findall(r"[a-z0-9']+", user_turns(t)[0].lower())[:3]) for t in tasks if user_turns(t))
    grams = [set().union(*(ngrams(u, 8) for u in user_turns(t))) for t in tasks]
    normed = [[norm(u) for u in user_turns(t)] for t in tasks]
    near = []
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            shared = len(grams[i] & grams[j])
            best = 0.0
            for x in normed[i]:
                for y in normed[j]:
                    sm = difflib.SequenceMatcher(None, x, y, autojunk=False)
                    if sm.real_quick_ratio() <= best or sm.quick_ratio() <= best:
                        continue
                    best = max(best, sm.ratio())
            if shared or best >= 0.85:
                near.append([tasks[i]["id"], tasks[j]["id"], shared, round(best, 3)])
    n = len(tasks)
    calls = [sum(len(t) for t in x["ground_truth"]) for x in tasks]
    return {"tasks": n, "distinct_signatures": len(sigs), "signature_repeat_max": max(sigs.values()) if sigs else 0,
            "per_class_tasks": dict(collections.Counter(c for t in tasks for c in t["involved_classes"])),
            "n_turns": dict(sorted(collections.Counter(len(t["question"]) for t in tasks).items())),
            "gold_calls_total": sum(calls), "gold_calls_per_task_mean": round(sum(calls) / n, 2) if n else None,
            "first_turn_opening_3gram_top5": first.most_common(5),
            "first_turn_top1_share": round(first.most_common(1)[0][1] / n, 4) if n else None,
            "within_set_near_duplicate_pairs": len(near), "within_set_near_duplicate_examples": near[:5]}


# --------------------------------------------------------------------------- row format
META_KEYS = {"task_id", "episode_id", "sample", "step", "action", "n_calls", "lookup_calls", "classes", "exercised"}


def row_conformance(rows):
    bad = collections.defaultdict(list)
    seen = {}
    conflicts = 0
    for i, r in enumerate(rows):
        m = r.get("messages") or []
        if len(m) < 3 or m[0].get("role") != "system":
            bad["bad_roles"].append(i)
            continue
        roles = [x.get("role") for x in m[1:]]
        if roles[-1] != "assistant" or any(roles[k] != ("user" if k % 2 == 0 else "assistant") for k in range(len(roles))):
            bad["bad_roles"].append(i)
        if any(not str(x.get("content", "")).strip() for x in m):
            bad["empty_message"].append(i)
        if "<|think|>" in str(m[0].get("content")):
            bad["think_marker_already_present"].append(i)
        tgt = m[-1].get("content")
        if not isinstance(tgt, str) or "```" in tgt:
            bad["target_not_str_or_fenced"].append(i)
        else:
            try:
                lst = json.loads(tgt)
                if not isinstance(lst, list) or any(not (isinstance(c, dict) and set(c) == {"function_name", "parameters"}
                                                        and isinstance(c["parameters"], dict)) for c in lst):
                    bad["target_not_strict_call_list"].append(i)
                elif (len(lst) == 0) != (r["meta"].get("action") == "end_turn") or len(lst) != r["meta"].get("n_calls"):
                    bad["meta_action_or_ncalls_mismatch"].append(i)
            except Exception:  # noqa: BLE001
                bad["target_not_json"].append(i)
        if m[-1].get("reasoning_content"):
            bad["json_only_row_has_reasoning"].append(i)
        if not META_KEYS <= set(r.get("meta") or {}):
            bad["meta_keys_missing"].append(i)
        key = json.dumps(m[:-1], sort_keys=True)
        if key in seen and seen[key] != tgt:
            conflicts += 1
        seen.setdefault(key, tgt)
    ids = [(r["meta"].get("episode_id"), r["meta"].get("step")) for r in rows]
    dup_keys = len(ids) - len(set(ids))
    hist_bad = 0
    for r_ in rows:
        for x in r_["messages"][1:-1]:
            if x.get("role") == "assistant":
                try:
                    ok_ = isinstance(json.loads(x.get("content") or ""), list) and "```" not in x.get("content", "")
                except Exception:  # noqa: BLE001
                    ok_ = False
                if not ok_:
                    hist_bad += 1
                    break
    return {"rows": len(rows), "rows_with_nonbare_json_assistant_history": hist_bad, "violations": {k: len(v) for k, v in bad.items()}, "violation_examples": {k: v[:5] for k, v in bad.items()},
            "identical_prompt_conflicting_target_rows": conflicts, "duplicate_episode_step_keys": dup_keys,
            "ok": bool(rows) and not bad and conflicts == 0 and dup_keys == 0}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", required=True)
    ap.add_argument("--total-cap", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-kept", type=int, default=40)
    ap.add_argument("--min-train-rows", type=int, default=100)
    ap.add_argument("--min-valid-rows", type=int, default=10)
    a = ap.parse_args(argv)
    D = pathlib.Path(a.attempt)
    from benchmark.distill_exclusions import ngrams, norm

    rep = {"attempt": str(D)}
    arep = json.load(open(D / "author" / "report.json"))
    spent_a, calls_a = arep.get("spent_usd"), arep.get("calls")
    slots, valid = arep["slots"], arep["valid"]
    rejects = arep["reject_reasons"]
    api_rej = sum(v for k, v in rejects.items() if k.startswith(("api", "deadline")))
    audited = load(D / "audited.jsonl")
    sa = json.load(open(D / "semantic_audit.json"))
    solve = [json.loads(l) for l in open(D / "solve" / "verdicts.jsonl") if l.strip()]
    done = [v for v in solve if "wall_s" in v]
    sp = json.load(open(D / "solve" / "pilot_report.json"))
    kept = load(D / "filtered" / "tasks_F.jsonl")
    fr = json.load(open(D / "filtered" / "F_report.json"))
    brep = json.load(open(D / "rows" / "build_log.json"))
    rows = load(D / "rows_clean" / "train.jsonl") + load(D / "rows_clean" / "valid.jsonl")
    conf = json.load(open(D / "rows_clean" / "conflicts.json"))
    prov = json.load(open(D / "provenance.json")) if (D / "provenance.json").exists() else {"pass": False, "missing_file": True}

    rep["funnel"] = {
        "slots": slots,
        "valid_tasks": ratio(valid, slots),
        "reject_reasons": rejects,
        "reject_share": {k: round(v / slots, 4) for k, v in rejects.items()},
        "api_or_deadline_rejects": ratio(api_rej, slots),
        "valid_by_repair_rounds": arep.get("valid_by_repair_rounds"),
        "audit_kept": ratio(len(audited), valid),
        "semantic_audit_issues": len(sa),
        "solve_episodes": len(done),
        "solve_episodes_verified": ratio(sum(bool(v["valid"]) and not v.get("api_error") for v in done), len(done)),
        "solve_greedy_verified": ratio(sum(bool(v["valid"]) and not v.get("api_error") for v in done if v["sample"] == 0),
                                       sum(1 for v in done if v["sample"] == 0)),
        "solve_api_error_episodes": sum(bool(v.get("api_error")) for v in done),
        "solve_skipped_verdicts": sum(1 for v in solve if "wall_s" not in v),
        "solve_stopped_early": sp.get("stopped_early"),
        "solve_episodes_run_vs_planned": [len(done), sp["tasks_in"] * sp["samples_per_task"]] if "samples_per_task" in sp else [len(done), None],
        "attempted_slots": slots - api_rej,
        "valid_tasks_per_attempted_slot": ratio(valid, slots - api_rej),
        "kept_per_attempted_slot": ratio(len(kept), slots - api_rej),
        "tasks_with_any_verified": ratio(sp["tasks_with_any_verified"], sp["tasks_in"]),
        "after_heldout_filter": {"verified_in": fr["verified"], "dropped": len(fr["dropped_vs_other_banks"]),
                                 "dropped_by_kind": dict(collections.Counter(d["why"][0] for d in fr["dropped_vs_other_banks"])),
                                 "kept_tasks": fr["F"]},
        "kept_per_authored_slot": ratio(len(kept), slots),
        "rows": {"train": conf["rows_out"]["train"], "valid": conf["rows_out"]["valid"],
                 "builder_train": brep["train_rows"], "builder_valid": brep["valid_rows"], "conflict_dedupe": conf, "tasks_used": brep["tasks_used"],
                 "mix_final": brep["mix_final"], "unparsed_dropped": brep["unparsed_dropped"],
                 "failed_action_targets_dropped": brep["failed_action_targets_dropped"],
                 "ungrounded_targets_dropped": brep["ungrounded_targets_dropped"],
                 "duplicate_rows_dropped": brep["duplicate_rows_dropped"], "quarantined_tasks": len(brep["quarantined_tasks"]),
                 "token_audit": brep.get("token_audit"), "leak_scan": brep["leak_scan"]},
    }
    files = [D / "author" / "log" / "calls.jsonl", D / "solve" / "calls.jsonl", D / "smoke" / "log" / "calls.jsonl", D / "judge" / "log" / "calls.jsonl"]
    led = ledger(files)
    n_kept = max(1, len(kept))
    rep["spend"] = {"ledger_note": "counts logged successful responses only; failed or retried requests the provider may have billed are not visible here",
                    **led, "author_reported_usd": spent_a, "author_reported_calls": calls_a,
                    "usd_per_kept_task": round(led["cost_usd"] / n_kept, 4),
                    "usd_per_row": round(led["cost_usd"] / max(1, len(rows)), 5), "total_cap_usd": a.total_cap}
    rep["diversity_kept_tasks"] = diversity(kept, ngrams, norm)
    per_task = collections.Counter(x["meta"]["task_id"] for x in rows)
    calls = sum(x["meta"]["n_calls"] for x in rows)
    rep["rows_clean_mix"] = {"rows": len(rows), "tasks": len(per_task), "rows_per_task_mean": round(len(rows) / max(1, len(per_task)), 2),
                             "max_rows_one_task": max(per_task.values()) if per_task else 0,
                             "stop_share": round(sum(x["meta"]["action"] == "end_turn" for x in rows) / max(1, len(rows)), 4),
                             "lookup_share_of_calls": round(sum(x["meta"]["lookup_calls"] for x in rows) / calls, 4) if calls else None,
                             "note": "computed from rows_clean (after conflict dedupe); the builder's mix_final is pre-dedupe"}
    rep["row_conformance"] = row_conformance(rows)
    rep["provenance"] = {k: prov[k] for k in ("n_rows", "n_train_tasks", "n_heldout_tasks", "pass") if k in prov}
    rep["provenance"]["failures"] = {k: (len(prov[k]) if isinstance(prov[k], list) else prov[k]) for k in
                                     ("id_overlap_with_heldout", "near_duplicate_pairs", "family_signature_overlap", "rows_with_heldout_8gram",
                                      "rows_with_heldout_turn_ratio_ge_thr", "tasks_rejected_by_check_task", "row_turns_rejected_by_check_task",
                                      "train_tasks_missing_text", "heldout_tasks_without_text", "rows_without_plain_user_turn_or_empty_message",
                                      "tasks_without_gold") if k in prov}
    rep["provenance"]["benchmark_scan"] = prov.get("benchmark_scan")
    rep["provenance"]["missing_file"] = prov.get("missing_file", False)
    valid_eps = {v["episode_id"] for v in done if v["valid"] and not v.get("api_error")}
    verified_ids = set(json.load(open(D / "filtered" / "verified_ids.json")))
    kept_ids = {t["id"] for t in kept}
    rows_ok = all(r["meta"]["episode_id"] in valid_eps and r["meta"]["task_id"] in verified_ids and r["meta"]["task_id"] in kept_ids for r in rows)
    pinned = "deepseek/deepseek-v4.1-flash-1017"
    flags = {}
    for name, p in (("author", D / "author" / "log" / "guard_flags.json"), ("solve", D / "solve" / "guard_flags.json")):
        flags[name] = json.load(open(p)) if p.exists() else {"missing_file": True, "missing_cost_responses": 1, "budget_hit": True}
    rep["guard_flags"] = flags
    earlier = 0.0
    earlier_missing = 0
    for other in sorted(D.parent.iterdir()):
        # every earlier timestamped attempt (Part 1, G2 smoke, G2 runs, ...), same file list as this attempt incl. the judge; not replay copies
        if not other.is_dir() or other.resolve() == D.resolve() or not re.match(r"^\d{8}_\d{6}_", other.name):
            continue
        e = ledger(list(g2_calls_files(other)))         # every calls.jsonl of that attempt; chunk logs of an aborted G2 attempt count, an aggregated copy does not
        earlier += e["cost_usd"]; earlier_missing += e["calls_missing_cost"]
    rep["spend"]["earlier_attempts_usd"] = round(earlier, 4)
    rep["spend"]["earlier_attempts_calls_missing_cost"] = earlier_missing
    rep["acceptance"] = {
        "A2_rows_only_from_checker_verified_episodes": bool(rows) and rows_ok,
        "A3_row_format_ok": rep["row_conformance"]["ok"],
        "A4_single_pinned_model_and_provider": led["api_models"] == [pinned] and len(led["api_providers"]) == 1
                                                and "deepinfra" in led["api_providers"][0].lower(),
        "A5_cost_complete_and_under_cap": led["calls_missing_cost"] == 0 and earlier_missing == 0 and led["cost_usd"] + earlier <= a.total_cap
                                          and all(f["missing_cost_responses"] == 0 for f in flags.values()),
        "A6_api_failures_low_and_solve_not_stopped": rep["funnel"]["api_or_deadline_rejects"]["rate"] is not None and rep["funnel"]["api_or_deadline_rejects"]["rate"] <= 0.02
                                                     and sp.get("stopped_early") is None and not any(f["budget_hit"] for f in flags.values()),
        "E0_sufficient_evidence": len(kept) >= a.min_kept and rep["funnel"]["rows"]["train"] >= a.min_train_rows and rep["funnel"]["rows"]["valid"] >= a.min_valid_rows,
        "provenance_pass": bool(prov.get("pass")),
    }
    rep["acceptance"]["all_pass_excluding_A1"] = all(rep["acceptance"].values())
    out = pathlib.Path(a.out) if a.out else D / "YIELD_REPORT.json"
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps({"funnel": {k: rep["funnel"][k] for k in ("slots", "valid_tasks", "tasks_with_any_verified", "kept_per_authored_slot")},
                      "rows": rep["funnel"]["rows"]["train"] + rep["funnel"]["rows"]["valid"], "spend_usd": led["cost_usd"],
                      "acceptance": rep["acceptance"]}, indent=1))
    return 0 if rep["acceptance"]["all_pass_excluding_A1"] else 4


if __name__ == "__main__":
    sys.exit(main())
