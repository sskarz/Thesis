#!/usr/bin/env python3
"""Offline self-test for yield_report.py (synthetic fixtures; run with the Thesis venv from the worktree root)."""
import json
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import yield_report as yr  # noqa: E402

PIN = "deepseek/deepseek-v4.1-flash-1017"


def task(i, user, calls, cls="MathAPI"):
    return {"id": f"distill_y_{i:04d}", "question": [[{"role": "user", "content": user}]], "ground_truth": [calls],
            "involved_classes": [cls], "initial_config": {}}


def row(tid, ep, step, target, action="execute", n=1):
    return {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": f"u {tid} {step}"},
                         {"role": "assistant", "content": target}],
            "meta": {"task_id": tid, "episode_id": ep, "sample": 0, "step": step, "action": action, "n_calls": n, "lookup_calls": 0,
                     "classes": ["MathAPI"], "exercised": ["MathAPI"]}}


GOOD = json.dumps([{"function_name": "add", "parameters": {"a": 1, "b": 2}}], indent=2)


def build(d, rows, cost=0.002, model=PIN, provider="DeepInfra", prov_pass=True, api_err=False, stopped=None, budget_hit=False):
    for sub in ("author/log", "solve", "filtered", "rows", "smoke/log"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    json.dump({"slots": 100, "valid": 12, "valid_by_repair_rounds": {"0": 12}, "reject_reasons": {"gold_error_t0": 80, "dup_in_batch": 8},
               "per_class": {}, "spent_usd": 0.004, "calls": 2}, open(d / "author" / "report.json", "w"))
    tasks = [task(1, "please add one and two now", ["add(a=1, b=2)"]), task(2, "multiply three by four please", ["multiply(a=3, b=4)"])]
    for p in (d / "audited.jsonl", d / "filtered" / "tasks_F.jsonl"):
        p.write_text("".join(json.dumps(t) + "\n" for t in tasks))
    json.dump({}, open(d / "semantic_audit.json", "w"))
    v = [{"episode_id": f"distill_y_{i:04d}__s0", "task_id": f"distill_y_{i:04d}", "sample": 0, "valid": True, "api_error": api_err, "wall_s": 1.0} for i in (1, 2)]
    (d / "solve" / "verdicts.jsonl").write_text("".join(json.dumps(x) + "\n" for x in v))
    json.dump({"tasks_in": 2, "tasks_with_any_verified": 2, "samples_per_task": 1, "stopped_early": stopped}, open(d / "solve" / "pilot_report.json", "w"))
    json.dump({"verified": 2, "dropped_vs_other_banks": [], "F": 2}, open(d / "filtered" / "F_report.json", "w"))
    json.dump(["distill_y_0001", "distill_y_0002"], open(d / "filtered" / "verified_ids.json", "w"))
    json.dump({"train_rows": len(rows), "valid_rows": 0, "tasks_used": 2, "mix_final": {}, "unparsed_dropped": 0, "failed_action_targets_dropped": 0,
               "ungrounded_targets_dropped": 0, "duplicate_rows_dropped": 0, "quarantined_tasks": {}, "token_audit": {"max": 100}, "leak_scan": {"id_rows": 0}},
              open(d / "rows" / "build_log.json", "w"))
    (d / "rows" / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "rows" / "valid.jsonl").write_text("")
    (d / "rows_clean").mkdir(exist_ok=True)
    (d / "rows_clean" / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "rows_clean" / "valid.jsonl").write_text("")
    json.dump({"prompts_with_conflicting_targets": 0, "rows_in": {"train": len(rows), "valid": 0}, "rows_dropped": {"train": 0, "valid": 0},
               "rows_out": {"train": len(rows), "valid": 0}}, open(d / "rows_clean" / "conflicts.json", "w"))
    json.dump({"n_rows": len(rows), "n_train_tasks": 2, "n_heldout_tasks": 5, "pass": prov_pass}, open(d / "provenance.json", "w"))
    json.dump({"missing_cost_responses": 0, "budget_hit": budget_hit}, open(d / "author" / "log" / "guard_flags.json", "w"))
    json.dump({"missing_cost_responses": 0, "budget_hit": False}, open(d / "solve" / "guard_flags.json", "w"))
    call = {"cost_usd": cost, "api_model": model, "api_provider": provider, "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
    (d / "author" / "log" / "calls.jsonl").write_text(json.dumps(call) + "\n")
    (d / "solve" / "calls.jsonl").write_text(json.dumps(call) + "\n")


def run(d, minimum=True, **kw):
    build(d, **kw)
    extra = ["--min-kept", "1", "--min-train-rows", "1", "--min-valid-rows", "0"] if minimum else []
    r = subprocess.run([sys.executable, str(HERE / "yield_report.py"), "--attempt", str(d), *extra], capture_output=True, text=True, cwd=ROOT)
    rep = json.load(open(d / "YIELD_REPORT.json"))
    return r.returncode, rep


def main():
    assert yr.wilson(5, 10)[0] < 0.5 < yr.wilson(5, 10)[1] and yr.wilson(0, 0) is None
    with tempfile.TemporaryDirectory() as t:
        t = pathlib.Path(t)
        good_rows = [row("distill_y_0001", "distill_y_0001__s0", 0, GOOD), row("distill_y_0002", "distill_y_0002__s0", 0, "[]", "end_turn", 0)]
        rc, rep = run(t / "ok", rows=good_rows)
        assert rc == 0 and rep["acceptance"]["all_pass_excluding_A1"], rep["acceptance"]
        assert rep["funnel"]["valid_tasks"]["k"] == 12 and rep["funnel"]["api_or_deadline_rejects"]["k"] == 0
        # each violation must flip the right flag
        fenced = row("distill_y_0001", "distill_y_0001__s0", 0, "```json\n" + GOOD + "\n```")
        rc, rep = run(t / "fence", rows=[fenced, good_rows[1]])
        assert rc == 4 and not rep["acceptance"]["A3_row_format_ok"], rep["row_conformance"]
        bad_keys = row("distill_y_0001", "distill_y_0001__s0", 0, json.dumps([{"name": "add", "args": {}}]))
        rc, rep = run(t / "keys", rows=[bad_keys, good_rows[1]])
        assert not rep["acceptance"]["A3_row_format_ok"] and "target_not_strict_call_list" in rep["row_conformance"]["violations"]
        mism = row("distill_y_0001", "distill_y_0001__s0", 0, "[]", "execute", 1)
        rc, rep = run(t / "mism", rows=[mism, good_rows[1]])
        assert "meta_action_or_ncalls_mismatch" in rep["row_conformance"]["violations"]
        reason = dict(good_rows[0]); reason["messages"] = list(reason["messages"]); reason["messages"][-1] = dict(reason["messages"][-1], reasoning_content="x")
        rc, rep = run(t / "reason", rows=[reason, good_rows[1]])
        assert "json_only_row_has_reasoning" in rep["row_conformance"]["violations"]
        marker = row("distill_y_0001", "distill_y_0001__s0", 0, GOOD); marker["messages"][0]["content"] = "<|think|>\nsys"
        rc, rep = run(t / "marker", rows=[marker, good_rows[1]])
        assert "think_marker_already_present" in rep["row_conformance"]["violations"]
        conflict = row("distill_y_0002", "distill_y_0002__s1", 0, GOOD)
        conflict["messages"][1]["content"] = good_rows[1]["messages"][1]["content"]
        rc, rep = run(t / "conf", rows=[good_rows[0], good_rows[1], conflict])
        assert rep["row_conformance"]["identical_prompt_conflicting_target_rows"] == 1 and not rep["row_conformance"]["ok"]
        rc, rep = run(t / "model", rows=good_rows, model="deepseek/other-model")
        assert not rep["acceptance"]["A4_single_pinned_model_and_provider"]
        rc, rep = run(t / "cost", rows=good_rows, cost=None)
        assert not rep["acceptance"]["A5_cost_complete_and_under_cap"]
        rc, rep = run(t / "prov", rows=good_rows, prov_pass=False)
        assert not rep["acceptance"]["provenance_pass"] and rc == 4
        rc, rep = run(t / "apierr", rows=good_rows, api_err=True)        # rows from api-error episodes are rejected by the row-level check
        assert not rep["acceptance"]["A2_rows_only_from_checker_verified_episodes"]
        rc, rep = run(t / "norows", rows=[])
        assert not rep["acceptance"]["A2_rows_only_from_checker_verified_episodes"] and not rep["acceptance"]["A3_row_format_ok"]
        histrow = row("distill_y_0001", "distill_y_0001__s0", 0, GOOD)
        histrow["messages"] = [histrow["messages"][0], {"role": "user", "content": "u"}, {"role": "assistant", "content": "prose reply"},
                               {"role": "user", "content": "u2"}, {"role": "assistant", "content": GOOD}]
        rc, rep = run(t / "hist", rows=[histrow, good_rows[1]])
        assert rep["row_conformance"]["rows_with_nonbare_json_assistant_history"] == 1
        orphan = row("distill_y_0003", "distill_y_0003__s0", 0, GOOD)
        rc, rep = run(t / "orphan", rows=[good_rows[0], orphan])
        assert not rep["acceptance"]["A2_rows_only_from_checker_verified_episodes"]
        rc, rep = run(t / "tiny", rows=good_rows, minimum=False)          # default thresholds: tiny result cannot certify anything
        assert not rep["acceptance"]["E0_sufficient_evidence"] and rc == 4
        rc, rep = run(t / "empty", rows=[])
        assert not rep["acceptance"]["E0_sufficient_evidence"]
        rc, rep = run(t / "budget", rows=good_rows, budget_hit=True)       # latched budget stop certifies nothing
        assert not rep["acceptance"]["A6_api_failures_low_and_solve_not_stopped"]
        # earlier attempt with a cost-less call must fail A5 of a later attempt
        cur = t / "20261109_000001_part1.cur"; build(cur, rows=good_rows)
        sib = t / "20261109_000000_part1.sib"; (sib / "solve").mkdir(parents=True)
        (sib / "solve" / "calls.jsonl").write_text(json.dumps({"cost_usd": None}) + "\n")
        subprocess.run([sys.executable, str(HERE / "yield_report.py"), "--attempt", str(cur), "--min-kept", "1", "--min-train-rows", "1", "--min-valid-rows", "0"], capture_output=True, cwd=ROOT)
        rep = json.load(open(cur / "YIELD_REPORT.json"))
        assert rep["spend"]["earlier_attempts_calls_missing_cost"] == 1 and not rep["acceptance"]["A5_cost_complete_and_under_cap"]
        # an aborted G2 attempt (chunk logs, no aggregated author/log/calls.jsonl) must count in the earlier spend
        ab = t / "20261108_000000_g2full.ab"; (ab / "author" / "a" / "chunk_0" / "log").mkdir(parents=True)
        (ab / "author" / "a" / "chunk_0" / "log" / "calls.jsonl").write_text(json.dumps({"cost_usd": 0.7, "api_model": "x", "api_provider": "y"}) + "\n")
        subprocess.run([sys.executable, str(HERE / "yield_report.py"), "--attempt", str(cur), "--min-kept", "1", "--min-train-rows", "1", "--min-valid-rows", "0"], capture_output=True, cwd=ROOT)
        rep2 = json.load(open(cur / "YIELD_REPORT.json"))
        assert rep2["spend"]["earlier_attempts_usd"] >= 0.7 - 1e-9, rep2["spend"]
        rc, rep = run(t / "stopped", rows=good_rows, stopped=["budget"])
        assert not rep["acceptance"]["A6_api_failures_low_and_solve_not_stopped"]
    print("test_yield_report: OK")


if __name__ == "__main__":
    main()
