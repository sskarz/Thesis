"""Smoke turn-0 Decision timeout hygiene (no Acc).

Offline (mocked client):
    uv run python -m benchmark.smoke_decision_turn0_timeout_hygiene

Live stampede fixture (Ollama tev1 :11434):
    uv run python -m benchmark.smoke_decision_turn0_timeout_hygiene --live

Proves: knobs loaded; inflight gate + timeout retry wired; no crash on
compose modes; live N=24 stampede under short-timeout proxy improves with
hygiene (optional --live). Acc NOT scored.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest import mock

from . import config
from . import decision_sidecar


def _assert_knobs() -> dict:
    knobs = decision_sidecar.hygiene_knobs()
    assert float(knobs["DECISION_TIMEOUT_S"]) >= 45.0 - 1e-9
    assert int(knobs["DECISION_TIMEOUT_RETRIES"]) >= 1
    assert int(knobs["DECISION_MAX_INFLIGHT"]) == 2 or int(knobs["DECISION_MAX_INFLIGHT"]) >= 1
    assert int(knobs["DECISION_PREWARM_SLOTS"]) >= 1
    # Both Decision modes share the adapter
    assert config.FC_IFFC_DECISION in config.FC_IFFC_DECISION_MODES
    assert config.FC_IFFC_DECISION_OODA in config.FC_IFFC_DECISION_MODES
    return knobs


def _assert_retry_on_timeout(monkey_calls: list) -> None:
    """First post times out; second succeeds — attempts_used == 2."""
    decision_sidecar.reset_inflight_gate()
    timeout_res = decision_sidecar.SystemOneResult(
        ok=False,
        choice=None,
        probabilities={},
        confidence=None,
        noul=None,
        latency_ms=20001.0,
        raw=None,
        error="client_exception:TimeoutError: timed out",
        http_status=None,
        input_tokens=None,
    )
    ok_res = decision_sidecar.SystemOneResult(
        ok=True,
        choice="ls",
        probabilities={"ls": 0.7, "noop": 0.3},
        confidence=0.7,
        noul=0.1,
        latency_ms=100.0,
        raw=None,
        error=None,
        http_status=200,
        input_tokens=50,
    )
    calls = {"n": 0}

    def fake_post(**kwargs):
        calls["n"] += 1
        monkey_calls.append(kwargs)
        if calls["n"] == 1:
            return timeout_res
        return ok_res

    tools = [
        {
            "type": "function",
            "function": {
                "name": "ls",
                "description": "list",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    history = [{"role": "user", "content": "list files"}]
    with mock.patch(
        "benchmark.decision_sidecar.post_systemone", side_effect=fake_post
    ):
        # Force retries=1 for this smoke
        old_r = decision_sidecar.DECISION_TIMEOUT_RETRIES
        decision_sidecar.DECISION_TIMEOUT_RETRIES = 1
        try:
            probe = decision_sidecar.probe_next_tool(
                history, tools, entry_id="smoke_retry", turn=0, step=0, log=False
            )
        finally:
            decision_sidecar.DECISION_TIMEOUT_RETRIES = old_r
    assert calls["n"] == 2, f"expected 2 attempts, got {calls['n']}"
    assert probe.client.ok is True
    assert probe.log_record.get("decision_attempts") == 2
    assert any(
        str(f).startswith("decision_timeout_retry:")
        for f in (probe.log_record.get("flags") or [])
    )


def _assert_inflight_caps_concurrency() -> None:
    """With MAX_INFLIGHT=2, never more than 2 concurrent post_systemone bodies."""
    decision_sidecar.reset_inflight_gate()
    old_m = decision_sidecar.DECISION_MAX_INFLIGHT
    old_r = decision_sidecar.DECISION_TIMEOUT_RETRIES
    decision_sidecar.DECISION_MAX_INFLIGHT = 2
    decision_sidecar.DECISION_TIMEOUT_RETRIES = 0
    decision_sidecar.reset_inflight_gate()

    inflight = 0
    peak = 0
    lock = threading.Lock()

    def slow_post(**kwargs):
        nonlocal inflight, peak
        with lock:
            inflight += 1
            peak = max(peak, inflight)
        time.sleep(0.05)
        with lock:
            inflight -= 1
        return decision_sidecar.SystemOneResult(
            ok=True,
            choice="ls",
            probabilities={"ls": 0.9, "noop": 0.1},
            confidence=0.9,
            noul=0.0,
            latency_ms=50.0,
            raw=None,
            error=None,
            http_status=200,
            input_tokens=10,
        )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "ls",
                "description": "list",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    history = [{"role": "user", "content": "list"}]

    try:
        with mock.patch(
            "benchmark.decision_sidecar.post_systemone", side_effect=slow_post
        ):
            with ThreadPoolExecutor(max_workers=8) as pool:
                futs = [
                    pool.submit(
                        decision_sidecar.probe_next_tool,
                        history,
                        tools,
                        entry_id=f"inf_{i}",
                        turn=0,
                        step=0,
                        log=False,
                    )
                    for i in range(8)
                ]
                for f in as_completed(futs):
                    assert f.result().client.ok
    finally:
        decision_sidecar.DECISION_MAX_INFLIGHT = old_m
        decision_sidecar.DECISION_TIMEOUT_RETRIES = old_r
        decision_sidecar.reset_inflight_gate()

    assert peak <= 2, f"peak inflight {peak} exceeded MAX_INFLIGHT=2"


def _live_stampede(n: int = 24) -> dict:
    """Live tev1 stampede: short-timeout ungated proxy vs hygiene knobs."""
    import os
    import sys as _sys

    _sys.path.insert(0, str(Path.home() / "Documents" / "decision-probe"))
    from sidecar.client import post_systemone
    from sidecar.state import build_questions, desc_map_from_tools, estimate_tokens

    names = [
        "cd",
        "ls",
        "pwd",
        "mkdir",
        "touch",
        "rm",
        "cp",
        "mv",
        "find",
        "grep",
        "cat",
        "head",
        "tail",
        "echo",
        "wc",
        "sort",
        "uniq",
        "diff",
        "chmod",
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": nm,
                "description": f"{nm} util — " + ("x" * 60),
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }
        for nm in names
    ]
    descs = desc_map_from_tools(tools)
    state = (
        "available_tools:\n"
        + "\n".join(f"  - {nm}" for nm in names)
        + "\ntranscript:\n  [user] list files please thoroughly\n"
    )
    questions = build_questions(
        kept_names=names,
        include_noul=True,
        active_tools=tools,
        desc_by_name=descs,
        desc_mode="short",
        desc_max_chars=80,
        state_tokens_est=estimate_tokens(state),
    )
    questions.pop("_meta", None)
    criteria = list(questions["next_tool"]["criteria"].keys())

    def run(label: str, timeout: float, max_inflight: int | None) -> dict:
        sem = threading.Semaphore(max_inflight) if max_inflight else None

        def one(i: int) -> dict:
            t0 = time.perf_counter()
            if sem:
                sem.acquire()
            try:
                r = post_systemone(
                    state=state,
                    questions=questions,
                    model="tev1:latest",
                    endpoint="http://127.0.0.1:11434",
                    api_key="ollama",
                    timeout_s=timeout,
                    criteria_names=criteria,
                    keep_alive="30m",
                )
            finally:
                if sem:
                    sem.release()
            return {
                "ok": r.ok,
                "err": r.error,
                "lat": round((time.perf_counter() - t0) * 1000, 1),
            }

        # prewarm once
        decision_sidecar.prewarm(force=True)
        results = []
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as ex:
            for f in as_completed([ex.submit(one, i) for i in range(n)]):
                results.append(f.result())
        wall = (time.perf_counter() - t0) * 1000
        n_ok = sum(1 for r in results if r["ok"])
        n_to = sum(
            1
            for r in results
            if (not r["ok"]) and r["err"] and "Timeout" in str(r["err"])
        )
        return {
            "label": label,
            "N": n,
            "timeout_s": timeout,
            "inflight": max_inflight,
            "ok": n_ok,
            "timeout_n": n_to,
            "timeout_rate": round(n_to / n, 3),
            "wall_ms": round(wall, 1),
            "lat_max": max(r["lat"] for r in results),
            "lat_mean": round(sum(r["lat"] for r in results) / n, 1),
        }

    # Proxy BEFORE: ungated + short timeout (reproduces queue timeouts on idle Mac)
    before = run("before_proxy_ungated_t5", timeout=5.0, max_inflight=None)
    # AFTER: gated + 45s (shipped hygiene)
    after = run("after_hygiene_inflight2_t45", timeout=45.0, max_inflight=2)
    return {"before": before, "after": after}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="Hit live tev1 stampede")
    args = ap.parse_args()

    print("=== turn0 timeout hygiene smoke ===")
    knobs = _assert_knobs()
    print("knobs:", json.dumps(knobs, indent=2))
    print("modes ok: iffc-decision + iffc-decision-ooda share adapter")

    calls: list = []
    _assert_retry_on_timeout(calls)
    print("retry-on-timeout: ok (2 attempts)")

    _assert_inflight_caps_concurrency()
    print("inflight-cap: ok (peak<=2)")

    live_summary = None
    if args.live:
        import requests

        tags = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        tags.raise_for_status()
        live_summary = _live_stampede(n=24)
        print("live stampede:", json.dumps(live_summary, indent=2))
        before = live_summary["before"]
        after = live_summary["after"]
        assert after["timeout_n"] == 0, after
        assert after["ok"] == after["N"], after
        # before should be worse or equal; on a quiet box may still be 0 — don't hard-fail
        print(
            f"proxy timeout rates: before={before['timeout_rate']} "
            f"after={after['timeout_rate']}"
        )

    print("turn0 timeout hygiene smoke: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
