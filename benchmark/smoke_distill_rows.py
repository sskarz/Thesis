"""Offline smoke for the row exporter (synthetic solve logs, no network).

    uv run python -m benchmark.smoke_distill_rows
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from . import distill_rows as R


def _write(d: Path, n_tasks: int = 6) -> None:
    d.mkdir(parents=True)
    verdicts, calls = [], []
    for t in range(n_tasks):
        for s in range(3):  # 3 verified samples per task; exporter keeps 2
            eid = f"task_{t}__s{s}"
            verdicts.append({"episode_id": eid, "task_id": f"task_{t}", "sample": s,
                             "valid": True, "classes": ["TradingBot"]})
            for step, txt in enumerate([
                    '[{"function_name": "get_account_info", "parameters": {}}]', "[]"]):
                calls.append({"episode_id": eid, "response_text": txt, "ts": "2026-11-06T00:00:00+00:00",
                              "api_model": "m-1", "api_provider": "P",
                              "messages": [{"role": "system", "content": "sys"},
                                           {"role": "user", "content": f"novel request number {t} {step} about quokka seating"}]})
    verdicts.append({"episode_id": "bad__s0", "task_id": "bad", "sample": 0, "valid": False, "classes": []})
    calls.append({"episode_id": "bad__s0", "response_text": "[]", "messages": []})
    (d / "verdicts.jsonl").write_text("\n".join(map(json.dumps, verdicts)) + "\n")
    (d / "calls.jsonl").write_text("\n".join(map(json.dumps, calls)) + "\n")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d, out = Path(tmp) / "solve", Path(tmp) / "out"
        _write(d)
        log = R.build([d], out, max_samples_per_task=2, stop_cap=0.4)
        assert log["verified_episodes"] == 18 and log["episodes_used"] == 12, log
        assert log["tasks_used"] == 6 and log["mix_before_stop_cap"]["stop_share"] == 0.5
        assert log["mix_final"]["stop_share"] <= 0.4 and log["stop_rows_dropped"] > 0
        tr = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
        va = [json.loads(l) for l in (out / "valid.jsonl").read_text().splitlines()]
        assert tr and va and not ({r["meta"]["task_id"] for r in tr} & {r["meta"]["task_id"] for r in va})
        assert tr[0]["messages"][-1]["role"] == "assistant"
        assert all(r["messages"][-1]["content"] == "[]" or r["messages"][-1]["content"].startswith("[")
                   for r in tr)
        assert log["leak_scan"]["id_rows"] == 0 and "No BFCL" in log["statement"]
        # a leaked row stops the build
        bad = json.loads((d / "calls.jsonl").read_text().splitlines()[0])
        from . import distill_exclusions as X
        bad["messages"][1]["content"] = next(iter(X.load_excluded()["texts"].values()))
        lines = (d / "calls.jsonl").read_text().splitlines()
        lines[0] = json.dumps(bad)
        (d / "calls.jsonl").write_text("\n".join(lines) + "\n")
        try:
            R.build([d], Path(tmp) / "out2")
            raise AssertionError("leak should abort the build")
        except SystemExit:
            assert not (Path(tmp) / "out2" / "train.jsonl").exists()
    print("smoke_distill_rows: OK")


if __name__ == "__main__":
    main()
