"""Offline smoke: teacher solves a validated task end to end against a mock API.

    uv run python -m benchmark.smoke_distill_solve
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path

from . import distill_solve as S
from . import distill_tasks as T
from . import teacher_api
from .smoke_distill_tasks import GOOD
from .smoke_teacher_api import FAKE_KEY, _H, _ok, _serve


def _reply(text: str) -> tuple:
    r = _ok(text)
    r[1]["usage"]["cost"] = 0.002
    return r


def main() -> None:
    task, why = T.validate(copy.deepcopy(GOOD), ["GorillaFileSystem"], "distill_smoke_solve_0001")
    assert task, why
    call = lambda n, **kw: {"function_name": n, "parameters": kw}  # noqa: E731
    script = [
        _reply(json.dumps([call("cd", folder="home"), call("cat", file_name="zebra_notes.txt")])),
        _reply("[]"),
        _reply(json.dumps([call("mv", source="zebra_notes.txt", destination="plans")])),
        _reply("[]"),
    ]
    os.environ[teacher_api.OPENROUTER_KEY_ENV] = FAKE_KEY
    srv, base = _serve(script)
    try:
        with tempfile.TemporaryDirectory() as d:
            tasks = Path(d) / "tasks.jsonl"
            tasks.write_text(json.dumps(task) + "\n")
            rep = S.pilot(tasks, Path(d) / "logs", max_cost=1.0, max_minutes=5, workers=1,
                          fc_mode="iffc-ooda-card", limit=None, base_url=base)
            assert rep["episodes_run"] == 1 and rep["verified"] == 1, rep
            assert rep["api_calls"] == 4 and rep["tokens"]["prompt_tokens"] == 400
            assert rep["decision_mix"]["stop_share"] == 0.5
            assert rep["decision_mix"]["tool_calls"] == 3
            assert rep["decision_mix"]["lookup_calls"] == 1
            assert rep["api_models"] == ["DeepSeek-V4.1-Flash-SMOKE"]
            assert abs(rep["cost_usd_solve"] - 0.008) < 1e-9
            # the harness card reached the model (what the student will see too)
            sent = json.dumps(_H.seen[0]["body"]["messages"])
            assert "[OODA]" in sent or "OODA" in sent
            v = json.loads((Path(d) / "logs" / "verdicts.jsonl").read_text().splitlines()[0])
            assert v["valid"] and not v["flags"]
            for p in (Path(d) / "logs").iterdir():
                assert FAKE_KEY not in p.read_text()

        # a wrong answer is caught by the stock checker
        srv2_script = [_reply(json.dumps([call("cd", folder="home")])), _reply("[]"), _reply("[]")]
        srv.shutdown(); srv.server_close()
        srv, base = _serve(srv2_script)
        with tempfile.TemporaryDirectory() as d:
            tasks = Path(d) / "tasks.jsonl"
            t2 = copy.deepcopy(task); t2["id"] = "distill_smoke_solve_0002"
            tasks.write_text(json.dumps(t2) + "\n")
            rep = S.pilot(tasks, Path(d) / "logs", max_cost=1.0, max_minutes=5, workers=1,
                          fc_mode="iffc-ooda-card", limit=None, base_url=base)
            assert rep["verified"] == 0 and rep["fail_errors"], rep
    finally:
        os.environ.pop(teacher_api.OPENROUTER_KEY_ENV, None)
        srv.shutdown(); srv.server_close()
    print("smoke_distill_solve: OK")


if __name__ == "__main__":
    main()
