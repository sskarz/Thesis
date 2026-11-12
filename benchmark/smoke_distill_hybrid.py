"""Offline smoke: scripted student prefix + mock teacher continuation, verified by the stock checker.

    uv run python -m benchmark.smoke_distill_hybrid
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path

from . import distill_hybrid as H
from . import distill_tasks as T
from . import teacher_api
from .smoke_distill_tasks import GOOD
from .smoke_teacher_api import FAKE_KEY, _H, _ok, _serve


def main() -> None:
    task, why = T.validate(copy.deepcopy(GOOD), ["GorillaFileSystem"], "distill_smoke_hy_0001")
    assert task, why
    cd = [{"function_name": "cd", "parameters": {"folder": "home"}},
          {"function_name": "cat", "parameters": {"file_name": "zebra_notes.txt"}}]
    mv = [{"function_name": "mv", "parameters": {"source": "zebra_notes.txt", "destination": "plans"}}]
    student_prefix = [{"episode_id": "e1", "response_text": json.dumps(cd), "reasoning_content": "student thinks"}]
    os.environ[teacher_api.OPENROUTER_KEY_ENV] = FAKE_KEY
    srv, base = _serve([_ok("[]"), _ok(json.dumps(mv)), _ok("[]")])  # teacher: stop, mv, stop
    try:
        with tempfile.TemporaryDirectory() as d:
            cfg = teacher_api.TeacherConfig.openrouter(base_url=base, log_dir=Path(d) / "lg", max_cost_usd=1.0)
            c = H.HybridClient(cfg, sleep=lambda s: None)
            teacher_api.install(c)
            c.set_script("distill_smoke_hy_0001__h1__s0", student_prefix)
            t = dict(task); t["id"] = "distill_smoke_hy_0001__h1"
            from .distill_solve import solve_one
            v = solve_one(t, c, "iffc-ooda-card", 0, None)
            teacher_api.install(None)
            assert v["valid"], v
            # the scripted first reply never reached the teacher; its reasoning is in the later history
            assert len(_H.seen) == 3, len(_H.seen)
            assert "student thinks" in json.dumps(_H.seen[0]["body"]["messages"])
            calls = [json.loads(l) for l in (Path(d) / "lg" / "calls.jsonl").read_text().splitlines()]
            assert len(calls) == 3 and calls[0]["response_text"] == "[]"
    finally:
        teacher_api.install(None)
        os.environ.pop(teacher_api.OPENROUTER_KEY_ENV, None)
        srv.shutdown(); srv.server_close()
    print("smoke_distill_hybrid: OK")


if __name__ == "__main__":
    main()
