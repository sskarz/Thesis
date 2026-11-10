"""Offline smoke for rationalization: mock local server that answers differently with/without the note.

    uv run python -m benchmark.smoke_distill_rationalize
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from . import distill_rationalize as R
from . import teacher_api
from .smoke_teacher_api import _H, _serve

ACT = [{"function_name": "get_account_info", "parameters": {}}]


def _resp(content, reasoning):
    return (200, {"model": "m", "choices": [{"message": {"role": "assistant", "content": content,
            "reasoning_content": reasoning}, "finish_reason": "stop"}], "usage": {}}, {})


def main() -> None:
    row = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "check my balance"},
                        {"role": "assistant", "content": json.dumps(ACT, indent=2)}], "meta": {"task_id": "t"}}
    wrong = json.dumps([{"function_name": "get_stock_info", "parameters": {"symbol": "X"}}])
    right = json.dumps(ACT)
    # greedy wrong, 3 sampled wrong, hinted: leak, then good
    script = [_resp(wrong, "thinking a"), _resp(wrong, "b"), _resp(wrong, "c"), _resp(wrong, "d"),
              _resp(right, "Per the private note I should call it."), _resp(right, "The user wants the balance, so I call get_account_info.")]
    srv, base = _serve(script)
    try:
        with tempfile.TemporaryDirectory() as d:
            c = teacher_api.TeacherClient(teacher_api.TeacherConfig.local(base_url=base, log_dir=Path(d)),
                                          sleep=lambda s: None)
            out = R.rationalize_row(row, c)
            assert out and out["meta"]["native_method"] == "hinted", out
            last = out["messages"][-1]
            assert last["reasoning_content"].startswith("The user wants") and "private" not in last["reasoning_content"].lower()
            assert "Private note" in _H.seen[4]["body"]["messages"][-1]["content"]   # hint only in the hinted request
            assert "Private note" not in json.dumps(out["messages"][:-1])               # never in the kept prompt
    finally:
        srv.shutdown(); srv.server_close()
    srv, base = _serve([_resp(right, "I check the balance first.")])                  # greedy already right
    try:
        with tempfile.TemporaryDirectory() as d:
            c = teacher_api.TeacherClient(teacher_api.TeacherConfig.local(base_url=base, log_dir=Path(d)),
                                          sleep=lambda s: None)
            out = R.rationalize_row(row, c)
            assert out["meta"]["native_method"] == "greedy"
    finally:
        srv.shutdown(); srv.server_close()
    print("smoke_distill_rationalize: OK")


if __name__ == "__main__":
    main()
