"""Offline smoke for on-policy relabeling: synthetic student logs + a scripted mock teacher.

    uv run python -m benchmark.smoke_distill_onpolicy
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path

from . import distill_onpolicy as OP
from . import distill_tasks as T
from . import teacher_api
from .smoke_distill_tasks import GOOD
from .smoke_teacher_api import FAKE_KEY, _H, _ok, _serve

CALLS0 = [{"function_name": "cd", "parameters": {"folder": "home"}},
          {"function_name": "cat", "parameters": {"file_name": "zebra_notes.txt"}}]


def main() -> None:
    task, why = T.validate(copy.deepcopy(GOOD), ["GorillaFileSystem"], "distill_smoke_op_0001")
    assert task, why
    sysm = {"role": "system", "content": "sys prompt with tools"}
    u0 = {"role": "user", "content": "Go into the home folder and show me what the zebra notes say."}
    res = {"role": "user", "content": "Function execution results:\n- cat(file_name='zebra_notes.txt') -> {\"file_content\": \"stripes\"}\n\nRespond with a JSON list"}
    u1 = {"role": "user", "content": "Now move those notes into the plans folder for me please."}
    a0 = {"role": "assistant", "content": "thinking... " + json.dumps(CALLS0)}
    recs = [
        {"episode_id": "e1", "messages": [sysm, u0], "response_text": json.dumps(CALLS0)},
        {"episode_id": "e1", "messages": [sysm, u0, a0, res], "response_text": "[]"},
        {"episode_id": "e1", "messages": [sysm, u0, a0, res, {"role": "assistant", "content": "[]"}, u1], "response_text": "[]"},
    ]
    verdict = {"episode_id": "e1", "task_id": task["id"], "valid": False, "sample": 0,
               "result": [[["cd(folder='home')", "cat(file_name='zebra_notes.txt')"]], []], "classes": ["GorillaFileSystem"]}
    mv = [{"function_name": "mv", "parameters": {"source": "zebra_notes.txt", "destination": "plans"}}]
    os.environ[teacher_api.OPENROUTER_KEY_ENV] = FAKE_KEY
    srv, base = _serve([_ok(json.dumps([CALLS0[0]])), _ok("[]"), _ok(json.dumps(mv))])
    try:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "stu").mkdir()
            (d / "stu" / "calls.jsonl").write_text("\n".join(map(json.dumps, recs)) + "\n")
            (d / "stu" / "verdicts.jsonl").write_text(json.dumps(verdict) + "\n")
            (d / "tasks.jsonl").write_text(json.dumps(task) + "\n")
            cfg = teacher_api.TeacherConfig.openrouter(base_url=base, log_dir=d / "lg", max_cost_usd=1.0)
            client = teacher_api.TeacherClient(cfg, sleep=lambda s: None)
            rep = OP.label_dir(d / "tasks.jsonl", d / "stu", d / "out", client, workers=1)
            assert rep["requests"] == 3 and rep["stats"]["act_ok"] == 2 and rep["stats"]["stop_correct"] == 1, rep
            labels = [json.loads(l) for l in (d / "out" / "labels.jsonl").read_text().splitlines()]
            assert [x["decision"] for x in labels] == ["keep"] * 3
            assert labels[0]["same_as_student"] is False and labels[1]["turn"] == 0 and labels[2]["turn"] == 1
            # the teacher saw the student's exact messages, reasoning text included
            assert "thinking..." in json.dumps(_H.seen[1]["body"]["messages"])
        # a parse-retry prompt must not count as a new task turn
        from .iffc import IFFC_RETRY_USER_MESSAGE
        assert OP.turn_index([sysm, u0, {"role": "user", "content": IFFC_RETRY_USER_MESSAGE}]) == 0
        assert OP.turn_index([sysm, u0, a0, res, u1]) == 1
        # label checks in isolation
        flat = [["cd(folder='home')", "cat(file_name='zebra_notes.txt')"]]
        a_early = {"executed_before": 0, "turn": 0, "rec": {"messages": [sysm, u0]}}
        assert OP.check_label(task, a_early, flat, [0], [])[1] == "stop_early"
        a_done = {"executed_before": 1, "turn": 0, "rec": {"messages": [sysm, u0, a0, res]}}
        assert OP.check_label(task, a_done, flat, [0], CALLS0[:1])[1] == "act_when_done"
        bad = [{"function_name": "cat", "parameters": {"file_name": "nope.txt"}}]
        a_mid = {"executed_before": 0, "turn": 0, "rec": {"messages": [sysm, u0]}}
        assert OP.check_label(task, a_mid, flat, [0], bad)[1] == "ungrounded"
    finally:
        os.environ.pop(teacher_api.OPENROUTER_KEY_ENV, None)
        srv.shutdown(); srv.server_close()
    print("smoke_distill_onpolicy: OK")


if __name__ == "__main__":
    main()
