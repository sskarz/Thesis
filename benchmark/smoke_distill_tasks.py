"""Offline smoke for teacher-authored task validation (no network, no key).

    uv run python -m benchmark.smoke_distill_tasks
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path

from . import distill_exclusions as X
from . import distill_tasks as T
from . import teacher_api
from .smoke_teacher_api import FAKE_KEY, _H, _ok, _serve

GOOD = {
    "initial_config": {"GorillaFileSystem": {"root": {"alex": {"type": "directory", "contents": {
        "home": {"type": "directory", "contents": {
            "zebra_notes.txt": {"type": "file", "content": "stripes and savanna"},
            "plans": {"type": "directory", "contents": {}}}}}}}}},
    "turns": [
        {"user": "Go into the home folder and show me what the zebra notes say.",
         "gold_calls": ["cd(folder='home')", "cat(file_name='zebra_notes.txt')"]},
        {"user": "Now move those notes into the plans folder for me please.",
         "gold_calls": ["mv(source='zebra_notes.txt', destination='plans')"]},
    ],
}


def check_validate() -> None:
    cls = ["GorillaFileSystem"]
    task, why = T.validate(copy.deepcopy(GOOD), cls, "distill_smoke_0001")
    assert task and not why, why
    assert task["ground_truth"][1] == ["mv(source='zebra_notes.txt', destination='plans')"]

    bad = copy.deepcopy(GOOD)
    bad["turns"][1]["gold_calls"] = ["cat(file_name='nope.txt')"]
    assert T.validate(bad, cls, "distill_smoke_0002")[1].startswith("gold_error_t1")

    one = copy.deepcopy(GOOD); one["turns"] = one["turns"][:1]
    assert T.validate(one, cls, "distill_smoke_0003")[1] == "bad_turn_count"

    wrong = copy.deepcopy(GOOD); wrong["initial_config"] = {"TradingBot": {}}
    assert T.validate(wrong, cls, "distill_smoke_0004")[1] == "bad_initial_config"

    copied = copy.deepcopy(GOOD)
    copied["turns"][0]["user"] = next(iter(X.load_excluded()["texts"].values()))
    assert T.validate(copied, cls, "distill_smoke_0005")[1].startswith("leak:")

    dup = T.validate(copy.deepcopy(GOOD), cls, "distill_smoke_0006",
                     accepted_turns=[[t["user"] for t in GOOD["turns"]]])
    assert dup[1] == "dup_in_batch"


def check_parse() -> None:
    assert T.parse_authored("no json here") is None
    lit = "{'initial_config': {'TradingBot': {'orders': {12345: {'x': None, 'ok': True}}}}, 'turns': []}"
    got = T.parse_authored(lit)
    assert got and got["initial_config"]["TradingBot"]["orders"] == {"12345": {"x": None, "ok": True}}, got
    assert "Expecting" in T.parse_error('{"a": 1,, "turns": []}')
    obj = T.parse_authored("Sure!\n```json\n" + json.dumps(GOOD) + "\n```")
    assert obj and len(obj["turns"]) == 2


def check_author_loop() -> None:
    os.environ[teacher_api.OPENROUTER_KEY_ENV] = FAKE_KEY
    reply = _ok(json.dumps(GOOD), reasoning="hidden")
    reply[1]["usage"]["cost"] = 0.001
    srv, base = _serve([reply, _ok("not json"), reply, reply])
    prev_combos = T.class_combos
    T.class_combos = lambda: [("GorillaFileSystem",)]
    try:
        with tempfile.TemporaryDirectory() as d:
            cfg = teacher_api.TeacherConfig.openrouter(
                base_url=base, log_dir=Path(d) / "logs", max_cost_usd=1.0)
            c = teacher_api.TeacherClient(cfg, sleep=lambda s: None)
            out = Path(d) / "tasks.jsonl"
            rep = T.author(3, out, client=c, seed=1, workers=1, prefix="distill_smoke_a")
            assert rep["valid"] == 1, rep
            assert rep["reject_reasons"] == {"dup_in_batch": 2}, rep  # slot 1 repaired, then duplicate
            assert len(_H.seen) == 4
            rows = [json.loads(l) for l in out.read_text().splitlines()]
            assert rows[0]["ground_truth"] and rows[0]["meta"]["classes"] == ["GorillaFileSystem"]
            # prompt shown to the teacher names tools + state schema, not BFCL tasks
            sent = _H.seen[0]["body"]["messages"][1]["content"]
            assert "cd(folder: string)" in sent and "STATE SCHEMA" in sent
            assert "Rename the zebra" not in sent
            assert (Path(d) / "logs" / "calls.jsonl").exists()
    finally:
        T.class_combos = prev_combos
        os.environ.pop(teacher_api.OPENROUTER_KEY_ENV, None)
        srv.shutdown(); srv.server_close()


def check_repair_round() -> None:
    os.environ[teacher_api.OPENROUTER_KEY_ENV] = FAKE_KEY
    bad = copy.deepcopy(GOOD)
    bad["turns"][1]["gold_calls"] = ["cat(file_name='nope.txt')"]
    srv, base = _serve([_ok(json.dumps(bad)), _ok(json.dumps(GOOD))])
    prev_combos = T.class_combos
    T.class_combos = lambda: [("GorillaFileSystem",)]
    try:
        with tempfile.TemporaryDirectory() as d:
            cfg = teacher_api.TeacherConfig.openrouter(base_url=base, log_dir=Path(d) / "l")
            c = teacher_api.TeacherClient(cfg, sleep=lambda s: None)
            rep = T.author(1, Path(d) / "t.jsonl", client=c, seed=2, workers=1, prefix="distill_smoke_r")
            assert rep["valid"] == 1 and rep["valid_by_repair_rounds"] == {1: 1}, rep
            fb = _H.seen[1]["body"]["messages"][-1]["content"]
            assert "Validation failed" in fb and "nope.txt" in fb and "No such file" in fb, fb
    finally:
        T.class_combos = prev_combos
        os.environ.pop(teacher_api.OPENROUTER_KEY_ENV, None)
        srv.shutdown(); srv.server_close()


def main() -> None:
    check_repair_round()
    check_validate()
    check_parse()
    check_author_loop()
    print("smoke_distill_tasks: OK")


if __name__ == "__main__":
    main()
