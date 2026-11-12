#!/usr/bin/env python3
"""A reply that makes parse/validate raise must cost one slot (reason internal:<Type>), not the chunk. Run from a tree whose benchmark/ is the patched one."""
import contextlib, pathlib, sys, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from benchmark import distill_tasks as D

class Stub:
    @contextlib.contextmanager
    def episode(self, *a, **k): yield
    def post_chat(self, p): return {"choices": [{"message": {"content": "{}"}}]}

calls = {"n": 0}
def boom(text):
    calls["n"] += 1
    if calls["n"] % 2:
        raise RuntimeError("synthetic parse crash")
    return None
D.parse_authored = boom
with tempfile.TemporaryDirectory() as t:
    rep = D.author(4, pathlib.Path(t) / "o.jsonl", client=Stub(), seed=1, workers=2, prefix="x")
assert rep["slots"] == 4 and rep["reject_reasons"].get("internal", 0) >= 1, rep
print("test_author_catchall: OK", rep["reject_reasons"])
