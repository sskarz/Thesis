#!/usr/bin/env python3
"""Offline tests: GuardedClient halts at the first cost-less response; build_bank_f.filt survives skipped verdicts."""
import json, os, pathlib, sys, tempfile
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
os.environ["OPENROUTER_API_KEY"] = "test-not-a-real-key"
from benchmark import teacher_api
import yield_guard

class Resp:
    status_code = 200
    def __init__(self, cost): self._c = cost
    def raise_for_status(self): pass
    def json(self):
        u = {"prompt_tokens": 1, "completion_tokens": 1}
        if self._c is not None: u["cost"] = self._c
        return {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}], "usage": u, "model": "m", "provider": "p"}

class Sess:
    def __init__(self, costs): self.costs = list(costs); self.n = 0
    def post(self, *a, **k):
        self.n += 1; return Resp(self.costs.pop(0))

yield_guard.GuardedClient.MISSING_COST = 0
c = yield_guard.GuardedClient(teacher_api.TeacherConfig.openrouter(max_cost_usd=1.0), session=Sess([0.01, None, 0.01]))
c.post_chat({"messages": []}); assert yield_guard.GuardedClient.MISSING_COST == 0
c.post_chat({"messages": []}); assert yield_guard.GuardedClient.MISSING_COST == 1          # the cost-less response is returned (already paid) ...
try:
    c.post_chat({"messages": []}); raise SystemExit("third call should have been refused")   # ... and every later request is refused
except teacher_api.BudgetExceeded:
    pass
assert c._session.n == 2, c._session.n
yield_guard.GuardedClient.MISSING_COST = 0
# budget latch: the cap is reached after one response; the next request raises and the latch is recorded
c2 = yield_guard.GuardedClient(teacher_api.TeacherConfig.openrouter(max_cost_usd=0.005), session=Sess([0.01, 0.01]))
c2.post_chat({"messages": []}); assert not yield_guard.GuardedClient.BUDGET_HIT
try:
    c2.post_chat({"messages": []}); raise SystemExit("cap not enforced")
except teacher_api.BudgetExceeded:
    pass
assert yield_guard.GuardedClient.BUDGET_HIT
with tempfile.TemporaryDirectory() as dd:
    yield_guard.GuardedClient.write_flags(pathlib.Path(dd) / "sub" / "f.json")
    assert json.load(open(pathlib.Path(dd) / "sub" / "f.json")) == {"missing_cost_responses": 0, "budget_hit": True}
yield_guard.GuardedClient.BUDGET_HIT = False

sys.path.insert(0, str(HERE))
import build_bank_f
with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    words = {1: "rename the quarterly budget spreadsheet before friday afternoon", 2: "send my dentist a reminder about tuesday morning appointment", 9: "convert yesterday temperature readings into celsius for the lab"}
    t = lambda i, call: {"id": f"t{i}", "question": [[{"role": "user", "content": words[i]}]],
                         "ground_truth": [[call]], "involved_classes": ["MathAPI"]}
    tasks = [t(1, "add(a=1,b=2)"), t(2, "multiply(a=1,b=2)")]
    (d / "aud.jsonl").write_text("".join(json.dumps(x) + "\n" for x in tasks))
    (d / "other.jsonl").write_text(json.dumps(t(9, "subtract(a=1,b=2)")) + "\n")
    (d / "solve").mkdir()
    (d / "solve" / "verdicts.jsonl").write_text("\n".join([json.dumps({"task_id": "t1", "valid": True, "api_error": False}),
                                                          json.dumps({"episode_id": "t2__s0", "valid": False, "error": "skipped:budget"})]) + "\n")
    build_bank_f.filt(str(d / "aud.jsonl"), str(d / "solve"), str(d / "out"), [str(d / "other.jsonl")])
    rep = json.load(open(d / "out" / "F_report.json"))
    assert rep["F"] == 1 and rep["skipped_verdicts_without_task_id"] == 1 and rep["verified"] == 1, rep
print("test_yield_guard: OK")
