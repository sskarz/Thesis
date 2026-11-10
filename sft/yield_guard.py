"""TeacherClient that stops spending as soon as one response arrives without cost accounting.

The stock client enforces the dollar cap from `usage.cost` of each response. If the API ever stops reporting cost, the cap silently
stops working. This subclass counts such responses (class-level, so wrappers that do not own the client can read it) and refuses every
later request once one has been seen.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import teacher_api  # noqa: E402


class GuardedClient(teacher_api.TeacherClient):
    MISSING_COST = 0
    BUDGET_HIT = False      # latched: generate_multi_turn turns BudgetExceeded into an ordinary episode error, so it must be recorded here

    def post_chat(self, payload):
        if GuardedClient.MISSING_COST:
            raise teacher_api.BudgetExceeded("a response had no cost accounting; halting all further requests")
        try:
            data = super().post_chat(payload)
        except teacher_api.BudgetExceeded:
            GuardedClient.BUDGET_HIT = True
            raise
        if (data.get("usage") or {}).get("cost") is None:
            GuardedClient.MISSING_COST += 1
        return data

    @classmethod
    def write_flags(cls, path):
        import json
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"missing_cost_responses": cls.MISSING_COST, "budget_hit": cls.BUDGET_HIT}, open(path, "w"))
