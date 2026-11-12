#!/usr/bin/env python3
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import yield_smoke_check as m
ok = {"cost_usd": 0.001, "api_model": m.PIN, "api_provider": "DeepInfra", "finish_reason": "stop"}
assert m.check([ok, dict(ok)]) == []
assert m.check([]) != []
assert m.check([dict(ok, cost_usd=None)]) != [] and m.check([dict(ok, cost_usd=0)]) != []
assert m.check([dict(ok, api_model="deepseek/deepseek-v4.1-flash")]) != []
assert m.check([dict(ok, api_provider="Other")]) != []
assert m.check([ok, dict(ok, api_provider="Other")]) != []
assert m.check([dict(ok, finish_reason=None)]) != []
print("test_yield_smoke_check: OK")
