#!/usr/bin/env python3
"""G1 diagnostic (reported, never gating): row-level in-sample agreement. For every G1 training row, greedy reply of the served model to the row's prompt, compared with the
row's target after JSON normalisation.  usage: g1_rowcheck.py --rows g1_rows.jsonl --out rowcheck_NAME.json [--workers 4]   (server on :8080)
Counts: parsed (reply is JSON), exact (parsed reply == parsed target), by_task (rows exact per task), and for 'exact' also the share where target is [] (stop rows)."""
import argparse, json, sys, requests
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser(); ap.add_argument("--rows", required=True); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
rows = [json.loads(l) for l in open(a.rows, encoding="utf-8") if l.strip()]
def loads(s):
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[4:] if s.lower().startswith("json") else s
    try:
        return json.loads(s.strip())
    except (json.JSONDecodeError, ValueError):
        return None
def one(r):
    try:
        c = requests.post("http://127.0.0.1:8080/v1/chat/completions", json={"messages": r["messages"][:-1], "temperature": 0, "max_tokens": 1024}, timeout=600).json()["choices"][0]
        txt = c["message"].get("content") or ""; fin = c.get("finish_reason")
    except Exception as e:
        return {"task": r["meta"]["task_id"], "error": type(e).__name__}
    got, want = loads(txt), loads(r["messages"][-1]["content"])
    return {"task": r["meta"]["task_id"], "parsed": got is not None, "exact": got is not None and got == want, "target_empty": want == [], "finish": fin, "reply_chars": len(txt)}
with ThreadPoolExecutor(max_workers=a.workers) as ex:
    res = list(ex.map(one, rows))
err = sum("error" in x for x in res)
ok = [x for x in res if "error" not in x]
by_task = {}
for x in ok:
    by_task.setdefault(x["task"], [0, 0]); by_task[x["task"]][0] += x["exact"]; by_task[x["task"]][1] += 1
rep = {"rows": len(rows), "errors": err, "parsed": sum(x["parsed"] for x in ok), "exact": sum(x["exact"] for x in ok),
       "exact_on_stop_rows": [sum(x["exact"] for x in ok if x["target_empty"]), sum(x["target_empty"] for x in ok)],
       "length_truncated": sum(x["finish"] == "length" for x in ok), "tasks_with_all_rows_exact": sum(v[0] == v[1] for v in by_task.values()), "tasks": len(by_task)}
json.dump({"summary": rep, "per_row": res}, open(a.out, "w")); print(json.dumps(rep))
