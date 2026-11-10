"""Dose probe on held-out TRAINING-valid prompts (never dev/confirm/mid20): greedy action under the served model.

    python sft/probe_action_change.py run   --rows AB_valid.jsonl --out probe_base.json   [--n 100]
    python sft/probe_action_change.py compare --a probe_base.json --b probe_lora.json
Reports parse rate, exact agreement with the row's target action, and how many greedy actions differ between two servers.
"""
import argparse, json, sys, pathlib
from concurrent.futures import ThreadPoolExecutor
import requests
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from benchmark.iffc import try_parse_iffc_response, iffc_calls_to_func_strings


def act(text):
    p = try_parse_iffc_response(text)
    return None if p is None else iffc_calls_to_func_strings(p)


def ask(msgs):
    r = requests.post("http://127.0.0.1:8080/v1/chat/completions",
                      json={"messages": msgs, "temperature": 0.0, "max_tokens": 1500}, timeout=600)
    return r.json()["choices"][0]["message"].get("content") or ""


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="c", required=True)
    r = sub.add_parser("run"); r.add_argument("--rows", required=True); r.add_argument("--out", required=True); r.add_argument("--n", type=int, default=100)
    c = sub.add_parser("compare"); c.add_argument("--a", required=True); c.add_argument("--b", required=True)
    a = ap.parse_args()
    if a.c == "run":
        rows = [json.loads(l) for l in open(a.rows) if l.strip()][: a.n]
        with ThreadPoolExecutor(4) as ex:
            outs = list(ex.map(lambda x: ask(x["messages"][:-1]), rows))
        res = [{"i": i, "pred": act(o), "target": act(x["messages"][-1]["content"]), "raw": o[:300]}
               for i, (x, o) in enumerate(zip(rows, outs))]
        json.dump(res, open(a.out, "w"))
        n = len(res)
        print(json.dumps({"n": n, "parsed": sum(x["pred"] is not None for x in res),
                          "exact_target": sum(x["pred"] == x["target"] and x["pred"] is not None for x in res)}))
    else:
        A, B = json.load(open(a.a)), json.load(open(a.b))
        diff = sum(x["pred"] != y["pred"] for x, y in zip(A, B))
        print(json.dumps({"n": len(A), "actions_differ": diff,
                          "exact_target_a": sum(x["pred"] == x["target"] and x["pred"] is not None for x in A),
                          "exact_target_b": sum(x["pred"] == x["target"] and x["pred"] is not None for x in B)}))

main()
