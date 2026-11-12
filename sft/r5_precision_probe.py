"""Round 5 A1: precision probe on FROZEN prompts. Greedy generation against the llama-server on :8080 (caller starts the server).
  freeze --out frozen.json : first 35 valid + first 35 train rows of mlx-AB-think (Round 3b selection), ids + sha256 of messages[:-1]
  run --frozen frozen.json --tag NAME --out out.json : for each prompt, /apply-template (rendered-prompt sha, <|think|> count) and /tokenize hash,
        then greedy chat completion (temperature 0, max_tokens 1500 as Round 3b) with HTTP status, finish_reason, usage logged.
  report --runs q4_base.json q4_ab.json bf16_base.json bf16_ab.json : agreement with target action, flips base->adapter per precision, rendered-prompt identity across servers."""
import argparse, hashlib, json, pathlib, sys
from concurrent.futures import ThreadPoolExecutor
import requests
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from benchmark.iffc import try_parse_iffc_response, iffc_calls_to_func_strings

U = "http://127.0.0.1:8080"
R = "sft/data/distill/r3/mlx-AB-think/"


def h(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def act(text):
    p = try_parse_iffc_response(text)
    return None if p is None else iffc_calls_to_func_strings(p)


def rows_for(frozen):
    out = []
    for f, key in ((R + "valid.jsonl", "valid"), (R + "train.jsonl", "train")):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        out += [(key, i, rows[i]) for i in range(35)]
    return out


def freeze(a):
    items = []
    for key, i, r in rows_for(None):
        items.append({"set": key, "i": i, "task_id": r["meta"]["task_id"], "arm": r["meta"]["arm"],
                      "prompt_sha": h(r["messages"][:-1]), "target_sha": h(r["messages"][-1])})
    json.dump({"source": R, "items": items}, open(a.out, "w"), indent=1)
    print(len(items), "prompts frozen;", len({x["task_id"] for x in items}), "tasks")


def run(a):
    fr = json.load(open(a.frozen))["items"]
    rows = rows_for(None)
    assert [h(r["messages"][:-1]) for _, _, r in rows] == [x["prompt_sha"] for x in fr], "frozen prompt mismatch"

    def one(t):
        key, i, r = t
        msgs = json.loads(json.dumps(r["messages"][:-1]))
        # the server's chat template adds <|think|> itself (thinking on); rows already carry it in the system content, which renders it TWICE
        # (found in the first A1 attempt). Strip the content marker so the rendered prompt has exactly the one marker the real harness gets.
        if msgs[0]["role"] == "system" and msgs[0]["content"].startswith("<|think|>\n"):
            msgs[0]["content"] = msgs[0]["content"][len("<|think|>\n"):]
        rec = {"set": key, "i": i}
        try:
            ap = requests.post(U + "/apply-template", json={"messages": msgs}, timeout=120).json()["prompt"]
            rec["rendered_sha"] = hashlib.sha256(ap.encode()).hexdigest(); rec["think_markers"] = ap.count("<|think|>")
            assert rec["think_markers"] == 1, "rendered prompt must contain exactly one think marker"
            tk = requests.post(U + "/tokenize", json={"content": ap, "add_special": False, "parse_special": True}, timeout=120).json()["tokens"]
            rec["n_prompt_tokens"] = len(tk); rec["token_sha"] = hashlib.sha256(json.dumps(tk).encode()).hexdigest()
            resp = requests.post(U + "/v1/chat/completions", json={"messages": msgs, "temperature": 0.0, "max_tokens": 1500}, timeout=1800)
            rec["http"] = resp.status_code
            c = resp.json()["choices"][0]
            rec["finish_reason"] = c.get("finish_reason"); rec["completion_tokens"] = resp.json().get("usage", {}).get("completion_tokens")
            txt = c["message"].get("content") or ""
            rec["pred"] = act(txt); rec["target"] = act(r["messages"][-1]["content"]); rec["raw_head"] = txt[:300]
        except Exception as e:  # logged, never silently dropped
            rec["error"] = repr(e)
        return rec
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one, rows))
    lora = requests.get(U + "/lora-adapters", timeout=30).json()
    props = requests.get(U + "/props", timeout=30).json()
    json.dump({"tag": a.tag, "lora_adapters": lora, "model_path": props.get("model_path"), "build_info": props.get("build_info"), "results": res},
              open(a.out, "w"), indent=1)
    ok = [x for x in res if "error" not in x]
    if len(ok) != len(res) or len(res) != len(fr) or len({(x["set"], x["i"]) for x in res}) != len(fr):
        print("INVALID RUN: errors or wrong cardinality", len(res) - len(ok)); sys.exit(1)
    print(a.tag, "n", len(res), "errors", len(res) - len(ok), "truncated(finish=length)", sum(x.get("finish_reason") == "length" for x in ok),
          "parsed", sum(x["pred"] is not None for x in ok),
          "exact_target", {s: sum(x["pred"] is not None and x["pred"] == x["target"] for x in ok if x["set"] == s) for s in ("valid", "train")})


def report(a):
    R_ = {p: json.load(open(p)) for p in a.runs}
    names = list(R_)
    out = {"runs": {}}
    for p, d in R_.items():
        res = d["results"]
        out["runs"][d["tag"]] = {"model_path": d["model_path"], "lora": [x.get("path") for x in d["lora_adapters"]], "errors": sum("error" in x for x in res),
                                 "truncated": sum(x.get("finish_reason") == "length" for x in res), "parsed": sum(x.get("pred") is not None for x in res),
                                 "think_markers_distinct": sorted({x.get("think_markers") for x in res}),
                                 "exact_target": {s: sum(x.get("pred") is not None and x["pred"] == x["target"] for x in res if x["set"] == s) for s in ("valid", "train")}}
    tags = [R_[p]["tag"] for p in names]

    def rd(tag):
        return {(x["set"], x["i"]): x for x in R_[names[tags.index(tag)]]["results"]}
    for prec in ("q4", "bf16"):
        b, ad = rd(prec + "_base"), rd(prec + "_ab")
        out[prec + "_base_vs_adapter"] = {s: {"actions_differ": sum(b[k].get("pred") != ad[k].get("pred") for k in b if k[0] == s),
                                              "adapter_minus_base_exact": sum((ad[k].get("pred") is not None and ad[k]["pred"] == ad[k]["target"]) - (b[k].get("pred") is not None and b[k]["pred"] == b[k]["target"]) for k in b if k[0] == s)} for s in ("valid", "train")}
    q, f = rd("q4_base"), rd("bf16_base")
    out["rendered_prompt_identity_q4_vs_bf16"] = {"same_rendered_sha": sum(q[k].get("rendered_sha") == f[k].get("rendered_sha") for k in q), "same_token_sha": sum(q[k].get("token_sha") == f[k].get("token_sha") for k in q), "n": len(q)}
    out["bf16_vs_q4_base_actions_differ"] = sum(q[k].get("pred") != f[k].get("pred") for k in q)
    dd = out["bf16_base_vs_adapter"]["valid"]["adapter_minus_base_exact"] - out["q4_base_vs_adapter"]["valid"]["adapter_minus_base_exact"]
    out["held_out_difference_in_differences_prompts"] = dd
    out["trigger_(>=4)_met"] = dd >= 4
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


ap = argparse.ArgumentParser(); s = ap.add_subparsers(dest="c", required=True)
x = s.add_parser("freeze"); x.add_argument("--out", required=True); x.set_defaults(f=freeze)
x = s.add_parser("run"); x.add_argument("--frozen", required=True); x.add_argument("--tag", required=True); x.add_argument("--out", required=True); x.set_defaults(f=run)
x = s.add_parser("report"); x.add_argument("--runs", nargs=4, required=True); x.add_argument("--out", required=True); x.set_defaults(f=report)
a = ap.parse_args(); a.f(a)
