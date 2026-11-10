"""MLX-side diagnostics for a trained adapter on frozen prompts (training/valid rows only).
usage: diag_mlx.py <adapter_dir|none> <rows.jsonl> <out.json> [n] [gen]
 - teacher-forced NLL split into reasoning tokens (before <channel|>) and answer tokens (after), per row and mean
 - optional greedy generation (gen=1): parsed action vs target action string, for parity vs llama.cpp server probes
Rows must already carry the <|think|> system prefix (mlx-*-think files)."""
import json, sys, pathlib
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from benchmark.iffc import try_parse_iffc_response, iffc_calls_to_func_strings

M = "/Users/sanskarthapa/Documents/GitHub/Thesis/sft/models/gemma-4-e2b-it-mlx"
ad, rows_p, out = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 35
gen = int(sys.argv[5]) if len(sys.argv) > 5 else 0
model, tok = load(M, adapter_path=None if ad == "none" else ad)
rows = [json.loads(l) for l in open(rows_p)][:n]
end_think = tok.convert_tokens_to_ids("<channel|>")
act = lambda t: (lambda p: None if p is None else iffc_calls_to_func_strings(p))(try_parse_iffc_response(t))
res = []
for i, r in enumerate(rows):
    m = r["messages"]
    p_ids = tok.apply_chat_template(m[:-1], add_generation_prompt=True, tokenize=True)
    f_ids = tok.apply_chat_template(m, tokenize=True)
    p_ids = p_ids["input_ids"] if hasattr(p_ids, "keys") else p_ids
    f_ids = f_ids["input_ids"] if hasattr(f_ids, "keys") else f_ids
    assert f_ids[:len(p_ids)] == p_ids, "prompt prefix mismatch"
    x = mx.array(f_ids)[None]
    logits = model(x[:, :-1]).astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tgt = x[:, 1:]
    nll = -mx.take_along_axis(lp, tgt[..., None], axis=-1)[..., 0][0]
    nll = nll[len(p_ids) - 1:]  # target positions only
    t_ids = f_ids[len(p_ids):]
    k = t_ids.index(end_think) if end_think in t_ids else len(t_ids)
    nl = nll.tolist()
    d = {"i": i, "arm": r["meta"]["arm"], "n_reason": k, "n_answer": len(t_ids) - k,
         "nll_reason": sum(nl[:k + 1]) / max(1, k + 1), "nll_answer": sum(nl[k + 1:]) / max(1, len(nl) - k - 1)}
    if gen:
        text = generate(model, tok, prompt=p_ids, max_tokens=1500)
        content = text.split("<channel|>")[-1]
        d["pred"] = act(content); d["target"] = act(m[-1]["content"])
    res.append(d)
    print(i, round(d["nll_reason"], 3), round(d["nll_answer"], 3), flush=True)
summ = {"n": len(res), "mean_nll_reason": sum(d["nll_reason"] for d in res) / len(res),
        "mean_nll_answer": sum(d["nll_answer"] for d in res) / len(res)}
if gen:
    summ["exact_target"] = sum(d["pred"] is not None and d["pred"] == d["target"] for d in res)
    summ["parsed"] = sum(d["pred"] is not None for d in res)
json.dump({"summary": summ, "rows": res}, open(out, "w"))
print(json.dumps(summ))
