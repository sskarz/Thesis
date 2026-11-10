#!/usr/bin/env python3
"""Render parity: the prompt the thinking-off llama-server builds (POST /apply-template) must equal the prompt the trainer builds (tokenizer, enable_thinking=False)
for every G1 row, and must contain no <|think|> marker.   usage: g1_parity.py --rows g1_rows.jsonl --out parity.json   (server on :8080 must be up)"""
import argparse, json, sys, requests
MODEL = "/Users/sanskarthapa/Documents/GitHub/Thesis/sft/models/gemma-4-e2b-it-mlx"
ap = argparse.ArgumentParser(); ap.add_argument("--rows", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
rows = [json.loads(l) for l in open(a.rows, encoding="utf-8") if l.strip()]
bad, markers, n_bos = [], 0, 0
def norm(s):
    return s[5:] if s.startswith("<bos>") else s
for r in rows:
    m = r["messages"][:-1]
    srv = requests.post("http://127.0.0.1:8080/apply-template", json={"messages": m}, timeout=60).json()["prompt"]
    loc = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    markers += srv.count("<|think|>")
    n_bos += srv.startswith("<bos>") != loc.startswith("<bos>")
    if norm(srv) != norm(loc):
        bad.append(r["meta"]["task_id"])
res = {"rows": len(rows), "mismatches": len(bad), "mismatch_tasks": sorted(set(bad))[:10], "server_think_markers": markers, "bos_prefix_differs_rows": n_bos}
json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res))
sys.exit(0 if not bad and markers == 0 else 1)
