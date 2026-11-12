#!/usr/bin/env python3
"""Round 6 D3: build "self-target" rows. Same prompts as a training set, but the target (reasoning and answer)
is the UNTUNED base model's own greedy reply to that exact prompt.

usage: r6_self_targets.py --rows-dir <dir with train.jsonl, valid.jsonl> --out-dir <dir> [--url http://127.0.0.1:8080] [--workers 4]

Requirements checked before any request: the server has NO LoRA adapter loaded (GET /lora-adapters == []), and for every prompt
the rendered template (POST /apply-template) contains exactly one "<|think|>" marker. Rows carry the marker in the system
content, so it is stripped for the request (the server's template adds it) and restored in the stored row.
No explicit max_tokens is sent. A row is dropped only if finish_reason != "stop" or the reply has empty content or empty
reasoning; all drops are counted in the manifest. No other filtering (no correctness filter, no reasoning-length cap).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

MARK = "<|think|>\n"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def parse_ok(text: str) -> bool:
    try:
        from benchmark.iffc import try_parse_iffc_response
        return try_parse_iffc_response(text) is not None
    except Exception:  # parser unavailable in the offline test
        try:
            return isinstance(json.loads(text), list)
        except Exception:
            return False


def check_server(url: str) -> dict:
    la = requests.get(f"{url}/lora-adapters", timeout=30).json()
    if la != []:
        raise SystemExit(f"server has adapters loaded: {la}")
    props = requests.get(f"{url}/props", timeout=30).json()
    return {"model_path": props.get("model_path"), "build_info": props.get("build_info")}


def ask(url: str, prompt: list[dict]) -> dict:
    msgs = copy.deepcopy(prompt)
    if not (msgs and msgs[0]["role"] == "system" and msgs[0]["content"].startswith(MARK)):
        raise ValueError("system content does not start with the think marker")
    msgs[0]["content"] = msgs[0]["content"][len(MARK):]
    ap = requests.post(f"{url}/apply-template", json={"messages": msgs}, timeout=120).json()["prompt"]
    n = ap.count("<|think|>")
    if n != 1:
        raise ValueError(f"rendered prompt has {n} think markers")
    r = requests.post(f"{url}/v1/chat/completions", json={"messages": msgs, "temperature": 0.0}, timeout=1800).json()
    ch = r["choices"][0]
    m = ch["message"]
    return {"content": (m.get("content") or "").strip(), "reasoning": (m.get("reasoning_content") or "").strip(),
            "finish": ch.get("finish_reason")}


def build(rows: list[dict], url: str, workers: int):
    prompts = [r["messages"][:-1] for r in rows]
    with ThreadPoolExecutor(workers) as ex:
        outs = list(ex.map(lambda p: ask(url, p), prompts))
    kept, stats = [], {"n": len(rows), "dropped_not_stop": 0, "dropped_empty_content": 0, "dropped_empty_reasoning": 0,
                       "kept": 0, "kept_unparseable_answer": 0, "reasoning_chars": []}
    for r, o in zip(rows, outs):
        if o["finish"] != "stop":
            stats["dropped_not_stop"] += 1
            continue
        if not o["content"]:
            stats["dropped_empty_content"] += 1
            continue
        if not o["reasoning"]:
            stats["dropped_empty_reasoning"] += 1
            continue
        if not parse_ok(o["content"]):
            stats["kept_unparseable_answer"] += 1  # kept on purpose: null-test style, unfiltered own behaviour
        msgs = copy.deepcopy(r["messages"][:-1]) + [{"role": "assistant", "content": o["content"], "reasoning_content": o["reasoning"]}]
        meta = dict(r.get("meta") or {})
        meta["r6_self_target"] = True
        kept.append({"messages": msgs, "meta": meta})
        stats["kept"] += 1
        stats["reasoning_chars"].append(len(o["reasoning"]))
    rc = sorted(stats.pop("reasoning_chars"))
    stats["reasoning_chars_median"] = rc[len(rc) // 2] if rc else None
    stats["reasoning_chars_max"] = rc[-1] if rc else None
    return kept, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    src, out = pathlib.Path(a.rows_dir), pathlib.Path(a.out_dir)
    if src.resolve() == out.resolve():
        raise SystemExit("refusing to write into the source rows dir")
    out.mkdir(parents=True, exist_ok=True)
    server = check_server(a.url)
    manifest = {"server": server, "inputs": {}, "outputs": {}, "stats": {}}
    for split in ("train", "valid"):
        sp = src / f"{split}.jsonl"
        rows = [json.loads(l) for l in open(sp) if l.strip()]
        kept, stats = build(rows, a.url, a.workers)
        op = out / f"{split}.jsonl"
        with open(op, "w") as f:
            for k in kept:
                f.write(json.dumps(k) + "\n")
        manifest["inputs"][split] = {"path": str(sp), "sha256": sha(sp), "rows": len(rows)}
        manifest["outputs"][split] = {"path": str(op), "sha256": sha(op), "rows": len(kept)}
        manifest["stats"][split] = stats
        print(split, json.dumps(stats), flush=True)
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
