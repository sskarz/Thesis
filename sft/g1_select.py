#!/usr/bin/env python3
"""G1 (overfit gate) data selection. CPU only, no API, no GPU.

usage: g1_select.py --calibration CAL.json --tasks tasks_F.jsonl --rows-dir ROWS --out-dir OUT [--exclude ID,ID,...] [--exclude-file F] [--n 40] [--max-tokens 6999]
Pre-registered procedure (docs/DISTILLATION_G1_PREREGISTRATION.md):
 1. candidates = calibration.json survivor_ids (filter survivors that Codex/subagents did not mark WRONG in the first review) minus --exclude
    (tasks any reviewer of the fresh sample marked WRONG);
 2. order candidates by sha256(task_id) ascending, take the first N;
 3. rows = every Part 1 row (train.jsonl and valid.jsonl) of those tasks; drop rows longer than --max-tokens tokens when rendered with
    thinking OFF (the registered '5% longest rows' fallback of the memory preflight);
Writes OUT/g1_tasks.jsonl (task records for the scoring bank), OUT/g1_rows.jsonl, OUT/g1_selection.json (ids, counts, hashes).
"""
import argparse, hashlib, json, pathlib, sys

MODEL = "/Users/USER/Documents/GitHub/Thesis/sft/models/gemma-4-e2b-it-mlx"


def render_ids(tok, messages, add_generation_prompt=False):
    out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt, enable_thinking=False, return_dict=False)
    return list(out["input_ids"]) if isinstance(out, dict) else list(out)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", required=True); ap.add_argument("--tasks", required=True); ap.add_argument("--rows-dir", required=True)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--exclude", default=""); ap.add_argument("--exclude-file")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--max-tokens", type=int, default=6999)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    cal = json.load(open(a.calibration))
    excl = {x for x in a.exclude.split(",") if x}
    if a.exclude_file:
        excl |= {l.strip() for l in open(a.exclude_file) if l.strip()}
    cand = [t for t in cal["survivor_ids"] if t not in excl]
    cand.sort(key=lambda t: hashlib.sha256(t.encode()).hexdigest())
    chosen = cand[: a.n]
    assert len(chosen) == a.n, f"only {len(chosen)} candidates"
    tasks = {}
    for l in open(a.tasks, encoding="utf-8"):
        if l.strip():
            r = json.loads(l); tasks[r["id"]] = r
    rows, dropped_long, per_task = [], 0, {}
    for fn in ("train.jsonl", "valid.jsonl"):
        for l in open(pathlib.Path(a.rows_dir) / fn, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            if r["meta"]["task_id"] not in chosen:
                continue
            n = len(render_ids(tok, r["messages"]))
            if n > a.max_tokens:
                dropped_long += 1
                continue
            r["_tokens"] = n; r["_src"] = fn
            rows.append(r); per_task[r["meta"]["task_id"]] = per_task.get(r["meta"]["task_id"], 0) + 1
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "g1_tasks.jsonl", "w", encoding="utf-8") as f:
        for t in chosen:
            f.write(json.dumps(tasks[t], ensure_ascii=False) + "\n")
    with open(out / "g1_rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    toks = sorted(r["_tokens"] for r in rows)
    info = {"n_tasks": len(chosen), "task_ids": chosen, "excluded_from_candidates": sorted(excl), "candidates": len(cand),
            "rows": len(rows), "rows_dropped_over_max_tokens": dropped_long, "max_tokens": a.max_tokens,
            "tasks_without_rows": sorted(set(chosen) - set(per_task)), "rows_per_task_min_max": [min(per_task.values()), max(per_task.values())],
            "tokens_median": toks[len(toks) // 2], "tokens_max": toks[-1], "tokens_sum": sum(toks),
            "sha256": {"g1_tasks.jsonl": sha(out / "g1_tasks.jsonl"), "g1_rows.jsonl": sha(out / "g1_rows.jsonl")}}
    json.dump(info, open(out / "g1_selection.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k not in ("task_ids",)}))
    return 0 if not info["tasks_without_rows"] else 3


if __name__ == "__main__":
    sys.exit(main())
