"""Prepend the Gemma-4 thinking marker to the system message of every training row.

The served prompt (llama-server --jinja, thinking on by default) renders `<|turn>system\\n<|think|>\\n<system text>`;
mlx-lm renders training prompts with enable_thinking=False, i.e. WITHOUT `<|think|>`. Prepending "<|think|>\\n" to the
system content makes the MLX-rendered prompt identical to the served one (checked on all rows against /apply-template).

usage: add_think_marker.py <in_dir> <out_dir>      (copies train.jsonl / valid.jsonl)
"""
import json
import pathlib
import sys

src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
for name in ("train.jsonl", "valid.jsonl"):
    n = 0
    with open(dst / name, "w", encoding="utf-8") as f:
        for line in open(src / name, encoding="utf-8"):
            row = json.loads(line)
            m = row["messages"][0]
            assert m["role"] == "system" and isinstance(m["content"], str) and not m["content"].startswith("<|think|>")
            m["content"] = "<|think|>\n" + m["content"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(name, n)
