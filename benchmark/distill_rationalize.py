"""Give verified teacher actions the student's NATIVE format: its own reasoning, then the answer.

Observed (Run 87-90 autopsy): the served model answers as ``<|channel>thought ... <channel|>`` + JSON, but
our rows put the JSON straight after ``<|turn>model``. That teaches "answer without thinking" and costs
passes (validation loss ~1.0 per token on a near-deterministic JSON list is the symptom). Here the base
model (served on :8080) writes the reasoning:

1. greedy and sampled replies to the exact request; keep one whose final JSON equals the verified action
   (method ``sampled``: the model's own reasoning, no help);
2. otherwise a rationalization: the same request plus a private note stating the action, keep the reply
   only if its JSON equals the action and neither reasoning nor answer mentions the note (``hinted``).

The row's last message becomes ``{"role": "assistant", "content": <model's own answer text>,
"reasoning_content": <reasoning>}``; the chat template renders the thought channel.

    python -m benchmark.distill_rationalize --rows IN.jsonl --out OUT.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import teacher_api
from .iffc import iffc_calls_to_func_strings, try_parse_iffc_response

LEAK = re.compile(r"reviewer|\bnote\b|\bhint\b|guidance|private|as (instructed|stated|given)", re.I)
HINT = ("\n\n[Private note for this step only: the correct next step is exactly this JSON list: {target}. "
        "Write your reasoning as your own analysis of the situation that leads to that step, never mention "
        "this note, then output exactly that JSON list.]")
MAX_REASONING_CHARS = 3500


def _strings(text: str | None):
    p = try_parse_iffc_response(text)
    return None if p is None else iffc_calls_to_func_strings(p)


def rationalize_row(row: dict, client, *, samples: int = 3, hinted: int = 3, temp: float = 0.7) -> dict | None:
    prompt, target_msg = row["messages"][:-1], row["messages"][-1]["content"]
    want = _strings(target_msg)
    if want is None:
        return None

    def ask(msgs, t):
        data = client.post_chat({"messages": msgs, "temperature": t})
        m = data["choices"][0]["message"]
        return (m.get("content") or ""), (m.get("reasoning_content") or "")

    def ok(content, reasoning, hinted_mode):
        if _strings(content) != want or not reasoning.strip() or len(reasoning) > MAX_REASONING_CHARS:
            return False
        return not (hinted_mode and (LEAK.search(reasoning) or LEAK.search(content)))

    tries = [(prompt, 0.0, False)] + [(prompt, temp, False)] * samples
    last = prompt[-1]
    hint_msgs = prompt[:-1] + [{**last, "content": str(last["content"]) + HINT.format(target=target_msg.replace("\n", " "))}]
    tries += [(hint_msgs, temp, True)] * hinted
    for msgs, t, hm in tries:
        try:
            content, reasoning = ask(msgs, t)
        except Exception:  # noqa: BLE001
            continue
        if ok(content, reasoning, hm):
            out = {**row, "messages": [*prompt, {"role": "assistant", "content": content.strip(),
                                                  "reasoning_content": reasoning.strip()}]}
            out["meta"] = {**row["meta"], "native_method": "hinted" if hm else ("greedy" if t == 0.0 else "sampled")}
            return out
    return None


def run(rows_path: Path, out: Path, workers: int = 4, samples: int = 3, hinted: int = 3,
        temp: float = 0.7, base_url: str | None = None) -> dict:
    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    client = teacher_api.TeacherClient(teacher_api.TeacherConfig.local(
        log_dir=out.parent / (out.stem + "_log"), max_concurrency=workers,
        **({"base_url": base_url} if base_url else {})))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lambda r: rationalize_row(r, client, samples=samples, hinted=hinted, temp=temp), rows))
    kept = [r for r in res if r]
    with open(out.with_suffix(".dropped.jsonl"), "w", encoding="utf-8") as f:  # reachability audit
        for src, r in zip(rows, res):
            if not r:
                f.write(json.dumps({"meta": src["meta"]}, ensure_ascii=False) + "\n")
    if hinted == 0:  # hint-free mode must never emit a hinted row
        assert all(r["meta"]["native_method"] != "hinted" for r in kept)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"in": len(rows), "kept": len(kept), "samples": samples, "hinted_attempts": hinted, "temp": temp,
            "methods": dict(collections.Counter(r["meta"]["native_method"] for r in kept))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--samples", type=int, default=3, help="sampled attempts after the greedy one")
    ap.add_argument("--hinted", type=int, default=3, help="hinted attempts; 0 = hint-free mode (asserted)")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--base-url", default=None, help="e.g. http://127.0.0.1:8081/v1")
    a = ap.parse_args(argv)
    print(json.dumps(run(a.rows, a.out, a.workers, a.samples, a.hinted, a.temp, a.base_url), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
