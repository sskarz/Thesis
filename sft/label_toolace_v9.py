"""Attach ``action_labels`` / ``stop_label`` onto ToolACE (and other) rows.

ToolACE formatted rows are often unlabeled at conversation level. This helper
walks assistant turns with ``action_class_v9`` and writes labels onto a copy
of each record. Used by tests and optional pre-mix inspection — ``build_mix_v9``
does not require pre-labeled rows (classification happens at unroll).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sft.action_class_v9 import attach_action_labels, classify_assistant_action
from sft.format_apigen_mt import label_assistant_content
from sft.io import read_jsonl, write_jsonl


def label_record(record: dict) -> dict:
    out = attach_action_labels(record)
    messages = list(out.get("messages") or [])
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            out["stop_label"] = label_assistant_content(messages[i].get("content"))
            out["action_class"] = classify_assistant_action(messages, i)
            break
    return out


def label_rows(rows: list[dict]) -> list[dict]:
    return [label_record(r) for r in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Attach action_labels / stop_label onto formatted JSONL rows."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = read_jsonl(args.input)
    labeled = label_rows(rows)
    n = write_jsonl(args.out, labeled)
    print(f"labeled {n} rows → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
