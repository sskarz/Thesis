"""Stage B DiaTool-style anti-loop preference pairs — **stub only** (SFT v9).

Schema for a later preference pass **after** a healthy v9 SFT. This module
does **not** train DPO/ORPO. CLI ``--dry-run`` writes counts / sample pairs
to stdout (and optional JSON), never launches preference training.

Pair kinds (DiaTool-DPO inspired, arXiv:2504.02882):

* ``advance_vs_loop`` — chosen = advancing next call or ``final_after_obs``;
  rejected = observe→same-call / consecutive identical
* ``ask_vs_wrong_call`` — chosen = clarify when slots missing; rejected =
  hallucinated tool call
* ``refuse_vs_force_call`` — chosen = refuse / ``[]`` on out-of-scope;
  rejected = forced irrelevant tool call

Explicit gate: Stage B only if v9 finish-200 beats v4 15% **or** probe gate
PASS with healthy under-call/state proxies. Size target later: ~2–5k pairs
mined from APIGen + synthetic Type-2/3 — **not** from BFCL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from sft.action_class_v9 import (
    ACTION_ASK,
    ACTION_FINAL_AFTER_OBS,
    ACTION_MT_TOOL_CHAIN_ADVANCE,
    ACTION_OBSERVE_SAME,
    ACTION_REFUSE,
    ACTION_CONSECUTIVE,
    classify_assistant_action,
)
from sft.config import V9_PREF_PAIRS_SCHEMA_VERSION
from sft.drop_identical_call import is_identical_to_previous_assistant_call, is_observe_same_call
from sft.io import read_jsonl

PAIR_ADVANCE_VS_LOOP = "advance_vs_loop"
PAIR_ASK_VS_WRONG = "ask_vs_wrong_call"
PAIR_REFUSE_VS_FORCE = "refuse_vs_force_call"
PAIR_KINDS = (PAIR_ADVANCE_VS_LOOP, PAIR_ASK_VS_WRONG, PAIR_REFUSE_VS_FORCE)


def pair_schema_example() -> dict[str, Any]:
    return {
        "schema_version": V9_PREF_PAIRS_SCHEMA_VERSION,
        "prompt": [{"role": "user", "content": "...prefix through last user..."}],
        "chosen": {"role": "assistant", "content": "...advancing or final_after_obs..."},
        "rejected": {"role": "assistant", "content": "...observe→same or loop..."},
        "pair_kind": PAIR_ADVANCE_VS_LOOP,
        "parent_id": "apigen_mt:123",
        "note": "stub — not for training until v9 SFT gate PASS",
    }


def mine_advance_vs_loop_candidates(
    conversations: Iterable[dict],
) -> list[dict[str, Any]]:
    """Dry-run miner: find prefixes where gold advances and a loop reject exists.

    Returns stub pair dicts (counts for --dry-run). Does not synthesize new NL.
    """
    pairs: list[dict[str, Any]] = []
    for rec in conversations:
        messages = list(rec.get("messages") or [])
        parent_id = str(rec.get("id") or "")
        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            cls = classify_assistant_action(messages, i)
            if cls not in {
                ACTION_MT_TOOL_CHAIN_ADVANCE,
                ACTION_FINAL_AFTER_OBS,
            }:
                continue
            # Look for a later observe→same / consecutive in the same traj as reject.
            reject = None
            for j in range(i + 1, len(messages)):
                if messages[j].get("role") != "assistant":
                    continue
                if is_observe_same_call(messages, j) or is_identical_to_previous_assistant_call(
                    messages, j
                ):
                    reject = messages[j]
                    break
            if reject is None:
                continue
            pairs.append(
                {
                    "schema_version": V9_PREF_PAIRS_SCHEMA_VERSION,
                    "prompt": messages[:i],
                    "chosen": {"role": "assistant", "content": msg.get("content")},
                    "rejected": {
                        "role": "assistant",
                        "content": reject.get("content"),
                    },
                    "pair_kind": PAIR_ADVANCE_VS_LOOP,
                    "parent_id": parent_id,
                    "chosen_action": cls,
                    "rejected_action": (
                        ACTION_OBSERVE_SAME
                        if is_observe_same_call(messages, j)
                        else ACTION_CONSECUTIVE
                    ),
                }
            )
    return pairs


def dry_run_report(conversations: list[dict]) -> dict[str, Any]:
    pairs = mine_advance_vs_loop_candidates(conversations)
    by_kind = {PAIR_ADVANCE_VS_LOOP: len(pairs)}
    # ask / refuse miners are stubs (count source labels only).
    n_ask = n_refuse = 0
    for rec in conversations:
        messages = list(rec.get("messages") or [])
        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            cls = classify_assistant_action(messages, i)
            if cls == ACTION_ASK:
                n_ask += 1
            elif cls == ACTION_REFUSE:
                n_refuse += 1
    return {
        "schema_version": V9_PREF_PAIRS_SCHEMA_VERSION,
        "n_conversations": len(conversations),
        "pairs_mined": len(pairs),
        "by_kind": by_kind,
        "ask_assistant_turns_seen": n_ask,
        "refuse_assistant_turns_seen": n_refuse,
        "sample_schema": pair_schema_example(),
        "sample_pairs_head": pairs[:3],
        "training": False,
        "note": (
            "Stage B stub only. Do not DPO/ORPO until v9 SFT probe/finish gate. "
            f"Also reserved kinds: {PAIR_ASK_VS_WRONG}, {PAIR_REFUSE_VS_FORCE}."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stage B anti-loop preference-pair stub for sft-v9. "
            "Dry-run only — no preference training."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Mine counts / sample pairs; never train (default).",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=None,
        help="Optional conversation JSONL dir or file to scan.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON path for the dry-run report.",
    )
    args = parser.parse_args(argv)

    conversations: list[dict] = []
    if args.splits is not None:
        if args.splits.is_dir():
            for name in ("train.jsonl", "valid.jsonl", "corpus_test.jsonl"):
                f = args.splits / name
                if f.exists():
                    conversations.extend(read_jsonl(f))
        elif args.splits.is_file():
            conversations = read_jsonl(args.splits)

    report = dry_run_report(conversations)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n"
        )
        print(f"wrote dry-run report → {args.out}")
    print(
        "pref_pairs_antiloop_v9: stub dry-run complete "
        f"(pairs_mined={report['pairs_mined']}; training=False)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
