"""Mine exact-same-prefix ``advance_vs_loop`` preference pairs for Stage B.

Stage B is the primary post-v9-quota path. This module is data engineering
only: it never trains a model. Every accepted pair has one shared prompt that
ends in a successful tool observation, the source gold next action as
``chosen``, and a constructed IFFC ``rejected`` response copied from the
immediately preceding executed assistant call set. A response from a later
turn is never reused at an earlier prefix.

The miner excludes retry-valid or ambiguous observations, BFCL-looking
records, canonical chosen/rejected collisions, and parents that cross source
splits. ``--dry-run`` prints counts, samples, and gates. JSONL is written only
with ``--write``. No preference policy should be initialized from failed v9,
v9.1, or v9.2 smoke adapters: the intended policy and frozen reference both
start from ``google/gemma-4-e2b-it``. No training is authorized here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from sft.action_class_v9 import (
    ACTION_FINAL_AFTER_OBS,
    ACTION_MT_TOOL_CHAIN_ADVANCE,
    classify_assistant_action,
)
from sft.config import SEED_V9, V9_PREF_PAIRS_SCHEMA_VERSION
from sft.drop_identical_call import canonical_tool_calls, is_exec_observation
from sft.io import read_jsonl, write_jsonl
from sft.split_data import bfcl_leak_reasons

PAIR_ADVANCE_VS_LOOP = "advance_vs_loop"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "data" / "pref-v9"
SPLIT_NAMES = ("train", "valid", "corpus_test")

# Prefer false negatives over unsafe preference rows.
_BAD_OBSERVATION_CUES = (
    "timed out",
    "timeout",
    "time out",
    "tool error",
    "execution error",
    "exception",
    "traceback",
    "failed to",
    "failure",
    "incomplete",
    "partial result",
    "try again",
    "retry",
)
_RETRY_METADATA_KEYS = (
    "retry_valid",
    "allow_retry",
    "explicit_retry",
    "observation_incomplete",
    "non_deterministic",
    "nondeterministic",
)
_NONDETERMINISTIC_TOOL_CUES = (
    "current_time",
    "current_date",
    "random",
    "weather",
    "stock_price",
    "market_price",
    "live_search",
    "web_search",
)
_EXPLICIT_RETRY_CUES = ("retry", "try again", "rerun", "run it again")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _parent_id(record: Mapping[str, Any]) -> str:
    return str(record.get("parent_id") or record.get("id") or "").strip()


def _metadata_marks_retry_valid(record: Mapping[str, Any]) -> bool:
    containers = [record]
    for key in ("metadata", "meta", "labels"):
        nested = record.get(key)
        if isinstance(nested, Mapping):
            containers.append(nested)
    return any(
        bool(container.get(key))
        for container in containers
        for key in _RETRY_METADATA_KEYS
    )


def _executed_tool_is_nondeterministic(
    record: Mapping[str, Any], calls: tuple[tuple[str, str], ...]
) -> bool:
    """Conservatively detect live/random tools from names or supplied schemas."""
    names = {name.lower() for name, _ in calls}
    if any(cue in name for name in names for cue in _NONDETERMINISTIC_TOOL_CUES):
        return True
    schemas = record.get("tools") or record.get("functions") or []
    for schema in schemas if isinstance(schemas, list) else []:
        if not isinstance(schema, Mapping):
            continue
        function = schema.get("function") if isinstance(schema.get("function"), Mapping) else schema
        name = str(function.get("name") or "").lower()
        if name not in names:
            continue
        description = str(function.get("description") or "").lower()
        if any(
            cue in description
            for cue in ("non-deterministic", "nondeterministic", "real-time", "live data", "random")
        ):
            return True
    return False


def _successful_nonretry_observation(record: Mapping[str, Any], content: Any) -> bool:
    """Require an explicit, apparently successful execution observation."""
    if not is_exec_observation(content) or _metadata_marks_retry_valid(record):
        return False
    text = str(content or "").strip().lower()
    if any(cue in text for cue in _BAD_OBSERVATION_CUES):
        return False
    lines = [line.strip() for line in text.splitlines()]
    try:
        start = next(i for i, line in enumerate(lines) if "function execution results" in line) + 1
    except StopIteration:
        return False
    result_lines: list[str] = []
    for line in lines[start:]:
        if line.startswith("respond with") or line.startswith("please continue"):
            break
        if line:
            result_lines.append(line)
    payload = "\n".join(result_lines)
    return bool(payload) and "->" in payload and "no functions were executed" not in payload


def _iffc_from_canonical(calls: tuple[tuple[str, str], ...]) -> str:
    payload: list[dict[str, Any]] = []
    for name, params_json in calls:
        try:
            params = json.loads(params_json)
        except json.JSONDecodeError:
            return ""
        payload.append({"function_name": name, "parameters": params})
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def canonical_response(content: Any) -> tuple[str, Any]:
    """Canonical response form for tool-call and text collision checks."""
    calls = canonical_tool_calls(content)
    if calls is not None:
        return "calls", calls
    return "text", str(content or "").strip()


def _tool_cap_key(calls: tuple[tuple[str, str], ...]) -> str:
    return "+".join(sorted({name for name, _ in calls}))


def pair_schema_example() -> dict[str, Any]:
    """Return a compact schema example for manifests and documentation."""
    return {
        "schema_version": V9_PREF_PAIRS_SCHEMA_VERSION,
        "prompt": [{"role": "user", "content": "...successful observation..."}],
        "chosen": {"role": "assistant", "content": "...advance or final..."},
        "rejected": {"role": "assistant", "content": "...copied prior call..."},
        "pair_kind": PAIR_ADVANCE_VS_LOOP,
        "parent_id": "source:conversation-id",
    }


def _candidate_at(
    record: Mapping[str, Any], index: int
) -> tuple[dict[str, Any] | None, str]:
    messages = list(record.get("messages") or [])
    if index < 2 or messages[index].get("role") != "assistant":
        return None, "not_assistant_after_call_observation"
    observation = messages[index - 1]
    executed = messages[index - 2]
    if observation.get("role") != "user" or executed.get("role") != "assistant":
        return None, "prefix_not_immediate_call_observation"
    if not _successful_nonretry_observation(record, observation.get("content")):
        return None, "retry_or_ambiguous_observation"

    executed_calls = canonical_tool_calls(executed.get("content"))
    if executed_calls is None:
        return None, "preceding_assistant_not_tool_calls"
    if _executed_tool_is_nondeterministic(record, executed_calls):
        return None, "nondeterministic_tool"
    chosen_content = messages[index].get("content")
    chosen_action = classify_assistant_action(messages, index)
    if chosen_action not in {ACTION_MT_TOOL_CHAIN_ADVANCE, ACTION_FINAL_AFTER_OBS}:
        return None, "chosen_not_advance_or_final"
    if chosen_action == ACTION_FINAL_AFTER_OBS and any(
        cue in str(chosen_content or "").lower() for cue in _EXPLICIT_RETRY_CUES
    ):
        return None, "explicit_retry_chosen"
    chosen_calls = canonical_tool_calls(chosen_content)
    if chosen_action == ACTION_MT_TOOL_CHAIN_ADVANCE and (
        chosen_calls is None or chosen_calls == executed_calls
    ):
        return None, "chosen_does_not_advance"

    rejected_content = _iffc_from_canonical(executed_calls)
    if not rejected_content:
        return None, "reject_serialization_failed"
    if canonical_response(chosen_content) == canonical_response(rejected_content):
        return None, "chosen_rejected_collision"

    prompt = messages[:index]
    prompt_hash = _sha256(prompt)
    pair = {
        "schema_version": V9_PREF_PAIRS_SCHEMA_VERSION,
        "prompt": prompt,
        "chosen": {"role": "assistant", "content": chosen_content},
        "rejected": {"role": "assistant", "content": rejected_content},
        "pair_kind": PAIR_ADVANCE_VS_LOOP,
        "parent_id": _parent_id(record),
        "source": str(record.get("source") or "unknown"),
        "chosen_action": chosen_action,
        "rejected_action": "repeat_preceding_executed_call",
        "prompt_sha256": prompt_hash,
        "chosen_prompt_sha256": prompt_hash,
        "rejected_prompt_sha256": prompt_hash,
        "executed_call_sha256": _sha256(executed_calls),
        "tool_key": _tool_cap_key(executed_calls),
    }
    return pair, "accepted"


def validate_pair(pair: Mapping[str, Any]) -> list[str]:
    """Return invariant failures for one mined pair."""
    failures: list[str] = []
    prompt = pair.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        return ["missing_prompt"]
    prompt_hash = _sha256(prompt)
    if (
        pair.get("chosen_prompt_sha256") != prompt_hash
        or pair.get("rejected_prompt_sha256") != prompt_hash
    ):
        failures.append("prompt_mismatch")
    if (
        prompt[-1].get("role") != "user"
        or not is_exec_observation(prompt[-1].get("content"))
    ):
        failures.append("prompt_not_ending_in_observation")
    chosen = (pair.get("chosen") or {}).get("content")
    rejected = (pair.get("rejected") or {}).get("content")
    if canonical_response(chosen) == canonical_response(rejected):
        failures.append("chosen_equals_rejected")
    prior = prompt[-2] if len(prompt) >= 2 else {}
    if (
        prior.get("role") != "assistant"
        or canonical_response(prior.get("content"))
        != canonical_response(rejected)
    ):
        failures.append("reject_not_preceding_executed_call")
    return failures


def mine_advance_vs_loop_candidates(
    conversations: Iterable[dict[str, Any]],
    *,
    max_per_parent: int = 2,
    max_per_tool: int = 200,
    stats: Counter[str] | None = None,
) -> list[dict[str, Any]]:
    """Mine constructed-repeat pairs after parent splitting, then dedupe/cap."""
    counts = stats if stats is not None else Counter()
    raw: list[dict[str, Any]] = []
    for record in conversations:
        if bfcl_leak_reasons(record):
            counts["excluded_bfcl"] += 1
            continue
        if not _parent_id(record):
            counts["excluded_missing_parent_id"] += 1
            continue
        for index, message in enumerate(record.get("messages") or []):
            if message.get("role") != "assistant":
                continue
            pair, reason = _candidate_at(record, index)
            counts[reason] += 1
            if pair is not None:
                raw.append(pair)

    raw.sort(
        key=lambda p: (
            p["parent_id"],
            p["prompt_sha256"],
            _sha256(p["chosen"]),
            _sha256(p["rejected"]),
        )
    )
    seen: set[str] = set()
    by_parent: Counter[str] = Counter()
    by_tool: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    for pair in raw:
        digest = _sha256([pair["prompt"], pair["chosen"], pair["rejected"]])
        if digest in seen:
            counts["excluded_duplicate"] += 1
            continue
        seen.add(digest)
        if by_parent[pair["parent_id"]] >= max_per_parent:
            counts["excluded_parent_cap"] += 1
            continue
        if by_tool[pair["tool_key"]] >= max_per_tool:
            counts["excluded_tool_cap"] += 1
            continue
        if validate_pair(pair):
            counts["excluded_invariant_failure"] += 1
            continue
        by_parent[pair["parent_id"]] += 1
        by_tool[pair["tool_key"]] += 1
        kept.append(pair)
    counts["pairs_kept"] += len(kept)
    return kept


def _hash_split(parent_id: str) -> str:
    digest = hashlib.sha256(f"{SEED_V9}:{parent_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "train" if bucket < 90 else "valid" if bucket < 95 else "corpus_test"


def split_conversations_by_parent(
    conversations: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Assign whole parents to deterministic splits before pair expansion."""
    result = {name: [] for name in SPLIT_NAMES}
    assigned: dict[str, str] = {}
    for record in conversations:
        parent = _parent_id(record)
        split = assigned.setdefault(parent, _hash_split(parent))
        result[split].append(record)
    return result


def assert_parent_disjoint(
    splits: Mapping[str, Iterable[Mapping[str, Any]]]
) -> None:
    owner: dict[str, str] = {}
    for split, records in splits.items():
        for record in records:
            parent = _parent_id(record)
            if not parent:
                continue
            previous = owner.setdefault(parent, split)
            if previous != split:
                raise ValueError(f"parent_id {parent!r} crosses {previous}/{split}")


def mine_splits(
    splits: Mapping[str, list[dict[str, Any]]],
    *,
    max_per_parent: int = 2,
    max_per_tool: int = 200,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    assert_parent_disjoint(splits)
    mined: dict[str, list[dict[str, Any]]] = {}
    exclusions: dict[str, dict[str, int]] = {}
    for name in SPLIT_NAMES:
        counts: Counter[str] = Counter()
        mined[name] = mine_advance_vs_loop_candidates(
            splits.get(name, []),
            max_per_parent=max_per_parent,
            max_per_tool=max_per_tool,
            stats=counts,
        )
        exclusions[name] = dict(sorted(counts.items()))
    parents = [{p["parent_id"] for p in mined[name]} for name in SPLIT_NAMES]
    invariant_failures = sum(
        len(validate_pair(pair)) for rows in mined.values() for pair in rows
    )
    no_bfcl = all(
        not bfcl_leak_reasons(
            {
                "id": pair["parent_id"],
                "source": pair["source"],
                "messages": pair["prompt"] + [pair["chosen"], pair["rejected"]],
            }
        )
        for rows in mined.values()
        for pair in rows
    )
    gate_report = {
        "zero_bfcl_leak": no_bfcl,
        "parent_disjoint": not any(
            parents[i] & parents[j]
            for i in range(3)
            for j in range(i + 1, 3)
        ),
        "prompt_hashes_identical": all(
            p["chosen_prompt_sha256"]
            == p["rejected_prompt_sha256"]
            == p["prompt_sha256"]
            for rows in mined.values()
            for p in rows
        ),
        "pair_invariants": invariant_failures == 0,
        "invariant_failure_count": invariant_failures,
    }
    report = {
        "schema_version": V9_PREF_PAIRS_SCHEMA_VERSION,
        "training": False,
        "counts": {name: len(mined[name]) for name in SPLIT_NAMES},
        "chosen_actions": dict(
            sorted(
                Counter(
                    p["chosen_action"] for rows in mined.values() for p in rows
                ).items()
            )
        ),
        "exclusions": exclusions,
        "gate_report": gate_report,
        "sample_pairs_head": [
            p for name in SPLIT_NAMES for p in mined[name]
        ][:3],
        "note": (
            "Stage B is primary after v9 quotas; scaffold only. Initialize "
            "policy/reference from google/gemma-4-e2b-it, never failed v9 adapters."
        ),
    }
    return mined, report


def dry_run_report(conversations: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the same report as the CLI without writing files."""
    _, report = mine_splits(split_conversations_by_parent(conversations))
    report["pairs_mined"] = sum(report["counts"].values())
    report["sample_schema"] = pair_schema_example()
    return report


def _load_splits(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {name: [] for name in SPLIT_NAMES}
    if path.is_file():
        return split_conversations_by_parent(read_jsonl(path))
    if path.is_dir():
        found = {
            name: read_jsonl(path / f"{name}.jsonl")
            for name in SPLIT_NAMES
            if (path / f"{name}.jsonl").exists()
        }
        if found:
            return {name: found.get(name, []) for name in SPLIT_NAMES}
    raise ValueError(f"No JSONL source splits found at {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mine exact-prefix Stage B advance-vs-loop pairs; never trains."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print report only (default unless --write).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Explicitly write mined JSONL and gate_report.json.",
    )
    parser.add_argument(
        "--splits", type=Path, default=None, help="Conversation JSONL or split dir."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-per-parent", type=int, default=2)
    parser.add_argument("--max-per-tool", type=int, default=200)
    args = parser.parse_args(argv)
    if args.max_per_parent < 1 or args.max_per_tool < 1:
        parser.error("caps must be positive")

    splits = _load_splits(args.splits)
    mined, report = mine_splits(
        splits,
        max_per_parent=args.max_per_parent,
        max_per_tool=args.max_per_tool,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.write:
        required = (
            "zero_bfcl_leak",
            "parent_disjoint",
            "prompt_hashes_identical",
            "pair_invariants",
        )
        if not all(report["gate_report"][key] for key in required):
            parser.error("refusing to write: preference gates failed")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in mined.items():
            write_jsonl(args.out_dir / f"{name}.jsonl", rows)
        (args.out_dir / "gate_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"wrote preference JSONL -> {args.out_dir}")
    elif not args.dry_run:
        print("dry-run is the default; pass --write to create JSONL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
