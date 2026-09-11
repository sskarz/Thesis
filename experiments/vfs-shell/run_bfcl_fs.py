#!/usr/bin/env python3
"""CLI: BFCL GorillaFileSystem slice, shell dialect → multi_turn_checker.

Does not change NFC (--fc-mode nfc) or IFFC (--fc-mode iffc) defaults.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
# Script dir first (vfs_shell), then repo root so `benchmark.config` imports
# under `uv run python experiments/vfs-shell/run_bfcl_fs.py`.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ROOT))

from vfs_shell.bfcl_fs_eval import (  # noqa: E402
    bfcl_execute,
    flatten_recorded_emissions,
    generate_shell_for_entry,
    llama_generate_fn,
    load_jsonl,
    load_jsonl_map,
    score_shell_results,
)
from vfs_shell.models import ScriptedModel  # noqa: E402
from vfs_shell.slice import SLICES, filter_slice, involved_classes  # noqa: E402


_CATEGORY_FILES = {
    "multi_turn_base": "BFCL_v4_multi_turn_base.json",
    "multi_turn_miss_func": "BFCL_v4_multi_turn_miss_func.json",
    "multi_turn_miss_param": "BFCL_v4_multi_turn_miss_param.json",
    "multi_turn_long_context": "BFCL_v4_multi_turn_long_context.json",
}


def _bfcl_paths(category: str) -> tuple[Path, Path]:
    """Resolve BFCL jsonl paths without changing NFC/IFFC config."""
    try:
        from benchmark import config

        info = config.MULTI_TURN_CATEGORIES[category]
        data = config.BFCL_DATA_DIR / info["file"]
        answers = config.BFCL_POSSIBLE_ANSWER_DIR / info["file"]
        return data, answers
    except ImportError:
        pass
    import bfcl_eval

    fname = _CATEGORY_FILES.get(category, f"BFCL_v4_{category}.json")
    root = Path(bfcl_eval.__file__).parent / "data"
    return root / fname, root / "possible_answer" / fname


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Shell-dialect eval on the GorillaFileSystem slice of BFCL "
            "multi_turn_base. NFC/IFFC pipelines are not used."
        )
    )
    parser.add_argument(
        "--slice",
        choices=SLICES,
        required=True,
        help="fs-only = involved_classes == [GorillaFileSystem] (~13); "
        "any-fs = GorillaFileSystem in involved_classes (~50)",
    )
    parser.add_argument("--category", default="multi_turn_base")
    parser.add_argument("--data", type=Path, help="Override BFCL test jsonl")
    parser.add_argument("--answers", type=Path, help="Override possible-answer jsonl")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching entry IDs; no model, no execute",
    )
    parser.add_argument(
        "--recorded",
        type=Path,
        help="JSONL of {id, emissions: [[cmd, ...], ...]} instead of llama-server",
    )
    parser.add_argument(
        "--model",
        default="gemma4-e2b",
        help="Registry key (default: gemma4-e2b). Used for live API model name.",
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8080",
        help="llama-server base URL (default: http://localhost:8080)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, help="Write result JSONL here")
    parser.add_argument(
        "--skip-score",
        action="store_true",
        help="Generate mapped traces only; do not call multi_turn_checker",
    )
    args = parser.parse_args(argv)

    if args.data:
        data_path = args.data
        answer_path = args.answers
    else:
        try:
            data_path, default_answers = _bfcl_paths(args.category)
        except ImportError as exc:
            print(
                "bfcl_eval/benchmark.config not importable; pass --data (and --answers).",
                file=sys.stderr,
            )
            print(exc, file=sys.stderr)
            return 2
        answer_path = args.answers or default_answers

    if not data_path.exists():
        print(f"data file not found: {data_path}", file=sys.stderr)
        return 2

    entries = filter_slice(load_jsonl(data_path), args.slice)
    if args.limit is not None:
        entries = entries[: args.limit]

    print(
        f"[shell-fs] category={args.category} slice={args.slice} "
        f"n={len(entries)} data={data_path}"
    )
    mixed = sum(1 for e in entries if involved_classes(e) != ["GorillaFileSystem"])
    if mixed:
        print(
            f"[shell-fs] note: {mixed} entries also involve non-FS APIs; "
            "this ablation only maps file-system commands."
        )

    if args.dry_run:
        for entry in entries:
            classes = ",".join(involved_classes(entry))
            print(f"{entry['id']}\t{classes}")
        return 0

    recorded_map: dict[str, dict] = {}
    generate_live = None
    api_model = args.model
    if args.recorded:
        recorded_map = load_jsonl_map(args.recorded)
    else:
        try:
            from benchmark import config

            if args.model in config.MODELS:
                config.select_model(args.model)
                api_model = config.MODEL_NAME
        except ImportError:
            api_model = args.model
        generate_live = llama_generate_fn(args.server_url, api_model)

    exec_model_name = f"{api_model}_shell_fs_gen"
    result_rows: list[dict] = []
    for entry in entries:
        if args.recorded:
            rec = recorded_map.get(entry["id"])
            if rec is None:
                print(f"  [skip] no recorded emissions for {entry['id']}", file=sys.stderr)
                continue
            model = ScriptedModel(flatten_recorded_emissions(rec["emissions"]))
            generate_fn = model.generate
        else:
            generate_fn = generate_live

        turn_results, timing = generate_shell_for_entry(
            entry,
            args.category,
            generate_fn,
            bfcl_execute,
            exec_model_name=exec_model_name,
        )
        result_rows.append({
            "id": entry["id"],
            "result": turn_results,
            "flags": timing.get("flags", ""),
        })
        n_calls = sum(len(step) for turn in turn_results for step in turn)
        print(f"  {entry['id']}: steps={n_calls} flags={timing.get('flags') or '-'}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as fh:
            for row in result_rows:
                fh.write(json.dumps(row) + "\n")
        print(f"[shell-fs] wrote {len(result_rows)} traces → {args.out}")

    if args.skip_score:
        print("[shell-fs] scoring skipped")
        return 0
    if not answer_path or not Path(answer_path).exists():
        print(f"[shell-fs] no answers at {answer_path}; skip score", file=sys.stderr)
        return 0

    # Checker reads initial_config off the original test entries.
    all_tests = load_jsonl_map(data_path)
    answers = load_jsonl_map(Path(answer_path))
    summary = score_shell_results(
        category=args.category,
        test_entries=all_tests,
        answers=answers,
        results=result_rows,
        # Distinct from NFC `{MODEL_NAME}_eval` instance keys.
        model_name=f"{api_model}_shell_fs",
    )
    print(
        f"[shell-fs] SCORE {args.slice}: "
        f"{summary['correct']}/{summary['total']} = {summary['accuracy']:.1%}"
    )
    print(
        "[shell-fs] NFC/IFFC controls (full 200 multi_turn_base, unchanged):\n"
        "  NFC:  uv run python -m benchmark.run_all --model gemma4-e2b "
        "--only-multi-turn --categories multi_turn_base --fc-mode nfc\n"
        "  IFFC: uv run python -m benchmark.run_all --model gemma4-e2b "
        "--only-multi-turn --categories multi_turn_base --fc-mode iffc\n"
        "Compare FS-slice accuracy on the IDs listed by --dry-run, not vs the full 200."
    )
    if summary["errors"]:
        print(f"[shell-fs] {len(summary['errors'])} errors")
        for err in summary["errors"][:10]:
            print(f"  {err['id']}: {err['error']}")
    return 0 if summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
