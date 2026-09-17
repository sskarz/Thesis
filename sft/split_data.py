"""Frozen train / val / corpus_test split for SFT v1.

Locked policy (do not reopen):

* Pool = ALL ToolACE + stratified 25,000 xLAM rows (oversample parallel).
* Split by conversation/example id, never by turn. seed=42.
* Stratify on source, is_multi_turn, n_tools_in_gold (1 vs ≥2).
* ~90% train / ~5% val / ~5% corpus_test.
* BFCL v4 ``multi_turn_base`` is the thesis test — never in these files.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sft.config import (
    CHAT_FORMAT,
    CORPUS_TEST_FRAC,
    CORPUS_TEST_JSONL,
    FORMATTED_TOOLACE,
    FORMATTED_XLAM,
    GGUF_QUANT,
    GGUF_REPO,
    HF_INSTRUCT_ID,
    MANIFEST_PATH,
    MLX_CONVERT_HINT,
    MLX_CONVERT_QLORA_HINT,
    MLX_VLM_4BIT_ID,
    SEED,
    SKIPPED_DATASETS_V1,
    SOURCE_TOOLACE,
    SOURCE_XLAM,
    SPLITS_DIR,
    THESIS_TEST,
    TRAIN_FRAC,
    TRAIN_JSONL,
    VAL_FRAC,
    VALID_JSONL,
    XLAM_SAMPLE_SIZE,
    BFCL_FORBIDDEN_CONTENT_PATTERNS,
    BFCL_FORBIDDEN_ID_PATTERNS,
)
from sft.io import read_jsonl, sha256_file, write_jsonl

_BFCL_ID_RE = re.compile(
    "|".join(re.escape(p) for p in BFCL_FORBIDDEN_ID_PATTERNS),
    re.IGNORECASE,
)
_BFCL_CONTENT_RE = re.compile(
    "|".join(re.escape(p) for p in BFCL_FORBIDDEN_CONTENT_PATTERNS),
    re.IGNORECASE,
)


def bfcl_leak_reasons(record: dict[str, Any]) -> list[str]:
    """Return reasons this record looks like BFCL eval material."""
    reasons: list[str] = []
    example_id = str(record.get("id") or "")
    if _BFCL_ID_RE.search(example_id):
        reasons.append(f"id:{example_id}")
    source = str(record.get("source") or "")
    if _BFCL_ID_RE.search(source):
        reasons.append(f"source:{source}")
    for msg in record.get("messages") or []:
        content = str(msg.get("content") or "")
        if _BFCL_CONTENT_RE.search(content):
            reasons.append("message_content")
            break
        if "tool_calls" in msg:
            reasons.append("native_tool_calls")
            break
    return reasons


def assert_no_bfcl(records: list[dict[str, Any]]) -> None:
    leaks = []
    for rec in records:
        reasons = bfcl_leak_reasons(rec)
        if reasons:
            leaks.append((rec.get("id"), reasons))
    if leaks:
        preview = ", ".join(f"{i}:{r}" for i, r in leaks[:5])
        raise ValueError(
            f"Refusing to write splits: {len(leaks)} record(s) look like BFCL "
            f"eval material ({preview}). Thesis test is {THESIS_TEST}."
        )


def sample_xlam_rows(
    rows: list[dict[str, Any]],
    n: int = XLAM_SAMPLE_SIZE,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Oversample parallel / multi-answer rows, then fill from single-call.

    Policy: take **all** ``n_tools_in_gold >= 2`` rows (shuffled), then fill
    the remainder of *n* from ``n_tools_in_gold <= 1``. If parallel already
    exceeds *n*, sample *n* from parallel only.
    """
    if n <= 0 or not rows:
        return []
    parallel = [r for r in rows if int(r.get("n_tools_in_gold") or 0) >= 2]
    single = [r for r in rows if int(r.get("n_tools_in_gold") or 0) < 2]

    def _sorted(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(items, key=lambda r: str(r.get("id", "")))

    rng = random.Random(seed)
    parallel = _sorted(parallel)
    single = _sorted(single)
    rng.shuffle(parallel)
    rng.shuffle(single)

    if len(parallel) >= n:
        return parallel[:n]
    remaining = n - len(parallel)
    return parallel + single[:remaining]


def stratum_key(record: dict[str, Any]) -> tuple[str, str, str]:
    source = str(record.get("source") or "unknown")
    multi = "1" if record.get("is_multi_turn") else "0"
    n_bin = str(record.get("n_tools_bin") or ("ge2" if int(record.get("n_tools_in_gold") or 0) >= 2 else "1"))
    return source, multi, n_bin


def split_counts(
    n: int,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = CORPUS_TEST_FRAC,
) -> tuple[int, int, int]:
    """Return (n_train, n_val, n_test) with remainder going to train."""
    if n <= 0:
        return 0, 0, 0
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test
    # Tiny strata: keep at least one train row when possible.
    if n_train <= 0 and n >= 1:
        n_train = 1
        leftover = n - 1
        n_val = leftover // 2
        n_test = leftover - n_val
    return n_train, n_val, n_test


def stratified_split(
    records: list[dict[str, Any]],
    *,
    seed: int = SEED,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = CORPUS_TEST_FRAC,
) -> dict[str, list[dict[str, Any]]]:
    """Split by example id within each (source, is_multi_turn, n_tools_bin) stratum."""
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        buckets[stratum_key(rec)].append(rec)

    rng = random.Random(seed)
    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "valid": [],
        "corpus_test": [],
    }
    for key in sorted(buckets):
        items = sorted(buckets[key], key=lambda r: str(r.get("id", "")))
        rng.shuffle(items)
        n_train, n_val, n_test = split_counts(
            len(items), train_frac, val_frac, test_frac
        )
        splits["train"].extend(items[:n_train])
        splits["valid"].extend(items[n_train : n_train + n_val])
        splits["corpus_test"].extend(items[n_train + n_val : n_train + n_val + n_test])

    for name in splits:
        splits[name].sort(key=lambda r: str(r.get("id", "")))
    return splits


def _count_by(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for rec in records:
        if field == "stratum":
            counter[",".join(stratum_key(rec))] += 1
        else:
            counter[str(rec.get(field))] += 1
    return dict(sorted(counter.items()))


def build_manifest(
    splits: dict[str, list[dict[str, Any]]],
    paths: dict[str, Path],
    *,
    seed: int = SEED,
    xlam_pool: int = 0,
    xlam_sampled: int = 0,
    toolace_n: int = 0,
) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for name, rows in splits.items():
        counts[name] = {
            "n": len(rows),
            "by_source": _count_by(rows, "source"),
            "by_is_multi_turn": _count_by(rows, "is_multi_turn"),
            "by_n_tools_bin": _count_by(rows, "n_tools_bin"),
            "by_stratum": _count_by(rows, "stratum"),
        }
    checksums = {
        paths[name].name: sha256_file(paths[name])
        for name in ("train", "valid", "corpus_test")
        if name in paths and paths[name].exists()
    }
    return {
        "experiment": "sft-v1",
        "seed": seed,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "corpus_test_frac": CORPUS_TEST_FRAC,
        "chat_format": CHAT_FORMAT,
        "thesis_test": THESIS_TEST,
        "xlam_sample_size": XLAM_SAMPLE_SIZE,
        "xlam_oversample": "all_parallel_then_fill_single",
        "xlam_pool_formatted": xlam_pool,
        "xlam_sampled": xlam_sampled,
        "toolace_n": toolace_n,
        "skipped_datasets_v1": list(SKIPPED_DATASETS_V1),
        "split_unit": "conversation/example id (never by turn)",
        "stratify_on": ["source", "is_multi_turn", "n_tools_in_gold (1 vs >=2)"],
        "base_model": {
            "hf_instruct": HF_INSTRUCT_ID,
            "gguf_repo": GGUF_REPO,
            "gguf_quant": GGUF_QUANT,
            "mlx_vlm_4bit_not_for_lora": MLX_VLM_4BIT_ID,
            "mlx_convert": MLX_CONVERT_HINT,
            "mlx_convert_qlora": MLX_CONVERT_QLORA_HINT,
        },
        "counts": counts,
        "checksums_sha256": checksums,
        "files": {name: str(path) for name, path in paths.items()},
    }


def build_pool(
    toolace: list[dict[str, Any]],
    xlam: list[dict[str, Any]],
    *,
    xlam_n: int = XLAM_SAMPLE_SIZE,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], int]:
    sampled = sample_xlam_rows(xlam, n=xlam_n, seed=seed)
    pool = list(toolace) + sampled
    pool.sort(key=lambda r: str(r.get("id", "")))
    return pool, len(sampled)


def write_splits(
    splits: dict[str, list[dict[str, Any]]],
    *,
    out_dir: Path = SPLITS_DIR,
    seed: int = SEED,
    xlam_pool: int = 0,
    xlam_sampled: int = 0,
    toolace_n: int = 0,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": out_dir / TRAIN_JSONL.name,
        "valid": out_dir / VALID_JSONL.name,
        "corpus_test": out_dir / CORPUS_TEST_JSONL.name,
        "manifest": out_dir / MANIFEST_PATH.name,
    }
    all_rows = [r for rows in splits.values() for r in rows]
    assert_no_bfcl(all_rows)
    ids = [r["id"] for r in all_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate example ids in splits")

    write_jsonl(paths["train"], splits["train"])
    write_jsonl(paths["valid"], splits["valid"])
    write_jsonl(paths["corpus_test"], splits["corpus_test"])
    manifest = build_manifest(
        splits,
        paths,
        seed=seed,
        xlam_pool=xlam_pool,
        xlam_sampled=xlam_sampled,
        toolace_n=toolace_n,
    )
    paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build frozen IFFC train/valid/corpus_test splits."
    )
    parser.add_argument("--formatted-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--xlam-n", type=int, default=XLAM_SAMPLE_SIZE)
    args = parser.parse_args(argv)

    toolace_path = FORMATTED_TOOLACE
    xlam_path = FORMATTED_XLAM
    if args.formatted_dir:
        toolace_path = args.formatted_dir / "toolace.jsonl"
        xlam_path = args.formatted_dir / "xlam.jsonl"

    if not toolace_path.exists() and not xlam_path.exists():
        parser.error(
            f"No formatted files at {toolace_path} / {xlam_path}. "
            "Run: uv run python -m sft.format_iffc"
        )

    toolace = read_jsonl(toolace_path) if toolace_path.exists() else []
    xlam = read_jsonl(xlam_path) if xlam_path.exists() else []
    pool, n_sampled = build_pool(toolace, xlam, xlam_n=args.xlam_n, seed=args.seed)
    splits = stratified_split(pool, seed=args.seed)
    manifest = write_splits(
        splits,
        out_dir=args.out_dir,
        seed=args.seed,
        xlam_pool=len(xlam),
        xlam_sampled=n_sampled,
        toolace_n=len(toolace),
    )
    counts = manifest["counts"]
    print(
        "splits: "
        f"train={counts['train']['n']} "
        f"valid={counts['valid']['n']} "
        f"corpus_test={counts['corpus_test']['n']}"
    )
    print(f"manifest → {args.out_dir / MANIFEST_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
