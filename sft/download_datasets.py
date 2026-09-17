"""Download ToolACE + xLAM-60k to ``sft/data/raw/``.

xLAM is gated: set ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` or run
``huggingface-cli login`` first.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from sft.config import (
    RAW_DIR,
    RAW_TOOLACE,
    RAW_XLAM,
    TOOLACE_HF_ID,
    XLAM_HF_ID,
)
from sft.io import write_jsonl


def _hf_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )


def _require_datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "The HuggingFace ``datasets`` package is required to download SFT "
            "corpora.\n"
            "  uv sync --extra sft\n"
            "  # or: uv pip install datasets huggingface_hub"
        ) from exc
    return load_dataset


def _row_to_dict(row: Any, index: int) -> dict[str, Any]:
    if hasattr(row, "keys"):
        rec = {k: row[k] for k in row.keys()}
    else:
        rec = dict(row)
    rec.setdefault("row_index", index)
    return rec


def download_hf_split(
    hf_id: str,
    out_path: Path,
    *,
    split: str = "train",
    token: str | None = None,
    gated: bool = False,
) -> int:
    load_dataset = _require_datasets()
    kwargs: dict[str, Any] = {"split": split}
    if token:
        kwargs["token"] = token
    try:
        ds = load_dataset(hf_id, **kwargs)
    except Exception as exc:
        if gated:
            raise SystemExit(
                f"Failed to download gated dataset {hf_id}.\n"
                "Pass a HuggingFace token via HF_TOKEN or run:\n"
                "  huggingface-cli login\n"
                "then accept the dataset license on the Hub page.\n"
                f"Original error: {exc}"
            ) from exc
        raise SystemExit(f"Failed to download {hf_id}: {exc}") from exc

    rows = [_row_to_dict(ds[i], i) for i in range(len(ds))]
    n = write_jsonl(out_path, rows)
    print(f"wrote {n} rows → {out_path}")
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download ToolACE + xLAM-60k into sft/data/raw/."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory for raw JSONL (default: sft/data/raw)",
    )
    parser.add_argument(
        "--skip-toolace",
        action="store_true",
        help="Skip Team-ACE/ToolACE",
    )
    parser.add_argument(
        "--skip-xlam",
        action="store_true",
        help="Skip Salesforce/xlam-function-calling-60k",
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    token = _hf_token()

    if not args.skip_toolace:
        download_hf_split(
            TOOLACE_HF_ID,
            out_dir / RAW_TOOLACE.name,
            token=token,
            gated=False,
        )
    if not args.skip_xlam:
        download_hf_split(
            XLAM_HF_ID,
            out_dir / RAW_XLAM.name,
            token=token,
            gated=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
