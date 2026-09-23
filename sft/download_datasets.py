"""Download ToolACE + xLAM-60k (+ optional APIGen-MT-5k) to ``sft/data/raw/``.

xLAM and APIGen-MT are Hub-gated: set ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN``
or run ``huggingface-cli login`` first.

APIGen-MT-5k is **CC-BY-NC-4.0**. Sanskar must Accept the Hub license on
``Salesforce/APIGen-MT-5k`` before ``--apigen-mt`` / ``--apigen-mt-only``.
Cite Prabhakar et al. (APIGen-MT, arXiv:2504.03601). Do not commit the JSONL.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from sft.config import (
    APIGEN_MT_CONFIG_NAME,
    APIGEN_MT_HF_ID,
    APIGEN_MT_LICENSE,
    RAW_APIGEN_MT,
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
    config_name: str | None = None,
    config_name_fallback: str | None = None,
) -> int:
    load_dataset = _require_datasets()
    kwargs: dict[str, Any] = {"split": split}
    if token:
        kwargs["token"] = token

    def _load(name: str | None):
        if name:
            return load_dataset(hf_id, name, **kwargs)
        return load_dataset(hf_id, **kwargs)

    names: list[str | None] = [config_name]
    if config_name_fallback and config_name_fallback != config_name:
        names.append(config_name_fallback)

    last_exc: Exception | None = None
    ds = None
    for name in names:
        try:
            ds = _load(name)
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
    if ds is None:
        extra = f" Tried configs {names!r}."
        if gated:
            raise SystemExit(
                f"Failed to download gated dataset {hf_id}.{extra}\n"
                "Pass a HuggingFace token via HF_TOKEN or run:\n"
                "  huggingface-cli login\n"
                "then accept the dataset license on the Hub page "
                f"({APIGEN_MT_LICENSE} for APIGen-MT).\n"
                f"Original error: {last_exc}"
            ) from last_exc
        raise SystemExit(f"Failed to download {hf_id}:{extra} {last_exc}") from last_exc

    rows = [_row_to_dict(ds[i], i) for i in range(len(ds))]
    n = write_jsonl(out_path, rows)
    print(f"wrote {n} rows → {out_path}")
    return n


def _download_apigen_mt(out_dir: Path, token: str | None) -> int:
    return download_hf_split(
        APIGEN_MT_HF_ID,
        out_dir / RAW_APIGEN_MT.name,
        token=token,
        gated=True,
        config_name=None,
        config_name_fallback=APIGEN_MT_CONFIG_NAME,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download ToolACE + xLAM-60k into sft/data/raw/. "
            "Add --apigen-mt / --apigen-mt-only for Salesforce/APIGen-MT-5k "
            f"({APIGEN_MT_LICENSE}, Hub gated)."
        )
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
    parser.add_argument(
        "--apigen-mt",
        action="store_true",
        help=(
            f"Also download {APIGEN_MT_HF_ID} ({APIGEN_MT_LICENSE}, Hub gated). "
            "Accept the Hub license first."
        ),
    )
    parser.add_argument(
        "--apigen-mt-only",
        action="store_true",
        help=f"Only download {APIGEN_MT_HF_ID} (skip ToolACE and xLAM).",
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    token = _hf_token()

    if args.apigen_mt_only:
        _download_apigen_mt(out_dir, token)
        return 0

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
    if args.apigen_mt:
        _download_apigen_mt(out_dir, token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
