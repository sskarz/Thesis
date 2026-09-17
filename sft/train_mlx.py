"""Thin wrapper around ``mlx_lm.lora`` for the IFFC SFT v1 mix.

Unrolls conversation-level splits so ``--mask-prompt`` trains every assistant
turn. Does **not** run training in CI; on Linux this prints the command.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from sft.config import (
    ADAPTER_DIR,
    FUSED_DIR,
    HF_INSTRUCT_ID,
    LORA_BATCH_SIZE,
    LORA_CONFIG_PATH,
    LORA_ITERS_FULL,
    LORA_ITERS_SMOKE,
    LORA_LEARNING_RATE,
    LORA_MAX_SEQ_LENGTH,
    LORA_NUM_LAYERS,
    LORA_RANK,
    MLX_CONVERT_HINT,
    MLX_CONVERT_QLORA_HINT,
    MLX_DATA_DIR,
    MLX_MODEL_DIR,
    SMOKE_TRAIN_ROWS,
    SMOKE_VALID_ROWS,
    SPLITS_DIR,
    TRAIN_JSONL,
    VALID_JSONL,
)
from sft.format_iffc import unroll_records
from sft.io import read_jsonl, write_jsonl


def prepare_mlx_data(
    *,
    splits_dir: Path = SPLITS_DIR,
    mlx_dir: Path = MLX_DATA_DIR,
    smoke: bool = False,
    smoke_train_rows: int = SMOKE_TRAIN_ROWS,
    smoke_valid_rows: int = SMOKE_VALID_ROWS,
) -> Path:
    """Unroll train/valid conversations into mlx-lm ``{train,valid}.jsonl``."""
    train = read_jsonl(splits_dir / TRAIN_JSONL.name)
    valid = read_jsonl(splits_dir / VALID_JSONL.name)
    train_u = unroll_records(train)
    valid_u = unroll_records(valid)
    if smoke:
        train_u = train_u[:smoke_train_rows]
        valid_u = valid_u[:smoke_valid_rows]
    if mlx_dir.exists():
        shutil.rmtree(mlx_dir)
    mlx_dir.mkdir(parents=True, exist_ok=True)
    n_train = write_jsonl(mlx_dir / "train.jsonl", train_u)
    n_valid = write_jsonl(mlx_dir / "valid.jsonl", valid_u)
    print(
        f"mlx data: unrolled train={n_train} valid={n_valid} → {mlx_dir} "
        f"(from {len(train)}/{len(valid)} conversations; smoke={smoke})"
    )
    return mlx_dir


def build_lora_command(
    *,
    model: str,
    data_dir: Path,
    adapter_path: Path,
    config_path: Path,
    iters: int,
    batch_size: int = LORA_BATCH_SIZE,
    num_layers: int = LORA_NUM_LAYERS,
    learning_rate: float = LORA_LEARNING_RATE,
    max_seq_length: int = LORA_MAX_SEQ_LENGTH,
    extra: list[str] | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        model,
        "--train",
        "--data",
        str(data_dir),
        "--config",
        str(config_path),
        "--adapter-path",
        str(adapter_path),
        "--iters",
        str(iters),
        "--batch-size",
        str(batch_size),
        "--num-layers",
        str(num_layers),
        "--learning-rate",
        str(learning_rate),
        "--max-seq-length",
        str(max_seq_length),
        "--mask-prompt",
        "--save-every",
        "100",
    ]
    if extra:
        cmd.extend(extra)
    return cmd


def fuse_command(model: str, adapter_path: Path, save_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "mlx_lm.fuse",
        "--model",
        model,
        "--adapter-path",
        str(adapter_path),
        "--save-path",
        str(save_path),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare unrolled mlx-lm data and print/run LoRA training."
    )
    parser.add_argument("--smoke", action="store_true", help="100 iters + tiny slice")
    parser.add_argument("--full", action="store_true", help="Full default iters")
    parser.add_argument("--run", action="store_true", help="Exec mlx_lm.lora (Mac)")
    parser.add_argument("--fuse", action="store_true", help="Print/run fuse after train")
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--rank", type=int, default=LORA_RANK, help="Documented in yaml (16–32)")
    parser.add_argument("--model", type=str, default=str(MLX_MODEL_DIR))
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--mlx-dir", type=Path, default=MLX_DATA_DIR)
    parser.add_argument("--adapter-path", type=Path, default=ADAPTER_DIR)
    parser.add_argument("--batch-size", type=int, default=LORA_BATCH_SIZE)
    parser.add_argument("--max-seq-length", type=int, default=LORA_MAX_SEQ_LENGTH)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only unroll JSONL; do not invoke mlx_lm",
    )
    args = parser.parse_args(argv)

    if not (args.splits_dir / TRAIN_JSONL.name).exists():
        parser.error(
            f"Missing {args.splits_dir / TRAIN_JSONL.name}. "
            "Run download → format → split first (see sft/README.md)."
        )

    smoke = args.smoke and not args.full
    iters = args.iters
    if iters is None:
        iters = LORA_ITERS_SMOKE if smoke else LORA_ITERS_FULL

    data_dir = prepare_mlx_data(
        splits_dir=args.splits_dir,
        mlx_dir=args.mlx_dir,
        smoke=smoke,
    )

    if args.rank != LORA_RANK:
        print(
            f"note: --rank {args.rank} is recorded here; mlx-lm reads rank from "
            f"{LORA_CONFIG_PATH}. Edit lora_parameters.rank in that yaml."
        )

    model = args.model
    if model == str(MLX_MODEL_DIR) and not MLX_MODEL_DIR.exists():
        print(
            f"MLX weights not found at {MLX_MODEL_DIR}. Convert on the Mac:\n"
            f"  {MLX_CONVERT_HINT}\n"
            f"QLoRA (4-bit):\n"
            f"  {MLX_CONVERT_QLORA_HINT}\n"
            f"HF instruct id: {HF_INSTRUCT_ID}"
        )

    cmd = build_lora_command(
        model=model,
        data_dir=data_dir,
        adapter_path=args.adapter_path,
        config_path=LORA_CONFIG_PATH,
        iters=iters,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )
    print("train command:")
    print(" ", " ".join(cmd))

    fused = FUSED_DIR / "gemma-4-e2b-it-iffc"
    fuse_cmd = fuse_command(model, args.adapter_path, fused)
    print("fuse command:")
    print(" ", " ".join(fuse_cmd))

    if args.prepare_only:
        return 0

    if not args.run:
        print("dry-run (pass --run on Apple Silicon to execute mlx_lm.lora)")
        return 0

    if os.environ.get("SFT_SKIP_MLX") == "1":
        print("SFT_SKIP_MLX=1; not executing mlx_lm")
        return 0

    args.adapter_path.mkdir(parents=True, exist_ok=True)
    print("running mlx_lm.lora …")
    subprocess.run(cmd, check=True)
    if args.fuse:
        fused.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(fuse_cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
