"""Thin wrapper around ``mlx_lm.lora`` for the IFFC SFT mix.

Unrolls conversation-level splits so ``--mask-prompt`` trains every assistant
turn, then applies the SFT v5 curriculum by default (v4 empty-after-exec +
recovery + first-assistant / clarifying ``[]`` drops, plus turn-0 completeness
and env-state consistency; see ``sft.filter_empty_after_exec``). Pass
``--recipe v6`` for v4 empty/recovery plus only ``drop_state_inconsistent``.
Pass ``--recipe v7`` for the APIGen-MT mix (defaults ``splits-v7`` /
``mlx-v7`` / ``adapters-v7``; v4 empty/recovery on filler, no v5
complete-turn0, then ``sft.cap_nontool_v7`` so mlx-v7 ``pct_nontool`` is
~15%). Pass ``--recipe v8`` for lever 1 (defaults ``splits-v8`` /
``mlx-v8`` / ``adapters-v8``; same v4 empty/recovery, **no** v5
complete-turn0, identical-call drop, then non-tool cap ~30% seed 48).
Does **not** run training in CI; on Linux this prints the
command.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from sft.config import (
    ADAPTER_DIR,
    ADAPTER_DIR_V7,
    ADAPTER_DIR_V8,
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
    MLX_DATA_DIR_V7,
    MLX_DATA_DIR_V8,
    MLX_MODEL_DIR,
    SMOKE_TRAIN_ROWS,
    SMOKE_VALID_ROWS,
    SPLITS_DIR,
    SPLITS_DIR_V7,
    SPLITS_DIR_V8,
    TRAIN_JSONL,
    VALID_JSONL,
)
from sft.filter_empty_after_exec import (
    RECIPE_IDS,
    RECIPE_V7,
    RECIPE_V8,
    RECIPES_APIGEN_MIX,
    RECIPES_WITH_RECOVERY,
    compare_recipe_unroll,
    format_recipe_comparison,
    infer_recipe,
    recipe_drop_flags,
    recipe_v4_drop_flags,
    recipe_v5_flags,
    resolve_recipe,
    unroll_and_filter,
)
from sft.io import read_jsonl, write_jsonl


def prepare_mlx_data(
    *,
    splits_dir: Path = SPLITS_DIR,
    mlx_dir: Path = MLX_DATA_DIR,
    smoke: bool = False,
    smoke_train_rows: int = SMOKE_TRAIN_ROWS,
    smoke_valid_rows: int = SMOKE_VALID_ROWS,
    drop_nonterminal_empty_after_exec: bool = True,
    drop_terminal_empty_after_exec: bool = True,
    drop_first_assistant_empty: bool | None = None,
    drop_clarifying_empty_after_user: bool | None = None,
    drop_underspecified_turn0: bool | None = None,
    drop_state_inconsistent: bool | None = None,
    synthesize_complete_turn0: bool | None = None,
    synthesize_recovery_stop: bool = True,
    recipe: str | None = None,
    compare_recipe: str | None = None,
    max_seq_length: int = LORA_MAX_SEQ_LENGTH,
    budget_tokenizer: Any | None = None,
) -> Path:
    """Unroll train/valid conversations into mlx-lm ``{train,valid}.jsonl``.

    Default SFT v5 keeps the v4 empty-after-exec drop + recovery curriculum
    and first-assistant / clarifying ``[]`` drops, then adds turn-0
    completeness and env-state consistency. Pass ``recipe="v6"`` for v4
    empty/recovery plus only ``drop_state_inconsistent``. Pass ``recipe="v7"``
    for the APIGen-MT mix (v4 empty/recovery on filler; no complete-turn0;
    then cap non-tool unrolls and hard-gate token-budgeted targets — splits-v7
    conversations stay full). Pass ``recipe="v8"`` for lever 1 (same v4
    empty/recovery; identical-call drop; non-tool cap ~30% seed 48; token
    budget — splits-v8 conversations stay full).
    Conversation-level splits are unchanged.
    """
    if recipe is None:
        recipe = infer_recipe(
            drop_nonterminal_empty_after_exec=drop_nonterminal_empty_after_exec,
            drop_terminal_empty_after_exec=drop_terminal_empty_after_exec,
            drop_first_assistant_empty=drop_first_assistant_empty,
            drop_clarifying_empty_after_user=drop_clarifying_empty_after_user,
            drop_underspecified_turn0=drop_underspecified_turn0,
            drop_state_inconsistent=drop_state_inconsistent,
            synthesize_complete_turn0=synthesize_complete_turn0,
        )
    drop_first_v4, drop_clarify_v4 = recipe_v4_drop_flags(recipe)
    drop_u_v5, drop_s_v5, synth_c_v5, drop_abs_v5 = recipe_v5_flags(recipe)
    if drop_first_assistant_empty is None:
        drop_first_assistant_empty = drop_first_v4
    if drop_clarifying_empty_after_user is None:
        drop_clarifying_empty_after_user = drop_clarify_v4
    if drop_underspecified_turn0 is None:
        drop_underspecified_turn0 = drop_u_v5
    if drop_state_inconsistent is None:
        drop_state_inconsistent = drop_s_v5
    if synthesize_complete_turn0 is None:
        synthesize_complete_turn0 = synth_c_v5
    filter_kwargs = dict(
        drop_nonterminal_empty_after_exec=drop_nonterminal_empty_after_exec,
        drop_terminal_empty_after_exec=drop_terminal_empty_after_exec,
        drop_first_assistant_empty=drop_first_assistant_empty,
        drop_clarifying_empty_after_user=drop_clarifying_empty_after_user,
        drop_underspecified_turn0=drop_underspecified_turn0,
        drop_state_inconsistent=drop_state_inconsistent,
        synthesize_complete_turn0=synthesize_complete_turn0,
        drop_absorbed_independent_followup=drop_abs_v5 and synthesize_complete_turn0,
        synthesize_recovery_stop=synthesize_recovery_stop,
        recipe=recipe,
    )
    train = read_jsonl(splits_dir / TRAIN_JSONL.name)
    valid = read_jsonl(splits_dir / VALID_JSONL.name)
    train_u, train_stats = unroll_and_filter(train, **filter_kwargs)
    valid_u, valid_stats = unroll_and_filter(valid, **filter_kwargs)
    train_cap = valid_cap = None
    train_idrop = valid_idrop = None
    if recipe == RECIPE_V8:
        from sft.drop_identical_call import (
            drop_identical_call_unrolls,
            format_identical_call_report,
        )

        train_u, train_idrop = drop_identical_call_unrolls(train_u)
        valid_u, valid_idrop = drop_identical_call_unrolls(valid_u)
    if recipe in RECIPES_APIGEN_MIX:
        from sft.cap_nontool_v7 import (
            cap_kwargs_for_recipe,
            cap_nontool_unrolls,
            format_cap_report,
        )

        cap_kwargs = cap_kwargs_for_recipe(recipe)
        train_u, train_cap = cap_nontool_unrolls(train_u, **cap_kwargs)
        valid_u, valid_cap = cap_nontool_unrolls(valid_u, **cap_kwargs)
    format_kwargs = dict(
        recipe=recipe,
        drop_nonterminal_empty_after_exec=drop_nonterminal_empty_after_exec,
        drop_terminal_empty_after_exec=drop_terminal_empty_after_exec,
        drop_first_assistant_empty=drop_first_assistant_empty,
        drop_clarifying_empty_after_user=drop_clarifying_empty_after_user,
        synthesize_recovery_stop=synthesize_recovery_stop,
        drop_underspecified_turn0=drop_underspecified_turn0,
        drop_state_inconsistent=drop_state_inconsistent,
        synthesize_complete_turn0=synthesize_complete_turn0,
    )
    print(train_stats.format("train", **format_kwargs))
    print(valid_stats.format("valid", **format_kwargs))
    if train_idrop is not None:
        print(format_identical_call_report(train_idrop))
        print(format_identical_call_report(valid_idrop))
    if train_cap is not None:
        print(format_cap_report(train_cap))
        print(format_cap_report(valid_cap))
    if compare_recipe:
        print(format_recipe_comparison(
            compare_recipe_unroll(
                train, recipe=recipe, baseline=compare_recipe, check_bfcl_leak=False
            ),
            "train",
        ))
        print(format_recipe_comparison(
            compare_recipe_unroll(
                valid, recipe=recipe, baseline=compare_recipe, check_bfcl_leak=False
            ),
            "valid",
        ))
    if smoke:
        train_u = train_u[:smoke_train_rows]
        valid_u = valid_u[:smoke_valid_rows]
    if recipe in RECIPES_APIGEN_MIX:
        from sft.budget_seq import (
            DEFAULT_MODEL_DIR,
            budget_unrolled_rows,
            format_budget_report,
        )

        if budget_tokenizer is None and not DEFAULT_MODEL_DIR.exists():
            print(
                f"token budget ({recipe}): skipped "
                f"(no tokenizer and {DEFAULT_MODEL_DIR} missing; "
                "Mac prepare with local MLX weights will hard-gate)"
            )
        else:
            train_u, train_budget = budget_unrolled_rows(
                train_u,
                max_seq_length=max_seq_length,
                tokenizer=budget_tokenizer,
            )
            valid_u, valid_budget = budget_unrolled_rows(
                valid_u,
                max_seq_length=max_seq_length,
                tokenizer=budget_tokenizer,
            )
            print(format_budget_report(train_budget, f"{recipe} train"))
            print(format_budget_report(valid_budget, f"{recipe} valid"))
    if mlx_dir.exists():
        shutil.rmtree(mlx_dir)
    mlx_dir.mkdir(parents=True, exist_ok=True)
    n_train = write_jsonl(mlx_dir / "train.jsonl", train_u)
    n_valid = write_jsonl(mlx_dir / "valid.jsonl", valid_u)
    print(
        f"mlx data: unrolled train={n_train} valid={n_valid} → {mlx_dir} "
        f"(from {len(train)}/{len(valid)} conversations; smoke={smoke}; "
        f"recipe={recipe}; "
        f"drop_nonterminal_empty_after_exec={drop_nonterminal_empty_after_exec}; "
        f"drop_terminal_empty_after_exec={drop_terminal_empty_after_exec}; "
        f"drop_first_assistant_empty={drop_first_assistant_empty}; "
        f"drop_clarifying_empty_after_user={drop_clarifying_empty_after_user}; "
        f"drop_underspecified_turn0={drop_underspecified_turn0}; "
        f"drop_state_inconsistent={drop_state_inconsistent}; "
        f"synthesize_complete_turn0={synthesize_complete_turn0}; "
        f"synthesize_recovery_stop={synthesize_recovery_stop}"
        + (
            f"; {recipe}_nontool_cap=on pct_nontool "
            f"{(train_cap or {}).get('pct_nontool_before')}→"
            f"{(train_cap or {}).get('pct_nontool_after')}"
            if train_cap is not None
            else ""
        )
        + (
            f"; identical_call_drop={train_idrop.get('dropped')}"
            if train_idrop is not None
            else ""
        )
        + ")"
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
    parser.add_argument(
        "--recipe",
        choices=list(RECIPE_IDS),
        default=None,
        help="Unroll recipe id. Default v5. v8/v7/v6/v4/v3/v2/v1 are other recipes.",
    )
    parser.add_argument(
        "--compare-recipe",
        choices=list(RECIPE_IDS),
        default=None,
        help="Also unroll under this baseline and print row deltas (e.g. v4).",
    )
    parser.add_argument(
        "--keep-nonterminal-empty-after-exec",
        action="store_true",
        help=(
            "SFT v1 unroll: keep every assistant turn (disables the "
            "empty-after-exec drop, including non-terminal [])."
        ),
    )
    parser.add_argument(
        "--keep-terminal-empty-after-exec",
        action="store_true",
        help=(
            "SFT v2 unroll: drop mid-trajectory [] after tool results, but "
            "keep true terminal empty-after-exec."
        ),
    )
    parser.add_argument(
        "--keep-first-assistant-empty",
        action="store_true",
        help=(
            "Keep unrolled rows whose gold is [] and that assistant is the "
            "first in the prefix (disable the v4 turn-0 empty drop)."
        ),
    )
    parser.add_argument(
        "--keep-clarifying-empty-after-user",
        action="store_true",
        help=(
            "Keep user→[] clarifying / chitchat targets when the previous "
            "message is not an IFFC tool-result observation. Pass together "
            "with --keep-first-assistant-empty to ablate back to v3."
        ),
    )
    parser.add_argument(
        "--no-synthetic-recovery",
        action="store_true",
        help=(
            "Do not insert a history-only [] after tool results on the next "
            "user→call unroll row (v3/v4/v5/v6 default inserts it)."
        ),
    )
    parser.add_argument(
        "--keep-underspecified-turn0",
        action="store_true",
        help=(
            "Keep first-call rows whose opening user names ≥2 catalog tools "
            "but gold has one call (disable the v5 underspecified drop)."
        ),
    )
    parser.add_argument(
        "--keep-state-inconsistent",
        action="store_true",
        help=(
            "Keep continue-after-exec gold that uses a user-named entity "
            "the exec observation never confirmed (disable the v5 state drop)."
        ),
    )
    parser.add_argument(
        "--no-synthetic-complete-turn0",
        action="store_true",
        help=(
            "Do not merge independently-knowable later first-episode calls "
            "into the first nonempty gold. Incomplete first-turns are dropped "
            "instead. Pass with --keep-underspecified-turn0 for v6 "
            "(state-inconsistent drop only); pass with both v5 keep-flags "
            "for v4."
        ),
    )
    args = parser.parse_args(argv)

    recipe = resolve_recipe(
        keep_nonterminal_empty_after_exec=args.keep_nonterminal_empty_after_exec,
        keep_terminal_empty_after_exec=args.keep_terminal_empty_after_exec,
        keep_first_assistant_empty=args.keep_first_assistant_empty,
        keep_clarifying_empty_after_user=args.keep_clarifying_empty_after_user,
        keep_underspecified_turn0=args.keep_underspecified_turn0,
        keep_state_inconsistent=args.keep_state_inconsistent,
        no_synthetic_complete_turn0=args.no_synthetic_complete_turn0,
        recipe=args.recipe,
    )
    splits_dir = args.splits_dir
    mlx_dir = args.mlx_dir
    adapter_path = args.adapter_path
    if recipe == RECIPE_V7:
        if splits_dir == SPLITS_DIR:
            splits_dir = SPLITS_DIR_V7
        if mlx_dir == MLX_DATA_DIR:
            mlx_dir = MLX_DATA_DIR_V7
        if adapter_path == ADAPTER_DIR:
            adapter_path = ADAPTER_DIR_V7
    elif recipe == RECIPE_V8:
        if splits_dir == SPLITS_DIR:
            splits_dir = SPLITS_DIR_V8
        if mlx_dir == MLX_DATA_DIR:
            mlx_dir = MLX_DATA_DIR_V8
        if adapter_path == ADAPTER_DIR:
            adapter_path = ADAPTER_DIR_V8

    if not (splits_dir / TRAIN_JSONL.name).exists():
        parser.error(
            f"Missing {splits_dir / TRAIN_JSONL.name}. "
            "Run download → format → split first (see sft/README.md). "
            "For v7: download --apigen-mt-only, format --apigen-mt, "
            "build_mix_v7 → sft/data/splits-v7. "
            "For v8: same ingest, then build_mix_v8 → sft/data/splits-v8."
        )

    smoke = args.smoke and not args.full
    iters = args.iters
    if iters is None:
        iters = LORA_ITERS_SMOKE if smoke else LORA_ITERS_FULL

    drop_nt, drop_t = recipe_drop_flags(recipe)
    drop_first, drop_clarify = recipe_v4_drop_flags(recipe)
    drop_u, drop_s, synth_c, _drop_abs = recipe_v5_flags(recipe)
    if args.keep_first_assistant_empty:
        drop_first = False
    if args.keep_clarifying_empty_after_user:
        drop_clarify = False
    if args.keep_underspecified_turn0:
        drop_u = False
    if args.keep_state_inconsistent:
        drop_s = False
    if args.no_synthetic_complete_turn0:
        synth_c = False
    synthesize = recipe in RECIPES_WITH_RECOVERY and not args.no_synthetic_recovery
    data_dir = prepare_mlx_data(
        splits_dir=splits_dir,
        mlx_dir=mlx_dir,
        smoke=smoke,
        drop_nonterminal_empty_after_exec=drop_nt,
        drop_terminal_empty_after_exec=drop_t,
        drop_first_assistant_empty=drop_first,
        drop_clarifying_empty_after_user=drop_clarify,
        drop_underspecified_turn0=drop_u,
        drop_state_inconsistent=drop_s,
        synthesize_complete_turn0=synth_c,
        synthesize_recovery_stop=synthesize,
        recipe=recipe,
        compare_recipe=args.compare_recipe,
        max_seq_length=args.max_seq_length,
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
        adapter_path=adapter_path,
        config_path=LORA_CONFIG_PATH,
        iters=iters,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )
    print("train command:")
    print(" ", " ".join(cmd))

    fused_name = {
        RECIPE_V7: "gemma-4-e2b-it-iffc-v7",
        RECIPE_V8: "gemma-4-e2b-it-iffc-v8",
    }.get(recipe, "gemma-4-e2b-it-iffc")
    fused = FUSED_DIR / fused_name
    fuse_cmd = fuse_command(model, adapter_path, fused)
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

    adapter_path.mkdir(parents=True, exist_ok=True)
    print("running mlx_lm.lora …")
    subprocess.run(cmd, check=True)
    if args.fuse:
        fused.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(fuse_cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
