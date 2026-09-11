"""Orchestrator: generate → evaluate → collect metrics → visualize.

Function-calling modes (``--fc-mode``):

- ``nfc`` (default): native OpenAI ``tools`` / ``tool_calls``
- ``iffc``: Instruction-Followed Function Calling (arXiv:2608.22472 Appendix B)
  JSON-in-text; used by multi-turn generation only
"""

import argparse
import sys
import time

import requests

from . import config, db
from . import generate_responses
from . import generate_multi_turn
from . import evaluate_responses
from . import evaluate_multi_turn
from . import collect_metrics
from . import visualize


def check_server() -> bool:
    """Return True if the llama-server is reachable."""
    try:
        resp = requests.get(f"{config.SERVER_URL}/v1/models", timeout=5)
        resp.raise_for_status()
        return True
    except Exception:
        return False


def _notes_with_run_meta(args: argparse.Namespace) -> str | None:
    extra: list[str] = []
    if args.fc_mode != config.FC_NFC:
        extra.append(f"fc_mode={args.fc_mode}")
    if args.categories:
        extra.append(f"categories={','.join(args.categories)}")
    if not extra:
        return args.notes
    tag = " ".join(extra)
    return f"{args.notes}; {tag}" if args.notes else tag


def main():
    parser = argparse.ArgumentParser(description="Run the BFCL benchmark pipeline.")
    parser.add_argument(
        "--model",
        choices=config.MODELS.keys(),
        default=config.DEFAULT_MODEL,
        help="Model to benchmark (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-multi-turn",
        action="store_true",
        help="Skip multi-turn categories (e.g., for FunctionGemma)",
    )
    parser.add_argument(
        "--only-multi-turn",
        action="store_true",
        help="Run only multi-turn categories (skip single-turn)",
    )
    parser.add_argument(
        "--fc-mode",
        choices=config.FC_MODES,
        default=config.FC_NFC,
        help=(
            "Function-calling mode for multi-turn. "
            "nfc = native OpenAI tools/tool_calls (default); "
            "iffc = Instruction-Followed Function Calling "
            "(arXiv:2608.22472 Appendix B JSON-in-text)."
        ),
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        metavar="CATEGORY",
        default=None,
        help=(
            "Subset of multi-turn categories to run "
            f"(choices: {', '.join(config.MULTI_TURN_CATEGORIES)}). "
            "Example: --categories multi_turn_base"
        ),
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Freeform annotation for this run",
    )
    args = parser.parse_args()
    if args.skip_multi_turn and args.only_multi_turn:
        parser.error("--skip-multi-turn and --only-multi-turn are mutually exclusive")
    if args.categories and args.skip_multi_turn:
        parser.error("--categories applies to multi-turn; cannot combine with --skip-multi-turn")
    try:
        selected_mt = config.resolve_multi_turn_categories(args.categories)
    except ValueError as e:
        parser.error(str(e))
    config.select_model(args.model)
    config.set_fc_mode(args.fc_mode)

    do_single = not args.only_multi_turn
    do_multi = not args.skip_multi_turn
    n_steps = (2 if do_single else 0) + (2 if do_multi else 0) + 3  # server + metrics + viz

    notes = _notes_with_run_meta(args)

    # Create a new run in the database
    run_id = db.create_run(model_key=args.model, notes=notes)
    config.RUN_ID = run_id

    print("=" * 60)
    print("BFCL Benchmark Pipeline")
    print(f"Model : {config.MODEL_DISPLAY}")
    print(f"Server: {config.SERVER_URL}")
    print(f"Run ID: {run_id}")
    print(
        f"FC mode: {args.fc_mode}"
        + (
            " (IFFC = arXiv:2608.22472 Appendix B JSON-in-text)"
            if args.fc_mode == config.FC_IFFC
            else " (NFC = native tools)"
        )
    )
    if not do_single:
        print("Single-turn: SKIPPED")
    if not do_multi:
        print("Multi-turn: SKIPPED")
    elif args.categories:
        print(f"Multi-turn categories: {', '.join(selected_mt)}")
    print("=" * 60)

    step = 0

    # 1. Verify server
    step += 1
    print(f"\n[{step}/{n_steps}] Checking server...", end=" ")
    if not check_server():
        print("FAILED")
        print(f"Server not reachable at {config.SERVER_URL}")
        print("Start llama-server first (see CLAUDE.md for the command).")
        sys.exit(1)
    print("OK")

    t0 = time.time()

    # Generate single-turn responses
    if do_single:
        step += 1
        print(f"\n[{step}/{n_steps}] Generating single-turn responses")
        generate_responses.run()

    # Generate multi-turn responses
    if do_multi:
        step += 1
        print(f"\n[{step}/{n_steps}] Generating multi-turn responses")
        generate_multi_turn.run(categories=selected_mt, fc_mode=args.fc_mode)

    # Evaluate single-turn responses
    if do_single:
        step += 1
        print(f"\n[{step}/{n_steps}] Evaluating single-turn responses")
        evaluate_responses.run()

    # Evaluate multi-turn responses (scores saved directly to DB by each evaluator)
    if do_multi:
        step += 1
        print(f"\n[{step}/{n_steps}] Evaluating multi-turn responses")
        evaluate_multi_turn.run(categories=selected_mt)

    # Collect metrics
    step += 1
    print(f"\n[{step}/{n_steps}] Collecting metrics")
    collect_metrics.run(
        include_single=do_single,
        include_multi=do_multi,
        multi_categories=selected_mt if do_multi else None,
    )

    # Visualize
    step += 1
    print(f"\n[{step}/{n_steps}] Generating charts")
    visualize.run(include_single=do_single, include_multi=do_multi)

    # Finish the run
    db.finish_run(run_id)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} minutes.")
    print(f"Results in DB (run_id={run_id})")
    print(f"Figures in: {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
