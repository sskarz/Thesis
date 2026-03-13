"""Orchestrator: generate → evaluate → collect metrics → visualize."""

import argparse
import csv
import sys
import time

import requests

from . import config
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
    args = parser.parse_args()
    if args.skip_multi_turn and args.only_multi_turn:
        parser.error("--skip-multi-turn and --only-multi-turn are mutually exclusive")
    config.select_model(args.model)

    do_single = not args.only_multi_turn
    do_multi = not args.skip_multi_turn
    n_steps = (2 if do_single else 0) + (2 if do_multi else 0) + 3  # server + metrics + viz

    print("=" * 60)
    print("BFCL Benchmark Pipeline")
    print(f"Model : {config.MODEL_DISPLAY}")
    print(f"Server: {config.SERVER_URL}")
    if not do_single:
        print("Single-turn: SKIPPED")
    if not do_multi:
        print("Multi-turn: SKIPPED")
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
    st_csv = None
    if do_single:
        step += 1
        print(f"\n[{step}/{n_steps}] Generating single-turn responses")
        generate_responses.run()

    # Generate multi-turn responses
    if do_multi:
        step += 1
        print(f"\n[{step}/{n_steps}] Generating multi-turn responses")
        generate_multi_turn.run()

    # Evaluate single-turn responses
    if do_single:
        step += 1
        print(f"\n[{step}/{n_steps}] Evaluating single-turn responses")
        st_csv = evaluate_responses.run()

    # Evaluate multi-turn responses & combine accuracy.csv
    multi_turn_rows: list[dict] = []
    if do_multi:
        step += 1
        print(f"\n[{step}/{n_steps}] Evaluating multi-turn responses")
        multi_turn_rows = evaluate_multi_turn.run()

    # Append multi-turn rows to accuracy.csv if present
    if multi_turn_rows and st_csv:
        with open(st_csv, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["category", "correct", "total", "accuracy"]
            )
            writer.writerows(multi_turn_rows)
        print(f"[evaluate] Multi-turn scores appended to {st_csv.name}")
    elif multi_turn_rows and not st_csv:
        # Only multi-turn was run — write a standalone accuracy.csv
        config.SCORES_DIR.mkdir(parents=True, exist_ok=True)
        st_csv = config.SCORES_DIR / "accuracy.csv"
        with open(st_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["category", "correct", "total", "accuracy"]
            )
            writer.writeheader()
            writer.writerows(multi_turn_rows)
        print(f"[evaluate] Multi-turn scores saved to {st_csv.name}")

    # Collect metrics
    step += 1
    print(f"\n[{step}/{n_steps}] Collecting metrics")
    collect_metrics.run(include_single=do_single, include_multi=do_multi)

    # Visualize
    step += 1
    print(f"\n[{step}/{n_steps}] Generating charts")
    visualize.run(include_single=do_single, include_multi=do_multi)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} minutes.")
    print(f"Results in: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
