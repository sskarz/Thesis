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
    args = parser.parse_args()
    config.select_model(args.model)

    do_multi = not args.skip_multi_turn
    n_steps = 7 if do_multi else 5

    print("=" * 60)
    print("BFCL Benchmark Pipeline")
    print(f"Model : {config.MODEL_DISPLAY}")
    print(f"Server: {config.SERVER_URL}")
    if not do_multi:
        print("Multi-turn: SKIPPED")
    print("=" * 60)

    # 1. Verify server
    print(f"\n[1/{n_steps}] Checking server...", end=" ")
    if not check_server():
        print("FAILED")
        print(f"Server not reachable at {config.SERVER_URL}")
        print("Start llama-server first (see CLAUDE.md for the command).")
        sys.exit(1)
    print("OK")

    t0 = time.time()

    # 2. Generate single-turn responses
    print(f"\n[2/{n_steps}] Generating single-turn responses")
    generate_responses.run()

    # 3. Generate multi-turn responses (optional)
    if do_multi:
        print(f"\n[3/{n_steps}] Generating multi-turn responses")
        generate_multi_turn.run()

    # 4. Evaluate single-turn responses
    step_eval_st = 4 if do_multi else 3
    print(f"\n[{step_eval_st}/{n_steps}] Evaluating single-turn responses")
    st_csv = evaluate_responses.run()

    # 5. Evaluate multi-turn responses (optional) & combine accuracy.csv
    multi_turn_rows: list[dict] = []
    if do_multi:
        print(f"\n[5/{n_steps}] Evaluating multi-turn responses")
        multi_turn_rows = evaluate_multi_turn.run()

    # Append multi-turn rows to accuracy.csv if present
    if multi_turn_rows:
        with open(st_csv, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["category", "correct", "total", "accuracy"]
            )
            writer.writerows(multi_turn_rows)
        print(f"[evaluate] Multi-turn scores appended to {st_csv.name}")

    # 6. Collect metrics
    step_metrics = 6 if do_multi else 4
    print(f"\n[{step_metrics}/{n_steps}] Collecting metrics")
    collect_metrics.run()

    # 7. Visualize
    step_viz = 7 if do_multi else 5
    print(f"\n[{step_viz}/{n_steps}] Generating charts")
    visualize.run()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} minutes.")
    print(f"Results in: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
