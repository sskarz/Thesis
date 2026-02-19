"""Orchestrator: generate → evaluate → collect metrics → visualize."""

import argparse
import sys
import time

import requests

from . import config
from . import generate_responses
from . import evaluate_responses
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
    args = parser.parse_args()
    config.select_model(args.model)

    print("=" * 60)
    print("BFCL Benchmark Pipeline")
    print(f"Model : {config.MODEL_DISPLAY}")
    print(f"Server: {config.SERVER_URL}")
    print("=" * 60)

    # 1. Verify server
    print("\n[1/5] Checking server...", end=" ")
    if not check_server():
        print("FAILED")
        print(f"Server not reachable at {config.SERVER_URL}")
        print("Start llama-server first (see CLAUDE.md for the command).")
        sys.exit(1)
    print("OK")

    t0 = time.time()

    # 2. Generate responses
    print("\n[2/5] Generating responses")
    generate_responses.run()

    # 3. Evaluate responses
    print("\n[3/5] Evaluating responses")
    evaluate_responses.run()

    # 4. Collect metrics
    print("\n[4/5] Collecting metrics")
    collect_metrics.run()

    # 5. Visualize
    print("\n[5/5] Generating charts")
    visualize.run()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} minutes.")
    print(f"Results in: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
