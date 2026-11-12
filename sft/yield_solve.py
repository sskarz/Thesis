#!/usr/bin/env python3
"""distill_solve.main with the cost-guarded client (see yield_guard.py). Same arguments as `python -m benchmark.distill_solve`.
Exit 3 if any response arrived without cost accounting."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import config, distill_solve, teacher_api  # noqa: E402
from yield_guard import GuardedClient  # noqa: E402

if __name__ == "__main__":
    config.SERVER_URL = "http://127.0.0.1:9"      # fail closed: a non-teacher request must never reach the local llama-server on :8080
    teacher_api.TeacherClient = GuardedClient       # distill_solve.pilot looks the class up at call time
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--log-dir")
    known, _ = ap.parse_known_args(sys.argv[1:])
    rc = distill_solve.main(sys.argv[1:])
    if known.log_dir:
        GuardedClient.write_flags(Path(known.log_dir) / "guard_flags.json")
    if GuardedClient.MISSING_COST or GuardedClient.BUDGET_HIT:
        print("ERROR: a response without cost or the budget cap was hit; the solve is incomplete or its spend cap cannot be trusted", file=sys.stderr)
        sys.exit(3)
    sys.exit(rc)
