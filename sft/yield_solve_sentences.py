#!/usr/bin/env python3
"""distill_solve.main with the cost-guarded client PLUS teacher reasoning sentences (see g2_sentences.py). Same arguments as `python -m benchmark.distill_solve`.
The local llama-server URL is closed (fail closed), exit 3 if any response arrived without cost accounting or the budget cap was hit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import config, distill_solve, teacher_api  # noqa: E402
from yield_guard import GuardedClient  # noqa: E402
import g2_sentences  # noqa: E402

if __name__ == "__main__":
    config.SERVER_URL = "http://127.0.0.1:9"
    teacher_api.TeacherClient = g2_sentences.make_client_class(GuardedClient)
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
