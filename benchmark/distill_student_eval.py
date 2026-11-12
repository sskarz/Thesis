"""Run a SERVED student (llama-server on :8080) over a tasks file with full recording.

Same loop and stock checker as the teacher solver, but the model is the local student and every
request/response (including Gemma's reasoning_content) is logged, so failure modes can be read
from raw outputs. Used for the dev / confirm banks (never for the BFCL gate tasks).

    python -m benchmark.distill_student_eval --tasks T.jsonl --log-dir D --label name
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config, teacher_api
from .distill_solve import decision_mix, solve_one


def run(tasks_path: Path, log_dir: Path, *, label: str, workers: int = 4,
        fc_mode: str = config.FC_IFFC_OODA_CARD, limit: int | None = None,
        samples: tuple[int, ...] = (0,), temp: float = 0.7) -> dict:
    tasks = [json.loads(l) for l in tasks_path.read_text().splitlines() if l.strip()]
    if limit:
        tasks = tasks[:limit]
    client = teacher_api.TeacherClient(teacher_api.TeacherConfig.local(
        log_dir=log_dir, max_concurrency=workers))
    teacher_api.install(client)
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            # sample 0 is greedy (gate-like); every other sample id is a temperature-`temp` draw
            # sample-major order: concurrent workers see different tasks (same-task requests share a prompt
            # prefix and thrash llama-server's per-slot prompt cache)
            jobs = [(t, k) for k in samples for t in tasks]
            verdicts = list(ex.map(lambda j: solve_one(j[0], client, fc_mode, j[1],
                                                       None if j[1] == 0 else temp), jobs))
    finally:
        teacher_api.install(None)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "verdicts.jsonl", "w") as f:
        for v in verdicts:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    ids = {v["episode_id"] for v in verdicts}
    kinds = collections.Counter()
    for v in verdicts:
        if not v["valid"]:
            m = v.get("error") or ""
            kinds["empty_turn" if "list is empty" in m else "state_mismatch" if "state" in m
                  else "missing_results" if "does not contain all" in m else "other"] += 1
    n = len(verdicts)
    rep = {"label": label, "tasks": len(tasks), "samples": list(samples), "temp": temp, "episodes": n, "passed": sum(v["valid"] for v in verdicts),
           "pass_ids": sorted(v["task_id"] for v in verdicts if v["valid"]),
           "fail_kinds": dict(kinds), "wall_s": round(time.time() - t0, 1),
           "api_error_episodes": sum(bool(v.get("api_error")) for v in verdicts),
           "api_calls": sum(v["api_calls"] for v in verdicts),
           "decision_mix": decision_mix(log_dir / "calls.jsonl", ids)}
    (log_dir / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--samples", default="0", help="comma list of sample ids; 0 = greedy, others sampled")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--fc-mode", default=config.FC_IFFC_OODA_CARD, choices=config.FC_MODES)
    a = ap.parse_args(argv)
    print(json.dumps(run(a.tasks, a.log_dir, label=a.label, workers=a.workers, limit=a.limit,
                         samples=tuple(int(x) for x in a.samples.split(",")), temp=a.temp,
                         fc_mode=a.fc_mode), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
