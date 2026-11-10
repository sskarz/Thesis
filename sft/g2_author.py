#!/usr/bin/env python3
"""G2 per-process authoring driver. Runs one authoring process as consecutive CHUNKS of ~500 slots (each chunk is one stock `yield_author.py` call), so that
(a) accepted / duplicate / other rejects are known per 500 slots while the run is going, (b) a killed run keeps every finished chunk (Part 1 wrote its output only at the end),
(c) the process cap is enforced cumulatively (each chunk gets only the remaining cap), (d) within the process, later chunks see earlier chunks' tasks through --existing,
so the duplicate rule is unchanged. Cross-process duplicates are handled later by yield_merge.py.

usage: g2_author.py --proc a --seed-base 7101000 --slots 3434 --chunk 500 --deadline-epoch E --cap USD --workers 4 --out-dir D --existing F... [--author-cmd "python -B sft/yield_author.py"]
Writes D/chunk_K/{tasks_raw.jsonl,report.json,log/}, D/progress.jsonl (one line per finished chunk), D/tasks_raw.jsonl (concatenation), D/proc_report.json.
Exit: 0 all chunks done, or stopped by the deadline (the chain gates that on the 2% rule); 3 cost guard fired in a chunk; 4 other chunk failure; 5 process cap reached before all slots were worked (incomplete); 143 on SIGTERM.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

CHILD = None


def _term(signum, frame):
    global CHILD
    if CHILD is not None and CHILD.poll() is None:
        try:
            os.killpg(CHILD.pid, signal.SIGTERM)
            time.sleep(5)
            os.killpg(CHILD.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    sys.exit(143)


def main(argv=None) -> int:
    global CHILD
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", required=True); ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--slots", type=int, required=True); ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--deadline-epoch", type=float, required=True); ap.add_argument("--cap", type=float, required=True)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--existing", type=Path, nargs="*", default=[])
    ap.add_argument("--variant", default="b1"); ap.add_argument("--weights-file", default=None)
    ap.add_argument("--author-cmd", default="python -B sft/yield_author.py")
    a = ap.parse_args(argv)
    signal.signal(signal.SIGTERM, _term); signal.signal(signal.SIGINT, _term)
    eff_deadline = a.deadline_epoch - 180            # no new slot starts in the last 3 minutes, so in-flight requests finish before the chain's outer kill at deadline + 60 s
    a.out_dir.mkdir(parents=True, exist_ok=True)
    prog = open(a.out_dir / "progress.jsonl", "a")
    n_chunks = math.ceil(a.slots / a.chunk)
    spent = 0.0; done_files: list[Path] = []; stop = "all_chunks_done"; rc_out = 0
    skipped_slots = 0
    for k in range(n_chunks):
        n = min(a.chunk, a.slots - k * a.chunk)
        left_s = eff_deadline - time.time()
        if left_s <= 30:
            stop = "deadline"; skipped_slots = a.slots - k * a.chunk; break
        rem = a.cap - spent
        if rem < 0.02:
            stop = "process_cap_reached"; rc_out = 5; skipped_slots = a.slots - k * a.chunk; break          # authoring is INCOMPLETE: the chain must not continue
        cd = a.out_dir / f"chunk_{k}"; cd.mkdir(exist_ok=True)
        cmd = shlex.split(a.author_cmd) + ["--n", str(n), "--out", str(cd / "tasks_raw.jsonl"), "--log-dir", str(cd / "log"),
               "--seed", str(a.seed_base + k), "--prefix", f"distill_g2{a.proc}{k}", "--existing", *map(str, a.existing + done_files),
               "--max-cost", f"{rem:.4f}", "--workers", str(a.workers), "--deadline-minutes", f"{left_s / 60:.2f}",
               "--report-out", str(cd / "report.json"), "--variant", a.variant] + (["--weights-file", a.weights_file] if a.weights_file else [])
        t0 = time.time()
        with open(cd / "author.out", "w") as out:
            CHILD = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            rc = CHILD.wait()
        CHILD = None
        rep = {}
        if (cd / "report.json").exists():
            rep = json.load(open(cd / "report.json"))
        rj = rep.get("reject_reasons", {})
        line = {"proc": a.proc, "chunk": k, "slots": n, "rc": rc, "seconds": round(time.time() - t0), "valid": rep.get("valid"),
                "spent_usd": rep.get("spent_usd"), "calls": rep.get("calls"), "reject_reasons": rj,
                "deadline_skipped": rj.get("deadline", 0), "api_rejects": sum(v for kk, v in rj.items() if kk.startswith("api"))}
        prog.write(json.dumps(line) + "\n"); prog.flush()
        if rep.get("spent_usd") is not None:
            spent += float(rep["spent_usd"])
        if (cd / "tasks_raw.jsonl").exists():
            done_files.append(cd / "tasks_raw.jsonl")
        if rc == 3:
            stop = "cost_guard"; rc_out = 3; skipped_slots = a.slots - (k + 1) * a.chunk; break
        if rc != 0:
            stop = f"chunk_failed_rc{rc}"; rc_out = 4; skipped_slots = a.slots - (k + 1) * a.chunk; break
    with open(a.out_dir / "tasks_raw.jsonl", "w", encoding="utf-8") as f:
        for p in done_files:
            f.write(Path(p).read_text(encoding="utf-8"))
    json.dump({"proc": a.proc, "slots_planned": a.slots, "chunks_done": len(done_files), "spent_usd": round(spent, 4),
               "stop_reason": stop, "slots_not_started": max(0, skipped_slots), "rc": rc_out}, open(a.out_dir / "proc_report.json", "w"), indent=1)
    print(json.dumps({"proc": a.proc, "chunks_done": len(done_files), "spent_usd": round(spent, 4), "stop_reason": stop}))
    return rc_out


if __name__ == "__main__":
    sys.exit(main())
