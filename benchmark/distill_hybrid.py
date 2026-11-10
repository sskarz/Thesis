"""Hybrid rollouts: replay the student until its first mistake, let the teacher finish the episode.

Why (Run 87/t2 autopsy): the stock BFCL checker requires the model's execution results to CONTAIN the
gold's results, so skipped steps (a lookup the gold needs, a guessed id) are fatal while extra calls are
free. Teacher labels that merely "execute without error" taught the concise teacher style and cost
passes. Here a teacher step becomes a training row only if the WHOLE episode (student prefix + teacher
continuation) is accepted by the stock checker, which is the outcome the gate scores.

For each failed student episode:
  1. r = index of the first request whose teacher label was a kept correction (from onpolicy labels);
  2. run the harness loop with a scripted client: requests < r return the student's recorded replies
     (content + reasoning_content, so history is byte-identical), requests >= r go to the teacher;
  3. verify with the stock checker; keep the teacher's calls as rows only if valid.

    python -m benchmark.distill_hybrid run --tasks T --student-dir D --labels L --out O --k 3 --max-cost 1.0
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config, teacher_api
from .distill_solve import solve_one


class HybridClient(teacher_api.TeacherClient):
    """Teacher client that first replays a recorded prefix (per episode), then calls the teacher."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.scripts: dict[str, dict] = {}
        self._script_lock = threading.Lock()

    def set_script(self, episode_id: str, replies: list[dict]) -> None:
        with self._script_lock:
            self.scripts[episode_id] = {"replies": replies, "i": 0}

    def post_chat(self, payload: dict) -> dict:
        ep = teacher_api._EPISODE.get()
        sc = self.scripts.get(ep["id"]) if ep else None
        if sc is not None and sc["i"] < len(sc["replies"]):
            r = sc["replies"][sc["i"]]
            sc["i"] += 1
            msg = {"role": "assistant", "content": r["response_text"]}
            if r.get("reasoning_content"):
                msg["reasoning_content"] = r["reasoning_content"]
            return {"choices": [{"message": msg, "finish_reason": "stop"}], "model": "scripted-student"}
        return super().post_chat(payload)


def run(tasks_path: Path, student_dir: Path, labels_path: Path, out: Path, *, k: int, max_cost: float,
        workers: int = 8, sample_temp: float = 0.7, fc_mode: str = config.FC_IFFC_OODA_CARD) -> dict:
    tasks = {json.loads(l)["id"]: json.loads(l) for l in tasks_path.read_text().splitlines() if l.strip()}
    verdicts = {json.loads(l)["episode_id"]: json.loads(l)
                for l in (student_dir / "verdicts.jsonl").read_text().splitlines() if l.strip()}
    recs = collections.defaultdict(list)
    for l in (student_dir / "calls.jsonl").read_text().splitlines():
        r = json.loads(l)
        recs[r["episode_id"]].append(r)
    first_fix: dict[str, int] = {}
    for l in labels_path.read_text().splitlines():
        x = json.loads(l)
        if x["decision"] == "keep" and not x.get("same_effect", x["same_as_student"]):
            e = x["episode_id"]
            first_fix[e] = min(first_fix.get(e, 10**9), x["req_index"])
    jobs = []
    for eid, v in verdicts.items():
        if v["valid"] or eid not in first_fix or v["task_id"] not in tasks:
            continue
        r = first_fix[eid]
        for s in range(k):
            jobs.append((tasks[v["task_id"]], eid, r, s))
    cfg = teacher_api.TeacherConfig.openrouter(log_dir=out, max_cost_usd=max_cost, max_concurrency=workers)
    client = HybridClient(cfg)
    teacher_api.install(client)

    def work(job):
        task, eid, r, s = job
        sid = f"{eid}__h{r}"                 # <source student episode>__h<switch>; solve_one appends __s<sample>
        client.set_script(f"{sid}__s{s}", recs[eid][:r])
        t = dict(task); t["id"] = sid
        try:
            v = solve_one(t, client, fc_mode, s, None if s == 0 else sample_temp)
        except teacher_api.BudgetExceeded:
            return {"episode_id": f"{sid}__s{s}", "valid": False, "error": "budget", "switch": r, "student_episode": eid}
        v["switch"], v["student_episode"], v["task_id"] = r, eid, task["id"]
        return v

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(work, jobs))
    finally:
        teacher_api.install(None)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "verdicts.jsonl", "w") as f:
        for v in res:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    by_ep = collections.defaultdict(list)
    for v in res:
        by_ep[v["student_episode"]].append(v["valid"])
    rep = {"failed_student_episodes_with_a_correction": len({j[1] for j in jobs}), "hybrid_episodes": len(res),
           "verified": sum(v["valid"] for v in res),
           "episodes_rescued": sum(any(x) for x in by_ep.values()),
           "spent_usd": round(client.stats["cost_usd"], 4)}
    (out / "hybrid_report.json").write_text(json.dumps(rep, indent=1))
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tasks", type=Path, required=True)
    r.add_argument("--student-dir", type=Path, required=True)
    r.add_argument("--labels", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--k", type=int, default=3)
    r.add_argument("--max-cost", type=float, required=True)
    a = ap.parse_args(argv)
    print(json.dumps(run(a.tasks, a.student_dir, a.labels, a.out, k=a.k, max_cost=a.max_cost), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
