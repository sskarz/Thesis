#!/usr/bin/env python3
"""Distillation gate: ONE mid20 and ONE fresh-20 bank eval under the exact bar harness.

Same protocol as sft/v13lk/bfcl_eval_v13lk.py (Mac, untracked): fc-mode iffc-ooda-card,
-np 4 slots, temp 0, categories=multi_turn_base, NEW run_id only (> 85), never resume.
The candidate GGUF must already be served on :8080 with the R69 argv
(--jinja -fa on -ngl 99 -c 65536 --temp 0.0 -np 4 -cb -ub 1024). Scores come from the DB only.

    python sft/distill_gate_eval.py --set mid20|fresh --out sft/data/distill/gate/eval_<set>.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL_KEY = "gemma4-e2b-distill-t1"  # default; override with --model-key
FC_MODE = "iffc-ooda-card"
MAX_OLD_RUN = 85


def health() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=5) as r:
            return "ok" in r.read().decode().lower()
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["mid20", "fresh"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--model-key", default=MODEL_KEY)
    ap.add_argument("--desc", default="distill-t1 candidate")
    a = ap.parse_args()
    os.chdir(ROOT)
    from benchmark import config, db, evaluate_multi_turn
    from benchmark import distill_exclusions as X
    from benchmark.generate_multi_turn import _generate_and_checkpoint, _load_func_docs
    from benchmark.generate_responses import load_jsonl
    import aiohttp
    from tqdm import tqdm

    ids = X.GATE_GROUPS["mid20" if a.set == "mid20" else "fresh_bank"]
    frozen = json.loads((ROOT / "sft/data/distill/gate/frozen_base.json").read_text())
    assert set(ids) == set(frozen["mid20_base_R69" if a.set == "mid20" else "fresh_base_R42"]["per_id"])
    assert len(ids) == 20 and len(set(ids)) == 20
    if not health():
        raise SystemExit("BLOCKER :8080 down")
    config.select_model(a.model_key)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a run was started earlier. Not resuming (manual decision).")
    notes = (f"{a.desc} {a.set} Acc exact R69 iffc-ooda-card; model_key={a.model_key}; "
             f"NEW run, base refs R69 mid20 / R42 fresh; ids={','.join(ids)}; categories=multi_turn_base "
             f"fc_mode={FC_MODE} serve=-np4 -c65536 --temp 0.0 -ngl 99 -cb -ub 1024 --jinja -fa on "
             f"NO_Decision NO_Choice NO_full200 NO_BoN {a.note}")
    rid = db.create_run(model_key=a.model_key, notes=notes)
    if rid <= MAX_OLD_RUN:
        raise SystemExit(f"REFUSING run_id={rid}")
    st = {"set": a.set, "run_id": rid, "ids": ids, "status": "GENERATING"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(st, indent=1))
    config.RUN_ID = rid
    config.set_fc_mode(FC_MODE)
    assert config.PARALLEL_SLOTS == 4
    data_file = config.BFCL_DATA_DIR / config.MULTI_TURN_CATEGORIES["multi_turn_base"]["file"]
    entries = [e for e in load_jsonl(data_file) if e["id"] in set(ids)]
    assert len(entries) == 20

    async def _run():
        sem = asyncio.Semaphore(config.PARALLEL_SLOTS)
        pbar = tqdm(total=20, desc=f"{a.set}-{rid}")
        async with aiohttp.ClientSession() as session:
            await asyncio.gather(*[_generate_and_checkpoint(
                session, sem, e, "multi_turn_base", _load_func_docs(e.get("involved_classes", [])),
                pbar, FC_MODE, rid) for e in entries])
        pbar.close()

    asyncio.run(_run())
    db.finish_run(rid)
    evaluate_multi_turn.run(categories=config.resolve_multi_turn_categories(["multi_turn_base"]))
    sc = [dict(s) for s in db.get_scores(rid) if s["category"] == "multi_turn_base"][0]
    conn = db.get_connection()
    errs = {r["entry_id"] for r in conn.execute(
        "SELECT entry_id FROM eval_errors WHERE run_id=? AND category=?", (rid, "multi_turn_base"))}
    resp = {r["entry_id"] for r in conn.execute(
        "SELECT entry_id FROM responses WHERE run_id=? AND category=? AND mode='multi'", (rid, "multi_turn_base"))}
    per = {e: ("MISSING" if e not in resp else (0 if e in errs else 1)) for e in ids}
    st.update({"status": "SCORED", "correct": sc["correct"], "total": sc["total"],
               "accuracy": sc["accuracy"], "per_entry": per})
    out.write_text(json.dumps(st, indent=1))
    print(json.dumps({"run_id": rid, "correct": sc["correct"], "total": sc["total"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
