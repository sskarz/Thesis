#!/usr/bin/env python3
"""Arm B rows for G2: the arm A rows (rows_clean, JSON call list target) joined with the teacher's reasoning sentences from the solve's sentences.jsonl.
usage: g2_rows_b.py --rows-dir rows_clean --solve-dir solve --out-dir rows_B
Join key: (episode_id, sha256 of the row's prompt messages, sha256 of the canonical call list). A row without a valid-sentence record is NOT in arm B (it stays in arm A);
nothing is invented or rewritten. Arm B row = prompt messages + {"role":"assistant","content": <the arm A target>, "reasoning_content": <sentences>}.
Reports counts per exclusion reason and the stop-row share of both arms. No API, no GPU."""
from __future__ import annotations
import argparse, collections, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import g2_sentences as S  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-dir", type=Path, required=True); ap.add_argument("--solve-dir", type=Path, required=True); ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    recs = collections.defaultdict(list)
    for l in open(a.solve_dir / "sentences.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l); recs[(r["episode_id"], r["messages_sha"], r["calls_sha"])].append(r)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter(); out_counts = {}
    for fn in ("train.jsonl", "valid.jsonl"):
        rows = [json.loads(l) for l in open(a.rows_dir / fn, encoding="utf-8") if l.strip()]
        keep = []
        for r in rows:
            stats["rows_arm_a"] += 1
            tgt = json.loads(r["messages"][-1]["content"])
            key = (r["meta"]["episode_id"], S.messages_sha(r["messages"][:-1]), S.calls_sha(tgt))
            cand = recs.get(key, [])
            if not cand:
                stats["excluded_no_record"] += 1; continue
            c = cand[0]
            if not c.get("split_ok"):
                stats["excluded_no_clean_split"] += 1; continue
            v = c.get("verdict") or {}
            if not v.get("valid"):
                stats["excluded_" + (v.get("reason") or "invalid").split(":")[0]] += 1; continue
            m = dict(r["messages"][-1]); m["reasoning_content"] = c["sentences"]
            keep.append({**r, "messages": [*r["messages"][:-1], m]})
            stats["rows_arm_b"] += 1
            stats["arm_b_stop_rows"] += (tgt == [])
        with open(a.out_dir / fn, "w", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out_counts[fn] = len(keep)
    rep = {"counts": dict(stats), "rows_b_files": out_counts,
           "arm_b_share_of_arm_a": round(stats["rows_arm_b"] / stats["rows_arm_a"], 4) if stats["rows_arm_a"] else None,
           "arm_b_tasks": len({json.loads(l)["meta"]["task_id"] for fn in out_counts for l in open(a.out_dir / fn) if l.strip()})}
    json.dump(rep, open(a.out_dir / "rows_b_report.json", "w"), indent=1); print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
