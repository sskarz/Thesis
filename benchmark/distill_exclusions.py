"""Exclusion list + leak checks for teacher-student distillation data.

Handoff ``docs/DISTILLATION_HANDOFF.md`` section 4. Nothing here calls a model
or the network. The excluded set is everything that must never become a
training row or a task a training row is derived from:

* all 200 ``multi_turn_base`` task IDs, their user-turn text and gold calls;
* the 20-task gate ("mid20"), the 10-task soft10 tripwire and the fresh
  20-task bank (all are subsets of the 200, listed explicitly so reports can
  show them);
* the other BFCL multi-turn splits (``miss_func``, ``miss_param``,
  ``long_context``): IDs and user-turn text, because they are variants of the
  same base questions.

Check thresholds (recorded in every report; Sans to confirm, see
``THRESHOLDS``):

* exact match after normalisation (lower-case, collapse whitespace/punctuation);
* any shared word 8-gram with an excluded user turn (same guard as the earlier
  ``sft/v11/leak_v11.py`` leak check, which found 0 rows);
* per-turn ``difflib`` ratio >= 0.85 against any excluded user turn;
* string literals in gold-call arguments copied from excluded gold calls are
  *reported* (fraction), not rejected by default: tasks over the same
  backends legitimately share defaults such as ``'document'``.

CLI::

    python -m benchmark.distill_exclusions build   # writes sft/data/distill/exclusion.json
    python -m benchmark.distill_exclusions scan ROWS.jsonl
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

from . import config
from .generate_responses import load_jsonl

# Sources (copied, not recomputed):
#   MID20, SOFT10: sft/v11/leak_v11.py (Mac). Gate bar 17/20 is Run 69.
#   FRESH_BANK: sft/v13lk/fresh_bank_ids.txt (Mac), the bank used for Run 77.
MID20_NUMS = [23, 45, 7, 12, 8, 14, 91, 113, 57, 138, 150, 100, 60, 83, 115,
              146, 112, 152, 186, 195]
SOFT10_NUMS = [11, 101, 126, 130, 6, 18, 26, 27, 32, 34]
FRESH_BANK_NUMS = [5, 28, 31, 40, 80, 90, 92, 93, 117, 121, 122, 134, 149, 156,
                   162, 165, 170, 176, 184, 199]

THRESHOLDS = {
    "ngram_n": 8,
    "ngram_reject_if_shared_ge": 1,
    "turn_ratio_reject_ge": 0.85,
    "literal_overlap_reject_ge": None,  # report only; set a float to enforce
}

OUT_DEFAULT = Path("sft/data/distill/exclusion.json")
_WORD = re.compile(r"[a-z0-9]+")
_LIT_MIN_LEN = 4


def _ids(nums: list[int]) -> list[str]:
    return [f"multi_turn_base_{n}" for n in nums]


GATE_GROUPS = {
    "mid20": _ids(MID20_NUMS),
    "soft10": _ids(SOFT10_NUMS),
    "fresh_bank": _ids(FRESH_BANK_NUMS),
}


def norm(text: str) -> str:
    return " ".join(_WORD.findall(str(text).lower()))


def ngrams(text: str, n: int = 8) -> set[tuple]:
    w = _WORD.findall(str(text).lower())
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def _user_turns(entry: dict) -> list[str]:
    out: list[str] = []
    for turn in entry.get("question", []):
        for m in turn:
            if m.get("role") == "user" and m.get("content"):
                out.append(str(m["content"]))
    return out


def _call_literals(call: str) -> set[str]:
    """String-literal argument values of a ``name(a='x', ...)`` call string."""
    try:
        node = ast.parse(call, mode="eval").body
    except SyntaxError:
        return set()
    lits: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            if len(sub.value) >= _LIT_MIN_LEN:
                lits.add(sub.value.lower())
    return lits


@lru_cache(maxsize=1)
def load_excluded(data_dir: str | None = None) -> dict:
    """Read every multi-turn split from the installed ``bfcl_eval`` data."""
    base = Path(data_dir) if data_dir else config.BFCL_DATA_DIR
    ids: dict[str, list[str]] = {}
    texts: dict[str, str] = {}  # normalised -> original (dedup across splits)
    literals: set[str] = set()
    for category, info in config.MULTI_TURN_CATEGORIES.items():
        f = base / info["file"]
        if not f.exists():
            raise FileNotFoundError(f"missing BFCL data file {f}")
        entries = load_jsonl(f)
        ids[category] = [e["id"] for e in entries]
        for e in entries:
            for t in _user_turns(e):
                texts.setdefault(norm(t), t)
        gt = base / "possible_answer" / info["file"]
        if gt.exists():
            for a in load_jsonl(gt):
                for turn in a.get("ground_truth", []):
                    for call in turn:
                        literals |= _call_literals(call)
    all_grams: set[tuple] = set()
    for t in texts.values():
        all_grams |= ngrams(t, THRESHOLDS["ngram_n"])
    return {
        "ids": ids,
        "texts": texts,
        "ngrams": frozenset(all_grams),
        "literals": frozenset(literals),
    }


def build_report(data_dir: str | None = None) -> dict:
    ex = load_excluded(data_dir)
    base_ids = set(ex["ids"].get("multi_turn_base", []))
    problems = []
    for name, group in GATE_GROUPS.items():
        missing = [g for g in group if g not in base_ids]
        if missing:
            problems.append(f"{name}: ids not in multi_turn_base: {missing}")
    all_ids = sorted({i for v in ex["ids"].values() for i in v})
    report = {
        "thresholds": THRESHOLDS,
        "counts": {
            "ids_per_split": {k: len(v) for k, v in ex["ids"].items()},
            "ids_total_unique": len(all_ids),
            "user_turn_texts_unique": len(ex["texts"]),
            "ngrams": len(ex["ngrams"]),
            "gold_string_literals": len(ex["literals"]),
            "gate_groups": {k: len(v) for k, v in GATE_GROUPS.items()},
        },
        "gate_groups": GATE_GROUPS,
        "gate_id_problems": problems,
        "ids": ex["ids"],
        "user_turn_texts": sorted(ex["texts"].values()),
    }
    report["sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in report.items()}, sort_keys=True).encode()
    ).hexdigest()
    return report


def check_task(
    user_turns: list[str],
    gold_calls: list[list[str]] | None = None,
    *,
    task_id: str | None = None,
    data_dir: str | None = None,
) -> dict:
    """Leak-check one generated task. ``reject`` True means drop it."""
    ex = load_excluded(data_dir)
    th = THRESHOLDS
    reasons: list[str] = []
    all_ids = {i for v in ex["ids"].values() for i in v}
    if task_id and task_id in all_ids:
        reasons.append(f"task_id {task_id} is an excluded BFCL id")
    excl_norm = list(ex["texts"].keys())
    shared_grams = 0
    max_ratio = 0.0
    exact = 0
    for t in user_turns:
        n = norm(t)
        if n in ex["texts"]:
            exact += 1
        shared_grams += len(ngrams(t, th["ngram_n"]) & ex["ngrams"])
        sm = difflib.SequenceMatcher(autojunk=False)
        sm.set_seq2(n)
        for e in excl_norm:
            sm.set_seq1(e)
            if sm.real_quick_ratio() <= max_ratio or sm.quick_ratio() <= max_ratio:
                continue
            max_ratio = max(max_ratio, sm.ratio())
    if exact:
        reasons.append(f"{exact} user turn(s) match an excluded turn exactly")
    if shared_grams >= th["ngram_reject_if_shared_ge"]:
        reasons.append(f"{shared_grams} shared {th['ngram_n']}-gram(s)")
    if max_ratio >= th["turn_ratio_reject_ge"]:
        reasons.append(f"turn similarity {max_ratio:.2f} >= {th['turn_ratio_reject_ge']}")
    lit_frac = None
    if gold_calls:
        lits: set[str] = set()
        for turn in gold_calls:
            for c in turn:
                lits |= _call_literals(c)
        if lits:
            lit_frac = len(lits & ex["literals"]) / len(lits)
            lim = th["literal_overlap_reject_ge"]
            if lim is not None and lit_frac >= lim:
                reasons.append(f"literal overlap {lit_frac:.2f} >= {lim}")
    return {
        "reject": bool(reasons),
        "reasons": reasons,
        "exact_turns": exact,
        "shared_ngrams": shared_grams,
        "max_turn_ratio": round(max_ratio, 4),
        "literal_overlap_frac": None if lit_frac is None else round(lit_frac, 4),
    }


def scan_rows(rows: list[dict], *, data_dir: str | None = None) -> dict:
    """Dataset-level leak scan for the build log (handoff section 4, item 3/4).

    Counts rows containing an excluded task ID, an 8-gram of an excluded user
    turn, or an exact excluded turn anywhere in the serialised row. Tool names
    are *not* flagged (training must use the BFCL backends, so those names are
    expected); the number of rows that mention one is reported for the record.
    """
    ex = load_excluded(data_dir)
    all_ids = {i for v in ex["ids"].values() for i in v}
    id_re = re.compile(r"multi_turn_[a-z_]+_\d+")
    n = int(THRESHOLDS["ngram_n"])
    out = {"rows": len(rows), "id_rows": 0, "ngram_rows": 0, "exact_turn_rows": 0}
    for r in rows:
        s = json.dumps(r, ensure_ascii=False)
        if any(m in all_ids for m in id_re.findall(s)):
            out["id_rows"] += 1
        # only user-role content is compared with user-turn grams
        user_txt = " ".join(
            str(m.get("content", ""))
            for m in (r.get("messages") or [])
            if isinstance(m, dict) and m.get("role") == "user"
        )
        if ngrams(user_txt, n) & ex["ngrams"]:
            out["ngram_rows"] += 1
        if any(norm(m.get("content", "")) in ex["texts"]
               for m in (r.get("messages") or [])
               if isinstance(m, dict) and m.get("role") == "user"):
            out["exact_turn_rows"] += 1
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, default=OUT_DEFAULT)
    s = sub.add_parser("scan")
    s.add_argument("rows", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        rep = build_report()
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
        print(json.dumps({"out": str(a.out), "counts": rep["counts"],
                          "problems": rep["gate_id_problems"],
                          "sha256": rep["sha256"]}, indent=1))
        return 1 if rep["gate_id_problems"] else 0
    rows = [json.loads(l) for l in a.rows.open() if l.strip()]
    print(json.dumps(scan_rows(rows), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
