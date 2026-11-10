"""Offline smoke for the distillation exclusion list and leak checks.

    uv run python -m benchmark.smoke_distill_exclusions

Reads only the installed ``bfcl_eval`` data. No model, no network.
"""

from __future__ import annotations

from . import distill_exclusions as X
from .config import MULTI_TURN_CATEGORIES


def main() -> None:
    rep = X.build_report()
    c = rep["counts"]
    assert c["ids_per_split"]["multi_turn_base"] == 200
    assert set(c["ids_per_split"]) == set(MULTI_TURN_CATEGORIES)
    assert c["gate_groups"] == {"mid20": 20, "soft10": 10, "fresh_bank": 20}
    assert not rep["gate_id_problems"], rep["gate_id_problems"]
    for g in X.GATE_GROUPS.values():
        assert len(set(g)) == len(g)

    ex = X.load_excluded()
    sample = next(iter(ex["texts"].values()))

    # exact copy, case/punctuation-changed copy, and a lightly edited copy all rejected
    assert X.check_task([sample])["reject"]
    assert X.check_task([sample.upper() + "!!"])["reject"]
    edited = sample + " Please."
    assert X.check_task([edited])["reject"]

    # excluded task id rejected even with novel text
    r = X.check_task(["Rename the zebra notes to giraffe notes."], task_id="multi_turn_base_5")
    assert r["reject"] and any("excluded BFCL id" in x for x in r["reasons"])

    # a clearly novel task passes
    ok = X.check_task(["Rename the zebra notes to giraffe notes."], [["mv(source='zebra', destination='giraffe')"]])
    assert not ok["reject"], ok
    assert ok["literal_overlap_frac"] is not None

    # row scan
    rows = [
        {"messages": [{"role": "user", "content": sample}]},
        {"messages": [{"role": "user", "content": "Rename the zebra notes to giraffe notes."}]},
        {"messages": [{"role": "user", "content": "x"}], "meta": {"src": "multi_turn_base_7"}},
    ]
    s = X.scan_rows(rows)
    assert s == {"rows": 3, "id_rows": 1, "ngram_rows": 1 if len(sample.split()) >= 8 else 0,
                 "exact_turn_rows": 1}, s
    print("smoke_distill_exclusions: OK", c["ids_total_unique"], "ids,",
          c["user_turn_texts_unique"], "turn texts")


if __name__ == "__main__":
    main()
