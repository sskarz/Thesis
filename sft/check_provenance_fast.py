"""Provenance / leak check for distillation training data against every bank we screen on.

usage (run from the worktree root with the Thesis venv):
  python sft/check_provenance.py --rows A.jsonl [B.jsonl ...] --heldout dev_all.jsonl \
         --task-sources tasks_audited.jsonl [more.jsonl] --out report.json

Checks (no model, no network, no writes except --out). A pass means these implemented checks found nothing;
it is not proof of absence of leakage.
 1. Row files -> benchmark exclusions (benchmark.distill_exclusions.scan_rows): excluded ids, 8-gram
    overlap of user turns with all BFCL multi-turn splits, exact-turn matches.
 2. Task-id disjointness: training task ids (row meta.task_id) vs held-out bank ids.
 3. Near-duplicate user turns between training tasks and held-out tasks: any shared word 8-gram, or
    per-turn difflib ratio >= 0.85 (same thresholds as the benchmark check). Training-task text is
    read from --task-sources by id; ids not found there are reported, not ignored.
 3b. Family separation: same gold-function-sequence signature (benchmark.distill_split.signature) on both sides.
 4. Content checks that do not trust row metadata: every non-system message of every row is compared with the
    held-out user turns (shared word 8-gram), and plain user turns (not tool results / OODA cards) with
    per-turn similarity >= 0.85. Also check_task() (which includes the 0.85 similarity criterion) is applied to
    each training task's user turns against the BFCL multi-turn splits.
 5. SHA-256 of each input file, of this script and of distill_exclusions.py, and a hash of the BFCL turn texts.
Exit code 1 if any check fails or any training task text is missing."""
import argparse, difflib, hashlib, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from benchmark import distill_exclusions as ex


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def turns(entry):
    return ex._user_turns(entry)


PLAIN_SKIP = ("Function execution results:", "[OODA]")


def plain_user_turns(r):
    """User-authored text: plain user messages, plus the goal line of [OODA] cards (which carries the user turn)."""
    out = []
    for m in r["messages"]:
        c = m.get("content")
        if m.get("role") != "user" or not isinstance(c, str):
            continue
        if c.startswith("[OODA]"):
            for line in c.splitlines():
                if line.startswith("goal:"):
                    out.append(line[len("goal:"):].strip())
        elif not c.startswith("Function execution results:"):
            out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--heldout", nargs="+", required=True)
    ap.add_argument("--task-sources", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = [r for p in a.rows for r in load(p)]
    held = {}
    for p in a.heldout:
        for e in load(p):
            if e["id"] in held:
                sys.exit(f"duplicate held-out id {e['id']}")
            held[e["id"]] = e
    src = {}
    for p in a.task_sources:
        for e in load(p):
            k = e.get("id") or e.get("task_id")
            if k in src:
                sys.exit(f"duplicate source id {k}")
            src[k] = e

    rep = {"inputs": {p: sha(p) for p in a.rows + a.heldout + a.task_sources}, "n_rows": len(rows)}
    rep["benchmark_scan"] = ex.scan_rows(rows)

    train_ids = sorted({r["meta"]["task_id"] for r in rows})
    rep["n_train_tasks"] = len(train_ids)
    rep["n_heldout_tasks"] = len(held)
    rep["id_overlap_with_heldout"] = sorted(set(train_ids) & set(held))

    clean = lambda ts: [t for t in ts if str(t).strip()]
    missing = [t for t in train_ids if t not in src or not clean(turns(src[t]))]
    rep["heldout_tasks_without_text"] = [h for h, e in held.items() if not clean(turns(e))]
    rep["rows_without_plain_user_turn_or_empty_message"] = [
        i for i, r in enumerate(rows)
        if not r.get("messages") or any(not str(m.get("content", "")).strip() for m in r["messages"])
        or not plain_user_turns(r)]
    rep["train_tasks_missing_text"] = missing
    n = int(ex.THRESHOLDS["ngram_n"])
    thr = ex.THRESHOLDS["turn_ratio_reject_ge"]
    held_grams = {}
    held_norm = {}
    for hid, he in held.items():
        ts = turns(he)
        held_norm[hid] = [ex.norm(t) for t in ts]
        g = set()
        for t in ts:
            g |= ex.ngrams(t, n)
        held_grams[hid] = g
    hits = []
    for tid in train_ids:
        if tid not in src:
            continue
        ts = turns(src[tid])
        tg = set()
        for t in ts:
            tg |= ex.ngrams(t, n)
        tn = [ex.norm(t) for t in ts]
        for hid in held:
            shared = len(tg & held_grams[hid])
            best = 0.0
            for x in tn:
                for y in held_norm[hid]:
                    sm = difflib.SequenceMatcher(None, x, y, autojunk=False)
                    if sm.real_quick_ratio() <= best or sm.quick_ratio() <= best:
                        continue
                    best = max(best, sm.ratio())
            if shared >= 1 or best >= thr:
                hits.append({"train": tid, "heldout": hid, "shared_8grams": shared, "max_turn_ratio": round(best, 3)})
    rep["near_duplicate_pairs"] = hits
    # family separation: same gold-function-sequence signature across training and held-out tasks
    from benchmark import distill_split as ds
    held_sig = {}
    for hid, he in held.items():
        held_sig.setdefault(ds.signature(he), []).append(hid)
    fam = []
    for tid in train_ids:
        if tid in src:
            sg = ds.signature(src[tid])
            if sg in held_sig:
                fam.append({"train": tid, "heldout": held_sig[sg]})
    rep["family_signature_overlap"] = fam
    bad_gold = [i for i, e in list(held.items()) + list(src.items()) if "ground_truth" not in e]
    rep["tasks_without_gold"] = bad_gold

    # content-level checks independent of row metadata
    all_held_grams = set().union(*held_grams.values()) if held_grams else set()
    all_held_norm = [x for v in held_norm.values() for x in v]
    gram_rows, ratio_rows = [], []
    turn_best = {}
    for ri, r in enumerate(rows):
        txt = " ".join(str(m.get("content", "")) + " " + str(m.get("reasoning_content") or "") for m in r["messages"] if m.get("role") != "system")
        if ex.ngrams(txt, n) & all_held_grams:
            gram_rows.append(ri)
        best = 0.0
        for t in plain_user_turns(r):
            x = ex.norm(t)
            if x not in turn_best:            # memoised per distinct normalised turn: the maximum over y does not depend on the row, so the row maximum is unchanged
                tb = 0.0
                for y in all_held_norm:
                    sm = difflib.SequenceMatcher(None, x, y, autojunk=False)
                    if sm.real_quick_ratio() <= tb or sm.quick_ratio() <= tb:
                        continue
                    tb = max(tb, sm.ratio())
                turn_best[x] = tb
            best = max(best, turn_best[x])
        if best >= thr:
            ratio_rows.append(ri)
    rep["rows_with_heldout_8gram"] = gram_rows
    rep["rows_with_heldout_turn_ratio_ge_thr"] = ratio_rows
    bench = []
    for tid in train_ids:
        if tid in src and turns(src[tid]):
            c = ex.check_task(turns(src[tid]), task_id=tid)
            if c["reject"]:
                bench.append({"task": tid, **c})
    rep["tasks_rejected_by_check_task"] = bench
    # BFCL similarity on the ACTUAL row text (plain user turns), not the source question
    row_bench = []
    seen = set()
    for ri, r in enumerate(rows):
        for t in plain_user_turns(r):
            k = ex.norm(t)
            if k in seen or not k:
                continue
            seen.add(k)
            c = ex.check_task([t])
            if c["reject"]:
                row_bench.append({"row": ri, **c})
    rep["row_turns_rejected_by_check_task"] = row_bench
    here = pathlib.Path(__file__).resolve()
    rep["code_sha256"] = {"check_provenance.py": sha(here),
                          "distill_exclusions.py": sha(here.parents[1] / "benchmark" / "distill_exclusions.py"),
                          "distill_split.py": sha(here.parents[1] / "benchmark" / "distill_split.py")}
    exl = ex.load_excluded()
    rep["bfcl_turn_texts_sha256"] = hashlib.sha256("\n".join(sorted(exl["texts"].values())).encode()).hexdigest()
    rep["bfcl_turn_texts_n"] = len(exl["texts"])
    # reasoning_content (arm B) must pass the same BFCL checks as user text: excluded ids, 8-grams of excluded user turns, exact excluded turns
    ex_all = ex.load_excluded()
    id_re = __import__("re").compile(r"multi_turn_[a-z_]+_\d+")
    all_bfcl_ids = {i for v in ex_all["ids"].values() for i in v}
    bfcl_gram = set(ex_all["ngrams"])
    reas_rows = []
    for ri, r in enumerate(rows):
        for m in r["messages"]:
            rc = m.get("reasoning_content")
            if rc and (any(x in all_bfcl_ids for x in id_re.findall(rc)) or (ex.ngrams(rc, n) & bfcl_gram) or ex.norm(rc) in ex_all["texts"]):
                reas_rows.append(ri)
    rep["rows_with_bfcl_text_in_reasoning"] = reas_rows
    rep["rows_with_reasoning_content"] = sum(1 for r in rows if any(m.get("reasoning_content") for m in r["messages"]))
    bs = rep["benchmark_scan"]
    rep["pass"] = (bs["id_rows"] == 0 and not reas_rows and bs["ngram_rows"] == 0 and bs["exact_turn_rows"] == 0
                   and not rep["id_overlap_with_heldout"] and not hits and not missing
                   and not gram_rows and not ratio_rows and not bench and not row_bench and not fam and not bad_gold
                   and not rep["heldout_tasks_without_text"] and not rep["rows_without_plain_user_turn_or_empty_message"])
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("inputs", "near_duplicate_pairs", "code_sha256")}, indent=1))
    print("near-duplicate pairs:", len(hits))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
