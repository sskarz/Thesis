"""Round 5 A0.4: eligible-state inventory and source-history guard audit (no model, no network, aggregate-only output).
usage: r5_state_audit.py --labels L1 L2 ... --drop-tasks drop.json --task-sources S1 S2 ... --heldout H1 H2 ... --out report.json --drop-out drop_r5.json
Eligible = decision keep, not in drop tasks, deduped on (messages, target) like build_recovery_rows, and (recovery OR correction):
  recovery = last user message has a tool-error result (build_recovery_rows.after_error); correction = canonical(label) != canonical(student_calls).
Guards (each source task is dropped if ANY hits): (a) task-level vs each held-out bank: id overlap, gold-signature equality, user-turn 8-gram share >=1,
per-turn similarity >=0.85; (b) state-level: 8-gram of any held-out bank user turn or any BFCL multi-turn user turn within the rendered NON-system
content (user, assistant, reasoning_content) of the state's messages, plus BFCL exact-turn matches. The report lists only counts; the drop list contains
training-pool task ids only (never held-out ids)."""
import argparse, collections, difflib, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from benchmark import distill_exclusions as ex, distill_split as ds
from sft.build_recovery_rows import after_error


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def canon(calls):
    out = []
    for c in calls or []:
        out.append((c["function_name"], json.dumps(c.get("parameters", {}), sort_keys=True, ensure_ascii=False)))
    return out


def turns(e):
    return [t for t in ex._user_turns(e) if str(t).strip()]


def rendered(msgs):
    parts = []
    for m in msgs:
        if m.get("role") == "system":
            continue
        parts.append(str(m.get("content") or ""))
        if m.get("reasoning_content"):
            parts.append(str(m["reasoning_content"]))
    return " ".join(parts)


def select_states(label_paths, drop):
    """Registered selection: keep AND (recovery OR canonical correction) -> sort -> dedupe on (messages, target) -> drop conflicting-target prompts."""
    info = collections.Counter(); cand = []
    for p in label_paths:
        for x in load(p):
            if x["decision"] != "keep":
                info["not_kept"] += 1; continue
            if x["task_id"] in drop:
                info["dropped_task"] += 1; continue
            rec = after_error(x["messages"]); cor = canon(x["label"]) != canon(x["student_calls"])
            x["_rec"], x["_cor"] = rec, cor
            if rec or cor:
                cand.append(x)
            else:
                info["not_recovery_or_correction"] += 1
    # registered order (r29): eligibility first, then sort by (task_id, episode_id, step), then dedupe on (messages, target) keeping the first
    cand.sort(key=lambda x: (x["task_id"], x["episode_id"], x["step"], x.get("req_index", 0)))
    seen, states = set(), []
    for x in cand:
        key = json.dumps([*x["messages"], x["label"]], sort_keys=True)
        if key in seen:
            info["duplicate"] += 1; continue
        seen.add(key); states.append(x)
    # identical prompts with conflicting targets are dropped (all occurrences)
    byp = collections.defaultdict(set)
    for x in states:
        byp[json.dumps(x["messages"], sort_keys=True)].add(json.dumps(canon(x["label"])))
    conflict = {k for k, v in byp.items() if len(v) > 1}
    info["conflicting_prompt_states_dropped"] = sum(json.dumps(x["messages"], sort_keys=True) in conflict for x in states)
    states = [x for x in states if json.dumps(x["messages"], sort_keys=True) not in conflict]
    return states, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--drop-tasks", required=True)
    ap.add_argument("--task-sources", nargs="+", required=True)
    ap.add_argument("--heldout", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop-out", required=True)
    a = ap.parse_args()
    n = int(ex.THRESHOLDS["ngram_n"]); thr = ex.THRESHOLDS["turn_ratio_reject_ge"]
    drop = set(json.load(open(a.drop_tasks)))
    src = {}
    for p in a.task_sources:
        for e in load(p):
            src[e.get("id") or e.get("task_id")] = e
    held = {}
    for p in a.heldout:
        for e in load(p):
            held[e["id"]] = (p, e)
    states, info = select_states(a.labels, drop)
    tasks = sorted({s["task_id"] for s in states})
    missing_src = [t for t in tasks if t not in src]
    # held-out grams / norms
    hg, hn, hsig, hbank = {}, {}, collections.defaultdict(list), {}
    for hid, (p, e) in held.items():
        ts = turns(e); g = set()
        for t in ts:
            g |= ex.ngrams(t, n)
        hg[hid] = g; hn[hid] = [ex.norm(t) for t in ts]; hsig[ds.signature(e)].append(hid); hbank[hid] = pathlib.Path(p).name
    exl = ex.load_excluded()
    task_hits = collections.defaultdict(set)  # task -> set of guard names
    for t in tasks:
        if t not in src:
            continue
        e = src[t]; ts = turns(e); tg = set()
        for x in ts:
            tg |= ex.ngrams(x, n)
        tn = [ex.norm(x) for x in ts]
        if t in held:
            task_hits[t].add("id_overlap")
        if ds.signature(e) in hsig:
            task_hits[t].add("signature_overlap_heldout")
        for hid in held:
            if tg & hg[hid]:
                task_hits[t].add("task_8gram_" + hbank[hid])
            best = 0.0
            for x in tn:
                for y in hn[hid]:
                    sm = difflib.SequenceMatcher(None, x, y, autojunk=False)
                    if sm.real_quick_ratio() <= best or sm.quick_ratio() <= best:
                        continue
                    best = max(best, sm.ratio())
            if best >= thr:
                task_hits[t].add("task_ratio_" + hbank[hid])
        c = ex.check_task(ts, task_id=t)
        if c["reject"]:
            task_hits[t].add("task_bfcl_check_task")
    all_hg = set().union(*hg.values()) if hg else set()
    all_hn = [y for v in hn.values() for y in v]
    from sft.check_provenance import plain_user_turns
    import re
    id_re = re.compile(r"multi_turn_[a-z_]+_\d+"); all_bfcl_ids = {i for v in exl["ids"].values() for i in v}
    state_hits = collections.Counter(); hit_states = set()
    for i, s in enumerate(states):
        tgt = json.dumps(s["label"], ensure_ascii=False)
        full = s["messages"] + [{"role": "assistant", "content": tgt}]
        txt = " ".join(str(v) for m in full if m.get("role") != "system" for k, v in m.items() if k in ("content", "reasoning_content", "reasoning") and v)
        g = ex.ngrams(txt, n)
        def hit(name, tid=s["task_id"]):
            state_hits[name] += 1; task_hits[tid].add(name); hit_states.add(i)
        if g & all_hg: hit("state_8gram_heldout")
        if g & exl["ngrams"]: hit("state_8gram_bfcl")
        if any(ex.norm(m.get("content", "")) in exl["texts"] for m in s["messages"] if m.get("role") == "user"): hit("state_exact_bfcl_user_turn")
        if any(m_ in all_bfcl_ids for m_ in id_re.findall(json.dumps(full, ensure_ascii=False))): hit("state_serialized_bfcl_id")
        pt = plain_user_turns({"messages": s["messages"]})
        best = 0.0
        for t in pt:
            x_ = ex.norm(t)
            for y in all_hn + list(exl["texts"].keys()):
                sm = difflib.SequenceMatcher(None, x_, y, autojunk=False)
                if sm.real_quick_ratio() <= best or sm.quick_ratio() <= best: continue
                best = max(best, sm.ratio())
        if best >= thr: hit("state_plain_turn_ratio")
    new_drops = sorted(t for t, h in task_hits.items() if h)
    drop_tasks_out = sorted(set(new_drops) | drop)
    kept_states = [s for s in states if s["task_id"] not in set(drop_tasks_out)]
    rep = {"labels": a.labels, "info": dict(info), "eligible_states_before_guards": len(states), "eligible_tasks_before_guards": len(tasks),
           "eligible_recovery": sum(s["_rec"] for s in states), "eligible_correction_only": sum((not s["_rec"]) and s["_cor"] for s in states),
           "task_sources_missing": missing_src, "heldout_banks": {pathlib.Path(p).name: sum(1 for h in held.values() if h[0] == p) for p in a.heldout},
           "bfcl_turn_texts_n": len(exl["texts"]), "state_level_hits": dict(state_hits), "states_with_any_state_hit": len(hit_states),
           "tasks_with_any_guard_hit": len(new_drops), "drop_list_includes_original_declared_drops": sorted(drop), "guard_hit_counts": dict(collections.Counter(g for h in task_hits.values() for g in h)),
           "eligible_states_after_dropping_hit_tasks": len(kept_states), "eligible_tasks_after": len({s["task_id"] for s in kept_states}),
           "eligible_recovery_after": sum(s["_rec"] for s in kept_states),
           "eligible_correction_only_after": sum((not s["_rec"]) and s["_cor"] for s in kept_states),
           "pass_no_missing_sources": not missing_src}
    json.dump(drop_tasks_out, open(a.drop_out, "w"))
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
