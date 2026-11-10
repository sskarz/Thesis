"""Round 5 B1-B3: eligible states -> student samples -> classified responses -> same-state preference pairs. Registered in docs/DISTILLATION_ROUND5_PLAN.md (v2.1 + v2.2).
  states --drop-r5 drop.json --out states.jsonl     : select_states (r5_state_audit) + guard drops; attaches executed history from the student rollout verdicts
  sample --states states.jsonl --out samples.jsonl  : per state greedy + 5 samples at T=0.7 on the served bar GGUF (no adapter), NO explicit max_tokens (the historical native sampler sent none; the server default applies), per-request timeout 1800 s, job deadline 14400 s, resumable (only successful records count)
  pairs  --states ... --samples ... --out-dir DIR   : matcher precedence (exact | execution_error | premature_stop | return_equivalent | unresolved; verified inferior = execution_error, premature_stop), pairs, split, diagnostics"""
import argparse, collections, hashlib, json, math, pathlib, sys, time
from concurrent.futures import ThreadPoolExecutor
import requests
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from benchmark.iffc import iffc_calls_to_func_strings
from benchmark.distill_onpolicy import _replay, turn_complete, same_effect
from benchmark.iffc_state import is_error_result
from sft.r5_state_audit import select_states

D = "sft/data/distill/"
STUDENT_DIRS = [D + "r3/student_pool_old", D + "r3/student_pool2", D + "g/student_pool"]
TASK_SOURCES = [D + "r3/pool_all.jsonl", D + "g/tasks_F.jsonl"]
LABELS = [D + "r3/labels_old/labels.jsonl", D + "r3/labels_new/labels.jsonl", D + "g/labels/labels.jsonl"]
MODEL = "/Users/sanskarthapa/Documents/GitHub/Thesis/sft/models/gemma-4-e2b-it-mlx"
MAX_REASONING_CHARS = 3500
URL = "http://127.0.0.1:8080/v1/chat/completions"


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def canon(calls):
    return [(c["function_name"], json.dumps(c.get("parameters", {}), sort_keys=True, ensure_ascii=False)) for c in calls or []]


def parse_strict(content):
    t = (content or "").strip()
    try:
        obj = json.loads(t)
    except Exception:
        return None
    if not isinstance(obj, list):
        return None
    for c in obj:
        if not (isinstance(c, dict) and isinstance(c.get("function_name"), str) and set(c) <= {"function_name", "parameters"}):
            return None
        if "parameters" in c and not isinstance(c["parameters"], dict):
            return None
    return obj


def cmd_states(a):
    drop = set(json.load(open(D + "g/drop_tasks.json"))) | set(json.load(open(a.drop_r5)))
    states, info = select_states(LABELS, set(json.load(open(D + "g/drop_tasks.json"))))
    states = [s for s in states if s["task_id"] not in drop]
    verd = {}
    for d in STUDENT_DIRS:
        for v in load(d + "/verdicts.jsonl"):
            verd.setdefault(v["episode_id"], v)
    tasks = {}
    for p in TASK_SOURCES:
        for t in load(p):
            tasks[t.get("id") or t.get("task_id")] = t
    out, miss = [], 0
    for i, s in enumerate(states):
        v = verd.get(s["episode_id"])
        if v is None or s["task_id"] not in tasks:
            miss += 1; continue
        flat, st_turns = [], []
        for ti, turn in enumerate(v["result"]):
            for st in turn:
                if isinstance(st, list) and st and not str(st[0]).startswith("ERROR"):
                    flat.append(st); st_turns.append(ti)
        out.append({"state_id": f"{s['episode_id']}@{s['step']}#{s['req_index']}", "task_id": s["task_id"], "episode_id": s["episode_id"], "step": s["step"],
                    "turn": s["turn"], "req_index": s["req_index"], "messages": s["messages"], "label": s["label"], "student_calls": s["student_calls"],
                    "recovery": bool(s["_rec"]), "correction": bool(s["_cor"]), "label_kind": s["kind"], "classes": s["classes"],
                    "flat_steps": flat, "step_turns": st_turns})
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"selected": len(states), "with_history": len(out), "missing_history": miss, "tasks": len({r["task_id"] for r in out}),
                      "recovery": sum(r["recovery"] for r in out), "correction_only": sum((not r["recovery"]) and r["correction"] for r in out), "info": dict(info)}))
    assert miss == 0, "states without executed history"


def cmd_sample(a):
    states = load(a.states)
    la = requests.get("http://127.0.0.1:8080/lora-adapters", timeout=30).json()
    assert la == [], f"adapter loaded on the sampling server: {la}"
    props = requests.get("http://127.0.0.1:8080/props", timeout=30).json()
    assert props.get("model_path", "").endswith("google_gemma-4-E2B-it-Q4_K_M.gguf") and "d344123fe" in str(props.get("build_info")), "sampling server is not the bar GGUF / pinned build"
    ap0 = requests.post("http://127.0.0.1:8080/apply-template", json={"messages": states[0]["messages"]}, timeout=60).json()["prompt"]
    assert ap0.count("<|think|>") == 1, "rendered prompt must contain exactly one think marker"
    manifest = {"model_path": props["model_path"], "build_info": props["build_info"], "lora": la, "temp_samples": a.temp, "n_samples": a.samples,
                "states_sha256": hashlib.sha256(open(a.states, "rb").read()).hexdigest(), "max_tokens": None, "timeout_s": 1800, "deadline_s": a.deadline}
    mp = pathlib.Path(a.out).with_suffix(".manifest.json")
    if mp.exists():
        old = json.load(open(mp)); assert {k: old[k] for k in ("model_path", "build_info", "states_sha256", "temp_samples", "n_samples")} == {k: manifest[k] for k in ("model_path", "build_info", "states_sha256", "temp_samples", "n_samples")}, "resume config differs from the original run"
    else:
        json.dump(manifest, open(mp, "w"), indent=1)
    done = set()
    if pathlib.Path(a.out).exists():
        done = {(r["state_id"], r["sample_idx"]) for r in load(a.out) if "error" not in r and r.get("http") == 200 and r.get("finish_reason") is not None}
    jobs = [(s, k) for s in states for k in range(1 + a.samples) if (s["state_id"], k) not in done]
    print(len(jobs), "generations to do;", len(done), "already done", flush=True)
    t0 = time.time()

    def one(j):
        s, k = j
        if time.time() - t0 > a.deadline:
            return None
        temp = 0.0 if k == 0 else a.temp
        try:
            r = requests.post(URL, json={"messages": s["messages"], "temperature": temp}, timeout=1800)
            r.raise_for_status()
            c = r.json()["choices"][0]; m = c["message"]
            assert "content" in m and c.get("finish_reason") is not None
            return {"state_id": s["state_id"], "sample_idx": k, "temp": temp, "http": r.status_code, "content": m.get("content") or "", "reasoning_content": m.get("reasoning_content") or "",
                    "finish_reason": c.get("finish_reason"), "completion_tokens": r.json().get("usage", {}).get("completion_tokens")}
        except Exception as e:
            return {"state_id": s["state_id"], "sample_idx": k, "temp": temp, "error": repr(e)}
    n = 0; skipped = 0
    with open(a.out, "a") as f, ThreadPoolExecutor(a.workers) as ex:
        for rec in ex.map(one, jobs):
            if rec is None:
                skipped += 1; continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); n += 1
            if n % 100 == 0:
                print(f"{n}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    recs = load(a.out)
    errs = sum("error" in r for r in recs)
    ok = authoritative(recs)
    missing = [(s["state_id"], k) for s in states for k in range(1 + a.samples) if (s["state_id"], k) not in ok]
    print(json.dumps({"records": len(recs), "error_records": errs, "skipped_by_deadline": skipped, "missing_keys": len(missing)}))
    assert not missing, "sampling incomplete (deadline or errors); rerun the same command to retry only the missing keys"


def authoritative(recs):
    """one successful record per (state_id, sample_idx): the LAST successful one in file order"""
    ok = {}
    for r in recs:
        if "error" not in r and r.get("http") == 200 and r.get("finish_reason") is not None:
            ok[(r["state_id"], r["sample_idx"])] = r
    return ok


def classify(s, task, r):
    """Registered precedence (plan v2.1 item 2 / v2.3): truncated | unparsable | no_reasoning (excluded) -> exact -> execution_error -> premature_stop -> return_equivalent -> unresolved.
    Verified inferior = execution_error | premature_stop. replay_exception (checker/infrastructure failure) is excluded and counted."""
    if r.get("error") or r.get("finish_reason") != "stop":
        return "truncated", None
    obj = parse_strict(r["content"])
    if obj is None:
        return "unparsable", None
    if not r["reasoning_content"].strip():
        return "no_reasoning", obj
    label = s["label"] or []
    if canon(obj) == canon(label):
        return "exact", obj
    flat, st_turns, step = s["flat_steps"], s["step_turns"], s["step"]
    if obj:
        try:
            res = _replay(task, flat[:step], iffc_calls_to_func_strings(obj))
        except Exception:
            return "replay_exception", obj
        if any(is_error_result(x) for x in res):
            return "execution_error", obj
    elif label:
        if not turn_complete(task, None, flat, step, s["turn"], st_turns):
            return "premature_stop", obj
        return "unresolved", obj
    try:
        if same_effect(task, flat, step, iffc_calls_to_func_strings(label), iffc_calls_to_func_strings(obj)):
            return "return_equivalent", obj
    except Exception:
        return "replay_exception", obj
    return "unresolved", obj


def cmd_pairs(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = lambda x: x["input_ids"] if hasattr(x, "keys") else x
    states = load(a.states)
    samples = collections.defaultdict(list)
    auth = authoritative(load(a.samples))
    for s_ in states:
        for k in range(6):
            assert (s_["state_id"], k) in auth, f"missing sample {s_['state_id']} #{k}"
    assert {k[0] for k in auth} <= {s_["state_id"] for s_ in states} and all(0 <= k[1] <= 5 for k in auth), "unknown state or sample index"
    for (sid, k), r in auth.items():
        samples[sid].append(r)
    tasks = {}
    for p in TASK_SOURCES:
        for t in load(p):
            tasks[t.get("id") or t.get("task_id")] = t
    diag = collections.Counter(); per_state = {}
    pairs_all = []
    states.sort(key=lambda x: (x["task_id"], x["episode_id"], x["step"], x["req_index"]))

    def work(s):
        task = tasks[s["task_id"]]
        out = []
        for r in sorted(samples[s["state_id"]], key=lambda r: r["sample_idx"]):
            cl, obj = classify(s, task, r)
            out.append((r, cl))
        return out
    with ThreadPoolExecutor(a.workers) as ex:
        cls = list(ex.map(work, states))
    for s, lst in zip(states, cls):
        per_state[s["state_id"]] = lst
        for r, c in lst:
            diag["response_" + c] += 1
    def with_marker(msgs):
        m = json.loads(json.dumps(msgs)); assert m[0]["role"] == "system" and not m[0]["content"].startswith("<|think|>")
        m[0]["content"] = "<|think|>\n" + m[0]["content"]; return m
    def resp(r):
        return {"content": r["content"].strip(), "reasoning_content": r["reasoning_content"].strip()}
    n_over = 0
    for s, lst in zip(states, cls):
        diag["exact_but_reasoning_over_3500"] += sum(1 for r, c in lst if c == "exact" and len(r["reasoning_content"]) > MAX_REASONING_CHARS)
        ch = [r for r, c in lst if c == "exact" and len(r["reasoning_content"]) <= MAX_REASONING_CHARS]
        rj = [(r, c) for r, c in lst if c in ("execution_error", "premature_stop")]
        diag["states_total"] += 1
        diag["states_with_exact"] += bool(ch); diag["states_with_inferior"] += bool(rj); diag["states_with_both"] += bool(ch and rj)
        for k, (c_, (r_, rc)) in enumerate(list(zip(ch, rj))[:2]):
            msgs = with_marker(s["messages"])
            full_c = ids(tok.apply_chat_template(msgs + [{"role": "assistant", **resp(c_)}], tokenize=True))
            full_r = ids(tok.apply_chat_template(msgs + [{"role": "assistant", **resp(r_)}], tokenize=True))
            if max(len(full_c), len(full_r)) > a.max_seq:
                n_over += 1; continue
            pairs_all.append({"pair_id": f"{s['state_id']}|{c_['sample_idx']}-{r_['sample_idx']}", "messages": msgs, "chosen": resp(c_), "rejected": resp(r_),
                              "meta": {"task_id": s["task_id"], "episode_id": s["episode_id"], "step": s["step"], "recovery": s["recovery"], "correction": s["correction"],
                                       "label_kind": s["label_kind"], "classes": s["classes"], "rejected_class": rc, "chosen_sample": c_["sample_idx"], "rejected_sample": r_["sample_idx"],
                                       "n_calls_target": len(s["label"] or []), "student_same": canon(s["student_calls"]) == canon(s["label"]), "len_chosen_tokens": len(full_c), "len_rejected_tokens": len(full_r),
                                       "reasoning_chars_chosen": len(c_["reasoning_content"]), "reasoning_chars_rejected": len(r_["reasoning_content"]),
                                       "chosen_temp": c_["temp"], "rejected_temp": r_["temp"]}})
    diag["pairs_before_split"] = len(pairs_all); diag["pairs_rejected_overlength"] = n_over
    ptasks = sorted({p["meta"]["task_id"] for p in pairs_all}, key=lambda t: hashlib.sha256(t.encode()).hexdigest())
    nval = math.ceil(0.10 * len(ptasks)); valt = set(ptasks[:nval])
    train = [p for p in pairs_all if p["meta"]["task_id"] not in valt]; val = [p for p in pairs_all if p["meta"]["task_id"] in valt]
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        with open(out / name, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    st = lambda rows, key: dict(collections.Counter(key(p) for p in rows))
    chosen_ids = collections.Counter(p["pair_id"].split("|")[0] + "|" + str(p["meta"]["chosen_sample"]) for p in train)
    chosen_content = collections.Counter(p["chosen"]["content"] + "|" + p["chosen"]["reasoning_content"] for p in train)
    rep = {"counts": dict(diag), "tasks_with_pairs": len(ptasks), "val_tasks": len(valt),
           "unique_chosen_response_contents": len(chosen_content), "chosen_content_multiplicity_max": max(chosen_content.values()) if chosen_content else 0,
           "student_agreed_with_teacher_originally_share": sum(p["meta"].get("student_same", False) for p in train) / max(1, len(train)),
           "mean_branch_len_diff_tokens": sum(p["meta"]["len_chosen_tokens"] - p["meta"]["len_rejected_tokens"] for p in train) / max(1, len(train)),
           "by_source": st(train, lambda p: p["meta"]["episode_id"].split("_")[1] if "_" in p["meta"]["episode_id"] else "?"), "train_pairs": len(train), "train_tasks": len({p["meta"]["task_id"] for p in train}),
           "val_pairs": len(val), "train_unique_states": len({p["pair_id"].split("|")[0] for p in train}), "train_unique_chosen_responses": len(chosen_ids),
           "chosen_multiplicity_max": max(chosen_ids.values()) if chosen_ids else 0,
           "by_rejected_class": st(train, lambda p: p["meta"]["rejected_class"]), "by_recovery": st(train, lambda p: "recovery" if p["meta"]["recovery"] else "correction"),
           "by_api_class": st(train, lambda p: p["meta"]["classes"][0] if p["meta"]["classes"] else None), "by_label_kind": st(train, lambda p: p["meta"]["label_kind"]),
           "mean_len_chosen_tokens": sum(p["meta"]["len_chosen_tokens"] for p in train) / max(1, len(train)), "mean_len_rejected_tokens": sum(p["meta"]["len_rejected_tokens"] for p in train) / max(1, len(train)),
           "mean_reasoning_chars_chosen": sum(p["meta"]["reasoning_chars_chosen"] for p in train) / max(1, len(train)), "mean_reasoning_chars_rejected": sum(p["meta"]["reasoning_chars_rejected"] for p in train) / max(1, len(train)),
           "chosen_greedy_share": sum(p["meta"]["chosen_temp"] == 0.0 for p in train) / max(1, len(train)),
           "floor_met_(>=150 pairs, >=50 tasks)": len(train) >= 150 and len({p["meta"]["task_id"] for p in train}) >= 50}
    json.dump(rep, open(out / "diagnostics.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
    if not rep["floor_met_(>=150 pairs, >=50 tasks)"]:
        print("PAIR FLOOR NOT MET (registered stop for stage B)"); sys.exit(2)


def cmd_expand(a):
    """both branches of every train+val pair as plain SFT-style rows (for sft/check_provenance.py)"""
    n = 0
    with open(a.out, "w") as f:
        for name in ("train.jsonl", "val.jsonl"):
            for p in load(pathlib.Path(a.pairs_dir) / name):
                for side in ("chosen", "rejected"):
                    f.write(json.dumps({"messages": p["messages"] + [{"role": "assistant", **p[side]}], "meta": {"task_id": p["meta"]["task_id"], "pair_id": p["pair_id"], "side": side}}, ensure_ascii=False) + "\n"); n += 1
    print("expanded rows", n)


ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="c", required=True)
x = sub.add_parser("states"); x.add_argument("--drop-r5", required=True); x.add_argument("--out", required=True); x.set_defaults(f=cmd_states)
x = sub.add_parser("sample"); x.add_argument("--states", required=True); x.add_argument("--out", required=True); x.add_argument("--samples", type=int, default=5)
x.add_argument("--temp", type=float, default=0.7); x.add_argument("--workers", type=int, default=4); x.add_argument("--deadline", type=int, default=14400); x.set_defaults(f=cmd_sample)
x = sub.add_parser("pairs"); x.add_argument("--states", required=True); x.add_argument("--samples", required=True); x.add_argument("--out-dir", required=True)
x.add_argument("--max-seq", type=int, default=9500); x.add_argument("--workers", type=int, default=4); x.set_defaults(f=cmd_pairs)
x = sub.add_parser("expand"); x.add_argument("--pairs-dir", required=True); x.add_argument("--out", required=True); x.set_defaults(f=cmd_expand)
a = ap.parse_args(); a.f(a)
