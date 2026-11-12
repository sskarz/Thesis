#!/usr/bin/env python3
"""G4 helpers. Subcommands:
  select  --a-attempt DIR --b-attempt DIR|none --out sel.json     choose the single trained model (higher FK run-1 composite, ties to A) and check its gates
  collect --results-dir DIR --out run.json --name U1               read one finished (or running, --partial) G4 run from its own benchmark.db, read-only
  parity  --think on|off --out parity.json [--ref slim_render.jsonl]   server /apply-template against the local tokenizer on the opening prompts of the 200 BFCL tasks
"""
import argparse, hashlib, json, os, re, sqlite3, sys

ERR_FLAG = re.compile(r"^(context_overflow|http_error_|iffc_unparseable)")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_select(a):
    arms = {}
    for arm, d in (("A", a.a_attempt), ("B", a.b_attempt)):
        if not d or d == "none":
            continue
        rp = os.path.join(d, f"G3_{arm}_REPORT.json")
        r1p = os.path.join(d, f"G3_{arm}_RUN1.json")
        if not os.path.exists(r1p):
            if arm == "B":
                sys.exit("arm B has no G3_B_RUN1.json yet: its FK screen is not finished; G4 does not start")
            sys.exit(f"{r1p} missing")
        r1 = json.load(open(r1p)); thr = r1["threshold_run1"]
        rep = json.load(open(rp)) if os.path.exists(rp) else {}
        run1 = r1["run1"]["composite"] if r1["run1"]["valid"] else -1
        run2 = rep.get("run2", {}).get("composite") if rep.get("run2", {}).get("valid") else None
        gate = run1 >= thr and run2 is not None and run2 >= thr
        gg = os.path.join(d, f"G3{arm}.gguf")
        rec = os.path.join(d, "runlogs", f"adapter_sha_G3{arm}run1.txt")
        sha = sha256_file(gg)
        recorded = open(rec).read().split()[0]
        arms[arm] = {"attempt": d, "run1": run1, "run2": run2, "threshold": thr, "gates_met": gate, "gguf": gg, "gguf_sha256": sha,
                     "screened_sha256": recorded, "sha_matches_screened": sha == recorded, "think": "off" if arm == "A" else "on"}
    if "B" not in arms:
        sys.exit("arm B's screen is not available; G4 does not start before it")
    pick = "B" if arms["B"]["run1"] > arms["A"]["run1"] else "A"          # ties go to A
    ch = arms[pick]
    ok = ch["gates_met"] and ch["sha_matches_screened"]
    out = {"chosen": pick, "rule": "higher FK run-1 composite of arm A and arm B, ties to A; must pass run 1 and the repeat (>= threshold) and be the screened GGUF", "arms": arms, "ok": ok}
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "arms"}), {x: (arms[x]["run1"], arms[x]["run2"]) for x in arms})
    sys.exit(0 if ok else 1)


def cmd_collect(a):
    db = os.path.join(a.results_dir, "benchmark.db")
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    runs = c.execute("select id, notes, started_at, finished_at from runs order by id").fetchall()
    if len(runs) != 1:
        sys.exit(f"expected exactly one run in {db}, found {len(runs)}")
    rid, notes, started, finished = runs[0]
    rows = c.execute("select entry_id, result, flags from responses where run_id=? and category='multi_turn_base' and mode='multi'", (rid,)).fetchall()
    failed = {r[0] for r in c.execute("select entry_id from eval_errors where run_id=? and category='multi_turn_base'", (rid,))}
    score = c.execute("select correct, total from scores where run_id=? and category='multi_turn_base'", (rid,)).fetchone()
    errored = sorted(e for e, res, fl in rows if "ERROR:" in res or any(ERR_FLAG.match(x) for x in json.loads(fl or "[]")))
    out = {"name": a.name, "run_id": rid, "notes": notes, "started_at": started, "finished_at": finished, "responses": len(rows),
           "score": list(score) if score else None, "counted_note": "passed_ids excludes errored tasks", "errored_tasks": len(errored), "errored_ids": errored[:30], "partial": bool(a.partial)}
    if not a.partial:
        stored_pass = sorted(e for e, _, _ in rows if e not in failed)
        passed = sorted(set(stored_pass) - set(errored))                  # an errored, unparsed or overflowed task counts as a fail whatever the checker said
        out["passed_ids"] = passed; out["stored_passes"] = len(stored_pass); out["counted_passes"] = len(passed)
        out["valid"] = bool(score) and score[1] == 200 and len(rows) == 200 and len(errored) <= 10 and finished is not None
        out["void_reason"] = None if out["valid"] else f"score={score} responses={len(rows)} errored={len(errored)} finished={finished}"
        if score and len(stored_pass) != score[0]:
            out["valid"] = False; out["void_reason"] = f"pass list {len(stored_pass)} != stored score {score[0]}"
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k not in ("passed_ids", "errored_ids")}))
    sys.exit(0)


def cmd_parity(a):
    sys.path.insert(0, os.getcwd())
    import requests
    from benchmark import config, generate_multi_turn as G
    from benchmark import ooda_card as OC
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/Users/USER/Documents/GitHub/Thesis/sft/models/gemma-4-e2b-it-mlx")
    think = a.think == "on"
    flt = getattr(G, "_filter_entry_ids", lambda x: x)
    entries = flt(G.load_jsonl(config.BFCL_DATA_DIR / "BFCL_v4_multi_turn_base.json"))
    ref = {}
    if a.ref:
        ref = {json.loads(l)["id"]: json.loads(l) for l in open(a.ref) if l.strip()}
    norm = lambda s: s[5:] if s.startswith("<bos>") else s
    bad, markers, ref_bad, ref_n = [], 0, [], 0
    for e in entries:
        tools = G._load_func_docs(e.get("involved_classes", []))
        hist = [dict(m) for m in e["question"][0]]
        G._prepare_step_messages(hist, tools, "iffc-state-ooda-slim")
        tracker = OC.OodaCardTracker() if G._uses_ooda("iffc-state-ooda-slim") else None
        msgs = G._build_chat_payload(hist, tools, "iffc-state-ooda-slim", ooda_tracker=tracker)["messages"]
        if a.local_only:
            srv = None
        else:
            srv = requests.post(os.environ.get("G4_SERVER", "http://127.0.0.1:8080") + "/apply-template", json={"messages": msgs}, timeout=60).json()["prompt"]
        loc = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
        if srv is not None:
            markers += srv.count("<|think|>")
            if norm(srv) != norm(loc):
                bad.append(e["id"])
        if think and e["id"] in ref:      # the offline render recorded 2026-11-09 for Runs 83 to 85 was made with the tokenizer's default (no thinking marker); compare the body without the marker
            ref_n += 1
            body = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            if hashlib.sha256(body.encode()).hexdigest() != ref[e["id"]]["text_sha"]:
                ref_bad.append(e["id"])
    res = {"think": a.think, "tasks": len(entries), "mismatches_server_vs_local": len(bad), "mismatch_ids": bad[:10], "server_think_markers": markers,
           "expected_markers": len(entries) if think else 0, "ref_compared": ref_n, "ref_mismatches": len(ref_bad), "ref_mismatch_ids": ref_bad[:10]}
    json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res))
    sys.exit(0 if not bad and (a.local_only or markers == res["expected_markers"]) and not ref_bad else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("select"); s.add_argument("--a-attempt", required=True); s.add_argument("--b-attempt", required=True); s.add_argument("--out", required=True)
    c = sp.add_parser("collect"); c.add_argument("--results-dir", required=True); c.add_argument("--out", required=True); c.add_argument("--name", required=True); c.add_argument("--partial", action="store_true")
    p = sp.add_parser("parity"); p.add_argument("--think", choices=["on", "off"], required=True); p.add_argument("--out", required=True); p.add_argument("--ref", default=""); p.add_argument("--local-only", action="store_true", help="tokenizer render against the stored reference only; no server (CPU rehearsal)")
    a = ap.parse_args()
    {"select": cmd_select, "collect": cmd_collect, "parity": cmd_parity}[a.cmd](a)
