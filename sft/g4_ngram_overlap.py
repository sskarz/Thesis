#!/usr/bin/env python3
"""G4 leakage counts: word n-gram overlap between training rows and the 200 BFCL multi_turn_base user turns.
usage: g4_ngram_overlap.py BFCL_QUESTIONS.json OUT.json  LABEL=rows.jsonl [LABEL=rows.jsonl ...]
For each row set: rows and tasks having at least one 5-gram / 8-gram of lowercase word tokens in common with any BFCL user turn,
checked on (a) user messages after the system prompt and (b) assistant text, including reasoning_content (the teacher sentences of arm B).
Template phrases shared by construction are not removed; the counts are an upper bound on copying, a lower bound on paraphrase overlap
(paraphrases and shared state templates are NOT detected). Offline, read-only."""
import json, re, sys
tok = lambda s: re.findall(r"[a-z0-9_']+", (s or "").lower())
def grams(t, n): return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}
q = sys.argv[1]; out = sys.argv[2]
raw = open(q).read().strip()
entries = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]
bf = {5: set(), 8: set()}; nturn = 0
for e in entries:
    for turn in e["question"]:
        for m in turn:
            if m.get("role") == "user":
                t = tok(m["content"]); nturn += 1
                for n in bf: bf[n] |= grams(t, n)
res = {"bfcl_entries": len(entries), "bfcl_user_turns": nturn, "bfcl_5grams": len(bf[5]), "bfcl_8grams": len(bf[8]), "sets": {}}
for spec in sys.argv[3:]:
    label, path = spec.split("=", 1)
    rows = [json.loads(l) for l in open(path) if l.strip()]
    r = {"rows": len(rows)}; hits = {}
    for part in ("user", "assistant"):
        for n in (5, 8):
            rr, tt, ex = set(), set(), []
            for i, row in enumerate(rows):
                msgs = [m for m in row["messages"] if (m["role"] == "user" if part == "user" else m["role"] == "assistant")]
                txt = " ".join((m.get("content") or "") + " " + (m.get("reasoning_content") or "") for m in msgs)
                g = grams(tok(txt), n) & bf[n]
                if g:
                    rr.add(i); tt.add(row["meta"]["task_id"])
                    if len(ex) < 3: ex.append(" ".join(sorted(g)[0]))
            r[f"{part}_{n}gram_rows"] = len(rr); r[f"{part}_{n}gram_tasks"] = len(tt); r[f"{part}_{n}gram_examples"] = ex
    res["sets"][label] = r
json.dump(res, open(out, "w"), indent=1); print(json.dumps({k: v for k, v in res.items() if k != "sets"})); [print(k, {a: b for a, b in v.items() if "examples" not in a}) for k, v in res["sets"].items()]
