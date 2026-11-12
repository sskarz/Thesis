#!/usr/bin/env python3
import copy, json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import g2_sentences as S

L = '[\n  {"function_name": "ls", "parameters": {}}\n]'
# split: two sentences + list
sent, call = S.split_reply("The user wants to see the folder. I will list its contents first.\n\n" + L)
assert sent == "The user wants to see the folder. I will list its contents first." and json.loads(call)[0]["function_name"] == "ls"
# empty list is a valid call list; bare list has no sentences
assert S.split_reply("[]") == ("", "[]")
assert S.split_reply(L)[0] == ""
# list on the same line as the sentences: no clean split -> sentences None, canonical call list
sent, call = S.split_reply("I will list the folder. " + L.replace("\n", " "))
assert sent is None and json.loads(call)[0]["function_name"] == "ls" and "I will" not in call
# list followed by trailing prose: parser may accept it; the returned call list must be canonical JSON only
sent, call = S.split_reply("I will list the folder. It is the first step.\n\n" + L + "\n\nThat is my answer.")
assert "That is my answer" not in call and json.loads(call)[0]["function_name"] == "ls", call
# no call list at all
assert S.split_reply("I do not know.") == (None, "I do not know.")
# validation
ok = S.validate_sentences("The user asked for the folder contents. Listing the directory is the first step.")
assert ok["valid"] and ok["n"] == 2
assert S.validate_sentences("One sentence only.")["reason"] == "sentence_count_1"
assert S.validate_sentences("A. B. C. D.")["reason"].startswith("sentence_count")
assert S.validate_sentences("The correct call is ls. It shows the files.")["reason"].startswith("leak_phrase")
assert S.validate_sentences("I follow the instruction. Then I list the files.")["reason"].startswith("leak_phrase")
assert S.validate_sentences("I call {x}. Then I list the files.")["reason"] == "json_in_sentences"
assert S.validate_sentences("")["reason"] == "no_sentences"

# client subclass with a stub base
class Base:
    logged = []
    def __init__(self): self.sent_log = []
    def _body(self, payload): return copy.deepcopy(payload)
    def _finish(self, payload, data, attempt, t0):
        Base.logged.append({"messages": payload["messages"], "response_text": data["choices"][0]["message"]["content"]}); return data
    def _log(self, name, rec): self.sent_log.append((name, rec))
C = S.make_client_class(Base)
c = C()
payload = {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "list the folder"}]}
before = copy.deepcopy(payload)
body = c._body(payload)
assert payload == before and body["messages"][-1]["content"].startswith("list the folder") and "Reply format for this step only" in body["messages"][-1]["content"]
assert body["messages"][0] == payload["messages"][0]
data = {"choices": [{"message": {"content": "The user wants the folder. Listing it is the first step.\n\n" + L}}]}
out = c._finish(payload, data, 0, 0.0)
assert json.loads(Base.logged[0]["response_text"]) == json.loads(L) and "Listing it" not in Base.logged[0]["response_text"]              # the stock log sees the call list only
assert Base.logged[0]["messages"] == before["messages"]                                                        # logged prompt has no instruction
name, rec = c.sent_log[0]
assert name == "sentences.jsonl" and rec["split_ok"] and rec["verdict"]["valid"] and rec["messages_sha"] == S.messages_sha(before["messages"])
# same-line reply: prose must not reach the stock log either
Base.logged.clear(); c2 = C()
data2 = {"choices": [{"message": {"content": "I will list the folder. " + L.replace("\n", " ")}}]}
c2._finish(payload, data2, 0, 0.0)
assert "I will" not in Base.logged[0]["response_text"] and json.loads(Base.logged[0]["response_text"])[0]["function_name"] == "ls"
rec2 = c2.sent_log[0][1]
assert rec2["split_ok"] is False and rec2["calls_sha"] == S.calls_sha([{"function_name": "ls", "parameters": {}}])
print("test_g2_sentences: OK")

# rows_b join
import subprocess, tempfile
with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d); (d / "rows").mkdir(); (d / "solve").mkdir()
    def row(ep, user, calls):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": user}]
        return {"messages": msgs + [{"role": "assistant", "content": json.dumps(calls, indent=2)}], "meta": {"episode_id": ep, "task_id": ep.split("__")[0], "action": "x"}}, msgs
    r1, m1 = row("t1__s0", "go one", [{"function_name": "ls", "parameters": {}}])
    r2, m2 = row("t2__s0", "go two", [])
    r3, m3 = row("t3__s0", "go three", [{"function_name": "pwd", "parameters": {}}])
    r4, m4 = row("t4__s0", "go four", [{"function_name": "pwd", "parameters": {}}])
    (d / "rows/train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (r1, r2, r3, r4))); (d / "rows/valid.jsonl").write_text("")
    sent = [
        {"episode_id": "t1__s0", "messages_sha": S.messages_sha(m1), "calls_sha": S.calls_sha([{"function_name": "ls", "parameters": {}}]), "split_ok": True, "sentences": "First sentence here. Second one follows.", "verdict": {"valid": True}},
        {"episode_id": "t2__s0", "messages_sha": S.messages_sha(m2), "calls_sha": S.calls_sha([]), "split_ok": True, "sentences": "Nothing more is needed. The request is complete.", "verdict": {"valid": True}},
        {"episode_id": "t3__s0", "messages_sha": S.messages_sha(m3), "calls_sha": S.calls_sha([{"function_name": "pwd", "parameters": {}}]), "split_ok": True, "sentences": "The correct call is pwd. Fine.", "verdict": {"valid": False, "reason": "leak_phrase:the correct call is"}},
    ]                                                                      # t4 has no record at all
    (d / "solve/sentences.jsonl").write_text("".join(json.dumps(x) + "\n" for x in sent))
    r = subprocess.run([sys.executable, str(HERE / "g2_rows_b.py"), "--rows-dir", str(d / "rows"), "--solve-dir", str(d / "solve"), "--out-dir", str(d / "b")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    rep = json.loads(r.stdout.strip().splitlines()[-1])
    assert rep["counts"]["rows_arm_a"] == 4 and rep["counts"]["rows_arm_b"] == 2 and rep["counts"]["excluded_leak_phrase"] == 1 and rep["counts"]["excluded_no_record"] == 1, rep
    b = [json.loads(l) for l in open(d / "b/train.jsonl")]
    assert b[0]["messages"][-1]["reasoning_content"].startswith("First sentence") and b[0]["messages"][-1]["content"] == r1["messages"][-1]["content"] and b[0]["messages"][:-1] == m1
print("test_g2_rows_b: OK")
