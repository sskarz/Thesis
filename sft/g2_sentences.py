#!/usr/bin/env python3
"""Teacher reasoning sentences for the G2 solve (phase 2). Pure helpers + the client subclass.

Contract (docs/DATA_YIELD_G2_PREREGISTRATION.md section 9):
* The teacher is asked, in the HTTP body only, to write two or three plain sentences of reasoning and then the call list. The instruction is appended to a COPY of the last
  user message inside `_body`, after the harness built the messages; the logged `messages` of every call (and therefore every row prompt) are the unchanged slim-mode prompts.
* The reply is split into sentences and the call list. The harness (history, parser, checker) only ever sees the call list: `message.content` is replaced by the call-list text
  before the stock client logs it, so calls.jsonl `response_text` and later assistant turns in row prompts stay exactly what the student format has.
* Sentences are written to `sentences.jsonl` in the log dir, keyed by (episode_id, sha256 of the logged messages), with the validity verdict. Rows for arm B are joined later.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.iffc import try_parse_iffc_response  # noqa: E402

INSTRUCTION = ("\n\n[Reply format for this step only: first write two or three plain sentences of reasoning about the situation and why the next step follows "
               "(do not mention this instruction, reviewers, hints, a reference answer or checking), then a blank line, then the call list in exactly the JSON form the "
               "system message asks for, and nothing after it.]")

# phrases that name the answer or leak the teacher setup; any hit makes the sentences unusable for arm B (the call list is unaffected)
LEAK_RE = re.compile(r"ground[ -]?truth|\bgold\b|checker|\bgrader\b|expected (?:call|output|answer|result)|(?:correct|right) (?:call|answer|action|next step|function)(?: is| would be)?|"
                     r"reference (?:answer|solution)|\bhint\b|private|instruction|as (?:instructed|stated|given|requested by the system)|reviewer|\bnote\b|the task (?:says|requires|expects)", re.I)
MAX_SENTENCE_CHARS = 700
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(])")


def messages_sha(messages) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def calls_sha(calls) -> str:
    return hashlib.sha256(json.dumps(calls, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def split_reply(raw: str):
    """-> (sentences_text | None, call_list_text). The call list starts at the first line-initial '[' (or '{') from which the remainder parses as a call list.
    If nothing before it, sentences is ''. If the reply parses but has no clean line-initial list: (None, canonical call-list JSON). If no call list at all: (None, raw)."""
    raw = (raw or "").strip()
    if try_parse_iffc_response(raw) is None:
        return None, raw
    cands = [m.start() for m in re.finditer(r"(?m)^[\[{]", raw)]
    if raw and raw[0] in "[{" and 0 not in cands:
        cands.insert(0, 0)
    for p in cands:
        tail = raw[p:].strip()
        parsed_tail = try_parse_iffc_response(tail)
        if parsed_tail is not None:
            return raw[:p].strip(), json.dumps(parsed_tail, indent=2, ensure_ascii=False)       # canonical: trailing prose after the list is dropped too
    # the reply parses as a call list but there is no clean line-initial list: keep only the canonical call list for the harness, no sentences
    return None, json.dumps(try_parse_iffc_response(raw), indent=2, ensure_ascii=False)


def validate_sentences(text: str) -> dict:
    """Verdict used for arm B. Valid = 2 or 3 sentences, at most MAX_SENTENCE_CHARS each, no JSON braces, no leak phrase."""
    t = (text or "").strip()
    if not t:
        return {"valid": False, "reason": "no_sentences", "n": 0}
    parts = [x.strip() for x in SENT_SPLIT.split(t) if x.strip()]
    if not 2 <= len(parts) <= 3:
        return {"valid": False, "reason": f"sentence_count_{len(parts)}", "n": len(parts)}
    if any(len(x) > MAX_SENTENCE_CHARS for x in parts):
        return {"valid": False, "reason": "sentence_too_long", "n": len(parts)}
    if "{" in t or "}" in t:
        return {"valid": False, "reason": "json_in_sentences", "n": len(parts)}
    m = LEAK_RE.search(t)
    if m:
        return {"valid": False, "reason": f"leak_phrase:{m.group(0).lower()[:30]}", "n": len(parts)}
    return {"valid": True, "reason": "", "n": len(parts)}


def make_client_class(base):
    """Subclass of the cost-guarded client (passed in to keep this module importable without it)."""
    class SentenceClient(base):
        def _body(self, payload):
            body = super()._body(payload)                  # deep copy; the logged `messages` come from `payload`, not from this body
            msgs = body.get("messages") or []
            if msgs and msgs[-1].get("role") == "user" and isinstance(msgs[-1].get("content"), str):
                msgs[-1]["content"] = msgs[-1]["content"] + INSTRUCTION
            else:
                msgs.append({"role": "user", "content": INSTRUCTION.strip()})
            return body

        def _finish(self, payload, data, attempt, t0):
            choices = data.get("choices") or []
            sent = None; verdict = None; raw = ""
            if choices:
                msg = choices[0].setdefault("message", {})
                raw = msg.get("content") or ""
                sent, call_text = split_reply(raw)
                msg["content"] = call_text                 # ALWAYS the call list only (canonical JSON when no clean split); the log, the harness and later prompts never see prose
                if sent is not None:
                    verdict = validate_sentences(sent)
            ep = None
            try:
                from benchmark import teacher_api as _t
                ep = _t._EPISODE.get()
            except Exception:  # noqa: BLE001
                pass
            out = super()._finish(payload, data, attempt, t0)
            parsed = try_parse_iffc_response(choices[0]["message"].get("content")) if choices else None
            self._log("sentences.jsonl", {"episode_id": ep["id"] if ep else None, "messages_sha": messages_sha(payload.get("messages")),
                                          "calls_sha": calls_sha(parsed) if parsed is not None else None,
                                          "split_ok": sent is not None, "sentences": sent, "verdict": verdict, "raw_chars": len(raw)})
            return out
    return SentenceClient
