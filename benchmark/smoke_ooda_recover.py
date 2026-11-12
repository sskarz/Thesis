"""Offline smoke for the Round 4 recovery mode (iffc-ooda-card-recover). No model, no network.

Checks: (1) the plain mode is byte-identical to before (no 'retry', no note, original addendum);
(2) recover mode changes ONLY on a step whose execution results contain an error, and only while the
last chat message is that execution message; (3) missing-param errors keep 'ask'; (4) the note is exact.
    python -m benchmark.smoke_ooda_recover
"""
from __future__ import annotations

from . import config, ooda_card as oc
from .iffc import format_iffc_execution_user_message

ERR = "{'error': \"Conversion from 'kg' to 'pounds' is not supported\"}"
OK = "{'result': 6.6}"
CALL = "imperial_si_conversion(value=3, unit_in='kg', unit_out='pounds')"


def hist(extra_user: str | None = None, err: bool = True, recover: bool = False):
    tr = oc.OodaCardTracker()
    res = ERR if err else OK
    tr.update([CALL], [res])
    note = oc.RECOVER_EXEC_NOTE if (recover and err) else None
    h = [{"role": "user", "content": "Convert 3 kg to pounds."},
         {"role": "assistant", "content": "[...]"},
         {"role": "user", "content": format_iffc_execution_user_message([CALL], [res], recover_note=note)}]
    if extra_user:
        h.append({"role": "user", "content": extra_user})
    return tr, h


# Golden strings produced by the pre-change code (commit 0f7762b, before H) for the fixed inputs above.
GOLD_CARD = ("[OODA]\ngoal: Convert 3 kg to pounds.\nslots: (none)\n"
             "last: imperial_si_conversion(value=3, unit_in=kg, unit_out=pounds)->error\nnext: call\n[/OODA]")
GOLD_MSG = ("Function execution results:\n- imperial_si_conversion(value=3, unit_in='kg', unit_out='pounds') -> "
            "{'error': \"Conversion from 'kg' to 'pounds' is not supported\"}\n\nRespond with a JSON list of function "
            'calls (each with "function_name" and "parameters"), or [] if no further function calling is needed.')
GOLD_SYS_SHA = "39f74f4dbcc4ae62a1f5ca70caf2cc8adab3a897f35e153f7a8c33e3631df893"
NOTE = ("Note: at least one call above returned an error. Read the error message and call again "
        "with corrected arguments, or call a lookup function to find valid values.")
ADD_RECOVER = ("A [OODA]...[/OODA] block prefixed on the latest user message is a harness-written one-pass snapshot "
               "(goal, known slots, last execution fact, next-mode hint). It is not model text and is not a second "
               "OODA pass. next is one of call, retry, ask, final, refuse: call = emit function calls; retry = the "
               "previous call returned an error, emit corrected function calls (or a lookup call to find valid "
               "values); ask / final / refuse = return []. Continue to answer only with a JSON list of dictionaries "
               'with "function_name" and "parameters", or [].')


def main() -> int:
    import hashlib
    # (0) exact golden checks against pre-change output (plain mode) and exact recover wording
    tr0, h0 = hist()
    assert oc.render_card(tr0, h0, has_tools=True) == GOLD_CARD
    assert h0[-1]["content"] == GOLD_MSG
    assert hashlib.sha256(oc.build_ooda_system_prompt(None).encode()).hexdigest() == GOLD_SYS_SHA
    assert oc.RECOVER_EXEC_NOTE == NOTE and oc.OODA_ADDENDUM_RECOVER == ADD_RECOVER
    _, hr = hist(recover=True)
    assert hr[-1]["content"] == GOLD_MSG.replace("\n\nRespond with", "\n\n" + NOTE + "\n\nRespond with")
    # a fresh user message that merely starts with the exec prefix must not trigger retry
    trf, hf = hist(extra_user="Function execution results: please convert 5 kg", recover=True)
    assert "next: call" in oc.render_card(trf, hf, has_tools=True, recover=True)
    # (1) plain mode unchanged
    tr, h = hist()
    plain = oc.render_card(tr, h, has_tools=True)
    assert "next: call" in plain and "retry" not in plain
    assert "retry" not in oc.OODA_ADDENDUM
    assert "Note: at least one call" not in h[-1]["content"]
    # (2) recover: error step -> retry + note
    tr, h = hist(recover=True)
    rec = oc.render_card(tr, h, has_tools=True, recover=True)
    assert "next: retry" in rec, rec
    assert oc.RECOVER_EXEC_NOTE in h[-1]["content"]
    assert rec.replace("next: retry", "next: call") == plain  # only the next field differs
    # no error -> identical to plain, no note
    tr, h = hist(err=False, recover=True)
    assert oc.render_card(tr, h, has_tools=True, recover=True) == oc.render_card(tr, h, has_tools=True)
    assert oc.RECOVER_EXEC_NOTE not in h[-1]["content"]
    # new user turn after an error: last message is not an execution message -> no retry
    tr, h = hist(extra_user="Now what is 5 plus 5?", recover=True)
    assert "next: call" in oc.render_card(tr, h, has_tools=True, recover=True)
    # (3) missing-param error keeps ask
    tr = oc.OodaCardTracker(); tr.update(["f()"], ["{'error': 'missing required parameter x'}"])
    h = [{"role": "user", "content": "go"}, {"role": "user", "content": "Function execution results:\n- f() -> err"}]
    assert "next: ask" in oc.render_card(tr, h, has_tools=True, recover=True)
    # mode registration + system addendum
    assert config.FC_IFFC_OODA_RECOVER in config.FC_MODES and config.FC_IFFC_OODA_RECOVER in config.FC_IFFC_OODA_MODES
    sysp = oc.build_ooda_system_prompt(None, recover=True)
    assert "retry = the previous call returned an error" in sysp
    assert oc.build_ooda_system_prompt(None) .endswith(oc.OODA_ADDENDUM)
    print("smoke_ooda_recover OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
