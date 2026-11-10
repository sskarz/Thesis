# Round 4 harness change H (frozen spec)

Mode name: `iffc-ooda-card-recover` (`config.FC_IFFC_OODA_RECOVER`). The existing `iffc-ooda-card` mode is unchanged (offline smoke asserts the plain mode renders as before). Scope approved by Sans 2026-11-08: generic nudge only; no benchmark-specific values or mappings; no extra attempt budget; no argument resolution; no termination changes.

## Trigger
A step is an *error step* if any execution result of the most recently executed step is an error per `iffc_state.is_error_result` (an `error` field that is not empty/false, or a string matching the existing error regex). Mixed success/error results count as an error step.

## Edits (these two only, plus the system-prompt line that explains the new value)
1. **Execution message note.** After an error step, the user-role execution message gets one extra paragraph before the usual footer:
   `Note: at least one call above returned an error. Read the error message and call again with corrected arguments, or call a lookup function to find valid values.`
2. **Card next-mode.** The `next:` field of the `[OODA]` card is `retry` instead of `call` only when (a) the existing logic would have said `call` (so `ask` for missing-parameter errors, `final`, `refuse` keep precedence), (b) the tracker's most recent step was an error step, and (c) the last chat message is that execution message (a new user turn clears it).
3. **System addendum** (recover mode only): `next is one of call, retry, ask, final, refuse: call = emit function calls; retry = the previous call returned an error, emit corrected function calls (or a lookup call to find valid values); ask / final / refuse = return [].` (the rest of the addendum is unchanged).

Nothing else changes: same system prompt body, same stop-on-repeat gate, same decode settings, same step limit.

## Not in H
No hard-coded mappings or values from the benchmark, no appended `.txt`, no unit canonicalization, no state-based argument resolution, no extra attempts, no change to when the harness ends a turn.

## Checks
`python -m benchmark.smoke_ooda_recover` (offline): plain mode byte-identical; recover mode differs only after an error step while the execution message is last; missing-param errors keep `ask`; exact note and addendum text. `smoke_ooda_card` and `smoke_iffc` still pass. Codex review of this spec and diff before any F run.

## Reporting
Per arm: model requests, tool invocations, mean API calls per episode, tokens and latency; share of post-error calls that are empty / new call; baseline-defined error and no-error subsets (episodes with an error result in B1 or B2 on F).
