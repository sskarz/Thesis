"""Orchestrator: generate → evaluate → collect metrics → visualize.

Function-calling modes (``--fc-mode``):

- ``nfc`` (default): native OpenAI ``tools`` / ``tool_calls``
- ``iffc``: Instruction-Followed Function Calling (arXiv:2608.22472 Appendix B)
  JSON-in-text; used by multi-turn generation only
- ``cd``: TagDispatch-style two-pass constrained decoding (arXiv:2601.04426
  approximation on stock llama-server). Not native tools GBNF.
- ``gst``: Guided-Structured Templates (arXiv:2509.18076) on the NFC channel;
  used by multi-turn generation only. Prompt-only; not GBNF.
- ``iffc-state``: IFFC JSON-in-text plus a harness-written STATE snapshot from
  actual tool execution results (METHODS §5). Not GST; not CD/GBNF.
- ``iffc-state-bon``: Best-of-N majority vote overlay on IFFC+STATE
  (METHODS §4; default N=5, temperature=0.7). Not NFC/GST/CD.
- ``xlam``: Salesforce xLAM JSON-in-text (``{"tool_calls":[{name, arguments}]}``);
  used by multi-turn generation only. No native tools. Fair channel for
  ``xlam-1b`` (Run 19 NFC 0/200 was format-channel collapse; Run 20
  ``--fc-mode xlam`` scored 6/200 = 3.0% --- still very low, not published
  ~78.94%).
- ``iffc-stop-repeat`` / ``iffc-ooda-card`` / ``iffc-decision`` /
  ``iffc-decision-ooda``: see ``benchmark.config`` / ``generate_multi_turn``.

Diagnostic (not a new fc-mode): ``--iffc-strip-empty-assistant`` omits
trailing IFFC assistant ``[]`` from ``chat_history`` before the next BFCL
user turn. Default off (Run 5–13 unchanged). Not a training method.
"""

import argparse
import re
import sys
import time

import requests

from . import config, db
from . import generate_responses
from . import generate_multi_turn
from . import evaluate_responses
from . import evaluate_multi_turn
from . import collect_metrics
from . import visualize


def check_server() -> bool:
    """Return True if the llama-server is reachable."""
    try:
        resp = requests.get(f"{config.SERVER_URL}/v1/models", timeout=5)
        resp.raise_for_status()
        return True
    except Exception:
        return False


_FC_MODE_IN_NOTES = re.compile(r"\bfc_mode=([a-z0-9_.-]+)")
_BON_N_IN_NOTES = re.compile(r"\bbon_n=(\d+)")
_STRIP_EMPTY_IN_NOTES = re.compile(r"\biffc_strip_empty_assistant=([01])")


def _notes_with_run_meta(args: argparse.Namespace) -> str | None:
    extra: list[str] = []
    if args.fc_mode != config.FC_NFC:
        extra.append(f"fc_mode={args.fc_mode}")
    if args.fc_mode in config.FC_IFFC_DECISION_MODES:
        import os
        extra.append(
            "decision_model="
            + os.environ.get("DECISION_MODEL", "tev1:latest")
        )
    if args.fc_mode == config.FC_IFFC_STATE_BON:
        extra.append(f"bon_n={config.BON_N}")
    if args.iffc_strip_empty_assistant:
        extra.append("iffc_strip_empty_assistant=1")
        extra.append(
            "DIAGNOSTIC: IFFC strip trailing assistant [] "
            "(history bookkeeping ablation; not a training method)"
        )
    if args.categories:
        extra.append(f"categories={','.join(args.categories)}")
    if not extra:
        return args.notes
    tag = " ".join(extra)
    return f"{args.notes}; {tag}" if args.notes else tag


def fc_mode_from_notes(notes: str | None) -> str:
    """Parse ``fc_mode=...`` from run notes. Missing tag means NFC."""
    if not notes:
        return config.FC_NFC
    match = _FC_MODE_IN_NOTES.search(notes)
    if match and match.group(1) in config.FC_MODES:
        return match.group(1)
    return config.FC_NFC


def bon_n_from_notes(notes: str | None) -> int | None:
    """Parse ``bon_n=...`` from run notes. Missing tag means unset."""
    if not notes:
        return None
    match = _BON_N_IN_NOTES.search(notes)
    if not match:
        return None
    return int(match.group(1))


def iffc_strip_empty_assistant_from_notes(notes: str | None) -> bool:
    """Parse ``iffc_strip_empty_assistant=1`` from run notes. Missing tag means off."""
    if not notes:
        return False
    match = _STRIP_EMPTY_IN_NOTES.search(notes)
    return bool(match and match.group(1) == "1")


def resolve_pipeline_run(
    model_key: str,
    fc_mode: str,
    notes: str | None,
    resume: bool = False,
    resume_run_id: int | None = None,
    iffc_strip_empty_assistant: bool = False,
) -> tuple[int, bool]:
    """Create a new run or reuse an unfinished one.

    Resume continues the same ``runs`` row so already-checkpointed
    ``responses`` / ``multi_turn_timings`` stay attached. Returns
    ``(run_id, resumed)``.
    """
    if resume_run_id is not None:
        run = db.get_run(resume_run_id)
        if run is None:
            raise ValueError(f"No run with id={resume_run_id}")
        if run["model_key"] != model_key:
            raise ValueError(
                f"Run {resume_run_id} is model_key={run['model_key']!r}, "
                f"not {model_key!r}"
            )
        existing_mode = fc_mode_from_notes(run["notes"])
        if existing_mode != fc_mode:
            raise ValueError(
                f"Run {resume_run_id} is fc_mode={existing_mode}, not {fc_mode}"
            )
        existing_strip = iffc_strip_empty_assistant_from_notes(run["notes"])
        if existing_strip != iffc_strip_empty_assistant:
            raise ValueError(
                f"Run {resume_run_id} is iffc_strip_empty_assistant="
                f"{int(existing_strip)}, not {int(iffc_strip_empty_assistant)}"
            )
        return resume_run_id, True

    if resume:
        for run in db.get_unfinished_runs(model_key):
            if fc_mode_from_notes(run["notes"]) != fc_mode:
                continue
            if (
                iffc_strip_empty_assistant_from_notes(run["notes"])
                != iffc_strip_empty_assistant
            ):
                continue
            return run["id"], True
        strip_tag = (
            " iffc_strip_empty_assistant=1" if iffc_strip_empty_assistant else ""
        )
        raise ValueError(
            f"No unfinished run for model {model_key!r} with fc_mode={fc_mode}"
            f"{strip_tag}. "
            "Omit --resume to start a new run, or pass --resume-run-id."
        )

    return db.create_run(model_key=model_key, notes=notes), False


def main():
    parser = argparse.ArgumentParser(description="Run the BFCL benchmark pipeline.")
    parser.add_argument(
        "--model",
        choices=config.MODELS.keys(),
        default=config.DEFAULT_MODEL,
        help="Model to benchmark (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-multi-turn",
        action="store_true",
        help="Skip multi-turn categories (e.g., for FunctionGemma)",
    )
    parser.add_argument(
        "--only-multi-turn",
        action="store_true",
        help="Run only multi-turn categories (skip single-turn)",
    )
    parser.add_argument(
        "--fc-mode",
        choices=config.FC_MODES,
        default=config.FC_NFC,
        help=(
            "Function-calling mode for multi-turn. "
            "nfc = native OpenAI tools/tool_calls (default); "
            "iffc = Instruction-Followed Function Calling "
            "(arXiv:2608.22472 Appendix B JSON-in-text); "
            "cd = TagDispatch-style two-pass constrained decoding "
            "(name GBNF + per-tool json_schema; not native tools GBNF); "
            "gst = Guided-Structured Templates on NFC tools "
            "(arXiv:2509.18076; prompt-only, not GBNF); "
            "iffc-state = IFFC JSON-in-text + harness-written STATE "
            "(not GST; not CD/GBNF); "
            "iffc-state-bon = Best-of-N majority vote on IFFC+STATE "
            "(default N=5, temperature=0.7; not NFC/GST/CD); "
            "xlam = Salesforce xLAM JSON-in-text "
            '({"tool_calls":[{name, arguments}]}; no native tools); '
            "iffc-stop-repeat = IFFC + harness force-stop on consecutive "
            "identical calls, or executed-call re-emits after 6 assistant "
            "tool-turns in the current ask (baseline iffc unchanged); "
            "iffc-decision = IFFC + Decision sidecar (Ollama System One "
            "Choice next-tool/noop before Gemma args; plain R58 recipe); "
            "iffc-ooda-card = IFFC + one-pass [OODA] card stacked on "
            "stop-repeat MIN=6 (R42 recipe); "
            "iffc-decision-ooda = Decision sidecar + OODA card + "
            "stop-repeat MIN=6 (compose); "
            "iffc-state-ooda = full stack: IFFC + STATE + OODA card "
            "(STATE before card) + stop-repeat MIN=6."
        ),
    )
    parser.add_argument(
        "--bon-n",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Best-of-N sample count for --fc-mode iffc-state-bon "
            f"(default: {config.BON_N_DEFAULT}). Ignored by other modes."
        ),
    )
    parser.add_argument(
        "--iffc-strip-empty-assistant",
        action="store_true",
        help=(
            "Diagnostic-only IFFC history bookkeeping: omit trailing "
            "assistant [] stop turns from chat_history before the next "
            "BFCL user turn. Default off (Run 5–13 IFFC unchanged). "
            "Not a training method; notes record "
            "iffc_strip_empty_assistant=1. Figures go under "
            "iffc-strip-empty-assistant/ so they do not overwrite "
            "standard IFFC charts."
        ),
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        metavar="CATEGORY",
        default=None,
        help=(
            "Subset of multi-turn categories to run "
            f"(choices: {', '.join(config.MULTI_TURN_CATEGORIES)}). "
            "Example: --categories multi_turn_base"
        ),
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Freeform annotation for this run",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue the latest unfinished run for this model with the same "
            "--fc-mode (and --iffc-strip-empty-assistant if set). Multi-turn "
            "generation skips entry IDs already in "
            "responses. Kill the process and rerun with the same flags plus "
            "--resume."
        ),
    )
    resume_group.add_argument(
        "--resume-run-id",
        type=int,
        metavar="ID",
        default=None,
        help="Continue a specific runs.id (must match --model, --fc-mode, and the strip-empty flag).",
    )
    args = parser.parse_args()
    if args.skip_multi_turn and args.only_multi_turn:
        parser.error("--skip-multi-turn and --only-multi-turn are mutually exclusive")
    if args.categories and args.skip_multi_turn:
        parser.error("--categories applies to multi-turn; cannot combine with --skip-multi-turn")
    try:
        selected_mt = config.resolve_multi_turn_categories(args.categories)
    except ValueError as e:
        parser.error(str(e))
    config.select_model(args.model)
    config.set_fc_mode(args.fc_mode)
    if args.bon_n is not None:
        try:
            config.set_bon_n(args.bon_n)
        except ValueError as e:
            parser.error(str(e))
    config.set_iffc_strip_empty_assistant(args.iffc_strip_empty_assistant)
    config.apply_iffc_strip_empty_assistant_result_paths()

    do_single = not args.only_multi_turn
    do_multi = not args.skip_multi_turn
    n_steps = (2 if do_single else 0) + (2 if do_multi else 0) + 3  # server + metrics + viz

    notes = _notes_with_run_meta(args)

    try:
        run_id, resumed = resolve_pipeline_run(
            model_key=args.model,
            fc_mode=args.fc_mode,
            notes=notes,
            resume=args.resume,
            resume_run_id=args.resume_run_id,
            iffc_strip_empty_assistant=args.iffc_strip_empty_assistant,
        )
    except ValueError as e:
        parser.error(str(e))
    config.RUN_ID = run_id

    if resumed and args.fc_mode == config.FC_IFFC_STATE_BON:
        recorded_n = bon_n_from_notes(db.get_run(run_id)["notes"])
        if recorded_n is not None:
            if args.bon_n is not None and args.bon_n != recorded_n:
                parser.error(
                    f"Run {run_id} has bon_n={recorded_n}, not {args.bon_n}"
                )
            config.set_bon_n(recorded_n)

    print("=" * 60)
    print("BFCL Benchmark Pipeline")
    print(f"Model : {config.MODEL_DISPLAY}")
    print(f"Server: {config.SERVER_URL}")
    print(f"{'Resuming run ID' if resumed else 'Run ID'}: {run_id}")
    print(
        f"FC mode: {args.fc_mode}"
        + (
            " (IFFC = arXiv:2608.22472 Appendix B JSON-in-text)"
            if args.fc_mode == config.FC_IFFC
            else " (CD = TagDispatch-style two-pass; not native tools GBNF)"
            if args.fc_mode == config.FC_CD
            else " (GST = Guided-Structured Templates on NFC tools; arXiv:2509.18076)"
            if args.fc_mode == config.FC_GST
            else " (IFFC+STATE = IFFC JSON-in-text + harness-written STATE snapshot)"
            if args.fc_mode == config.FC_IFFC_STATE
            else (
                f" (IFFC+STATE+BoN = MV@N={config.BON_N} on IFFC+STATE; "
                f"temp={config.BON_TEMPERATURE}; ~N× latency)"
            )
            if args.fc_mode == config.FC_IFFC_STATE_BON
            else " (xLAM = JSON-in-text {\"tool_calls\":[...]} ; no native tools)"
            if args.fc_mode == config.FC_XLAM
            else " (iffc-stop-repeat = IFFC + harness force-stop on identical re-calls)"
            if args.fc_mode == config.FC_IFFC_STOP_REPEAT
            else " (iffc-decision = IFFC + Decision sidecar Choice/noop before Gemma args)"
            if args.fc_mode == config.FC_IFFC_DECISION
            else " (iffc-state-ooda = IFFC + STATE + OODA card + stop-repeat MIN=6)"
            if args.fc_mode == config.FC_IFFC_STATE_OODA
            else " (NFC = native tools)"
        )
    )
    if args.iffc_strip_empty_assistant:
        print(
            "IFFC strip empty assistant: ON "
            "(DIAGNOSTIC history bookkeeping; not a training method)"
        )
    if not do_single:
        print("Single-turn: SKIPPED")
    if not do_multi:
        print("Multi-turn: SKIPPED")
    elif args.categories:
        print(f"Multi-turn categories: {', '.join(selected_mt)}")
    print("=" * 60)

    step = 0

    # 1. Verify server
    step += 1
    print(f"\n[{step}/{n_steps}] Checking server...", end=" ")
    if not check_server():
        print("FAILED")
        print(f"Server not reachable at {config.SERVER_URL}")
        print("Start llama-server first (see CLAUDE.md for the command).")
        sys.exit(1)
    print("OK")

    t0 = time.time()

    # Generate single-turn responses
    if do_single:
        step += 1
        print(f"\n[{step}/{n_steps}] Generating single-turn responses")
        generate_responses.run()

    # Generate multi-turn responses
    if do_multi:
        step += 1
        print(f"\n[{step}/{n_steps}] Generating multi-turn responses")
        generate_multi_turn.run(categories=selected_mt, fc_mode=args.fc_mode)

    # Evaluate single-turn responses
    if do_single:
        step += 1
        print(f"\n[{step}/{n_steps}] Evaluating single-turn responses")
        evaluate_responses.run()

    # Evaluate multi-turn responses (scores saved directly to DB by each evaluator)
    if do_multi:
        step += 1
        print(f"\n[{step}/{n_steps}] Evaluating multi-turn responses")
        evaluate_multi_turn.run(categories=selected_mt)

    # Collect metrics
    step += 1
    print(f"\n[{step}/{n_steps}] Collecting metrics")
    collect_metrics.run(
        include_single=do_single,
        include_multi=do_multi,
        multi_categories=selected_mt if do_multi else None,
    )

    # Visualize
    step += 1
    print(f"\n[{step}/{n_steps}] Generating charts")
    visualize.run(include_single=do_single, include_multi=do_multi)

    # Finish the run
    db.finish_run(run_id)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} minutes.")
    print(f"Results in DB (run_id={run_id})")
    print(f"Figures in: {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
