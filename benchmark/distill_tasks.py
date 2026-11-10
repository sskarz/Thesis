"""Fresh multi-turn tasks over the real BFCL backends, authored by the teacher.

Pipeline (handoff section 5, step 3): the teacher writes user turns plus a gold
call sequence and a small ``initial_config``; this module validates them
offline and deterministically:

1. structure (known classes, 2-5 turns, 1+ calls per turn);
2. the gold calls run turn by turn in the real BFCL backend with no error;
3. the stock ``multi_turn_checker`` accepts the gold against itself (round trip);
4. ``distill_exclusions.check_task`` finds no overlap with any BFCL task, and no
   overlap with tasks already accepted in this batch.

Only tasks that pass all four are written. The teacher is shown each backend's
tool docs and its documented state schema (``_load_scenario`` docstring), never
a BFCL task.

    python -m benchmark.distill_tasks author --n 25 --out sft/data/distill/pilot/tasks.jsonl
"""

from __future__ import annotations

import argparse
import ast
import collections
import inspect
import importlib
import json
import random
import re
import sys
import time
import uuid
from pathlib import Path

from bfcl_eval.constants.executable_backend_config import (
    CLASS_FILE_PATH_MAPPING,
    STATELESS_CLASSES,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
    multi_turn_checker,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
)

from . import config, distill_exclusions as X, teacher_api
from .generate_multi_turn import _load_func_docs
from .generate_responses import load_jsonl

CLASSES = (
    "GorillaFileSystem", "MathAPI", "MessageAPI", "TwitterAPI",
    "TicketAPI", "TradingBot", "TravelAPI", "VehicleControlAPI",
)
MIN_TURNS, MAX_TURNS = 2, 5
STATE_DOC_CHARS = 4500
DEFAULT_STATE_CHARS = 2200
MAX_REPAIRS = 4

THEMES = [
    "a small bakery", "a hiking club", "a freelance designer", "a university lab",
    "a family road trip", "a neighbourhood book swap", "a startup's first hires",
    "a community garden", "a weekend photography workshop", "a food-truck owner",
    "an indie game studio", "a retired teacher organising memoirs",
    "a local theatre troupe", "a bike-repair shop", "a podcast team",
    "a wedding planner", "a research assistant", "a remote support desk",
]

SYSTEM = (
    "You write test tasks for a tool-calling assistant. You output ONLY one JSON "
    "object, no prose, no code fences."
)

PROMPT = """Write one new multi-turn task for an assistant that can call the tools below.

Backends involved: {classes}
Theme for flavour (use loosely, invent specific names and values): {theme}
Number of user turns: {n_turns}

TOOLS (name(required params): description)
{tools}

STATE SCHEMA for each stateful backend (how initial_config is written)
{state_docs}

BACKEND DATA (methods with built-in lookup tables: only the values in these tables work)
{data_excerpts}

Style example (different backend, only to show tone and gold size; never reuse its names):
user: "Go into the home folder and show me what the zebra notes say." -> gold_calls: ["cd(folder='home')", "cat(file_name='zebra_notes.txt')"]
user: "Now move those notes into the plans folder." -> gold_calls: ["mv(source='zebra_notes.txt', destination='plans')"]

Return exactly this JSON shape:
{{"initial_config": {{"<Backend>": <state dict per the schema above>, ...}},
 "turns": [{{"user": "<what the user says>", "gold_calls": ["tool_name(arg='value', ...)", ...]}}, ...]}}

Rules:
- initial_config has one key per backend listed (use {{}} for MathAPI and any backend that needs no state). Keep states small but real: only items the task touches plus a few distractors.
- user turns are written the way a real person talks to an assistant: short, natural, sometimes referring to earlier results ("that file", "the same account"). Never mention tool names or parameter names.
- turns build on each other: later turns depend on state changed by earlier ones.
- every value a call needs must be stated by the user, present in the initial state, or learnable by an earlier call. When the user does not give a value (an id, a path, a ticker, a code), the gold calls must look it up first with a read-only tool.
- at least one turn must need a look-up before the action, and in that turn the user must NOT state the looked-up value (an id, a path, a code): the assistant has to find it.
- be explicit about operations in plain words, the way a user who knows what they want does ("go into jobs, read job_115, count its lines"). The gold is exactly the minimal natural sequence any competent assistant would run for that request: no optional parameters, no extra checks, no look-up whose result the next call does not use, no arithmetic or unit conversion the user did not ask for.
- whenever a call writes free text (tweet, comment, message, ticket resolution or title, file content), the user must dictate the exact text in quotes in that turn or an earlier one, and the gold must use exactly that text, character for character. The checker compares written text exactly, so text the assistant would have to compose can never match.
- prefer making the user STATE a specific value (name, number, code) over adding a look-up for it, except in the one look-up turn above.
- gold_calls: Python call syntax with keyword args, using ONLY single quotes inside each call string (door=['driver'], never double quotes, so the JSON stays valid), in execution order, using only the tools above, each valid and successful from the state at that point. 1 to 6 calls per turn, no redundant calls, no placeholders.
- all state changes the user asks for must happen in gold_calls; do not ask for things the tools cannot do.
- if one backend is a file system: one directory level per cd, no paths in file names.
- many backends need a logged-in or authenticated user before most actions: either put that state in initial_config or include the login/auth call in gold_calls first.
- tools with built-in tables (routes, distances, prices, symbols, users) only accept values that really exist in the schema or the BACKEND DATA above; invent nothing outside them.
{fs_hint}{focus_rules}- do not copy well-known benchmark prompts.
"""


# --------------------------------------------------------------------- prompts
def _tool_lines(classes: list[str]) -> str:
    out = []
    for fn in _load_func_docs(list(classes)):
        f = fn["function"] if "function" in fn else fn
        props = f.get("parameters", {}).get("properties", {})
        req = ", ".join(
            f"{r}: {props.get(r, {}).get('type', '?')}"
            + (f"[{props[r]['items'].get('type', '?')}]" if props.get(r, {}).get("items") else "")
            for r in f.get("parameters", {}).get("required", []))
        desc = re.sub(r"\s+", " ", f.get("description", "")).split("Tool description:")[-1].strip()
        out.append(f"- {f['name']}({req}): {desc[:220]}")
    return "\n".join(out)


def _state_doc(cls_name: str) -> str:
    if cls_name in STATELESS_CLASSES:
        return f"{cls_name}: stateless, use {{}}."
    mod = importlib.import_module(CLASS_FILE_PATH_MAPPING[cls_name])
    cls = getattr(mod, cls_name)
    # full source (docstring + the keys the loader actually reads), not just the docstring
    doc = inspect.getsource(cls._load_scenario)
    default = getattr(mod, "DEFAULT_STATE", None)
    shape = ""
    if default is not None and cls_name != "GorillaFileSystem":
        shape = "\nExample of a valid state (the backend's own default; same key names and types):\n" + repr(default)[:DEFAULT_STATE_CHARS]
    return f"{cls_name}:\n{doc[:STATE_DOC_CHARS]}{shape}"


def _data_excerpts(cls_name: str, per_class_chars: int = 9000) -> str:
    """Source of public methods that embed lookup tables (many literals)."""
    mod = importlib.import_module(CLASS_FILE_PATH_MAPPING[cls_name])
    src = inspect.getsource(mod)
    tree = ast.parse(src)
    out, used = [], 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and not fn.name.startswith("_"):
                    n_const = sum(isinstance(x, ast.Constant) and isinstance(x.value, (str, int, float))
                                  for x in ast.walk(fn))
                    if n_const >= 40 or re.search(r"auth|login|token", fn.name, re.I):
                        seg = ast.get_source_segment(src, fn) or ""
                        seg = seg[:5000]
                        if used + len(seg) > per_class_chars:
                            continue
                        used += len(seg)
                        out.append(seg)
    return "\n\n".join(out) if out else "(none)"


FS_HINT = (
    "- file-system tasks must exercise navigation: folders two levels deep, going up with "
    "cd(folder='..') to reach a sibling folder or the top, files that must be found with "
    "ls/find before acting, copy vs move (the user says which), and at least one action in a "
    "folder other than the starting one.\n"
)


# Round 4 Stage B curriculum focus (default None = prompt byte-identical to rounds 1-3 and to bank F).
# batch 2 (G2b): wider theme list, none of them in THEMES (flavour only; no BFCL text)
THEMES_B2 = [
    "a pet-grooming salon", "a community radio station", "a coworking space manager", "a school science fair", "a yoga studio",
    "a farmers' market stall", "a small law office", "a charity fun run", "a dentist's front desk", "a language-exchange meetup",
    "a drone-photography hobbyist", "a neighbourhood tool library", "a boutique hotel", "a vintage record shop", "a climbing gym",
    "a city-council intern", "a mobile coffee cart", "a high-school debate team", "a ceramics studio", "a night-shift nurse planning errands",
    "a travelling musician", "a moving-day coordinator", "a museum volunteer", "a software-bug triage rota", "an urban beekeeper",
    "a sailing club", "a tutoring centre", "a bookkeeping side business", "a film-festival volunteer", "a houseplant swap group",
    "a car-sharing co-op", "a stand-up comedy open mic", "a robotics club", "a catering crew", "a lost-and-found desk",
    "a toy-library", "a community choir", "a repair-cafe", "a trail-running group", "a freelance translator",
]
FOCUS_RULES = {
    "b2": (
        "- whenever the starting state has an id counter (tweet_counter, order_counter, ticket_counter or similar), set it strictly greater than every id of that kind that already "
        "exists in the state, so that an object created by the gold calls gets a fresh id and never overwrites an existing one.\n"
    ),
    "recovery": (
        "- (focus) make at least one turn depend on an exact spelling the backend accepts (a unit abbreviation, "
        "a file name with its extension, a ticker symbol or code as listed in the backend data) that the user "
        "does not spell out exactly, so that a plausible wrong spelling would be rejected by the backend; and "
        "make at least one turn need a read-only look-up before acting. The gold stays the minimal successful "
        "sequence.\n"
    ),
}


def build_prompt(classes: list[str], theme: str, n_turns: int, focus: str | None = None) -> str:
    return PROMPT.format(
        focus_rules=FOCUS_RULES[focus] if focus else "",
        fs_hint=FS_HINT if "GorillaFileSystem" in classes else "",
        classes=", ".join(classes),
        theme=theme,
        n_turns=n_turns,
        tools=_tool_lines(classes),
        state_docs="\n\n".join(_state_doc(c) for c in classes),
        data_excerpts="\n\n".join(
            f"[{c}]\n{_data_excerpts(c)}" for c in classes),
    )


def class_combos() -> list[tuple[str, ...]]:
    """Structural prior only: which backends co-occur in BFCL multi_turn_base."""
    f = config.BFCL_DATA_DIR / config.MULTI_TURN_CATEGORIES["multi_turn_base"]["file"]
    return [tuple(sorted(e["involved_classes"])) for e in load_jsonl(f)]


# ---------------------------------------------------------------------- parsing
def _normalise(obj: dict) -> dict:
    """JSON round trip: int keys become strings, tuples lists (what we store)."""
    return json.loads(json.dumps(obj))


def parse_authored(text: str) -> dict | None:
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "turns" in obj:
            return obj
    # Lenient fallback: Python literals (None/True, int keys, single quotes).
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        try:
            obj = ast.literal_eval(text[i : j + 1])
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):  # TypeError: a set literal holding a dict (unhashable) crashed g2 process b, 2026-11-09
            return None
        if isinstance(obj, dict) and "turns" in obj:
            try:
                return _normalise(obj)
            except (TypeError, ValueError):
                return None
    return None


def parse_error(text: str) -> str:
    """Human-readable reason a reply was not usable (for the repair prompt)."""
    i = text.find("{")
    if i < 0:
        return "no JSON object found"
    try:
        json.JSONDecoder().raw_decode(text[i:])
        return "JSON parsed but has no 'turns' list at the top level"
    except json.JSONDecodeError as e:
        ctx = text[max(0, i + e.pos - 40) : i + e.pos + 20].replace("\n", " ")
        return f"{e.msg} near ...{ctx}..."


def semantic_issue(call: str, result) -> str | None:
    """Independent check that a gold call did what the words say it did.

    The self round trip cannot catch a gold that runs cleanly but does the
    wrong thing. Known case: file-system ``cp``/``mv`` to a name that is not a
    child folder of the current directory silently creates a FILE with that
    name. A destination without an extension is meant as a folder, so the
    backend must report ``... to '<dest>/<name>'``.
    """
    m = re.match(r"\s*(cp|mv)\(", call)
    if not m:
        return None
    try:
        node = ast.parse(call, mode="eval").body
        kw = {k.arg: k.value.value for k in node.keywords if isinstance(k.value, ast.Constant)}
    except (SyntaxError, AttributeError):
        return None
    dest = kw.get("destination")
    if isinstance(dest, str) and (dest.strip() in ("..", ".", "/") or "/" in dest or ".." in dest):
        return (f"{m.group(1)} destination '{dest}' is a path or parent-directory reference; the "
                "backend does not support it (it creates a file with that odd name)")
    if isinstance(dest, str) and "." not in dest.rsplit("/", 1)[-1]:
        if f"to '{dest}/" not in str(result):
            return f"{m.group(1)} destination '{dest}' is not a folder in the current directory (it would create a file named '{dest}')"
    return None


def audit_task(task: dict) -> list[str]:
    """Re-run the gold under a fresh id and list semantic issues (empty = ok)."""
    classes = list(task["involved_classes"])
    vid = f"{task['id']}_audit_{time.time_ns()}"
    issues: list[str] = []
    for ti, calls in enumerate(task["ground_truth"]):
        res, _ = execute_multi_turn_func_call(
            func_call_list=calls, initial_config=task["initial_config"],
            involved_classes=classes, model_name="distill_audit", test_entry_id=vid,
            long_context=False, is_evaL_run=False)
        for c, r in zip(calls, res):
            msg = semantic_issue(c, r)
            if msg:
                issues.append(f"turn {ti}: {msg}")
    return issues


def _is_error(res) -> bool:
    if isinstance(res, str):
        if res.startswith("Error during execution"):
            return True
        try:
            res = json.loads(res)
        except (json.JSONDecodeError, TypeError):
            return False
    return isinstance(res, dict) and "error" in res


# ------------------------------------------------------------------- validation
def validate(
    authored: dict,
    classes: list[str],
    task_id: str,
    accepted_turns: list[list[str]] | None = None,
    attempt: int = 0,
) -> tuple[dict | None, str]:
    """Return ``(task, "")`` if the authored task passes, else ``(None, reason)``."""
    turns = authored.get("turns")
    ic = authored.get("initial_config")
    if not isinstance(turns, list) or not (MIN_TURNS <= len(turns) <= MAX_TURNS):
        return None, "bad_turn_count"
    if not isinstance(ic, dict) or not set(ic) <= set(classes):
        return None, "bad_initial_config"
    users, gold = [], []
    for t in turns:
        u, g = (t or {}).get("user"), (t or {}).get("gold_calls")
        if not isinstance(u, str) or not u.strip() or not isinstance(g, list) \
                or not g or not all(isinstance(c, str) and "(" in c for c in g) \
                or len(g) > 6:
            return None, "bad_turn"
        users.append(u.strip()); gold.append(g)
    ic = {c: ic.get(c, {}) for c in classes}

    # BFCL caches backend instances by id in module globals, so every validation
    # attempt runs under its own execution id; the task keeps ``task_id``.
    vid = f"{task_id}_v{attempt}_{uuid.uuid4().hex[:6]}"
    entry = {
        "id": vid,
        "question": [[{"role": "user", "content": u}] for u in users],
        "initial_config": ic,
        "involved_classes": list(classes),
    }
    # 2. gold runs clean, turn by turn, in the real backend
    for ti, calls in enumerate(gold):
        try:
            res, _ = execute_multi_turn_func_call(
                func_call_list=calls, initial_config=ic, involved_classes=list(classes),
                model_name="distill_gold", test_entry_id=vid,
                long_context=False, is_evaL_run=False)
        except Exception as e:  # noqa: BLE001
            return None, f"gold_exception_t{ti}:{str(e)[:160]}"
        for c, r in zip(calls, res):
            if _is_error(r):
                return None, f"gold_error_t{ti}: call {c[:100]} returned {str(r)[:200]}"
            sem = semantic_issue(c, r)
            if sem:
                return None, f"gold_error_t{ti}: {sem}. Use a destination that is an existing folder inside the current directory (cd to its parent first), or a file name with an extension."
    # 3. round trip through the stock checker
    try:
        rt = multi_turn_checker(
            multi_turn_model_result_list_decoded=[[g] for g in gold],
            multi_turn_ground_truth_list=gold, test_entry=entry,
            test_category="multi_turn_base", model_name="distill_rt")
    except Exception as e:  # noqa: BLE001
        return None, f"roundtrip_exception:{str(e)[:80]}"
    if not rt.get("valid"):
        return None, "roundtrip_invalid"
    # 4. leak checks
    leak = X.check_task(users, gold, task_id=task_id)
    if leak["reject"]:
        return None, "leak:" + ";".join(leak["reasons"])[:100]
    # 3b. grounding: every specific value in the gold must be knowable
    probe = {"id": f"{task_id}_g{attempt}", "involved_classes": list(classes),
             "initial_config": ic, "ground_truth": gold,
             "question": entry["question"]}
    bad = ungrounded_literals(probe)
    if bad:
        shown = "; ".join(f"turn {t}: {v!r} in {c}" for t, c, v in bad[:4])
        return None, ("ungrounded: these gold values are not stated by the user in "
                      "that turn or earlier, and no earlier tool result shows them: "
                      f"{shown}. Either make the user state the value, or add a "
                      "read-only look-up call earlier whose result reveals it.")
    grams = set().union(*(X.ngrams(u) for u in users)) if users else set()
    for prev in accepted_turns or []:
        pg = set().union(*(X.ngrams(u) for u in prev))
        if grams & pg:
            return None, "dup_in_batch"
    task = {**entry, "id": task_id, "ground_truth": gold, "leak": leak}
    return task, ""


# --------------------------------------------------------------------- authoring
def author(
    n_slots: int,
    out: Path,
    *,
    client: teacher_api.TeacherClient,
    seed: int = 0,
    workers: int = 4,
    deadline: float | None = None,
    prefix: str = "distill_pilot",
    existing: list[Path] | None = None,
    focus: str | None = None,
    themes: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    rng = random.Random(seed)
    combos = class_combos()
    # Anchor rotation: every slot is anchored on one backend; the file system
    # appears twice per cycle (where the base model fails most), so no backend
    # is starved the way a plain draw from the BFCL combo prior starves them.
    anchors = ["GorillaFileSystem", "TradingBot", "TravelAPI", "VehicleControlAPI",
               "GorillaFileSystem", "TicketAPI", "MessageAPI", "TwitterAPI", "MathAPI"]
    slots = []
    for i in range(n_slots):
        if weights:       # batch 2: anchor backend and combination drawn with weights (structural co-occurrence from the BFCL backend sets only)
            anchor = rng.choices(sorted(weights), weights=[weights[b] for b in sorted(weights)])[0]
            pool = [c for c in combos if anchor in c] or combos
            combo = list(rng.choices(pool, weights=[sum(weights.get(b, 1.0) for b in c) / len(c) for c in pool])[0])
        else:
            anchor = anchors[(i + seed) % len(anchors)]
            pool = [c for c in combos if anchor in c] or combos
            combo = list(rng.choice(pool))
        slots.append((i, combo, rng.choice(themes or THEMES), rng.randint(MIN_TURNS, MAX_TURNS)))
    accepted: list[dict] = []
    earlier: list[list[str]] = []
    for pth in existing or []:
        for line in Path(pth).read_text().splitlines():
            if line.strip():
                t = json.loads(line)
                earlier.append([m["content"] for q in t["question"] for m in q])
    log: list[dict] = []
    lock = __import__("threading").Lock()

    def deadline_hit() -> bool:
        return bool(deadline and time.time() > deadline)

    def work(slot):
        # one bad reply must cost one slot, not a whole chunk (G2 process b lost 222 slots to one uncaught TypeError, 2026-11-09)
        try:
            return _work(slot)
        except Exception as e:  # noqa: BLE001
            return {"slot": slot[0], "ok": False, "reason": f"internal:{type(e).__name__}", "classes": slot[1]}

    def _work(slot):
        i, combo, theme, nt = slot
        if deadline and time.time() > deadline:
            return {"slot": i, "ok": False, "reason": "deadline"}
        tid = f"{prefix}_{i:04d}"
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": build_prompt(combo, theme, nt, focus)}]
        why, task = "unparseable", None
        for attempt in range(MAX_REPAIRS + 1):
            try:
                with client.episode(f"author_{tid}_a{attempt}", kind="author", attempt=attempt):
                    data = client.post_chat({"messages": messages,
                                             "temperature": 0.8, "max_tokens": 3500})
            except Exception as e:  # noqa: BLE001
                return {"slot": i, "ok": False, "reason": f"api:{type(e).__name__}"}
            text = (data["choices"][0]["message"].get("content") or "")
            obj = parse_authored(text)
            if obj is None:
                why = ("unparseable: " + parse_error(text) + ". Use strict JSON: double-quoted "
                       "keys and strings, string keys for ids (\"12345\"), null/true/false, "
                       "balanced braces, and one top-level object with initial_config and turns.")
            else:
                with lock:
                    prev = earlier + [[m["content"] for q in t["question"] for m in q] for t in accepted]
                task, why = validate(obj, combo, tid, prev, attempt=attempt)
                if task is not None:
                    break
            if why.startswith(("leak", "dup_in_batch")) or deadline_hit():
                break  # not fixable by editing details; drop the slot
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content":
                    f"Validation failed: {why}\nFix this (use only values the backend "
                    "accepts, include any needed login, follow the state schema exactly) "
                    "and return the complete corrected JSON object only."}]
        if task is None:
            return {"slot": i, "ok": False, "reason": why, "classes": combo}
        task["meta"] = {"theme": theme, "slot": i, "classes": combo, "repairs": attempt}
        with lock:
            # re-check duplicates against everything accepted while we validated
            g = set().union(*(X.ngrams(m["content"]) for q in task["question"] for m in q))
            for other in earlier + [[m["content"] for q in t["question"] for m in q] for t in accepted]:
                if g & set().union(*(X.ngrams(u) for u in other)):
                    return {"slot": i, "ok": False, "reason": "dup_in_batch", "classes": combo}
            accepted.append(task)
        return {"slot": i, "ok": True, "classes": combo}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        log = list(ex.map(work, slots))
    accepted.sort(key=lambda t: t["id"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for t in accepted:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    reasons = collections.Counter(
        r["reason"].split(":")[0] for r in log if not r["ok"])
    repairs = collections.Counter(t["meta"]["repairs"] for t in accepted)
    return {"slots": n_slots, "valid": len(accepted), "valid_by_repair_rounds": dict(repairs), "reject_reasons": dict(reasons),
            "per_class": dict(collections.Counter(c for t in accepted for c in t["involved_classes"]))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("author")
    a.add_argument("--n", type=int, default=25)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--log-dir", type=Path, required=True)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--prefix", default="distill_pilot")
    a.add_argument("--existing", type=Path, nargs="*", default=[])
    a.add_argument("--max-cost", type=float, required=True)
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--focus", choices=sorted(FOCUS_RULES), default=None)
    args = ap.parse_args(argv)
    cfg = teacher_api.TeacherConfig.openrouter(
        log_dir=args.log_dir, max_cost_usd=args.max_cost,
        max_concurrency=args.workers)
    client = teacher_api.TeacherClient(cfg)
    print(json.dumps(author(args.n, args.out, client=client, seed=args.seed, prefix=args.prefix, existing=args.existing,
                       workers=args.workers, focus=args.focus), indent=1))
    print("spent_usd", round(client.stats["cost_usd"], 4), "calls", client.stats["calls"])
    return 0



# ------------------------------------------------------------------ grounding
_EXT = {"txt", "pdf", "md", "csv", "log", "json", "doc", "docx", "png", "jpg"}


def _toks(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _contains(corpus: list[str], seq: list[str]) -> bool:
    n = len(seq)
    return n == 0 or any(corpus[i : i + n] == seq for i in range(len(corpus) - n + 1))


def ungrounded_literals(task: dict) -> list[tuple[int, str, str]]:
    """Gold-call argument values the user/assistant could not have known.

    A string or number argument is grounded if its tokens appear in the user
    turns so far, in any earlier tool result (this turn included), or in the
    tool docs (enum values). Returns ``(turn, call, value)`` for the rest.
    Offline; re-executes the gold under a fresh id.
    """
    classes = task["involved_classes"]
    doc_corpus = _toks(json.dumps(_load_func_docs(list(classes))))
    users = [m["content"] for q in task["question"] for m in q]
    seen_results: list[str] = []
    bad: list[tuple[int, str, str]] = []
    vid = f"{task['id']}_ground_{time.time_ns()}"
    for ti, calls in enumerate(task["ground_truth"]):
        user_corpus = _toks(" ".join(users[: ti + 1]))
        for c in calls:
            res, _ = execute_multi_turn_func_call(
                func_call_list=[c], initial_config=task["initial_config"],
                involved_classes=list(classes), model_name="distill_ground",
                test_entry_id=vid, long_context=False, is_evaL_run=False)
            try:
                node = ast.parse(c, mode="eval").body
            except SyntaxError:
                continue
            corpus = user_corpus + _toks(" ".join(seen_results)) + doc_corpus
            for kw in getattr(node, "keywords", []):
                for sub in ast.walk(kw.value):
                    if isinstance(sub, ast.Constant) and not isinstance(sub.value, bool):
                        v = sub.value
                        if isinstance(v, (int, float)) and abs(v) < 2:
                            continue
                        if not isinstance(v, (str, int, float)) or (isinstance(v, str) and len(v) < 3):
                            continue
                        if isinstance(v, str) and len(v.split()) >= 4:
                            # free text: must be dictated verbatim by the user
                            if " ".join(v.lower().split()) not in " ".join(
                                    " ".join(users[: ti + 1]).lower().split()):
                                bad.append((ti, c[:90], "free text not dictated: " + v[:40]))
                            continue
                        seq = [t for t in _toks(v) if t not in _EXT]
                        if not _contains(corpus, seq):
                            bad.append((ti, c[:90], str(v)[:40]))
            seen_results.append(str(res[0]) if res else "")
    return bad

if __name__ == "__main__":
    sys.exit(main())
