#!/usr/bin/env python3
"""Assemble thesis/main.tex from paper/main.tex.

Line ranges still copy the paper. Thesis-only overrides then do three things:
replacement blocks, cuts, and moves. Running this script rewrites thesis/main.tex.
Do not edit thesis/main.tex by hand.

Numbers in the replacement blocks are copied from paper/main.tex. The script checks
the abstract against that file before it writes.
"""
import re
import pathlib

here = pathlib.Path(__file__).resolve().parent
paper_text = (here.parent / "paper" / "main.tex").read_text()
src = paper_text.split("\n")
L = lambda a, b: "\n".join(src[a - 1 : b])  # 1-based inclusive

TITLES = {
    "Benchmark.": "Benchmark",
    "Output channel.": "Output Channel",
    "Constrained decoding.": "Constrained Decoding",
    "Harness-side control and test-time compute.": "Harness-Side Control and Test-Time Compute",
    "Training for tool use.": "Training for Tool Use",
    "Model and serving.": "Model and Serving",
    "Harness interventions.": "Harness Interventions",
    "Training.": "Training",
    "Error labels.": "Error Labels",
    "Statistics.": "Statistics",
    "One call is easy; a chain is not.": "One Call Is Easy; a Chain Is Not",
    "Most failures are about the world.": "Most Failures Are About the World",
    "Forcing valid syntax hurts.": "Forcing Valid Syntax Hurts",
    r"JSON text over the native API (+3.5\pp).": "JSON Text over the Native API",
    r"State summary (+3.5\pp).": "State Summary",
    r"Stop on repeats and the status card (minimal recipe, 35.5\%).": "Stop on Repeats and the Status Card",
    r"Full stack (38.5\% on the first run).": "Full Stack",
    "Shorter tool list, same tools.": "Shorter Tool List, Same Tools",
    r"Cost: majority vote (36.5\%).": "Majority Vote",
    "How sure are we?": "How Sure Are We?",
    "Public data taught the model to stop.": "Public Data Taught the Model to Stop",
    "A rebuilt mix collapsed.": "A Rebuilt Mix Collapsed",
    "A candidate cause that targeted retrains did not confirm.": "A Candidate Cause That Targeted Retrains Did Not Confirm",
    "State, not syntax.": "State, Not Syntax",
    "The guided-template tie.": "The Guided-Template Tie",
    "What we recommend.": "What We Recommend",
    "Limitations.": "Limitations",
    "Future work.": "Future Work",
    "Conclusion.": None,
    "JSON text (IFFC).": "JSON Text (IFFC)",
    "State summary.": "State Summary",
    "Stop on repeats.": "Stop on Repeats",
    r"Status card (tag \texttt{[OODA]}).": r"Status Card (Tag \texttt{[OODA]})",
    "Full stack.": "Full Stack",
    "Vote of five.": "Vote of Five",
    "Tool-name automaton.": "Tool-Name Automaton",
    "Version history.": "Version History",
    "Self-distillation counts.": "Self-Distillation Counts",
    "Stop targets in the rebuilt mix.": "Stop Targets in the Rebuilt Mix",
    "10-task tripwire.": "10-Task Tripwire",
    "Reproducibility.": "Reproducibility",
    "Glossary.": "Glossary",
}

# Appendix A headings must not repeat the chapter headings.
APP_A_TITLES = {
    "JSON text (IFFC).": "Calls Written as JSON Text",
    "State summary.": "The State Summary Block",
    "Stop on repeats.": "The Stop Rule",
    r"Status card (tag \texttt{[OODA]}).": "The Status Card",
    "Full stack.": "Combining the Full Stack",
    "Vote of five.": "Sampling Five Calls",
    "Constrained decoding.": "Two-Pass Constrained Decoding",
    "Tool-name automaton.": "The Tool-Name Automaton",
    "Training.": "Fine-Tuning Setup",
}

CHAPTER_LABELS = {
    "sec:intro",
    "sec:related",
    "sec:setup",
    "sec:fail",
    "sec:helps",
    "sec:train",
    "sec:discussion",
    "sec:conclusion",
}

SHORT_LIST = {
    "fig:vignette": "The multi-turn tool-calling loop",
    "fig:fail": "Why multi-turn tasks fail",
    "fig:ladder": "Multi-turn accuracy by condition",
    "fig:gate": "Scores on the 20-task gate",
    "tab:trim": "Shorter tool list against the original list",
    "tab:mcnemar": "Paired tests on the 200 tasks",
    "tab:failfull": "Primary error for each failed task",
    "tab:mcnemar-x": "Exploratory paired tests",
    "tab:board": "Every scored condition",
    "tab:runs": "Every scored condition",
}

ABSTRACT = (
    "Small language models can run on a laptop and call tools to act for a user, "
    "but they often fail when a task needs several calls in a row. This thesis asks why, "
    "and what helps, using Gemma-4-E2B-it, a model with about two billion effective parameters, "
    "on 200 multi-turn tasks from the Berkeley Function Calling Leaderboard. Tasks are graded "
    "by the final state of a simulated backend, for example whether a car's engine is on.\n"
    "\n"
    "The main finding is that the model fails at deciding what to do next, not at writing calls. "
    "It handles single calls well (91.9\\%) but completes only 27.0\\% of multi-turn tasks, and "
    "forcing every call to be valid syntax made results worse. The remedy is a harness, the "
    "program around the model, with the weights untouched. It records the current state for the "
    "model, stops repeated calls, and adds a short card advising whether to call, ask, finish or "
    "refuse. This raised accuracy from 30.5\\% to 38.5\\%, a suggestive gain, and matched a "
    "five-sample vote at about a seventh of the generation time. Writing the tool list more "
    "compactly added about 7 points in two paired runs, the best reaching 44.0\\%. Wrong final "
    "states remain the largest failure type. Fine-tuning the model did not help; the best tuned "
    "model scored 15.0\\%, below the untuned model.\n"
)

# Strings that must already appear in paper/main.tex, with the same meaning.
ABSTRACT_SOURCES = [
    "about two billion effective parameters",
    "200 multi-turn",
    "91.9\\%",
    "27.0\\%",
    "30.5\\%",
    "38.5\\%",
    "44.0\\%",
    "15.0\\%",
    "about 7 points",
    "a seventh",
    "five-sample",
]

CONTRIBUTIONS = r"""Our contributions:

\begin{enumerate}
  \item \textbf{Where the failures are.} The model usually handles a single call, and it usually fails when the task needs several. A needed call never happens, or the backend ends in the wrong state. Forcing each call to be valid syntax makes this worse (Chapter~\ref{sec:fail}).
  \item \textbf{A harness that helps without training.} The weights stay fixed. The program around the model records the current state, stops repeated calls, and shows a short card advising the next move. Together these raise accuracy by a suggestive amount. A more compact tool list helps further, and the harness keeps up with a five-sample vote at a fraction of the generation time (Chapter~\ref{sec:helps}).
  \item \textbf{Fine-tuning did not help.} The best supervised fine-tune scores below the untuned model, and preference tuning does not beat it. A control fine-tune that should change nothing, if the machinery is sound, stays close to the untuned model. The later training set is the leading suspect, and it also differed in training length (Chapter~\ref{sec:train}).
\end{enumerate}"""

CH2_INTRO = (
    "This chapter names the benchmark and the ideas the experiments rely on: "
    "how a call is written, what it means to force that writing to be valid, "
    "what the program around the model can do, and how training a tool-calling model is usually done.\n\n"
)

CH2_CLOSER = (
    "\n\nThe following chapters use only these ideas. They ask where a small model fails, "
    "whether the harness can help with the weights left fixed, and whether fine-tuning can.\n"
)

MODEL_SERVING = r"""\section{Model and Serving}

All results use Gemma-4-E2B-it, quantized with Q4\_K\_M (a 4-bit weight format of llama.cpp), served by \texttt{llama-server} on a laptop. Decoding is greedy (temperature 0) except for majority voting. Greedy runs are not identical from one run to the next (Chapter~\ref{sec:helps}). The machine, the server build, and the flags are in Appendix~\ref{app:runs}. Most runs used four parallel server slots (requests the server processes at the same time) and the native API baseline used eight, a serving difference that could affect its results. This thesis did not test whether the eight-slot and four-slot greedy outputs match."""

HARNESS = r"""\section{Harness Interventions}

All leave the weights untouched. A task is correct only when the BFCL multi-turn checker returns valid. Appendix~\ref{app:method} gives the rule for each intervention except the shorter tool list. By design, the state summary targets lost state, the stop rule targets runaway repetition, and the status card targets the choice between acting and abstaining.
\begin{itemize}\setlength{\itemsep}{1pt}
  \item \textbf{JSON text} instead of the native API. The harness parses the calls.
  \item \textbf{State summary.} After each executed call, the harness writes a short \texttt{[STATE]} snapshot of facts from real results (current directory, known files, last calls) into the conversation.
  \item \textbf{Stop on repeats.} The harness ends the current turn when the model repeats its previous set of calls, or when it re-proposes calls already executed in this turn after at least six tool steps. An \emph{ungated} variant drops the six-step minimum.
  \item \textbf{Status card.} A four-line card (goal, known facts, last call and outcome, and an advisory next move: call, ask, finish, or refuse) is prefixed to the latest user message. The card prints finish as \texttt{final}. The harness fills the card from fixed rules on the conversation and executed results; it never sees reference answers. JSON text, stop on repeats, and the card, together, are the \emph{minimal recipe}.
  \item \textbf{Full stack.} JSON text, the state summary, stop on repeats, and the status card, decoded greedily. On its first run this is the harness we compare with JSON text.
  \item \textbf{Shorter tool list.} On the full stack, every tool stays in the prompt. The harness states each tool group's shared description once, instead of repeating it, and prints each schema without indentation. The model, the four pieces above, and the tasks stay the same. The shorter list is text in the context, so this lever, like the state summary and the status card, is harness-side control delivered through the context.
  \item \textbf{Vote of five.} On top of the state summary, at each step the harness samples five candidate calls at temperature 0.7 and executes the most common one. This is the compute-heavy comparison point.
  \item \textbf{Forced-valid decoding.} \emph{Constrained decoding} restricts output in two passes (tool names, then arguments by schema). A \emph{tool-name automaton} constrains only the name, inspired by~\cite{yu2026ptc}.
\end{itemize}

The state summary and the status card are short notes the harness writes from results it has already executed. A state summary looks like this:
\begin{quote}\singlespacing\footnotesize\ttfamily
[STATE]\\ cwd: /workspace/notes\\ files: todo.txt, log.csv\\ last: cd(folder='notes')->ok
\end{quote}
A status card, prefixed to the latest user message, looks like this:
\begin{quote}\singlespacing\footnotesize\ttfamily
[OODA]\\ goal: lock the doors and start the engine\\ slots: lockStatus=locked\\ last: startEngine()->error: brake pedal must be pressed\\ next: call
\end{quote}"""

STATISTICS = r"""\section{Statistics}

Each reported accuracy comes with a Wilson 95\% interval~\cite{wilson1927}, a range that stays reliable when few tasks are correct. When two conditions are compared on the same 200 tasks, we use an exact two-sided McNemar test~\cite{mcnemar1947}. It counts only the tasks the two conditions decide differently. The $p$-value is the chance of a gap at least that large if the conditions were equal.

Eight comparisons were planned as one family. The family covers the full stack and the minimal recipe against their baselines, the switch from the native API to JSON text, and the best fine-tune. Four of the eight use the full stack, and the family was fixed after that configuration was built. Four single-lever comparisons are exploratory and were left out of the family (Table~\ref{tab:mcnemar-x}): the state summary against JSON text, the vote against the state summary, stop on repeats against JSON text, and the minimal recipe against stop on repeats. We report the raw $p$-value, and we say when Holm's correction~\cite{holm1979} changes the conclusion. That correction keeps the chance of any false alarm among the eight tests at 5\%.

The shorter tool list is not one of the eight tests. It is judged at the 0.05 level set in the plan. It was run as two pairs, the second in the opposite order. To combine them, we count how many of its two runs each list solved for each task, and we apply an exact two-sided sign test to the tasks where the counts differ. Why the comparison uses a fresh rerun, rather than the first full-stack run, is recorded in Appendix~\ref{app:trim}."""

STATE_MISMATCH = (
    "A \\emph{state mismatch} is a task where the calls ran and the backend still ended wrong. "
    "The loop in Figure~\\ref{fig:vignette} is an example. The model locks the doors and then tries to start the engine. "
    "Both calls are valid. The backend says the brake pedal has to be pressed first. "
    "The model never presses it, so the engine stays off and the check fails.\n\n"
)

JSON_CHANNEL = r"""\section{JSON Text over the Native API}

Writing calls as JSON text scores 61/200 against 54/200 on the native API, 3.5 percentage points higher ($p=0.32$, not significant). Part of the gap may come from serving. A seven-step reasoning template on the native API also scored 61/200, tying JSON text: 19 tasks swap each way, and state mismatches rose to 79. The gap may therefore reflect the shape of the prompt as much as which channel carries the call. We do not claim the channel switch alone as a gain. Later steps are built on JSON text because the harness can parse and inspect those calls directly. The harness levers were not tested on the native API, or combined with the template."""

SHORTER = r"""\section{Shorter Tool List, Same Tools}

The full stack still sends every tool. The list is long because tools that share a group also share a description, and that description was repeated on every tool, with each schema indented. We state each group's shared description once and print the schemas without indentation. Nothing else changes: the model, the four pieces above, and the tasks stay the same. Like the state summary and the card, this reaches the model as text in its context.

We compared the shorter list with a fresh run of the original list, then repeated the comparison with the shorter list run first (Table~\ref{tab:trim}). The shorter list scores 84 of 200 (42.0\%) against 71 of 200 (35.5\%) in the first pair, and 88 of 200 (44.0\%) against 72 of 200 (36.0\%) in the second. The second pair is the best single run. The first pair falls just short of the 0.05 level set in the plan, and the second clears it. Counting each task once across both pairs, the shorter list does better on 33 tasks and the original list on 17, with 150 ties ($p=0.033$). The gain replicated. It is modest, about 7 points, and repeat runs still move individual tasks: the original list changed on 15 tasks and the shorter list on 18. Length and timing are in Appendix~\ref{app:trim}.

\begin{table}[H]
\centering
\footnotesize
\caption{Shorter tool list against the original list (same model, tasks, and other harness parts). Pair 1 compares the shorter list with a fresh run of the original list. Pair 2 runs the shorter list first. ``Shorter only'' counts tasks the shorter list solves and the original list misses. $p$ is an exact two-sided McNemar test, judged at the 0.05 level set in the plan, outside Table~\ref{tab:mcnemar}.}
\label{tab:trim}
\begin{tabular}{@{}lrrrrc@{}}
\toprule
 & Shorter only & Original only & Both & Neither & $p$ \\
\midrule
Pair 1 & 26 & 13 & 58 & 103 & 0.053 \\
Pair 2 & 26 & 10 & 62 & 102 & 0.011 \\
\bottomrule
\end{tabular}
\end{table}

Figure~\ref{fig:ladder} shows these runs on one scale. The top bar is the shorter tool list.

@@LADDER@@"""

HOW_SURE = r"""Each harness accuracy comes with a range that extends about six to seven percentage points on either side (Table~\ref{tab:board}). A gap smaller than that can be noise. Of the eight planned comparisons (Table~\ref{tab:mcnemar}), two harness gaps fall below 0.05 on their own: the full stack against JSON text ($p=0.023$) and the minimal recipe against the native API ($p=0.033$). Neither stays below 0.05 after correcting for the eight tests. The order of the harness rows is therefore tentative, and the full stack's gain of 8.0 percentage points over JSON text is suggestive.

Repeat runs of the full stack do not agree. It scored 77 of 200, then 71 of 200 and 72 of 200. A swing of that size is as large as most gaps between harness rows, so a difference from a single run should not be ranked. Against JSON text, the first of those later runs is ahead by 5.0 percentage points (27 tasks only that run solves, 17 only JSON text, $p=0.174$). The shorter tool list is the one change we ran twice, and that modest gain replicated.

Two contrasts are firm. The best supervised fine-tune solves only 7 tasks that JSON text misses and misses 38 that JSON text solves ($p<0.001$), and that gap survives the correction. The full stack's interval (32.0--45.4\%) does not overlap the best fine-tune (10.7--20.6\%) or the forced-valid output. Preference tuning (26.5\%) is not separable from JSON text."""

CH6_OPEN = r"""Fine-tuning is the standard way to adapt a small model to a task. Three terms come up in this chapter. The \emph{rebuilt mix} is a later training set of 30{,}000 rows that combines newly simulated, execution-checked episodes with cleaned public data. The \emph{gate} is a fixed 20-task check drawn from the same split and run with the status card; the bar is 17 of 20, the untuned model's score with the card. The \emph{null test} is a fine-tune that should change nothing if the machinery is sound: one pass over the untuned model's own logged answers.

Here no trained model beat the untuned model. The best supervised fine-tune scored 15.0\%, significantly below JSON text, and the DPO model scored 26.5\%, within noise of it. The runs are evidence about what goes wrong. The version history is in Appendix~\ref{app:train}."""

CH6_POINTER = (
    "The self-distillation bars in Figure~\\ref{fig:gate}, and the retrains aimed at one suspected cause, "
    "are described in Appendix~\\ref{app:train}. Those retrains did not confirm the cause."
)
CH6_CLOSER = (
    "Fine-tuning changed the weights and did not beat the untuned model. "
    "The next chapter asks what that means for someone choosing a method."
)

DISCUSSION_STATE = r"""\section{State, Not Syntax}

Across these conditions, the hard part is choosing the next step from the current state of the world, including whether to stop. Writing a valid call is the easy part. The levers that help are the ones that support that decision: a summary of real results, a gated stop, and a card that mainly controls whether the model acts. The state summary was meant to put lost facts back in view. What falls, instead, is the number of empty turns. None of the levers reduces state mismatches, and those remain the largest failure type, so the gains come from calling or stopping at the right moment. Tracking the state of the world is still unsolved. Levers that constrain the written form hurt, and training appeared to shift the balance between acting and stopping rather than to improve that tracking."""

RECOMMEND = r"""\section{What We Recommend}

For an on-device agent, use JSON text, a harness-written state summary, a gated stop rule, and a status card. All four are greedy and need no training data or training compute. The minimal recipe, which leaves out the state summary, is a smaller fallback. Writing the same tool list more compactly is a cheap addition: it helped across two paired runs, and it cut context per request by about 30\%. Before any fine-tune, screen on task-level checks rather than validation loss. Even the best run finishes fewer than half of the tasks, and a repeat of the same harness does not pass the same tasks every time. On this benchmark, this model is not yet a reliable multi-step agent."""

LIMITATIONS = r"""\section{Limitations}

\begin{itemize}\setlength{\itemsep}{1pt}
  \item \textbf{One model, one split, one device.} All multi-turn results use Gemma-4-E2B-it on the 200-task base split, on one laptop. The stop threshold, the card's fields, and the training filters were chosen on these same tasks. Other models, and the harder splits of the benchmark, are untested.
  \item \textbf{The harness was assembled on these tasks.} The full stack was put together after earlier results on the same 200 tasks were already known, and the family of eight tests was fixed after that. Its gain over JSON text does not survive that correction.
  \item \textbf{Runs move.} Repeating the same harness does not mark the same tasks correct every time. The shorter tool list was tested as two pairs on these same 200 tasks.
  \item \textbf{The screening gate was built from the same split.} The 20 gate tasks were picked using earlier per-task results from the same split, and the bar is the score of the setup the gate is screening. It is a screening check, not a confirmation.
\end{itemize}

The result is the same across the checks in this thesis. The model fails when it must decide the next step, the harness helps that decision, and forcing valid syntax or fine-tuning does not."""

CONCLUSION = (
    "A small model can carry out one tool call and still fail when the task needs a chain of them. "
    "The failure is in deciding the next step from the current state of the world, including whether "
    "to call, ask, stop, or refuse. Writing a call that only has to parse is the easy part, and forcing "
    "every call into valid syntax made the chain worse.\n"
    "\n"
    "What helped was the harness, the program around the model, with the weights left as they were. "
    "It records the current state, stops repeated calls, and shows a short card that advises the next move. "
    "That combination raised accuracy by a suggestive amount, and it kept up with a five-sample vote at a "
    "fraction of the generation time. Writing the same tool list more compactly helped further. Wrong final "
    "states remain the largest failure type. Fine-tuning did not help: the best tuned model scored below "
    "the untuned model. For a local agent, these harness changes are also far cheaper than voting or training.\n"
)

TRIM = r"""\section{Shorter Tool List Checks}
\label{app:trim}

A 10-task check scored 5 of 10 with the original list and 8 of 10 with the shorter list. The recorded calls differed from the first full-stack run on five tasks. The plan required none of those ten to differ, so that check failed. The plan had paired the test with the first full-stack run (77 of 200). After the check failed, the comparison used a fresh rerun of the original list instead.

The context window is 16{,}384 tokens per request (65{,}536 shared across four slots). The opening prompt on the original list is about 5{,}000 tokens. Its median length falls from 4{,}923.5 tokens to 2{,}888, a 41\% cut (41.3\%). Mean context per request falls from 7{,}107 tokens to 4{,}990, about 30\%. Invalid tool names stay at zero (0 of 1{,}276 calls with the shorter list, 0 of 1{,}244 with the original list). Invalid arguments are 2 of 1{,}276 against 9 of 1{,}244. Format errors are 1 of 1{,}801 model replies against 0 of 1{,}773. Mean server time per model request, prompt reading plus generation, with four slots, is 15.4\,s with the shorter list and 18.8\,s with the original list (medians 12.7\,s and 15.9\,s). Most of the drop is generation (16.2\,s to 13.5\,s). Prompt reading falls from 2.53\,s to 1.86\,s.

The second pair shows the same pattern. Mean context per request falls from 7{,}121 tokens to 5{,}001, again about 30\%, and mean server time per request from 19.4\,s to 15.4\,s, about 20\% faster. Invalid tool names and format errors are zero for both lists. Invalid arguments are 4 of 1{,}300 calls with the shorter list against 5 of 1{,}255 (0.31\% and 0.40\%).
"""


def para_to_section(text, level="section", titles=None):
    titles = titles or TITLES
    out = []
    for line in text.split("\n"):
        m = re.match(r"\\paragraph\{((?:[^{}]|\{[^{}]*\})*)\}\s?(.*)$", line)
        if m and m.group(1) in titles:
            t = titles[m.group(1)]
            if t is not None:
                out.append(f"\\{level}{{{t}}}")
                out.append("")
            if m.group(2):
                out.append(m.group(2))
            continue
        if m:
            raise SystemExit("unmapped paragraph heading: " + m.group(1))
        out.append(line)
    return "\n".join(out)


def fix(text, level="section", titles=None):
    text = para_to_section(text, level, titles)
    for lab in CHAPTER_LABELS:
        text = text.replace(f"Section~\\ref{{{lab}}}", f"Chapter~\\ref{{{lab}}}")
    text = re.sub(r"\\begin\{(figure|table)\}\[[^\]]*\]", r"\\begin{\1}[H]", text)
    text = text.replace("\\FloatBarrier\n", "")
    text = text.replace("this paper", "this thesis").replace("This paper", "This thesis")
    text = re.sub(
        r"\\texttt\{[^{}]*--[^{}]*\}",
        lambda m: m.group(0).replace("--", "-{}-"),
        text,
    )
    return text


def raw(a, b, level="section", titles=None):
    return fix(L(a, b), level, titles)


def add_short_captions(text):
    """List-of-figures / list-of-tables entries are short titles, not '1. TABLE 1. full caption'."""

    def repl_float(m):
        block = m.group(0)
        labels = re.findall(r"\\label\{([^}]+)\}", block)
        short = next((SHORT_LIST[lab] for lab in labels if lab in SHORT_LIST), None)
        if short is None:
            raise SystemExit("no short list title for " + ", ".join(labels))
        block2, n = re.subn(r"\\caption\{", r"\\caption[" + short + "]{", block, count=1)
        if n != 1:
            raise SystemExit("caption insert failed for " + ", ".join(labels))
        return block2

    return re.sub(
        r"\\begin\{(?:figure|table)\}.*?\\end\{(?:figure|table)\}",
        repl_float,
        text,
        flags=re.S,
    )


def _float_span(text, label):
    needle = "\\label{" + label + "}"
    for m in re.finditer(
        r"\\begin\{(?:figure|table)\}\[H\]\n.*?\\end\{(?:figure|table)\}\n\n",
        text,
        flags=re.S,
    ):
        if needle in m.group(0):
            return m
    raise AssertionError(label)


def move_float_to_token(text, label, token):
    m = _float_span(text, label)
    fl = m.group(0)
    text = text[: m.start()] + text[m.end() :]
    if token not in text:
        raise SystemExit("missing float token " + token)
    return text.replace(token, fl.rstrip("\n") + "\n", 1)


def allow_path_breaks(text):
    return re.sub(
        r"\\texttt\{[^{}]*/[^{}]*\}",
        lambda m: m.group(0).replace("/", r"/\allowbreak{}"),
        text,
    )


def must_replace(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: found {n} occurrences of {old[:120]!r}")
    return text.replace(old, new, 1)


def replace_span(text, start, end, new, label):
    i = text.find(start)
    if i < 0:
        raise SystemExit(f"{label}: start missing: {start[:80]!r}")
    j = text.find(end, i)
    if j < 0:
        raise SystemExit(f"{label}: end missing: {end[:80]!r}")
    j += len(end)
    return text[:i] + new + text[j:]


def replace_paragraph_starting(text, start, new, label):
    i = text.find(start)
    if i < 0:
        raise SystemExit(f"{label}: start missing: {start[:80]!r}")
    j = text.find("\n\n", i)
    if j < 0:
        j = len(text)
    return text[:i] + new.rstrip("\n") + text[j:]


def replace_section(text, title, new_text):
    pat = rf"\\section\{{{re.escape(title)}\}}\n\n.*?(?=\n\\section\{{|\Z)"
    m = re.search(pat, text, flags=re.S)
    if not m:
        raise SystemExit("section not found: " + title)
    return text[: m.start()] + new_text.strip() + "\n\n" + text[m.end() :]


def cut_heading_block(text, level, title):
    pat = rf"\\{level}\{{{re.escape(title)}\}}\n\n.*?(?=\n\\(?:sub)?section\{{|\Z)"
    m = re.search(pat, text, flags=re.S)
    if not m:
        raise SystemExit(f"{level} not found: {title}")
    body = m.group(0).strip() + "\n"
    text = text[: m.start()] + text[m.end() :]
    return text, body


def edit_glossary(text):
    text = must_replace(
        text,
        r"\item[BoN] Best-of-N; internal name for the vote of five." + "\n",
        "",
        "cut BoN glossary",
    )
    text = must_replace(
        text,
        r"\item[GST] Guided-structured template: the seven-step scaffold on the native API." + "\n",
        "",
        "cut GST glossary",
    )
    text = must_replace(
        text,
        r"\item[10-task tripwire] Fixed 10-task regression check; does not rank models." + "\n",
        "",
        "cut tripwire glossary",
    )
    text = must_replace(
        text,
        r"\item[Greedy decoding] Always picking the most likely next token (temperature 0). With four parallel server slots, repeat runs on our server still differed (77/200, 71/200 and 72/200 for the same full stack).",
        r"\item[Greedy decoding] Always picking the most likely next token (temperature 0). Repeat runs can still differ.",
        "greedy glossary",
    )
    text = must_replace(
        text,
        r"\item[IFFC] Instruction-followed function calling: calls written as JSON in ordinary text; called \emph{JSON text} in this thesis.",
        r"\item[JSON text] Calls written as a JSON list in ordinary text, which the harness parses. Logs abbreviate this IFFC.",
        "JSON text glossary",
    )
    text = must_replace(
        text,
        r"\item[NFC] Native function calling: structured \texttt{tool\_calls} via the server's tool API; called \emph{native API} in this thesis.",
        r"\item[Native API] Structured \texttt{tool\_calls} from the server's tool interface. Logs abbreviate this NFC.",
        "native API glossary",
    )
    return text


def apply_overrides(parts):
    # Chapter 1: keep the problem sentence, drop the numbered contribution list.
    parts["ch1"] = must_replace(
        parts["ch1"],
        "on a MacBook Pro. Through the server's built-in tool-call field, the native tool interface,",
        "on a laptop. Through the server's built-in tool-call field, the native API,",
        "ch1 hardware",
    )
    parts["ch1"] = replace_span(
        parts["ch1"],
        "Our contributions:",
        r"\end{enumerate}",
        CONTRIBUTIONS,
        "contributions",
    )

    # Chapter 2: one naming pair, a chapter opening, and a closer.
    parts["ch2"] = CH2_INTRO + parts["ch2"]
    parts["ch2"] = must_replace(
        parts["ch2"],
        r"With \emph{native function calling} (NFC), the server receives a tool list and the model returns structured \texttt{tool\_calls}, as in most production APIs. With \emph{instruction-followed function calling} (IFFC), the prompt asks the model to write its calls as a JSON list in ordinary text, which the harness parses.",
        r"With the \emph{native API}, the server receives a tool list and the model returns structured \texttt{tool\_calls}, as in most production APIs. With \emph{JSON text}, the prompt asks the model to write its calls as a JSON list in ordinary text, which the harness parses.",
        "output channel names",
    )
    parts["ch2"] = must_replace(
        parts["ch2"],
        r"Terms used in this thesis are collected in a glossary (Appendix~\ref{app:runs}).",
        "Terms used in this thesis are collected in the glossary before Chapter 1.",
        "glossary pointer",
    )
    parts["ch2"] = parts["ch2"].rstrip() + CH2_CLOSER

    # Chapter 3: hardware and commits leave the chapter; examples of the two notes move in.
    parts["ch3"] = replace_section(parts["ch3"], "Model and Serving", MODEL_SERVING)
    parts["ch3"] = replace_section(parts["ch3"], "Harness Interventions", HARNESS)
    parts["ch3"] = must_replace(
        parts["ch3"],
        r" A smaller 10-task regression check is described in Appendix~\ref{app:train}.",
        "",
        "cut tripwire pointer",
    )
    parts["ch3"] = replace_section(parts["ch3"], "Statistics", STATISTICS)

    # Chapter 4: a plain state-mismatch example; drop the draft caption clause.
    parts["ch4"] = must_replace(
        parts["ch4"],
        r"\section{Most Failures Are About the World}" + "\n\n",
        r"\section{Most Failures Are About the World}" + "\n\n" + STATE_MISMATCH,
        "state mismatch example",
    )
    parts["ch4"] = must_replace(
        parts["ch4"],
        " The shorter tool list is omitted. Checker rows exist for its runs (116 and 112 failed tasks) and the original-list reruns (129 and 128), but those rows were not labelled for this revision.",
        " The shorter tool list is omitted.",
        "fig 2 caption",
    )

    # Chapter 5: guided-template tie lives with the channel result; shorter list is one page;
    # the ladder figure waits until that list has been introduced.
    parts["ch5"] = must_replace(
        parts["ch5"],
        "Figure~\\ref{fig:ladder} shows the results. ",
        "",
        "uncite ladder early",
    )
    parts["ch5"] = must_replace(
        parts["ch5"],
        "Every change is greedy except the vote of five, and none touches the weights; exact counts",
        "Every change below is greedy except the vote of five, and none touches the weights. Exact counts",
        "ch5 opening",
    )
    parts["ch5"] = replace_section(parts["ch5"], "JSON Text over the Native API", JSON_CHANNEL)
    parts["ch5"] = must_replace(
        parts["ch5"],
        r"The harness-written snapshot scores 68/200 ($p=0.30$ against JSON text).",
        r"The harness-written snapshot scores 68/200, 3.5 percentage points above JSON text ($p=0.30$).",
        "state summary gap",
    )
    parts["ch5"] = must_replace(
        parts["ch5"],
        r"Adding the status card gives the minimal recipe, 71/200 (+5.0\pp over JSON text, $p=0.16$).",
        r"Adding the status card gives the minimal recipe, 71/200 (35.5\%, +5.0\pp over JSON text, $p=0.16$).",
        "minimal recipe percent",
    )
    parts["ch5"] = must_replace(
        parts["ch5"],
        r"Combining all four greedy changes scores 77/200 (38.5\%, interval 32.0--45.4\%). Paired against JSON text, it solves 30 tasks that JSON text misses and misses 14 that JSON text solves ($p=0.023$ uncorrected; not significant after correction). Against the minimal recipe, the state summary alone and the vote of five, the differences are within noise (Table~\ref{tab:mcnemar}). The configuration was assembled after we had seen earlier results on these tasks, so its lead is subject to tuning leakage (Chapter~\ref{sec:discussion}). Its harness code is in pull request \#90 (run at commit \texttt{344ce45}, merged as \texttt{83f35ee}).",
        r"Combining all four greedy changes scores 77/200 (38.5\%, interval 32.0--45.4\%). Paired against JSON text, it solves 30 tasks that JSON text misses and misses 14 that JSON text solves (Table~\ref{tab:mcnemar}). Against the minimal recipe, the state summary alone, and the vote of five, the differences are within noise. The configuration was assembled after we had seen earlier results on these tasks, so its lead is subject to tuning leakage (Chapter~\ref{sec:discussion}).",
        "full stack prose",
    )
    parts["ch5"] = replace_section(parts["ch5"], "Shorter Tool List, Same Tools", SHORTER)
    parts["ch5"] = must_replace(
        parts["ch5"],
        r"Voting over five samples on top of the state summary scores 73/200, four tasks fewer",
        r"Voting over five samples on top of the state summary scores 73/200 (36.5\%), four tasks fewer",
        "vote percent",
    )
    parts["ch5"] = replace_paragraph_starting(
        parts["ch5"],
        "Intervals for the harness rows are about",
        HOW_SURE,
        "how sure",
    )
    parts["ch5"] = must_replace(
        parts["ch5"],
        "200 tasks each, same task IDs",
        "200 tasks each, the same tasks",
        "mcnemar caption",
    )

    # Chapter 6: define the three terms, drop two asides, move the candidate-cause section.
    parts["ch6"] = replace_span(
        parts["ch6"],
        "Fine-tuning is the standard way to adapt a small model to a task.",
        r"the version history is in Appendix~\ref{app:train}.",
        CH6_OPEN,
        "ch6 opening",
    )
    parts["ch6"] = must_replace(
        parts["ch6"],
        r" (2{,}121 versus 596 continue)",
        "",
        "cut stop-target counts",
    )
    parts["ch6"] = must_replace(
        parts["ch6"],
        r" Adding the state summary to this fine-tune lowered it to 26/200, although the summary helps the untuned model.",
        "",
        "cut 26/200 aside",
    )
    parts["ch6"] = must_replace(
        parts["ch6"],
        r"To test the pipeline, we ran a \emph{null test}, a fine-tune that should change nothing if the machinery is sound: the untuned model trained for one pass (39 updates) on 156 of its own logged answers through the same template, adapter, merge, quantization and serving steps.",
        r"To test the pipeline, the null test trained the untuned model for one pass (39 updates) on 156 of its own logged answers, through the same template, adapter, merge, quantization, and serving steps.",
        "null test once",
    )
    parts["ch6"], cause = cut_heading_block(
        parts["ch6"], "section", "A Candidate Cause That Targeted Retrains Did Not Confirm"
    )
    cause = cause.replace("\\section{", "\\subsection{", 1)
    parts["ch6"] = must_replace(
        parts["ch6"],
        "so its data are the leading suspect but not the only one.\n",
        "so its data are the leading suspect but not the only one.\n\n" + CH6_POINTER + "\n\n@@GATE@@\n",
        "gate token",
    )
    parts["ch6"] = parts["ch6"].rstrip() + "\n\n" + CH6_CLOSER + "\n"
    parts["appB_train"], _trip = cut_heading_block(parts["appB_train"], "subsection", "10-Task Tripwire")
    marker = r"\subsection{Self-Distillation Counts}"
    if marker not in parts["appB_train"]:
        raise SystemExit("self-distillation heading missing")
    parts["appB_train"] = parts["appB_train"].replace(marker, cause.strip() + "\n\n" + marker, 1)

    # Chapter 7: guided-template section already moved; four limitations; a closer.
    parts["ch7"], _guided = cut_heading_block(parts["ch7"], "section", "The Guided-Template Tie")
    parts["ch7"] = replace_section(parts["ch7"], "State, Not Syntax", DISCUSSION_STATE)
    parts["ch7"] = replace_section(parts["ch7"], "What We Recommend", RECOMMEND)
    parts["ch7"] = replace_section(parts["ch7"], "Limitations", LIMITATIONS)

    parts["ch8_concl"] = CONCLUSION

    # Appendix A: examples moved to Chapter 3; drop the draft-note sentence.
    parts["appA"] = must_replace(
        parts["appA"],
        " It is not a model-written summary. Example:\n"
        "\\begin{quote}\\footnotesize\\ttfamily\n"
        "[STATE]\\\\ cwd: /workspace/notes\\\\ files: todo.txt, log.csv\\\\ last: cd(folder='notes')->ok\n"
        "\\end{quote}",
        r" It is not a model-written summary. An example is in Chapter~\ref{sec:setup}.",
        "move state example",
    )
    parts["appA"] = must_replace(
        parts["appA"],
        " It is not the multi-pass OODA-Tool of~\\cite{guo2026ooda}. Example:\n"
        "\\begin{quote}\\footnotesize\\ttfamily\n"
        "[OODA]\\\\ goal: lock the doors and start the engine\\\\ slots: lockStatus=locked\\\\ last: startEngine()->error: brake pedal must be pressed\\\\ next: call\n"
        "\\end{quote}",
        r" It is not the multi-pass OODA-Tool of~\cite{guo2026ooda}. An example is in Chapter~\ref{sec:setup}.",
        "move card example",
    )
    parts["appA"] = must_replace(
        parts["appA"],
        " How preferred and rejected responses were paired is not in the notes used for this thesis.",
        "",
        "cut notes phrase",
    )

    # Appendix B table: the reader-facing names, not the log abbreviations.
    parts["appB"] = must_replace(
        parts["appB"],
        "NFC = native API; IFFC = JSON text; ",
        "Native is the native API. JSON text is calls written as JSON text. ",
        "failure caption",
    )
    parts["appB"] = must_replace(
        parts["appB"],
        "Label & NFC & IFFC & Stop &",
        "Label & Native & JSON text & Stop &",
        "failure header",
    )

    parts["appC_repro"] = must_replace(
        parts["appC_repro"],
        "Hardware: MacBook Pro, Apple M4~Pro, 48~GB.",
        "Hardware: MacBook Pro, Apple M4~Pro, 48~GB, with Metal acceleration (Apple's GPU interface).",
        "metal in appendix",
    )
    parts["glossary"] = edit_glossary(parts["glossary"])
    parts["appC_trim"] = TRIM
    parts["appC_board"] = parts["appC_board"].replace(
        "\\begin{table}[H]\n",
        "\\begin{table}[H]\n\\renewcommand{\\arraystretch}{1.0}\n",
        1,
    )
    return parts


def check_abstract():
    for s in ABSTRACT_SOURCES:
        if s not in paper_text and s not in paper_text.replace("\\%", "%"):
            # "a seventh" and "five-sample" and "about 7 points" are literal.
            if s not in paper_text:
                raise SystemExit("abstract source missing from paper: " + s)


def check_output(tex):
    if "—" in tex or "---" in tex:
        raise SystemExit("em dash in thesis")
    if re.search(r"\bJava(Script)?\b", tex):
        raise SystemExit("Java benchmark mention")
    if tex.count(r"\placeholderinline{Committee member name}") != 2:
        raise SystemExit("committee placeholders changed")
    if "Dr. Mohammad Pourhomayoun, Committee Chair" not in tex:
        raise SystemExit("committee chair missing")
    if "Dr. Chengyu Sun, Department Chair" not in tex:
        raise SystemExit("department chair missing")
    if r"\chapter*{ACKNOWLEDGMENTS}" not in tex.split("%", 1)[-1] and "\\chapter*{ACKNOWLEDGMENTS}" in re.sub(
        r"(?m)^%.*$", "", tex
    ):
        raise SystemExit("acknowledgments were uncommented")
    # The acknowledgments heading may appear only inside a comment.
    live = re.sub(r"(?m)^%.*$", "", tex)
    if "ACKNOWLEDGMENTS" in live:
        raise SystemExit("acknowledgments heading is live")
    if "December 2026" not in tex:
        raise SystemExit("date missing")
    pre, _post = tex.split(r"\appendix", 1)
    for banned in (
        "evaluate_multi_turn",
        "not labelled for this revision",
        "not in the repository",
        "not in the notes used for this thesis",
        "pull request",
        "344ce45",
        "11c325c6e",
        "d344123fe",
        "scores.id",
        "2{,}121",
        "Pooled",
        "166\\,ms",
        "10-Task Tripwire",
        "@@LADDER@@",
        "@@GATE@@",
    ):
        if banned in pre:
            raise SystemExit("banned in main text: " + banned)
    if pre.count("77 of 200, then 71 of 200 and 72 of 200") != 1:
        raise SystemExit("repeat-run triple should be stated once")
    if pre.count("33 tasks") != 1:
        raise SystemExit("33 tasks should be stated once in the main text")
    if pre.count("p=0.023") != 1:
        raise SystemExit("p=0.023 should be stated once in the main text")


def main():
    check_abstract()
    parts = {
        "abstract": ABSTRACT,
        "ch1": raw(76, 103),
        "ch2": raw(108, 116),
        "ch3": raw(121, 139),
        "ch4": raw(144, 155),
        "ch5": raw(161, 228),
        "ch6": raw(232, 246),
        "ch7": raw(250, 268),
        "ch8_future": raw(273, 273),
        "ch8_concl": raw(275, 275),
        "appA": raw(286, 308, titles=APP_A_TITLES),
        "appB": raw(313, 347),
        "appB_train": raw(352, 358, level="subsection"),
        "appC_repro": raw(419, 419),
        "glossary": fix(L(423, 458)),
        "appC_board": raw(364, 417).replace(
            r"\parbox{\linewidth}{\scriptsize",
            r"\parbox{\linewidth}{\RaggedRight\scriptsize",
        ),
    }
    parts = apply_overrides(parts)
    for key, text in list(parts.items()):
        parts[key] = add_short_captions(text)
    parts["ch5"] = move_float_to_token(parts["ch5"], "fig:ladder", "@@LADDER@@")
    parts["ch6"] = move_float_to_token(parts["ch6"], "fig:gate", "@@GATE@@")
    parts["appC_repro"] = allow_path_breaks(parts["appC_repro"])

    tpl = (here / "main.tpl.tex").read_text()
    for k, v in parts.items():
        tpl = tpl.replace("%%" + k + "%%", v)
    if "%%" in re.sub(r"%%%+", "", tpl):
        raise SystemExit("unfilled slot")
    check_output(tpl)
    (here / "main.tex").write_text(tpl)
    print("wrote", here / "main.tex")


if __name__ == "__main__":
    main()
