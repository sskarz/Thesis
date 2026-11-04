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
    "Forcing valid syntax hurts.": "The Two Forced-Valid Methods We Tested Did Worse",
    r"JSON text over the native API (+3.5\pp).": "JSON Text over the Native API",
    r"State summary (+3.5\pp).": "State Summary",
    r"Stop on repeats and the status card (minimal recipe, 35.5\%).": "Stop on Repeats and the Status Card",
    r"Full stack (38.5\% on the first run).": "Full Stack",
    "Shorter tool list, same tools.": "Shorter Tool List, Same Tools",
    r"Cost: majority vote (36.5\%).": "Majority Vote",
    "How sure are we?": "How Sure Are We?",
    "Public data taught the model to stop.": "Stop Targets in the First Mixes",
    "A rebuilt mix collapsed.": "A Rebuilt Mix Collapsed",
    "A candidate cause that targeted retrains did not confirm.": "Two Targeted Retrains Did Not Restore the Score",
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

ABSTRACT = (
    "Small language models can run on a laptop and call tools for a user, but they often fail "
    "when a later call depends on an earlier one. This thesis studies Gemma-4-E2B-it, a model "
    "with about two billion effective parameters, on 200 multi-turn tasks from the Berkeley "
    "Function Calling Leaderboard. A task is correct only when a simulated backend ends in the "
    "right state, such as a car's engine being on.\n"
    "\n"
    "The best setup leaves the weights fixed. A harness, the program around the model, writes "
    "calls as JSON text, records the current state, stops repeated calls, and shows a short "
    "card advising the next move. On that harness, writing the same tool list more compactly "
    "scored 84 of 200 (42.0\\%) and 88 of 200 (44.0\\%) in two paired runs, against 71 and 72 "
    "of 200 for the original list. The gain repeated on these tasks. The compact list also cut "
    "mean context per request by about 30\\%. Reduced context is a plausible explanation, but "
    "these runs do not isolate the mechanism.\n"
    "\n"
    "Most failed tasks end with the backend in the wrong state or a required result missing, "
    "and the two forced-valid decoding methods we tested made results worse. Without the "
    "compact list, the first full-stack run scored 38.5\\% against 30.5\\% for JSON text; that "
    "gap does not survive correction for multiple comparisons. Fine-tuning did not beat the "
    "untuned model.\n"
)

# Strings that must already appear in paper/main.tex, with the same meaning.
ABSTRACT_SOURCES = [
    "about two billion effective parameters",
    "200 multi-turn",
    "engine is on",
    "88 of 200",
    "44.0\\%",
    "84 of 200",
    "42.0\\%",
    "71 and 72",
    "30.5\\%",
    "38.5\\%",
    "first run",
    "about 30\\%",
    "not significant after correcting for multiple comparisons",
]

CONTRIBUTIONS = r"""The individual techniques are borrowed or simple. JSON-text calls follow Taheri et al.~\cite{taheri2026iffc}, voting is a standard baseline~\cite{snell2024testtime}, and the state summary, stop rule, status card, and compact tool list are our own rule-based implementations of familiar ideas (Chapter~\ref{sec:related}). The contribution is empirical, a controlled comparison on one small model:

\begin{enumerate}
  \item \textbf{A harness plus a compact tool list.} The weights stay fixed. The program around the model records the current state, stops repeated calls, and shows a short card advising the next move. On that harness, writing the same tool list more compactly scored 84 and 88 of 200, against 71 and 72 for the original list, in two paired runs on the same tasks (Chapter~\ref{sec:helps}).
  \item \textbf{Where the failures are.} Most failed tasks end with a required result missing or the backend in the wrong state. The failure analysis motivated the harness. The two forced-valid decoding methods we tested made results worse (Chapter~\ref{sec:fail}).
  \item \textbf{Fine-tuning did not help.} The best supervised fine-tune scores below the untuned model, and preference tuning does not beat it. A short fine-tune on the untuned model's own logged answers, which should change little if the training steps are sound, stayed close to the untuned model on a 20-task check. The collapse of the rebuilt training mix therefore points at its data or its much longer training, and these runs do not separate the two. Two retrains that rebalanced stop and look-up targets also missed the bar (Chapter~\ref{sec:train}).
\end{enumerate}"""

CH2_INTRO = (
    "This chapter names the benchmark and the ideas the experiments rely on: "
    "how a call is written, what it means to force that writing to be valid, "
    "what the program around the model can do, and how training a tool-calling model is usually done.\n\n"
)

CH2_POSITION = r"""\section{How This Thesis Differs}

Three lines of work are closest. Context compression~\cite{kang2025acon} shortens the history an agent carries; here the state summary is written by fixed rules from executed results, and the compact tool list shortens only the tool descriptions, which repeat on every request. OODA-style controllers~\cite{guo2026ooda} structure each agent step; our status card is one advisory block from fixed rules and adds no model calls. Structured decoding~\cite{willard2023outlines,dong2026xgrammar2} guarantees valid syntax; we test two of our own implementations of that idea on stock \texttt{llama-server}, not those engines.

We separate three things. Borrowed: JSON-text calls~\cite{taheri2026iffc}, majority voting~\cite{snell2024testtime}, the idea of a tool-name automaton~\cite{yu2026ptc}, and LoRA and DPO training~\cite{hu2021lora,rafailov2023dpo}. Our implementation choices: the specific state summary, stop rule, status card, compact tool list, and two-pass constrained decoding described in Chapter~\ref{sec:setup}. The empirical contribution: measuring these harness levers, two forced-valid methods, and fine-tuning against each other on one small model and one benchmark split, with paired tests and repeat runs. We do not claim a new method."""

CH2_CLOSER = (
    "\n\n" + CH2_POSITION + "\n\nThe following chapters use only these ideas. They ask where a small model fails, "
    "whether a training-free harness with a compact tool list can help, and whether fine-tuning can.\n"
)

MODEL_SERVING = r"""\section{Model and Serving}

All results use Gemma-4-E2B-it, quantized with Q4\_K\_M (a 4-bit weight format of llama.cpp), served by \texttt{llama-server} on a laptop. Decoding is greedy (temperature 0) except for majority voting. Greedy runs are not identical from one run to the next (Chapter~\ref{sec:helps}). The machine, the server build, and the flags are in Appendix~\ref{app:runs}.

Two serving settings differ between runs, and neither difference was tested. Most runs used four parallel server slots, and the native API baseline used eight. The earlier runs, including JSON text and the single-lever harness rows, used an older llama.cpp build than the first full-stack run and the compact-list runs. Comparisons across those groups, such as the full stack against JSON text, therefore also cross a server change. Both runs in each compact-list pair used the same build and four slots."""

HARNESS = r"""\section{Harness Interventions}

All leave the weights untouched. A task is correct only when the multi-turn checker returns valid. Appendix~\ref{app:method} gives each rule except the compact tool list. Each lever was designed for one problem: the state summary for lost state, the stop rule for repetition, and the status card for whether to act or stop. These are design intents. The error labels in Chapter~\ref{sec:helps} show what changed, not why.
\begin{itemize}\setlength{\itemsep}{1pt}
  \item \textbf{JSON text} instead of the native API. The harness parses the calls.
  \item \textbf{State summary.} After each executed call, the harness writes a short \texttt{[STATE]} snapshot of facts from real results (current directory, known files, last calls) into the conversation.
  \item \textbf{Stop on repeats.} The harness ends the turn when the model repeats its previous calls, or when it re-proposes calls already executed in this turn after at least six tool steps. An \emph{ungated} variant drops that minimum.
  \item \textbf{Status card.} A four-line card is prefixed to the latest user message. It advises call, ask, finish, or refuse (the card prints finish as \texttt{final}) from fixed rules on the conversation and executed results, and it never sees reference answers. JSON text, stop on repeats, and the card, together, are the \emph{minimal recipe}.
  \item \textbf{Full stack.} JSON text, the state summary, stop on repeats, and the status card, decoded greedily. On its first run this is the harness we compare with JSON text.
  \item \textbf{Compact tool list.} On the full stack, every tool stays in the prompt. The harness states each tool group's shared description once, instead of repeating it, and prints each schema without indentation. It does not remove or retrieve tools. The model, the four pieces above, and the tasks stay the same.
  \item \textbf{Vote of five.} On top of the state summary, at each step the harness samples five candidate calls at temperature 0.7 and executes the most common one. This is the compute-heavy comparison point.
  \item \textbf{Forced-valid decoding.} \emph{Constrained decoding} restricts output in two passes (tool names, then arguments by schema). A \emph{tool-name automaton} constrains only the name, inspired by~\cite{yu2026ptc}.
\end{itemize}

A state summary, written from results the harness has already executed, looks like this:
\begin{quote}\singlespacing\footnotesize\ttfamily\setlength{\topsep}{2pt}\setlength{\partopsep}{0pt}
[STATE]\\ cwd: /workspace/notes\\ files: todo.txt, log.csv\\ last: cd(folder='notes')->ok
\end{quote}
\noindent\begin{minipage}{\linewidth}
A status card, prefixed to the latest user message, looks like this:
\begin{quote}\singlespacing\footnotesize\ttfamily\setlength{\topsep}{2pt}\setlength{\partopsep}{0pt}
[OODA]\\ goal: lock the doors and start the engine\\ slots: lockStatus=locked\\ last: startEngine()->error: brake pedal must be pressed\\ next: call
\end{quote}
\end{minipage}
"""

ERROR_LABELS = r"""Each failed task gets one primary label, taken from the error message of the BFCL multi-turn checker. The checker goes through the turns in order and stops at the first check that fails. Within a turn it runs three checks in this order, and the label names the first one that fails:
\begin{itemize}\setlength{\itemsep}{1pt}
  \item \textbf{Empty turn.} The model's response for a turn that needed calls is empty: no call was executed in that turn.
  \item \textbf{State mismatch.} After the turn, the state of at least one simulated backend differs from the reference state.
  \item \textbf{Under-call.} The backend states match, but the results of the model's calls do not include every result that the reference calls produce, for example because a needed look-up was never made.
\end{itemize}
Because the checker stops at the first failure, a task with several problems gets the label of its earliest failing turn, and within that turn an empty turn outranks a state mismatch, which outranks an under-call. Every failed task in the labelled runs carries one of these three messages.

Parse failures have no label of their own. With JSON text, an unparseable reply gets one retry. If the retry also fails, the turn ends without that step. If no call had run yet in the turn, the checker then sees an empty turn; if calls had already run, the task can fail as a state mismatch or an under-call instead. On the native API, a reply without a tool call ends the turn, and a tool call whose arguments are not valid JSON is still sent on, marked as an argument error, and fails when executed.

The labels describe outcomes, not causes. A state mismatch says the final state was wrong; it does not by itself show that the model lost track of the state."""

STATISTICS = r"""\section{Statistics}

Each reported accuracy comes with a Wilson 95\% interval~\cite{wilson1927}. It describes uncertainty from the sample of 200 tasks, treating them as drawn from a larger pool of similar tasks. It does not describe how much a rerun on the same tasks would change; that run-to-run variation is reported separately, from repeated runs (Chapter~\ref{sec:helps}).

When two conditions are compared on the same 200 tasks, we use an exact two-sided McNemar test~\cite{mcnemar1947}. It counts only the tasks the two conditions decide differently. The $p$-value is the chance of a split at least that uneven if each such task were equally likely to go either way. Each test uses one run per condition, so run-to-run flips enter it as if they were real differences, and it says nothing about tasks outside these 200. A $p$-value above 0.05, or two overlapping intervals, means the data do not separate the conditions. It does not show that they are equal.

Eight comparisons were corrected together with Holm's method~\cite{holm1979}, so that the chance of any false alarm among them stays at 5\%. That family was specified after the harness had been developed on these tasks, and no test in this thesis was preregistered. The compact tool list was a separate test, run twice in opposite order. Which comparisons are in the family, and how the two pairs were combined, is in Appendix~\ref{app:stats}."""

STATE_MISMATCH = (
    "The loop in Figure~\\ref{fig:vignette} shows how a task fails. The model locks the doors and then tries to start the engine. "
    "Both of those calls are valid. The backend says the brake pedal has to be pressed first. "
    "The model never presses the brake and retries, so the engine stays off. "
    "The vehicle's state then differs from the reference, and the checker labels the task a \\emph{state mismatch}. "
    "An \\emph{under-call} is the other common label: the backend state matches, but a result the reference needs, such as a look-up, is missing.\n\n"
)

JSON_CHANNEL = r"""\section{JSON Text over the Native API}

JSON text scores 61/200 against 54/200 on the native API, a lead of 3.5 percentage points that is not statistically significant ($p=0.32$). The native API run also used eight server slots instead of four (Chapter~\ref{sec:setup}). A seven-step reasoning template on the native API also scored 61/200, the same as JSON text: 19 tasks swap each way, and state mismatches rose to 79. The gap reflects the shape of the prompt, not only the channel that carries the call, so we do not claim the channel switch alone as a gain. Later steps are built on JSON text because the harness can parse and inspect those calls directly. The harness levers were evaluated on JSON text, the stronger baseline."""

SHORTER = r"""\section{The Best Setup: A Compact Tool List}

The best setup is the full stack with the tool list written more compactly. It scored 84 of 200 (42.0\%) in the first paired run and 88 of 200 (44.0\%) in the second. In each pair, the original list is a fresh rerun of the full stack. This is a repeat on the same 200 tasks, not a test on new tasks. The full stack still sends every tool. The list is long because tools that share a group also share a description, and that description was repeated on every tool, with each schema indented. We state each group's shared description once and print the schemas without indentation (Fig.~\ref{fig:compact}). It does not remove or retrieve tools. Nothing else changes. The model, the four pieces of the full stack, and the tasks stay the same. Like the state summary and the card, this is harness-side control delivered through the context.

\begin{figure}[H]
\centering
\noindent\textbf{Original list.} Median opening prompt 4{,}923.5 tokens.
\begin{lstlisting}
[
  {
    "name": "cat",
    "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Display the contents of a file of any extension from currrent directory.",
    "parameters": {
      "type": "object",
      "properties": {
        "file_name": {
          "type": "string",
          "description": "The name of the file from current directory to display. No path is allowed. "
        }
      },
      "required": [
        "file_name"
      ]
    }
  },
  ...
]
\end{lstlisting}
\par\vspace{6pt}
\noindent\textbf{Compact list.} Median opening prompt 2{,}888 tokens, a 41\% cut.
\begin{lstlisting}
This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc.
[{"name": "cat", "description": "Display the contents of a file of any extension from currrent directory.", "parameters": {"type": "object", "properties": {"file_name": {"type": "string", "description": "The name of the file from current directory to display. No path is allowed. "}}, "required": ["file_name"]}}, ...]
\end{lstlisting}
\caption{File-system tools from the multi-turn tasks, printed both ways. Both lists contain the same tools. The original list repeats the group's shared description on each tool and indents each schema. The compact list states that description once and prints the schemas without indentation. The excerpt is the first tool of the group; an ellipsis marks the rest. The labels are the measured median lengths of the full opening prompt, 4{,}923.5 and 2{,}888 tokens, a 41\% cut.}
\label{fig:compact}
\end{figure}

How the test was run matters for reading it. Before the first pair, we set the test as a McNemar comparison at the 0.05 level against the first full-stack run. A 10-task check showed that a new run of the original list no longer reproduced that first run's calls, so the first pair used a fresh rerun of the original list instead. After the first pair narrowly missed the 0.05 level, we ran a second pair in the opposite order, with the compact list run first, as a repeat (Table~\ref{tab:trim}). Both pairs use the same model, the same 200 tasks, and the same other harness parts. The 10-task check, and the token and timing counts, are in Appendix~\ref{app:trim}.

In the first pair the shorter list scores 84 of 200 (42.0\%, interval 35.4--48.9\%) and the original list scores 71 of 200 (35.5\%, interval 29.2--42.3\%). The shorter list solves 13 more tasks: 26 that the original list misses, and it misses 13 that the original list solves (58 both, 103 neither). An exact two-sided McNemar test gives $p=0.053$, just short of the 0.05 level.

In the second pair the shorter list scores 88 of 200 (44.0\%, interval 37.3--50.9\%), the best single run, and the original list scores 72 of 200 (36.0\%, interval 29.7--42.9\%). The shorter list solves 26 tasks that the original list misses and misses 10 that it solves (62 both, 102 neither), $p=0.011$.

To count each task once across both pairs, we used a sign test. The shorter list does better on 33 tasks and the original list on 17, with 150 ties (exact two-sided sign test, $p=0.033$). The notes do not record whether this combination was chosen before the second pair ran, so we report it as a follow-up analysis. The gain was 6.5 and 8.0 points and repeated in both pairs on the same tasks. Repeat runs still move individual tasks: between its two runs, the original list changed result on 15 tasks and the shorter list on 18.

\begin{table}[H]
\centering
\footnotesize
\caption{Shorter tool list against the original list (same model, tasks, and other harness parts). Pair 1 compares the shorter list with a fresh run of the original list. Pair 2 runs the shorter list first. ``Shorter only'' counts tasks the shorter list solves and the original list misses. $p$ is an exact two-sided McNemar test, outside Table~\ref{tab:mcnemar}. The sign test that counts each task once is in the text.}
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

The compact list reduced mean context per request by about 30\% (from about 7{,}100 to 5{,}000 tokens) and improved accuracy in both paired comparisons. Reduced context burden is a plausible explanation, but these experiments do not isolate the mechanism. They do not vary context length with the wording held fixed, and the compact list also changes the text the model reads. Its failure labels were not split out. Invalid tool names stay at zero for both lists.

Figure~\ref{fig:ladder} puts this setup on the same scale as the other conditions. The top two bars are the shorter tool list in its two pairs, and the two bars below them are the original list rerun in the same pairs.

@@LADDER@@"""

HOW_SURE = r"""The compact tool list is the one change we ran twice. Both pairs favored it, and the follow-up sign test that counts each task once gives $p=0.033$ (Chapter~\ref{sec:helps}). Its failure labels and a direct best-setup versus JSON text comparison are left to the confirmatory runs in Chapter~\ref{sec:conclusion}. The other harness gaps are less firm, and two separate kinds of uncertainty apply to them.

The first is which tasks happened to be in the sample. The Wilson interval of each harness accuracy extends about six to seven percentage points on either side (Table~\ref{tab:board}). It treats the 200 tasks as a sample, and it says nothing about what a second run of the same configuration on the same tasks would give.

The second is run-to-run variation, which the intervals do not include. The full stack scored 77 of 200, then 71 of 200 and 72 of 200 on the same tasks. Between its two runs, the original list changed result on 15 tasks and the compact list on 18. A swing of six tasks between runs of one configuration is as large as several gaps between harness rows, so single runs should not be ranked. Against JSON text, the 71-of-200 rerun is ahead by 5.0 percentage points (27 tasks only that run solves, 17 only JSON text, $p=0.174$).

The McNemar tests in Table~\ref{tab:mcnemar} use one run per condition, so they share that limit. Of the eight comparisons in the family, two harness gaps fall below 0.05 on their own: the full stack against JSON text ($p=0.023$) and the minimal recipe against the native API (McNemar $p=0.033$). Neither stays below 0.05 after correcting for the eight tests. The full stack's first run leads JSON text by 8.0 percentage points, but the order of the harness rows is not statistically separated. That does not make the rows equal; the data are too few to rank them.

One contrast survives the correction. The best supervised fine-tune solves only 7 tasks that JSON text misses and misses 38 that JSON text solves ($p<0.001$). The full stack's interval (32.0--45.4\%) also lies above those of the best fine-tune (10.7--20.6\%) and both forced-valid methods. Preference tuning (26.5\%) is not separated from JSON text by these data."""

CH6_OPEN = r"""Fine-tuning is the standard way to adapt a small model to a task. Three terms come up in this chapter. The \emph{rebuilt mix} is a later training set of 30{,}000 rows that combines newly simulated, execution-checked episodes with cleaned public data. The \emph{gate} is a fixed 20-task check drawn from the same split and run with the status card; the bar is 17 of 20, the untuned model's score with the card. The \emph{null test} is a fine-tune that should change little if the machinery is sound: one pass over the untuned model's own logged answers.

Here no trained model beat the untuned model. The best supervised fine-tune scored 15.0\%, significantly below JSON text, and the DPO model scored 26.5\%, which the paired test does not separate from JSON text's 30.5\%. The runs are evidence about what goes wrong. The version history and the training recipe are in Appendices~\ref{app:train} and~\ref{app:recipe}."""

CH6_POINTER = (
    "Self-distillation trains the model on its own answers to simulated tasks, keeping only the turns a checker marked correct. "
    "It also missed the bar, at 8 of 20. Its data had shifted toward stop targets and away from look-up calls. "
    "Two retrains that each corrected one side of that shift scored 14 and 12 of 20, still below the bar. "
    "Rebalancing the data in those two ways did not restore the untuned score, and these runs do not identify what caused the drop. "
    "The counts are in Appendix~\\ref{app:train}."
)
CH6_CLOSER = (
    "Fine-tuning changed the weights and did not beat the untuned model. "
    "The next chapter asks what that means for someone choosing a method."
)

DISCUSSION_STATE = r"""\section{State, Not Syntax}

Across these conditions, most failed tasks end with the backend in the wrong state or a needed result missing, and the two methods that forced valid syntax did worse. We read this as evidence that the harder part for this model is choosing the next step, including whether to stop, rather than writing a valid call. That reading rests on outcome labels, not on a direct measurement of what the model tracked.

The levers that helped were designed to support that decision: a summary of real results, a gated stop, and a card that advises whether to act. Their measured effect on the labels is mostly on empty turns. The state summary was meant to put lost facts back in view; what falls instead is the number of empty turns. Among the labelled runs, no lever reduces state mismatches, and those remain the largest label. A state mismatch records that the final state was wrong; it does not show by itself that the model lost track of state.

The compact list reduced mean context per request by approximately 30\% and improved accuracy in both paired comparisons. Reduced context burden is a plausible explanation, but these experiments do not isolate the mechanism (Chapter~\ref{sec:helps}). The trained models moved failures between labels without reducing state mismatches."""

RECOMMEND = r"""\section{What We Recommend}

On this model and these 200 tasks, the best setup we found is the full stack with the tool list written more compactly: JSON-text calls, a harness-written state summary, a gated stop rule, a status card, and the same tools with each group's shared description stated once. All of these are greedy and need no training data or training compute. The shorter list added 6.5 and 8.0 points in its two paired runs and cut context per request by about 30\%. The minimal recipe, which leaves out the state summary, is a smaller fallback. Before any fine-tune, screen on task-level checks rather than validation loss. Even the best run, the full stack with the shorter list, finishes 88 of 200, and a repeat of the same harness does not pass the same tasks every time. On this benchmark, this model still fails most multi-step tasks."""

LIMITATIONS = r"""\section{Limitations}

\begin{itemize}\setlength{\itemsep}{1pt}
  \item \textbf{One model, one split, one device.} All multi-turn results use Gemma-4-E2B-it on the 200-task base split, on one laptop. The stop threshold, the card's fields, and the training filters were chosen on these same tasks. Other models and harder splits are future work.
  \item \textbf{The harness and the tests were specified on these tasks.} The full stack was put together after earlier results on the same 200 tasks were already known. The family of eight tests was specified after that, and no test was preregistered. The full stack's gain over JSON text does not survive the eight-test correction.
  \item \textbf{Repeatability is not generalization.} Repeated runs of one harness differ on individual tasks. The compact-list gain repeated across two pairs on the same 200 tasks. That shows it was not one lucky run; it does not show that it holds on new tasks.
  \item \textbf{The shorter list.} It ran as two pairs outside the eight-test family. The second pair and the sign test that combines them came after the first pair missed the 0.05 level. Its failure labels were not split out, and its mechanism was not isolated. A held-out check is the first confirmatory run in Chapter~\ref{sec:conclusion}.
  \item \textbf{Serving differed between some runs.} The native API baseline used eight server slots and the other runs four, and the earlier runs used an older server build than the full-stack and compact-list runs. Neither difference was tested for its effect on greedy outputs.
  \item \textbf{The training controls are narrow.} The null test ran for 39 updates; the evaluated rebuilt-mix checkpoint came from iteration 22{,}000 of a much longer run. The two retrains each changed one share of the data. These runs do not separate training data from training length, and they do not rule out other causes.
  \item \textbf{The screening gate was built from the same split.} The 20 gate tasks were picked using earlier per-task results from the same split, and the bar is the score of the setup the gate is screening. It is designed as a fast screen, with confirmation left to full runs.
\end{itemize}"""

CONCLUSION = (
    "The best setup is the training-free harness with the same tool list written more compactly. "
    "It scored 42.0\\% (84 of 200) and 44.0\\% (88 of 200) in two paired runs. In the same pairs, "
    "the original list scored 35.5\\% and 36.0\\% (71 and 72 of 200). All of these runs used the "
    "same 200 tasks, so the gain repeated but has not been tested on new tasks. The compact list "
    "reduced mean context per request by approximately 30\\%. Reduced context burden is a "
    "plausible explanation, but these experiments do not isolate the mechanism. Most failed "
    "tasks end with the backend in the wrong state or a needed result missing, and the failure "
    "analysis motivated the harness. Without the compact list, the first full-stack run scored "
    "38.5\\% against 30.5\\% for JSON text. That gain does not survive correction for multiple "
    "comparisons. The two forced-valid decoding methods we tested made results worse, and "
    "fine-tuning did not beat the untuned model. For a local agent, these harness changes also "
    "cost far less than voting or training.\n"
)

REPRO = r"""\section{Reproducibility}

Hardware: MacBook Pro, Apple M4~Pro, 48~GB, with Metal acceleration (Apple's GPU interface). Temperature is 0 except for voting (0.7, $N{=}5$, one seed). The native API baseline used eight server slots; the runs below used four (Table~\ref{tab:repro}). BFCL v4 is read through the \texttt{bfcl-eval} package; \texttt{pyproject.toml} requires version 2025.8.6.2 or later, there is no lockfile, and the exact version per run was not logged. Failure labels come from the checker messages in the \texttt{eval\_errors} table of the benchmark database; that database and the run logs below are not in the repository, so the counts here are copied from the paper's tables. Fig. 2 and 5 are regenerated from these counts by \texttt{paper/figures/src/make\_refocus\_figs.py}. Fig. 4 is regenerated from the same scoreboard counts by \texttt{thesis/figures/src/make\_ladder.py}. The 91.9\% interval is the Wilson formula on 772/840 in the single-call log.

\begin{table}[H]
\centering
\footnotesize
\renewcommand{\arraystretch}{1.15}
\caption{Runs and logs of the full-stack and shorter-list tests.}
\label{tab:repro}
\begin{tabular}{@{}rlr>{\RaggedRight\arraybackslash}p{5.4cm}@{}}
\toprule
Run & What & scores.id & Log \\
\midrule
78 & Full stack, first run & 51 & \texttt{sft/v13/paper\_evals/runs/b\_fullstack\_mtb200.log} \\
79 & Single-call categories & & \texttt{sft/v13/paper\_evals/runs/a\_single\_turn\_gemma.log} \\
80 & 10-task original list, 5/10 & 59 & \texttt{R81\_slim/smoke\_off.log} \\
81 & 10-task shorter list, 8/10 & 60 & \texttt{R81\_slim/smoke\_on.log} \\
82 & 200-task original list & 61 & \texttt{R81\_slim/full\_off.log} \\
83 & 200-task shorter list & 62 & \texttt{R81\_slim/full\_on.log} \\
84 & 200-task shorter list, second pair & 63 & \texttt{R82\_confirm/full\_on.log} \\
85 & 200-task original list, second pair & 64 & \texttt{R82\_confirm/full\_off.log} \\
\bottomrule
\end{tabular}
\end{table}

Earlier runs used llama.cpp build 8111 (commit \texttt{11c325c6e}). Runs 78 and 80--85 used commit \texttt{d344123fe}. The full stack's harness was run at commit \texttt{344ce45} and merged as \texttt{83f35ee}. Server flags for those runs: \texttt{llama-server -{}-jinja -fa on -ngl 99 -c 65536 -{}-temp 0.0 -np 4 -cb -ub 1024}. Opening-prompt lengths are medians from \texttt{R81\_slim/*.metrics.jsonl}.

\section{Training Recipe}
\label{app:recipe}

This section collects the training settings that the repository records, from \texttt{sft/lora\_config.yaml}, \texttt{sft/config.py}, \texttt{sft/train\_mlx.py}, \texttt{sft/README.md}, and the run notes. Settings marked \emph{not recorded} could not be recovered from those files.

\textbf{Base model.} \texttt{google/gemma-4-e2b-it} from Hugging Face, converted to full-precision MLX weights with \texttt{mlx\_lm.convert} before training. The model revision was not pinned. The served base model is the Q4\_K\_M GGUF of the same model (\texttt{bartowski/google\_gemma-4-E2B-it-GGUF}).

\textbf{Data.} Rows are chat-format \texttt{messages}. The system message is the JSON-text prompt with the tool list, built by the same function the JSON-text evaluation uses (\texttt{benchmark/iffc.py}), so the model sees that prompt in training and at test time. The public sources are \texttt{Team-ACE/ToolACE}, \texttt{Salesforce/xlam-function-calling-60k} (a 25{,}000-row sample), and, in the rebuilt mix, \texttt{Salesforce/APIGen-MT-5k}; dataset revisions were not pinned. The early mixes pool all of ToolACE with the xLAM sample and split it about 90/5/5 into train, validation, and test with seed 42. Conversations are unrolled so that every assistant turn is a training target.

\textbf{Template and masking.} mlx-lm renders each row with the tokenizer's chat template, and \texttt{-{}-mask-prompt} restricts the loss to the assistant response. The exact template and tokenizer files used were not recorded.

\textbf{Common LoRA settings} (\texttt{sft/lora\_config.yaml}): \texttt{mlx\_lm.lora}, seed 42; rank 16, scale 32.0, dropout 0.0; adapters on the query and value projections (\texttt{self\_attn.q\_proj}, \texttt{self\_attn.v\_proj}) of 16 layers (\texttt{num\_layers} 16); learning rate $1.0\times10^{-5}$; batch size 1 with gradient accumulation 4, so one optimizer update per four rows; maximum sequence length 4{,}096 tokens; gradient checkpointing on; validation every 200 iterations on 25 batches; a checkpoint every 100 iterations. The optimizer and learning-rate schedule are not set in these files, so they are mlx-lm's defaults for the installed version, which was not recorded. An iteration is one row; with accumulation 4, the 39 updates of the null test correspond to its 156 rows. Later runs overrode some of these values on the command line; for example, an unscored v10 run used a maximum sequence length of 8{,}192 and seed 53.

\textbf{Checkpoint selection.} For SFT-v1 to SFT-v6 and the rebuilt mix, the checkpoint with the lowest validation loss is evaluated.

\textbf{Merge, quantize, serve.} \texttt{mlx\_lm.fuse} merges the adapter; llama.cpp's \texttt{convert\_hf\_to\_gguf.py} converts the fused model and \texttt{llama-quantize} makes the Q4\_K\_M file, which \texttt{llama-server} serves with the flags above. The llama.cpp build used for each conversion was not recorded.

\textbf{DPO.} One full run (Run 51) with $\beta{=}0.2$ on LoRA, evaluated with JSON text. An earlier smoke run used $\beta{=}0.2$, learning rate $1\times10^{-6}$, rank 16 and scale 32, for 100 iterations. For the full run, the learning rate, number of updates, and how preferred and rejected responses were paired are not recorded.

\begin{table}[H]
\centering
\footnotesize
\renewcommand{\arraystretch}{1.15}
\caption{Fine-tuned variants, with the settings that differ from the common ones. An iteration is one row; under the common settings, four rows make one update.}
\label{tab:recipe}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.0cm}>{\raggedright\arraybackslash}p{6.9cm}>{\raggedright\arraybackslash}p{3.3cm}@{}}
\toprule
Variant & Data and training & Evaluated checkpoint, result \\
\midrule
SFT-v1, SFT-v2 & First mixes: ToolACE and the xLAM sample; v2 drops mid-dialogue empty targets & v1 iteration 1{,}000; v2 iteration 2{,}000 (val.\ 0.211); 1/200 each \\
SFT-v2 strip & SFT-v2 checkpoint; trailing empty answers deleted at serving time & 4/200 \\
SFT-v4 (best) & Stop targets after a tool result removed, recovery examples added & Iteration 1{,}600 (val.\ 0.196); 30/200 \\
SFT-v5 & v4 plus first-turn and state filters & Iteration 1{,}800 (val.\ 0.147); 23/200 \\
SFT-v6 & v4 plus the state filter only & Iteration 1{,}800 (val.\ 0.142); 26/200 \\
Rebuilt mix (SFT-v11) & 30{,}000 train / 1{,}500 validation rows; 45/40/10/5\% simulated / APIGen-MT / ToolACE / xLAM; 32{,}000 iterations planned, stopped at 26{,}500 & Iteration 22{,}000 (val.\ 0.030); 2/20 \\
Layout continue-train & SFT-v11 trained 2{,}500 more iterations on tool lists laid out like the test & 2/20 \\
Null test (v13) & 156 of the untuned model's own logged answers, one pass, 39 updates & 15/20 \\
Self-distillation (v13-sd) & 440 rows of the model's checker-approved answers to simulated tasks & 8/20 \\
v13-sdb, v13-lk & Self-distillation with the stop share capped, or with simulated look-up episodes added & 14/20; 12/20 \\
DPO & See text & 53/200 \\
\bottomrule
\end{tabular}
\end{table}

For the rebuilt mix, the learning rate, gradient accumulation, maximum sequence length, and LoRA settings are not recorded separately, so whether it used the common settings above is not confirmed. For the v13 runs, the per-run settings and the checkpoint rule are not recorded. Per-run adapter configuration files were not kept in the repository.
"""

TRIM = r"""\section{Shorter Tool List Checks}
\label{app:trim}

A 10-task check scored 5 of 10 with the original list and 8 of 10 with the shorter list. The recorded calls differed from the first full-stack run on five tasks, so the 200-task comparison used a fresh rerun of the original list rather than that first run (77 of 200).

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


def abbreviate_figure_refs(text):
    """Body cross-references use the sample's 'Fig. N', including a bare figure number."""
    text = text.replace("Figure~\\ref", "Fig.~\\ref")
    text = text.replace("Figures 2 and 5", "Fig. 2 and 5")
    text = text.replace("Figure 4 uses", "Fig. 4 uses")
    return text


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
    parts["ch1"] = must_replace(
        parts["ch1"],
        "Two of these levers, a state summary and a status card, reach the model as text in its context, so the harness acts partly through the context; both are computed from executed results. ",
        "",
        "cut context sentence",
    )
    parts["ch1"] = must_replace(
        parts["ch1"],
        "it completes 27.0\\% of 200 multi-turn tasks.",
        "it completes 27.0\\% of 200 multi-turn tasks. The best setup we found, a harness plus a compact tool list, reaches 42.0\\% and 44.0\\% (84 and 88 of 200) in two paired runs on the same tasks.",
        "ch1 headline",
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
    parts["ch3"] = replace_paragraph_starting(
        parts["ch3"],
        "Each failed task gets one primary label",
        ERROR_LABELS,
        "shorter error labels",
    )
    parts["ch3"] = replace_paragraph_starting(
        parts["ch3"],
        "We train LoRA adapters",
        r"We train LoRA adapters on Gemma-4-E2B-it with MLX~\cite{mlx}, merge and quantize to Q4\_K\_M, and evaluate with JSON text. The \emph{rebuilt mix} has 30{,}000 rows, about 45\% newly simulated episodes plus cleaned public data; earlier mixes use ToolACE and sampled xLAM data. One DPO model is trained too. The \emph{gate} is 20 tasks from the same split, run with the status card; the bar is 17/20. The training settings the repository records are in Appendix~\\ref{app:recipe}.",
        "shorter training paragraph",
    )

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
        " The compact tool list is omitted.",
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
        ", suggesting that the summary mainly keeps the model calling.",
        "The labels are consistent with the summary mostly reducing turns with no call; they do not show why.",
        "state summary calling",
    )
    parts["ch5"] = must_replace(
        parts["ch5"],
        r"Adding the status card gives the minimal recipe, 71/200 (+5.0\pp over JSON text, $p=0.16$).",
        r"Adding the status card gives the minimal recipe, 71/200 (35.5\%, +5.0\pp over JSON text, $p=0.16$).",
        "minimal recipe percent",
    )
    parts["ch5"] = must_replace(
        parts["ch5"],
        "rise from 60 to 71, so the error labels suggest the card acts mainly as an act-versus-abstain control.",
        "rise from 60 to 71. The labels are consistent with the card acting mostly on whether the model calls at all, as it was designed to, but they do not isolate that mechanism.",
        "card acts",
    )
    parts["ch5"] = must_replace(
        parts["ch5"],
        r"Combining all four greedy changes scores 77/200 (38.5\%, interval 32.0--45.4\%). Paired against JSON text, it solves 30 tasks that JSON text misses and misses 14 that JSON text solves ($p=0.023$ uncorrected; not significant after correction). Against the minimal recipe, the state summary alone and the vote of five, the differences are within noise (Table~\ref{tab:mcnemar}). The configuration was assembled after we had seen earlier results on these tasks, so its lead is subject to tuning leakage (Chapter~\ref{sec:discussion}). Its harness code is in pull request \#90 (run at commit \texttt{344ce45}, merged as \texttt{83f35ee}).",
        r"Combining all four greedy changes scores 77/200 (38.5\%, interval 32.0--45.4\%). Paired against JSON text, it solves 30 tasks that JSON text misses and misses 14 that JSON text solves (Table~\ref{tab:mcnemar}). Against the minimal recipe, the state summary alone, and the vote of five, the paired tests do not separate the full stack from them, which does not show that they are equal. Because the configuration was assembled using earlier results on these tasks, its lead is reported as in-sample (Chapter~\ref{sec:discussion}). The next section keeps this harness and changes only how the tool list is written.",
        "full stack prose",
    )
    parts["ch5"] = replace_section(parts["ch5"], "Shorter Tool List, Same Tools", SHORTER)
    parts["ch5"] = must_replace(
        parts["ch5"],
        r"Voting over five samples on top of the state summary scores 73/200, four tasks fewer",
        r"The compact list stays greedy. Voting over five samples on top of the state summary scores 73/200 (36.5\%), four tasks fewer",
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
    parts["ch5"] = must_replace(
        parts["ch5"],
        r"Top bar: the full stack with a shorter tool list, the better of its two runs (88/200; the other scored 84/200, and two reruns of the original list scored 71/200 and 72/200; Table~\ref{tab:trim}).",
        r"Above the thin line: the four runs of the compact-list test. The top two bars are the full stack with a compact tool list in its two pairs; the two below them are the original list rerun in the same pairs (Table~\ref{tab:trim}).",
        "shorten ladder caption",
    )
    parts["ch6"] = must_replace(
        parts["ch6"],
        r"It scored 15/20 (interval 53--89\%), within noise of the bar (17/20; 64--95\%), while the rebuilt mix's 2/20 (3--30\%) is far outside it.",
        r"It scored 15 of 20, two below the bar of 17 of 20. Their intervals overlap, which does not make them equal, while the rebuilt mix's 2 of 20 is far below the bar. The exact ranges are in Appendix~\ref{app:train}.",
        "null ranges in words",
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
        parts["ch6"], "section", "Two Targeted Retrains Did Not Restore the Score"
    )
    cause = cause.replace("\\section{", "\\subsection{", 1)
    cause = must_replace(
        cause,
        r"Neither passed, so the data are \emph{consistent with}, but do not establish, a shift in the decision mix (stop versus continue, look up versus act) as the cause.",
        r"Neither passed. Correcting the shift in the decision mix (stop versus continue, look up versus act) in these two ways was not enough; whether a different rebalancing would work is untested.",
        "retrain sole cause",
    )
    parts["ch6"] = must_replace(
        parts["ch6"],
        "so its data are the leading suspect but not the only one.\n",
        "so these runs cannot say whether its data or its training length caused the collapse.\n\n" + CH6_POINTER + "\n",
        "gate token",
    )
    parts["ch6"] = must_replace(
        parts["ch6"],
        "scored 2/20 on the gate (Figure~\\ref{fig:gate}); validation loss did not predict the collapse.",
        "scored 2/20 on the gate (Figure~\\ref{fig:gate}).\n\n@@GATE@@\n\nValidation loss did not predict the collapse.",
        "gate beside citation",
    )
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
    parts["appA"], finetune = cut_heading_block(parts["appA"], "section", "Fine-Tuning Setup")
    finetune_body = finetune.split("\n\n", 1)[1].strip()
    parts["appB_train"] = (
        finetune_body
        + " The training recipe is in Appendix~\\ref{app:recipe}."
        + "\n\nThe null test scored 15 of 20 (interval 53--89\\%); the bar is 17 of 20 (64--95\\%). The intervals overlap, which does not show that the two are equal. "
        + "The rebuilt mix scored 2 of 20 (3--30\\%), far below that bar.\n\n"
        + parts["appB_train"].lstrip()
    )
    for old, new, label in (
        (r"Full stack, first run (\texttt{iffc-state-ooda}, PR \#90)", "Full stack, first run", "drop stack log name"),
        (r"Vote of five on state summary (BoN $N{=}5$ on IFFC+STATE)", "Vote of five on the state summary", "drop vote log name"),
        (r"State summary (IFFC+STATE)", "State summary on JSON text", "drop state log name"),
        (r"JSON text (IFFC)", "JSON text", "drop JSON log name"),
        (r"Guided template on native API (GST+NFC)", "Guided template on the native API", "drop template log name"),
        (r"Native API (NFC)", "Native API", "drop native log name"),
        (r"Tool-name automaton (PTC-A)", "Tool-name automaton", "drop automaton log name"),
        (r"Minimal recipe: card on stop on repeats (``R42'')", "Minimal recipe: card on stop on repeats", "drop R42"),
        (
            "The ID is the internal log id, with the internal name in parentheses. ",
            "The ID is the internal log id. ",
            "scoreboard caption",
        ),
        (
            "Recorded calls differed from the first full-stack run on tasks 55, 69, 26, 179, and 59. The plan's gate required 0 of 10 to differ, so the gate failed. ",
            "The recorded calls differed from the first full-stack run on five tasks, so the 200-task comparison used a fresh rerun. ",
            "drop task ids",
        ),
    ):
        parts["appC_board"] = must_replace(parts["appC_board"], old, new, label)
    parts["appC_repro"] = REPRO
    parts["glossary"] = edit_glossary(parts["glossary"])
    parts["appC_trim"] = TRIM
    parts["appC_board"] = board_legible(parts["appC_board"])
    parts["appC_board"] = parts["appC_board"].replace(
        "\\begin{table}[H]\n",
        "\\begin{table}[H]\n\\renewcommand{\\arraystretch}{1.0}\n",
        1,
    )
    return parts


def abstract_word_count(text):
    t = text.replace("\\%", "%")
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    t = t.replace("{", " ").replace("}", " ").replace("$", " ").replace("~", " ")
    t = re.sub(r"[^A-Za-z0-9.%'-]+", " ", t)
    words = [w for w in t.split() if w]
    return len(words)


def check_abstract():
    n = abstract_word_count(ABSTRACT)
    if not 150 <= n <= 250:
        raise SystemExit(f"abstract word count {n} is outside 150-250")
    for s in ABSTRACT_SOURCES:
        if s not in paper_text:
            raise SystemExit("abstract source missing from paper: " + s)


# Sentences that come from paper/main.tex and need the advisor-review wording. Each is
# matched once against the final text (after the Fig. and compact-list renames).
FINAL_EDITS = [
    (
        "all from ToolACE, so the model learned to stop after its first tool result while BFCL needs further calls.",
        "all from ToolACE, while BFCL needs further calls. The low scores are consistent with the model learning to stop after a tool result, but these runs do not isolate that cause.",
    ),
    (
        r"Chapter~\ref{sec:fail} shows it is mostly the latter: tasks fail because a needed tool never ran or the backend ended in the wrong state.",
        r"The failure labels in Chapter~\ref{sec:fail} suggest it is mostly the latter: most failed tasks end with a required result missing or the backend in the wrong state, and forcing valid syntax does not help.",
    ),
    (
        r"Chapter~\ref{sec:fail} tests whether forcing validity helps in a multi-turn loop.",
        r"Chapter~\ref{sec:fail} tests two such methods in a multi-turn loop.",
    ),
    (
        "Harness control mainly removes empty turns: the full stack has 20,",
        "With harness control, empty turns fall: the full stack has 20,",
    ),
    (
        "Because parse failures can only appear as empty turns, most of what remains is not a formatting problem.",
        r"Parse failures have no label of their own. They can appear as empty turns or, after earlier calls in the same turn, under another label (Chapter~\ref{sec:setup}). Where format errors were counted, in the compact-list runs, they were rare: at most 1 per run (Appendix~\ref{app:trim}). Those counts are not reported for the other conditions.",
    ),
    (
        "Harness control mostly removes empty turns (grey); wrong final states (blue) remain.",
        "With harness control, empty turns (grey) fall; wrong final states (blue) remain.",
    ),
    (
        "guaranteeing valid output should help. It does the opposite.",
        "guaranteeing valid output should help. In this setup, the two constrained-decoding implementations we tested performed worse.",
    ),
    (
        r"Our two-pass program, which forces each call into valid syntax, approximates rather than reproduces full grammar engines~\cite{dong2026xgrammar2}, but the effect is large and in the wrong direction: a valid call is not a correct next step.",
        r"Both are our own implementations on stock \texttt{llama-server}; the two-pass program approximates rather than reproduces full grammar engines~\cite{dong2026xgrammar2}. The result is about these two implementations in this loop, not about grammar-constrained decoding in general. A valid call is not necessarily a correct next step.",
    ),
    (
        "A greedy harness matches it at a fraction of the time, which matters on a local device.",
        "The paired test does not separate the two, which is not the same as showing that they are equal, and the greedy full stack needs a fraction of the generation time, which matters on a local device.",
    ),
    (
        r"The null test rules out those pipeline steps as the cause at small update budgets. The rebuilt mix differs from it in both content and training length (22{,}000 steps for the evaluated checkpoint), so",
        r"The null test suggests that those pipeline steps do not by themselves cause a collapse after a short run of 39 updates. It does not test them at the rebuilt mix's training length: the evaluated checkpoint came from iteration 22{,}000 of a run on 30{,}000 rows, far longer. The rebuilt mix differs from the null test in both content and training length, so",
    ),
    (
        "so stop share alone does not explain the collapse.",
        "so stop share alone does not appear to explain the collapse.",
    ),
]

# Short entries for the lists of tables and figures, keyed by float label.
SHORT_CAPTIONS = {
    "fig:vignette": "The multi-turn tool-calling loop",
    "fig:fail": "Primary error labels of failed tasks by condition",
    "fig:compact": "The original and compact tool lists",
    "fig:ladder": "Multi-turn accuracy by condition",
    "fig:gate": "Twenty-task gate results",
    "tab:trim": "Compact tool list against the original list",
    "tab:mcnemar": "The eight-test McNemar family",
    "tab:mcnemar-x": "McNemar tests outside the family",
    "tab:failfull": "Primary error label per failed task",
    "tab:board": "Every scored condition",
    "tab:repro": "Runs and logs of the compact-list test",
    "tab:recipe": "Fine-tuned variants",
}


def apply_final_edits(tex):
    for old, new in FINAL_EDITS:
        if tex.count(old) != 1:
            raise SystemExit("final edit not found once: " + old[:70])
        tex = tex.replace(old, new)
    return tex


def add_short_captions(tex):
    labels = set(re.findall(r"\\label\{((?:fig|tab):[^}]+)\}", tex))
    missing = labels - set(SHORT_CAPTIONS) - {"tab:runs"}  # tab:runs is a second label on tab:board
    if missing:
        raise SystemExit("float without a short caption: " + ", ".join(sorted(missing)))
    for label, short in SHORT_CAPTIONS.items():
        i = tex.find("\\label{" + label + "}")
        if i < 0:
            raise SystemExit("short caption label missing: " + label)
        start = max(tex.rfind("\\begin{figure", 0, i), tex.rfind("\\begin{table", 0, i))
        j = tex.rfind("\\caption{", start, i)
        if start < 0 or j < 0:
            j = tex.find("\\caption{", i)
            end = min(x for x in (tex.find("\\end{figure", i), tex.find("\\end{table", i)) if x > 0)
            if j < 0 or j > end:
                raise SystemExit("caption not found for " + label)
        tex = tex[:j] + "\\caption[" + short + "]{" + tex[j + len("\\caption{"):]
    return tex


def board_legible(board):
    """Table 5: footnote size instead of script size, and its notes outside the float."""
    board = must_replace(board, "\\scriptsize\n", "\\footnotesize\n", "board size")
    m = re.search(r"\\parbox\{\\linewidth\}\{\\RaggedRight\\scriptsize (.*?)\}\n\\end\{table\}", board, re.S)
    if not m:
        raise SystemExit("board notes not found")
    notes = m.group(1)
    board = board[: m.start()] + "\\end{table}\n\n{\\footnotesize\\RaggedRight\\noindent " + notes + "\\par}" + board[m.end():]
    return board


def rename_shorter_list(text):
    """The thesis name is compact list. Log paths such as R81_slim stay."""
    for old, new in (
        ("Shorter Tool List", "Compact Tool List"),
        ("Shorter tool list", "Compact tool list"),
        ("shorter tool list", "compact tool list"),
        ("shorter-list", "compact-list"),
        ("Shorter only", "Compact only"),
        ("Shorter list", "Compact list"),
        ("shorter list", "compact list"),
    ):
        text = text.replace(old, new)
    return text


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
        "set in the plan",
        "acts partly through the context",
    ):
        if banned in pre:
            raise SystemExit("banned in main text: " + banned)
    for banned in (
        "set in the plan",
        "plan's gate",
        "tasks 55, 69",
        "PTC-A",
        "R42",
        "iffc-state-ooda",
        "(IFFC)",
        "(NFC)",
    ):
        if banned in tex:
            raise SystemExit("banned in thesis: " + banned)
    if pre.count("77 of 200, then 71 of 200 and 72 of 200") != 1:
        raise SystemExit("repeat-run triple should be stated once")
    if pre.count("33 tasks") != 1:
        raise SystemExit("33 tasks should be stated once in the main text")
    if pre.count("p=0.023") != 1:
        raise SystemExit("p=0.023 should be stated once in the main text")
    if r"\section{The Best Setup: A Compact Tool List}" not in tex:
        raise SystemExit("best-setup section missing")
    if re.search(r"[Ss]horter(?: tool|-) list|[Ss]horter only", tex):
        raise SystemExit("shorter list name remains")
    for banned in (
        "helps because",
        "statistically level",
        "statistically indistinguishable",
        "within run-to-run variation",
        "ruled out",
        "rules out",
        "Sole Cause",
        "sole cause",
        "treat those two settings as equivalent",
        "can only appear as empty turns",
        "counts as no call",
        "Forcing Valid Syntax Hurts",
        "same runs as the",
        "leading cause",
    ):
        if banned in tex:
            raise SystemExit("overclaiming wording came back: " + banned)
    if "DRAFT" not in tex:
        raise SystemExit("draft marking missing")
    if "fig:compact" not in tex:
        raise SystemExit("compact-list figure missing")
    if r"\section{Shorter Tool List, Same Tools}" in tex:
        raise SystemExit("old shorter-list heading still present")
    for banned in (
        "which is why",
        "same kind of lever",
        "best setup for an on-device agent",
        "That is why the harness",
        "The best harness setup",
    ):
        if banned in tex:
            raise SystemExit("removed wording came back: " + banned)
    if "sign test, $p=0.033$" not in tex:
        raise SystemExit("sign test not named next to p=0.033")
    if "McNemar $p=0.033$" not in tex:
        raise SystemExit("McNemar not named next to p=0.033")
    if "Figure~\\ref" in tex or "Figures " in tex or re.search(r"\bFigure \d", tex):
        raise SystemExit("figure cross-reference still says Figure")
    if len(re.findall(r"\\caption\{", tex)) != 0:
        raise SystemExit("every float needs a short caption for the lists")


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
    parts["ch5"] = move_float_to_token(parts["ch5"], "fig:ladder", "@@LADDER@@")
    parts["ch6"] = move_float_to_token(parts["ch6"], "fig:gate", "@@GATE@@")
    parts["appC_repro"] = allow_path_breaks(parts["appC_repro"])

    tpl = (here / "main.tpl.tex").read_text()
    for k, v in parts.items():
        tpl = tpl.replace("%%" + k + "%%", v)
    tpl = abbreviate_figure_refs(tpl)
    tpl = rename_shorter_list(tpl)
    tpl = apply_final_edits(tpl)
    tpl = add_short_captions(tpl)
    if "%%" in re.sub(r"%%%+", "", tpl):
        raise SystemExit("unfilled slot")
    check_output(tpl)
    (here / "main.tex").write_text(tpl)
    print("wrote", here / "main.tex")


if __name__ == "__main__":
    main()
