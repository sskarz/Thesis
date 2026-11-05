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

ABSTRACT = (
    "Small language models can run on a laptop and call tools, but they often fail "
    "when a later call depends on an earlier one. This thesis studies Gemma-4-E2B-it, a model "
    "with about two billion effective parameters, on 200 multi-turn tasks from the Berkeley "
    "Function Calling Leaderboard. A task is correct only when a simulated backend ends in the "
    "right state, such as a car's engine being on.\n"
    "\n"
    "The best setup leaves the weights fixed. A harness, the program around the model, writes "
    "calls as JSON text, records the current state, stops repeated calls, and shows a short card "
    "advising the next move. On that harness, a compact tool list (the same tools, with each "
    "group's shared description stated once) gave the best result: 88 of 200 (44.0\\%) in one "
    "run and 84 of 200 (42.0\\%) in another. The original list scored 71 and 72 of 200 in the "
    "matching runs, so the gain held both times.\n"
    "\n"
    "The model mainly fails at choosing the next step, not at writing a call, and forcing valid "
    "syntax made results worse. The compact list makes the prompt sent with each request about "
    "30\\% shorter, which likely helps by keeping the state the model needs in view. "
    "Without the compact list, the harness raised accuracy from 30.5\\% for JSON text alone to "
    "38.5\\% in its first run, a gain that is not significant after correcting for multiple "
    "comparisons. Fine-tuning did not improve accuracy: the best fine-tuned model scored "
    "15.0\\%, below the untuned model's 30.5\\%.\n"
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
    "untested guess",
    "not significant after correcting for multiple comparisons",
    "15.0\\%",
    "about 30\\%",
]

CONTRIBUTIONS = r"""

\section{Contributions}

This thesis makes three contributions.

\begin{enumerate}
  \item \textbf{A harness plus a compact tool list.} The weights stay fixed. The program around the model records the current state, stops repeated calls, and shows a short card advising the next move. On that harness, writing the same tool list more compactly helps further, and that gain replicated (Chapter~\ref{sec:helps}).
  \item \textbf{Where the failures are.} The model fails at choosing the next step. A needed call never happens, or the backend ends in the wrong state. The failure analysis is why we built the harness. The compact list makes the prompt about 30\% shorter, which is the likely reason it helps (Chapter~\ref{sec:helps}). Forcing each call into valid syntax makes results worse (Chapter~\ref{sec:fail}).
  \item \textbf{Fine-tuning did not improve the model.} The results show that fine-tuning did not raise accuracy over the untuned model. The best supervised fine-tune scored 15.0\%, below the untuned model's 30.5\% with the same JSON-text calls, and preference tuning scored 26.5\%, statistically level with it. A fine-tune on the untuned model's own logged answers, which should change nothing if the training steps are sound, stayed close to the untuned model. The rebuilt training data is the leading suspect, with training length the other candidate (Chapter~\ref{sec:train}).
\end{enumerate}"""

CH2_INTRO = (
    "This chapter names the benchmark and the ideas the experiments rely on: "
    "how a call is written, what it means to force that writing to be valid, "
    "what the program around the model can do, and how training a tool-calling model is usually done.\n\n"
)

CH2_CLOSER = (
    "\n\nThe following chapters use only these ideas. They ask where a small model fails, "
    "whether a training-free harness with a compact tool list can help, and whether fine-tuning can.\n"
)

MODEL_SERVING = r"""\section{Model and Serving}

All results use Gemma-4-E2B-it, quantized with Q4\_K\_M (a 4-bit weight format of llama.cpp), served by \texttt{llama-server} on a laptop. Decoding is greedy (temperature 0) except for majority voting. Greedy runs are not identical from one run to the next (Chapter~\ref{sec:helps}). The machine, the server build, and the flags are in Appendix~\ref{app:runs}. Most runs used four parallel server slots, and the native API baseline used eight. We treat those two settings as equivalent; their greedy outputs were not compared directly."""

HARNESS = r"""\section{Harness Interventions}

All leave the weights untouched. A task is correct only when the multi-turn checker returns valid. The state summary targets lost state, the stop rule targets repetition, and the status card targets whether to act or stop.
\begin{itemize}\setlength{\itemsep}{1pt}
  \item \textbf{JSON text} instead of the native API. Following Appendix~B of~\cite{taheri2026iffc}, the tool list is left out of the API request, and the system prompt asks for a JSON list of \texttt{\{function\_name, parameters\}} objects in ordinary text. The harness parses the calls and maps them to the same BFCL function strings as native calling. Tool results return as user messages.
  \item \textbf{State summary.} After each executed call, the harness writes a short \texttt{[STATE]} snapshot of facts from real results (current directory, known files, last calls) into the conversation. The last two turns stay verbatim, and older raw outputs are dropped. It is not a model-written summary.
  \item \textbf{Stop on repeats.} After a tool result, the harness ends the turn if the model proposes the same calls as in its previous step, or if every proposed call was already executed in this user request and at least six non-empty tool steps have occurred in it. Otherwise it executes the calls. The six-step count resets with each new user request. An \emph{ungated} variant drops that minimum.
  \item \textbf{Status card.} A four-line card is prefixed to a working copy of the latest user message; the stored conversation is unchanged. Its fields are \texttt{goal} (the latest user request, truncated), \texttt{slots} (up to eight key--value facts from executed arguments and results), \texttt{last} (the last executed call and its outcome), and \texttt{next}, which advises \texttt{refuse} if no tools are available, \texttt{ask} if the last result signals a missing or invalid argument, \texttt{final} (finish) if the user's message is a closing ``thanks'' or ``done'', and \texttt{call} otherwise. The advice does not constrain decoding, and the card never sees reference answers. It is a single-pass card, not the multi-pass OODA-Tool of~\cite{guo2026ooda}. JSON text, stop on repeats, and the card, together, are the \emph{minimal recipe}.
  \item \textbf{Full stack.} JSON text, the state summary, stop on repeats, and the status card, decoded greedily. The summary is written into the stored conversation after each executed call, and the card is prefixed to a working copy of the latest user message. On its first run this is the harness we compare with JSON text.
  \item \textbf{Compact tool list.} On the full stack, every tool stays in the prompt. The harness states each tool group's shared description once, instead of repeating it, and prints each schema without indentation. It does not remove or retrieve tools. The model, the four pieces above, and the tasks stay the same.
  \item \textbf{Vote of five.} On top of the state summary, at each step the harness samples five candidate calls at temperature 0.7, executes the most common \texttt{(function\_name, arguments)} pair once, and continues. This is the compute-heavy comparison point, not a learned verifier.
  \item \textbf{Forced-valid decoding.} \emph{Constrained decoding} makes two requests per call: the first restricts output to an empty list or a list of valid tool names, and the second restricts the arguments to each tool's JSON schema. No native tool grammar is active; this approximates the tag-triggered dispatch of XGrammar-2~\cite{dong2026xgrammar2} on a stock llama.cpp server. A \emph{tool-name automaton}, inspired by~\cite{yu2026ptc}, restricts only the tool name to valid names while the arguments stay free. It runs on JSON text without stop on repeats, and its accuracy is not comparable to the F1 that work reports.
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

STATISTICS = r"""\section{Statistics}

Each reported accuracy comes with a Wilson 95\% interval~\cite{wilson1927}, a range that stays reliable when few tasks are correct. When two conditions are compared on the same 200 tasks, we use an exact two-sided McNemar test~\cite{mcnemar1947}. It counts only the tasks the two conditions decide differently. The $p$-value is the chance of a gap at least that large if the conditions were equal.

Eight comparisons were corrected together so that the chance of any false alarm among them stays at 5\%. The compact tool list was a separate test, run twice in opposite order. Which comparisons are in the family, and how the two pairs were combined, is in Appendix~\ref{app:stats}."""

STATE_MISMATCH = (
    "An \\emph{under-call} is a task where a required tool never ran. "
    "The loop in Figure~\\ref{fig:vignette} is one. The model locks the doors and then tries to start the engine. "
    "Both of those calls are valid. The backend says the brake pedal has to be pressed first. "
    "The model never presses the brake, so that required call never ran, the engine stays off, and the check fails. "
    "A \\emph{state mismatch} is the other kind: the tools did run, and the backend's final state still differs from the reference.\n\n"
)

JSON_CHANNEL = r"""\section{JSON Text over the Native API}

JSON text scores 61/200 against 54/200 on the native API, a lead of 3.5 percentage points that is not statistically significant ($p=0.32$). A seven-step reasoning template on the native API also scored 61/200, tying JSON text: 19 tasks swap each way, and state mismatches rose to 79. The gap reflects the shape of the prompt, not only the channel that carries the call, so we do not claim the channel switch alone as a gain. Later steps are built on JSON text because the harness can parse and inspect those calls directly. The harness levers were evaluated on JSON text, the stronger baseline."""

SHORTER = r"""\section{The Best Setup: A Compact Tool List}

The best setup is the full stack with the tool list written more compactly. It scored 88 of 200 (44.0\%) in its better run and 84 of 200 (42.0\%) in the other. This is a repeat on the same 200 tasks, not a test on new tasks. The full stack still sends every tool. The list is long because tools that share a group also share a description, and that description was repeated on every tool, with each schema indented. We state each group's shared description once and print the schemas without indentation (Fig.~\ref{fig:compact}). It does not remove or retrieve tools. Nothing else changes. The model, the four pieces of the full stack, and the tasks stay the same. Like the state summary and the card, this is harness-side control delivered through the context.

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
  {
    "name": "cd",
    "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Change the current working directory to the specified folder.",
    "parameters": {
      "type": "object",
      "properties": {
        "folder": {
          "type": "string",
          "description": "The folder of the directory to change to. You can only change one folder level at a time. "
        }
      },
      "required": [
        "folder"
      ]
    }
  }
]
...
\end{lstlisting}
\par\vspace{6pt}
\noindent\textbf{Compact list.} Median opening prompt 2{,}888 tokens, a 41\% cut.
\begin{lstlisting}
This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc.
[{"name": "cat", "description": "Display the contents of a file of any extension from currrent directory.", "parameters": {"type": "object", "properties": {"file_name": {"type": "string", "description": "The name of the file from current directory to display. No path is allowed. "}}, "required": ["file_name"]}}, {"name": "cd", "description": "Change the current working directory to the specified folder.", "parameters": {"type": "object", "properties": {"folder": {"type": "string", "description": "The folder of the directory to change to. You can only change one folder level at a time. "}}, "required": ["folder"]}}]
...
\end{lstlisting}
\caption{File-system tools from the multi-turn tasks, printed both ways. Both panels contain the same tools. The original list repeats the group's shared description on each tool and indents each schema. The compact list states that description once and prints the schemas without indentation. The excerpt is the first two tools of the group; an ellipsis marks the rest. The labels are the measured median lengths of the full opening prompt, 4{,}923.5 and 2{,}888 tokens, a 41\% cut.}
\label{fig:compact}
\end{figure}

We compared that shorter list with a fresh run of the original list, then repeated the comparison in the opposite order, with the shorter list run first (Table~\ref{tab:trim}). Both pairs use the same model, the same 200 tasks, and the same other harness parts. The 10-task check, and the token and timing counts, are in Appendix~\ref{app:trim}.

In the first pair the shorter list scores 84 of 200 (42.0\%, interval 35.4--48.9\%) and the original list scores 71 of 200 (35.5\%, interval 29.2--42.3\%). The shorter list solves 13 more tasks: 26 that the original list misses, and it misses 13 that the original list solves (58 both, 103 neither). An exact two-sided McNemar test gives $p=0.053$, just short of the 0.05 level.

The second pair is the confirmation. The shorter list scores 88 of 200 (44.0\%, interval 37.3--50.9\%), the best single run, and the original list scores 72 of 200 (36.0\%, interval 29.7--42.9\%). The shorter list solves 26 tasks that the original list misses and misses 10 that it solves (62 both, 102 neither), $p=0.011$, which clears the 0.05 level. The gain replicated. These 71 and 72 of 200 are the same runs as the 35.5\% and 36.0\% full-stack reruns.

The test we rely on counts each task once across both pairs. The shorter list does better on 33 tasks and the original list on 17, with 150 ties (exact two-sided sign test, $p=0.033$). The gain is about 7 points and replicated in both pairs. Repeat runs still move individual tasks: between its two runs, the original list changed result on 15 tasks and the shorter list on 18.

\begin{table}[H]
\centering
\footnotesize
\caption{Shorter tool list against the original list (same model, tasks, and other harness parts). Pair 1 compares the shorter list with a fresh run of the original list. Pair 2 runs the shorter list first. ``Shorter only'' counts tasks the shorter list solves and the original list misses. $p$ is an exact two-sided McNemar test, judged at the 0.05 level, outside Table~\ref{tab:mcnemar}. The sign test that counts each task once is in the text.}
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

The compact list cuts context use: mean context per request falls about 30\%, from about 7{,}100 to 5{,}000 tokens. The likely reason it helps is that a shorter context reduces the model's drift over long multi-turn episodes and leaves the state it needs in view. We did not test that explanation directly. The compact list is also text in the context. Invalid tool names stay at zero for both lists.

Figure~\ref{fig:ladder} puts this setup on the same scale as the other conditions. The top bar is the shorter tool list, the better of its two runs.

@@LADDER@@"""

HOW_SURE = r"""The compact tool list is the one change we ran twice, and its gain replicated (sign test across both pairs, $p=0.033$). Failure labels and a direct best-setup versus JSON text comparison are left to the confirmatory runs in Chapter~\ref{sec:conclusion}. The other harness gaps are less firm. Each harness accuracy comes with a range that extends about six to seven percentage points on either side (Table~\ref{tab:board}). Gaps smaller than that fall within run-to-run variation. Of the eight planned comparisons (Table~\ref{tab:mcnemar}), two harness gaps fall below 0.05 on their own: the full stack against JSON text ($p=0.023$) and the minimal recipe against the native API (McNemar $p=0.033$). Neither stays below 0.05 after correcting for the eight tests. The full stack leads JSON text by 8.0 percentage points, but the order of the harness rows is not statistically separated.

Repeat runs of the full stack do not agree. It scored 77 of 200, then 71 of 200 and 72 of 200. These 71 and 72 of 200 are the same runs as the 35.5\% and 36.0\% full-stack reruns. A swing of that size is as large as most gaps between harness rows, so a difference from a single run should not be ranked. Against JSON text, the first of those later runs is ahead by 5.0 percentage points (27 tasks only that run solves, 17 only JSON text, $p=0.174$).

Two contrasts are firm. The best supervised fine-tune solves only 7 tasks that JSON text misses and misses 38 that JSON text solves ($p<0.001$), and that gap survives the correction. The full stack's interval (32.0--45.4\%) does not overlap the best fine-tune (10.7--20.6\%) or the forced-valid output. Preference tuning (26.5\%) is not separable from JSON text."""

CH6_OPEN = r"""Fine-tuning is the standard way to adapt a small model to a task. Three terms come up in this chapter. The \emph{rebuilt mix} is a later training set of 30{,}000 rows that combines newly simulated, execution-checked episodes with cleaned public data. The \emph{gate} is a fixed 20-task check drawn from the same split and run with the status card; the bar is 17 of 20, the untuned model's score with the card. The \emph{null test} is a fine-tune that should change nothing if the machinery is sound: one pass over the untuned model's own logged answers.

Here no trained model beat the untuned model. The best supervised fine-tune scored 15.0\%, significantly below JSON text, and the DPO model scored 26.5\%, statistically level with it. The runs are evidence about what goes wrong. The version history is in Appendix~\ref{app:train}."""

CH6_POINTER = (
    "Self-distillation trains the model on its own answers to simulated tasks, keeping only the turns a checker marked correct. "
    "The suspected cause was a shift in that data, toward stopping and away from look-up calls. "
    "Two targeted retrains that corrected that shift scored 14 and 12 of 20, below the bar of 17 but not clearly separable from it on 20 tasks, "
    "so they neither confirm nor rule out the shift as the cause. The counts are in Appendix~\\ref{app:train}."
)
CH6_CLOSER = (
    "Fine-tuning changed the weights but did not improve accuracy over the untuned model. "
    "The next chapter asks what that means for someone choosing a method."
)

DISCUSSION_STATE = r"""\section{State, Not Syntax}

Across these conditions, the hard part is choosing the next step from the current state of the world, including whether to stop. Writing a valid call is the easy part. The levers that help are the ones that support that decision: a summary of real results, a gated stop, and a card that mainly controls whether the model acts. The compact list cuts context use by about 30\%, and the likely reason it helps is that this reduces drift over long episodes and leaves the state the model needs in view (Chapter~\ref{sec:helps}). The state summary was meant to put lost facts back in view. What falls, instead, is the number of empty turns. Among the labelled runs, none of the levers reduces state mismatches, and those remain the largest failure type, so the gains come from calling or stopping at the right moment. State tracking remains the main failure, as the state mismatch counts show. Levers that constrain the written form hurt, and training shifted the balance between acting and stopping rather than improving that tracking."""

RECOMMEND = r"""\section{What We Recommend}

On this model and this benchmark, the best setup we found is the full stack with the tool list written more compactly: JSON-text calls, a harness-written state summary, a gated stop rule, a status card, and the same tools with each group's shared description stated once. All of these are greedy and need no training data or training compute. The shorter list added about 7 points across two paired runs and cut context per request by about 30\%. The minimal recipe, which leaves out the state summary, is a smaller fallback. Before any fine-tune, screen on task-level checks rather than validation loss. Even the best run, the full stack with the shorter list, finishes 88 of 200, and a repeat of the same harness does not pass the same tasks every time. On this benchmark, this model still fails most multi-step tasks."""

LIMITATIONS = r"""\section{Limitations}

\begin{itemize}\setlength{\itemsep}{1pt}
  \item \textbf{One model, one split, one device.} All multi-turn results use Gemma-4-E2B-it on the 200-task base split, on one laptop. The stop threshold, the card's fields, and the training filters were chosen on these same tasks. Other models and harder splits are future work.
  \item \textbf{The harness was assembled on these tasks.} The full stack was put together after earlier results on the same 200 tasks were already known, and the family of eight tests was fixed after that. Its gain over JSON text does not survive that correction.
  \item \textbf{Run-to-run variation.} Repeated runs of one harness differ on individual tasks, so we report replicated pairs.
  \item \textbf{The shorter list.} It ran as two pairs on the same 200 tasks, outside the eight-test correction family. Its 44.0\% is the better of two replicated runs; a held-out check is the first confirmatory run in Chapter~\ref{sec:conclusion}.
  \item \textbf{The screening gate was built from the same split.} The 20 gate tasks were picked using earlier per-task results from the same split, and the bar is the score of the setup the gate is screening. It is designed as a fast screen, with confirmation left to full runs.
\end{itemize}"""

CONCLUSION = (
    "The best setup is the training-free harness with the same tool list written more compactly. "
    "It reached 44.0\\% (88 of 200) in its better run. The other scored 42.0\\% (84 of 200). "
    "The original list scored 71 and 72 of 200 in the same pairs. These 71 and 72 of 200 are "
    "the same runs as the 35.5\\% and 36.0\\% full-stack reruns. All of these runs used the same "
    "200 tasks. The gain from that shorter list replicated. The model fails at choosing the next "
    "step, not at writing a call. The failure analysis is why we built the harness. The compact "
    "list makes the prompt the model reads with each request about 30\\% shorter, and the likely "
    "reason it helps is that this leaves the state the model needs in view. Without the compact list, the harness raises "
    "accuracy from 30.5\\% to 38.5\\% in its first run; two reruns scored 35.5\\% and 36.0\\%. "
    "That gain is positive but does not survive correction for multiple comparisons. "
    "Forcing valid syntax made results worse. The results also show that fine-tuning did not "
    "improve the model: the best fine-tune scored 15.0\\%, below the untuned model's 30.5\\%. "
    "Among the labelled runs, wrong final states remain the largest failure type. For a local "
    "agent, these changes are also far cheaper than voting or training.\n"
)

REPRO = r"""\section{Reproducibility}

Hardware: MacBook Pro, Apple M4~Pro, 48~GB, with Metal acceleration (Apple's GPU interface). Temperature is 0 except for voting (0.7, $N{=}5$, one seed). The native API baseline used eight server slots and the other runs used four. Earlier runs used llama.cpp build 8111; the full-stack and compact-list runs used a later build, with the server flags \texttt{llama-server -{}-jinja -fa on -ngl 99 -c 65536 -{}-temp 0.0 -np 4 -cb -ub 1024}. BFCL v4 is read through the \texttt{bfcl-eval} package (the exact version per run was not logged). Fig. 2 and 5 are regenerated from these counts by \texttt{paper/figures/src/make\_refocus\_figs.py}, and Fig. 4 uses the same counts.
"""

TRIM = r"""\section{Shorter Tool List Checks}
\label{app:trim}

A 10-task check scored 5 of 10 with the original list and 8 of 10 with the shorter list. The recorded calls differed from the first full-stack run on five tasks, so the 200-task comparison used a fresh rerun of the original list rather than that first run (77 of 200).

The context window is 16{,}384 tokens per request (65{,}536 shared across four slots). The median opening prompt falls from 4{,}923.5 tokens to 2{,}888, a 41\% cut (41.3\%). Mean context per request falls from 7{,}107 tokens to 4{,}990, about 30\%. Invalid tool names stay at zero (0 of 1{,}276 calls with the shorter list, 0 of 1{,}244 with the original list). Invalid arguments are 2 of 1{,}276 against 9 of 1{,}244. Format errors are 1 of 1{,}801 model replies against 0 of 1{,}773. Mean server time per model request, prompt reading plus generation, with four slots, is 15.4\,s with the shorter list and 18.8\,s with the original list (medians 12.7\,s and 15.9\,s). Most of the drop is generation (16.2\,s to 13.5\,s). Prompt reading falls from 2.53\,s to 1.86\,s.

The second pair shows the same pattern: mean context per request falls from 7{,}121 tokens to 5{,}001, again about 30\%, and mean server time per request from 19.4\,s to 15.4\,s, about 20\% faster. Invalid tool names and format errors are zero for both lists, and invalid arguments are 4 of 1{,}300 calls with the shorter list against 5 of 1{,}255 (0.31\% and 0.40\%).
"""


TRAIN_SUMMARY = r"""The rebuilt mix has 30{,}000 training and 1{,}500 validation rows: 45/40/10/5\% simulated, APIGen, ToolACE, and xLAM data, with 22\% empty-list targets (all after a tool result) and no evaluation tool names copied into the simulations. DPO used $\beta=0.2$ with LoRA. The null test scored 15 of 20 (interval 53--89\%), statistically level with the bar of 17 of 20 (64--95\%). The rebuilt mix scored 2 of 20 (3--30\%), far below that bar.

Dropping only mid-dialogue empty targets from the first mix still scored 1/200, and deleting trailing empty answers at serving time scored 4/200. After the repaired mix (30/200), two further filtering variants scored 23/200 and 26/200 (Table~\ref{tab:board}); each changed several filters at once. The evaluated rebuilt-mix checkpoint is the best-validation one (loss 0.030 at step 22{,}000; training stopped at step 26{,}500 of 32{,}000).

Self-distillation scored 8/20; in 6 of 12 failures the model said an ID or price was missing and stopped instead of calling the tool that would supply it. Against the null-test data, look-up-type calls (a name-matching heuristic) fell from 37.9\% (67/177) to 19.7\% (78/395) of calls, and stop targets rose from 23.1\% (36/156) to 33.2\% (146/440). Capping the stop share scored 14/20 but left the look-up share low (20.6\%); restoring the look-up share (39.6\%) scored 12/20. Leak checks found no evaluation task IDs or tool names in either dataset."""


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
        "it completes 27.0\\% of 200 multi-turn tasks. The best setup we found, a harness plus a compact tool list, reaches 44.0\\% (88 of 200) in its better run.",
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
        r"Each failed task gets one primary label: \emph{empty turn} (no call when one was needed), \emph{under-call} (a required tool never ran), or \emph{state mismatch} (tools ran, but the final state differs). A call the harness cannot parse counts as no call.",
        "shorter error labels",
    )
    parts["ch3"] = replace_paragraph_starting(
        parts["ch3"],
        "We train LoRA adapters",
        r"We train LoRA adapters on Gemma-4-E2B-it with MLX~\cite{mlx}, merge and quantize to Q4\_K\_M, and evaluate with JSON text. The \emph{rebuilt mix} has 30{,}000 rows, about 45\% newly simulated episodes plus cleaned public data; earlier mixes use ToolACE and sampled xLAM data. One DPO model is trained too. The \emph{gate} is 20 tasks from the same split, run with the status card; the bar is 17/20.",
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
        "suggesting that the summary mainly keeps the model calling.",
        "so the summary mainly keeps the model calling.",
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
        "rise from 60 to 71: the card acts mainly as an act-versus-abstain control.",
        "card acts",
    )
    parts["ch5"] = must_replace(
        parts["ch5"],
        r"Combining all four greedy changes scores 77/200 (38.5\%, interval 32.0--45.4\%). Paired against JSON text, it solves 30 tasks that JSON text misses and misses 14 that JSON text solves ($p=0.023$ uncorrected; not significant after correction). Against the minimal recipe, the state summary alone and the vote of five, the differences are within noise (Table~\ref{tab:mcnemar}). The configuration was assembled after we had seen earlier results on these tasks, so its lead is subject to tuning leakage (Chapter~\ref{sec:discussion}). Its harness code is in pull request \#90 (run at commit \texttt{344ce45}, merged as \texttt{83f35ee}).",
        r"Combining all four greedy changes scores 77/200 (38.5\%, interval 32.0--45.4\%). Paired against JSON text, it solves 30 tasks that JSON text misses and misses 14 that JSON text solves (Table~\ref{tab:mcnemar}). Against the minimal recipe, the state summary alone, and the vote of five, the full stack is statistically indistinguishable. Because the configuration was assembled using earlier results on these tasks, its lead is reported as in-sample (Chapter~\ref{sec:discussion}). The next section keeps this harness and changes only how the tool list is written.",
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
        r"Top bar: the full stack with a compact tool list, the better of its two runs (Table~\ref{tab:trim}).",
        "shorten ladder caption",
    )
    parts["ch6"] = must_replace(
        parts["ch6"],
        r"It scored 15/20 (interval 53--89\%), within noise of the bar (17/20; 64--95\%), while the rebuilt mix's 2/20 (3--30\%) is far outside it.",
        r"It scored 15 of 20, statistically level with the bar of 17 of 20, while the rebuilt mix's 2 of 20 is far below it. The exact ranges are in Appendix~\ref{app:train}.",
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
        parts["ch6"], "section", "A Candidate Cause That Targeted Retrains Did Not Confirm"
    )
    cause = cause.replace("\\section{", "\\subsection{", 1)
    cause = must_replace(
        cause,
        r"Neither passed, so the data are \emph{consistent with}, but do not establish, a shift in the decision mix (stop versus continue, look up versus act) as the cause.",
        r"Neither reached the bar, but on 20 tasks neither is clearly separable from it, so these retrains neither confirm nor rule out that shift in the decision mix (stop versus continue, look up versus act) as the cause.",
        "retrain sole cause",
    )
    parts["ch6"] = must_replace(
        parts["ch6"],
        "so its data are the leading suspect but not the only one.\n",
        "so its data are the leading suspect, with training length the other candidate.\n\n" + CH6_POINTER + "\n",
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
        + "\n\nThe null test scored 15 of 20 (interval 53--89\\%), statistically level with the bar of 17 of 20 (64--95\\%). "
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
    parts["appC_board"] = parts["appC_board"].replace(
        "\\begin{table}[H]\n",
        "\\begin{table}[H]\n\\renewcommand{\\arraystretch}{1.0}\n",
        1,
    )
    # Advisor review (Nov 5): Appendix A folds into Chapter 3's harness list, the training
    # history becomes a short summary, and the run table and commit details leave Appendix C.
    parts.pop("appA")
    parts["appB_train"] = TRAIN_SUMMARY
    parts["appC_board"] = replace_span(
        parts["appC_board"],
        "$^{a}$Run 80 (scores.id 59)",
        "(sign test, $p=0.033$).",
        r"$^{a}$Runs 82 and 83 are pair 1 and runs 84 and 85 are pair 2 of the compact-list comparison (Table~\ref{tab:trim}); the 10-task check before them is in the last section of this appendix.",
        "trim footnote a",
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
    if "untested guess" in tex or "did not test why" in tex:
        raise SystemExit("compact-list hedge remains")
    for banned in ("list helps because", "sole cause", "leading cause"):
        if banned in tex:
            raise SystemExit("unsupported causal wording came back: " + banned)
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
    if r"\caption[" in tex:
        raise SystemExit("short captions would replace the full list entries")


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
    if "%%" in re.sub(r"%%%+", "", tpl):
        raise SystemExit("unfilled slot")
    check_output(tpl)
    (here / "main.tex").write_text(tpl)
    print("wrote", here / "main.tex")


if __name__ == "__main__":
    main()
