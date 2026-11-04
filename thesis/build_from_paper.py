#!/usr/bin/env python3
"""Assemble thesis/main.tex (CSULA format) from paper/main.tex body text.

Content is copied verbatim from paper/main.tex by line range; only structural
transforms are applied (run-in \\paragraph headings -> level-2 headings,
"Section~\\ref" -> "Chapter~\\ref" for targets that are now chapters,
"this paper" -> "this thesis", float placement -> [H]). Line ranges match
paper/main.tex after the shorter-tool-list results and PR #109. Fig. 2 already
follows the paragraph that cites it. The thesis pins every float with [H], so
the paper's [tb] on the McNemar table is not copied.
"""
import re, pathlib
here = pathlib.Path(__file__).resolve().parent
src = (here.parent / "paper" / "main.tex").read_text().split("\n")
L = lambda a, b: "\n".join(src[a-1:b])   # 1-based inclusive line range

TITLES = {  # \paragraph heading -> Title Case level-2 heading (sample style)
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
 r"JSON text over the native API (+3.5\pp).": r"JSON Text over the Native API (+3.5\pp)",
 r"State summary (+3.5\pp).": r"State Summary (+3.5\pp)",
 r"Stop on repeats and the status card (minimal recipe, 35.5\%).": r"Stop on Repeats and the Status Card (Minimal Recipe, 35.5\%)",
 r"Full stack (38.5\% on the first run).": r"Full Stack (38.5\% on the First Run)",
 "Shorter tool list, same tools.": "Shorter Tool List, Same Tools",
 r"Cost: majority vote (36.5\%).": r"Cost: Majority Vote (36.5\%)",
 "How sure are we?": "How Sure Are We?",
 "Public data taught the model to stop.": "Public Data Taught the Model to Stop",
 "A rebuilt mix collapsed.": "A Rebuilt Mix Collapsed",
 "A candidate cause that targeted retrains did not confirm.": "A Candidate Cause That Targeted Retrains Did Not Confirm",
 "State, not syntax.": "State, Not Syntax",
 "The guided-template tie.": "The Guided-Template Tie",
 "What we recommend.": "What We Recommend",
 "Limitations.": "Limitations",
 "Future work.": "Future Work",
 "Conclusion.": None,  # chapter title already says Conclusion; drop run-in label
 # appendix run-in labels
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
CHAPTER_LABELS = {"sec:intro","sec:related","sec:setup","sec:fail","sec:helps","sec:train","sec:discussion","sec:conclusion"}

def para_to_section(text, level="section"):
    out = []
    for line in text.split("\n"):
        m = re.match(r"\\paragraph\{((?:[^{}]|\{[^{}]*\})*)\}\s?(.*)$", line)
        if m and m.group(1) in TITLES:
            t = TITLES[m.group(1)]
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

def fix(text, level="section"):
    text = para_to_section(text, level)
    for lab in CHAPTER_LABELS:
        text = text.replace(f"Section~\\ref{{{lab}}}", f"Chapter~\\ref{{{lab}}}")
    text = re.sub(r"\\begin\{(figure|table)\}\[[^\]]*\]", r"\\begin{\1}[H]", text)
    text = text.replace("\\FloatBarrier\n", "")
    text = text.replace("this paper", "this thesis").replace("This paper", "This thesis")
    # T1 ligatures turn "--" inside \\texttt into an en dash. Flags such as
    # --jinja must print as two hyphens. Number ranges outside \\texttt stay.
    text = re.sub(
        r"\\texttt\{[^{}]*--[^{}]*\}",
        lambda m: m.group(0).replace("--", "-{}-"),
        text,
    )
    return text

def list_captions(text, env):
    """Give every \\caption a list entry: the label ("Fig. N." / "TABLE N.") plus the full caption, as in the sample's LOF/LOT."""
    lab = r"Fig.~\thefigure.\ " if env == "figure" else r"TABLE~\thetable.\ "
    out, i = [], 0
    while True:
        j = text.find("\\caption{", i)
        if j < 0:
            out.append(text[i:]); break
        k = j + len("\\caption{"); depth = 1; m = k; first = None
        while depth:
            c = text[m]
            if c == "{": depth += 1
            elif c == "}": depth -= 1
            m += 1
        out.append(text[i:j] + "\\caption[" + lab + text[k:m-1] + "]{" + text[k:m])
        i = m
    return "".join(out)

def captions(text):
    """Apply list_captions per float environment."""
    parts = re.split(r"(\\begin\{(?:figure|table)\}.*?\\end\{(?:figure|table)\})", text, flags=re.S)
    return "".join(list_captions(p, "figure" if p.startswith("\\begin{figure}") else "table") if p.startswith("\\begin{") else p for p in parts)

def body(a, b, level="section"):
    return captions(fix(L(a, b), level))

def allow_path_breaks(text):
    """Allow line breaks after "/" inside \\texttt paths (keeps long paths inside the margin)."""
    return re.sub(r"\\texttt\{[^{}]*/[^{}]*\}", lambda m: m.group(0).replace("/", r"/\allowbreak{}"), text)

parts = {
 "abstract": L(70, 70),
 "ch1": body(76, 103),
 "ch2": body(108, 116),
 "ch3": body(121, 139),
 "ch4": body(144, 155),
 "ch5": body(161, 228),
 "ch6": body(232, 246),
 "ch7": body(250, 268),
 "ch8_future": body(273, 273),
 "ch8_concl": body(275, 275),
 "appA": body(286, 308),
 "appB": body(313, 347),
 "appB_train": body(352, 358, level="subsection"),  # run-ins under B.1 Training History
 "appC_repro": allow_path_breaks(body(419, 419)),
 "appC_gloss": fix(L(423, 458)),   # glossary body; the template already supplies the heading
 "appC_board": body(364, 417).replace(r"\parbox{\linewidth}{\scriptsize", r"\parbox{\linewidth}{\RaggedRight\scriptsize"),  # note ragged like the body (no underfull lines)
}
tpl = (here / "main.tpl.tex").read_text()
for k, v in parts.items():
    tpl = tpl.replace("%%" + k + "%%", v)
assert "%%" not in re.sub(r"%%%+", "", tpl), "unfilled slot"
(here / "main.tex").write_text(tpl)
print("wrote", here / "main.tex")
