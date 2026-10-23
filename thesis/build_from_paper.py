#!/usr/bin/env python3
"""Assemble thesis/main.tex (CSULA format) from paper/main.tex body text.

Content is copied verbatim from paper/main.tex by line range; only structural
transforms are applied (run-in \\paragraph headings -> \\section* headings,
"Section~\\ref" -> "Chapter~\\ref" for targets that are now chapters,
"this paper" -> "this thesis", Fig. 2 after its citing paragraph,
float placement -> [H]). Run once; main.tex is then the maintained source.
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
 r"Full stack: the best measured result (38.5\%).": r"Full Stack: The Best Measured Result (38.5\%)",
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
    # em dash removal (house rule: no em dashes); the one occurrence becomes commas
    EM = "-" * 3  # LaTeX em-dash ligature
    text = text.replace(r"\emph{Self-distillation}" + EM + r"training on the model's own answers to simulated tasks, keeping only turns the task checker marked correct" + EM + "scored",
                        r"\emph{Self-distillation}, training on the model's own answers to simulated tasks, keeping only turns the task checker marked correct, scored")
    text = text.replace("this paper", "this thesis").replace("This paper", "This thesis")
    text = text.replace(r"\texttt{llama-server --jinja}", r"\texttt{llama-server -{}-jinja}")
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

def move_float_after_next_paragraph(text, label):
    """Place the float with this label after the paragraph that follows it (the paragraph that cites it)."""
    m = re.search(r"\\begin\{figure\}\[H\]\n(?:(?!\\end\{figure\}).)*?\\label\{" + re.escape(label) + r"\}.*?\\end\{figure\}\n\n", text, flags=re.S)
    assert m, label
    fl = m.group(0)
    rest = text[m.end():]
    n = rest.index("\n\n") + 2          # end of the citing paragraph (heading + blank + paragraph)
    n = rest.index("\n\n", n) + 2 if rest.startswith("\\section") else n
    return text[:m.start()] + rest[:n] + fl + rest[n:]

def body(a, b, level="section"):
    return captions(fix(L(a, b), level))

def allow_path_breaks(text):
    """Allow line breaks after "/" inside \\texttt paths (keeps long paths inside the margin)."""
    return re.sub(r"\\texttt\{[^{}]*/[^{}]*\}", lambda m: m.group(0).replace("/", r"/\allowbreak{}"), text)

parts = {
 "abstract": L(64, 64),
 "ch1": body(70, 97),
 "ch2": body(102, 110),
 "ch3": body(115, 132),
 "ch4": move_float_after_next_paragraph(body(137, 148), "fig:fail"),
 "ch5": body(154, 194),
 "ch6": body(199, 212),
 "ch7": body(217, 233),
 "ch8_future": body(238, 238),
 "ch8_concl": body(240, 240),
 "appA": body(251, 273),
 "appB": body(278, 313),
 "appB_train": body(317, 323, level="subsection"),  # run-ins under B.1 Training History
 "appC_repro": allow_path_breaks(body(332, 332)),
 "appC_gloss": fix(L(336, 370)),   # glossary without the \enlargethispage hack (line 335)
 "appC_board": body(372, 421),
}
tpl = (here / "main.tpl.tex").read_text()
for k, v in parts.items():
    tpl = tpl.replace("%%" + k + "%%", v)
assert "%%" not in re.sub(r"%%%+", "", tpl), "unfilled slot"
(here / "main.tex").write_text(tpl)
print("wrote", here / "main.tex")
