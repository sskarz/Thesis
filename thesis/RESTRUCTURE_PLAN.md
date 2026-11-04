# Restructure plan: draft thesis into the CSULA format of the chair's sample

Source draft: `paper/main.tex` (left untouched). Format model: the chair's sample thesis
(Mohammed Mustafa Kuko, M.S. Computer Science, Cal State LA; chair
Dr. Mohammad Pourhomayoun). Output: a separate document, `thesis/main.tex` built to
`thesis/Thesis_CSULA_Format.pdf`. The original `paper/main.tex` and the root
`Thesis_Paper.pdf` are not modified.

## 1. Format taken from the sample

| Item | Sample | Thesis implementation |
|:--|:--|:--|
| Page | US Letter; margins 1.25 in left/right, 1 in top/bottom | `geometry` with the same margins |
| Font | Times New Roman 12 pt | `mathptmx` (Times) 12 pt |
| Spacing | Double spaced body, 0.5 in first-line indent (also first paragraph) | `\setstretch{1.91}` (measured baseline pitch 27.6 pt, same as the sample's Word double spacing), also on the title page; `indentfirst`, 0.5 in indent; ragged right without hyphenation as in the sample; floats single spaced |
| Page numbers | Centered bottom; front matter lower roman, title page counted as i but not printed; arabic from Chapter 1 | Same |
| Front matter order | Title (i), Copyright (ii), Approval (iii), Abstract (iv), Acknowledgments (v), Table of Contents, List of Tables, List of Figures | Same order, except Acknowledgments is commented out and is not a page. A glossary follows the lists, before Chapter 1 |
| Title page | ALL-CAPS title / A Thesis / Presented to / The Faculty of the Department of Computer Science / California State University, Los Angeles / In Partial Fulfillment / of the Requirements for the Degree / Master of Science / in / Computer Science / By / name / date | Same lines, date "December 2026" |
| Copyright | "© year / name / ALL RIGHTS RESERVED", bottom of page | "© 2026 / Sanskar Thapa / ALL RIGHTS RESERVED" |
| Approval page | "The thesis of X is approved.", chair, members, Department Chair, university, date | Same line style. Chair "Dr. Mohammad Pourhomayoun, Committee Chair". Department Chair "Dr. Chengyu Sun, Department Chair". Two committee member lines stay inline placeholders |
| Abstract page | ABSTRACT heading, title, "By", name, text | Same |
| Front-matter headings | Centered, all caps, not bold | Same |
| Chapter openers | New page; "CHAPTER N" centered bold, title centered bold | Same |
| Level-2 headings | Flush left, bold, Title Case, unnumbered | Same (listed in TOC) |
| TOC | Front-matter entries with dense dot leaders; "Chapter" label; "1. Introduction" with level-2 entries indented; References | Same; an "Appendix" label line (like "Chapter") introduces A-C |
| LOT / LOF | "Table" / "Figure" label; numbered entries with page numbers | Same style: "1. TABLE 1. full caption" / "1. Fig. 1. full caption"; entries single spaced with a blank line between entries |
| Numbering | Continuous: Table 1..n, Fig. 1..n | Same (`\counterwithout`) |
| Captions | Figures below, small, italic "Fig. N."; tables above, centered, "TABLE N." with small-caps caption text | Same |
| References | "REFERENCES", numbered [n] in order of citation | `unsrtnat` numbered, order of citation |
| Appendices | None in the sample | Kept after References (draft content), as Appendix A-C |

## 2. Section mapping (draft to thesis)

| Draft (`paper/main.tex`) | Thesis (`thesis/main.tex`) | Change |
|:--|:--|:--|
| `\title`, `\author`, `\date{...}`, title block draft/presentation line and repo URL | Title page, copyright page | Reformatted to the sample's title page; date "December 2026"; draft line and URL dropped from the title page |
| (none) | Approval page | New, chair from sample, other names placeholders |
| Abstract | ABSTRACT (p. iv) | Same text; heading block as in sample |
| (none) | ACKNOWLEDGMENTS commented out | Not printed. There is no Acknowledgments page. Uncomment the three lines in `main.tpl.tex` to restore the page and its table-of-contents entry |
| (none) | TABLE OF CONTENTS, LIST OF TABLES, LIST OF FIGURES, then GLOSSARY | Generated. The glossary is the old Appendix C glossary, moved forward. List entries use a short title, not "1. TABLE 1." |
| 1 Introduction (incl. Figure 1 vignette) | Chapter 1 Introduction | Contributions and the roadmap have no result counts. The roadmap is three sentences |
| 2 Background and Related Work | Chapter 2 Background and Related Work | Run-in `\paragraph` labels become level-2 headings |
| 3 Setup and Methods | Chapter 3 Setup and Methods | Same |
| 4 Where Small Models Fail | Chapter 4 Where Small Models Fail | Same |
| 5 What Helps: Harness-Side Control | Chapter 5 What Helps: Harness-Side Control | Same |
| 6 Fine-Tuning Did Not Help | Chapter 6 Fine-Tuning Did Not Help | Same, one em-dash pair changed to commas |
| 7 Discussion and Limitations | Chapter 7 Discussion (Limitations is a level-2 heading inside it) | Chapter title shortened |
| 8 Future Work and Conclusion | Chapter 8 Future Work and Conclusion | Future Work section, then a Conclusion section. The conclusion is last |
| References | REFERENCES | Same bibliography (`paper/references.bib`) |
| Appendix A Intervention details | Appendix A Intervention Details | Title Case |
| Appendix B Failure labels, exploratory tests and training history (B.1 Training history) | Appendix B Failure Labels, Exploratory Tests and Training History (section "Training History"; its run-in labels such as Version History are subsections under it) | No lone "B.1" heading. Chapters 3 and 6 point at this section with `\ref{app:train}` |
| Appendix C Full scoreboard, reproducibility and glossary | Appendix C Full Scoreboard and Reproducibility | Glossary moved to the front matter. Shorter-list checks (the 10-task protocol and the token and timing counts) are a section here. The pooled row and the first-token timing check are not copied |

Figures and tables follow `paper/main.tex`: Fig. 1 (loop vignette), Fig. 2 (failure anatomy),
Fig. 3 (ladder), Fig. 4 (20-task gate); Table 1 (shorter tool list), Table 2 (McNemar),
Table 3 (failure labels), Table 4 (exploratory McNemar), Table 5 (scoreboard). Every float
is pinned with `[H]`. The paper's `[tb]` on the McNemar table is not copied. Fig. 2 already
follows the paragraph that cites it in the current paper.

## 3. Placeholders (all marked `[PLACEHOLDER: ...]` in bold)

1. Approval page: Committee member name (1 of 2)
2. Approval page: Committee member name (2 of 2)

The committee chair is Dr. Mohammad Pourhomayoun. The department chair is Dr. Chengyu Sun. Acknowledgments are commented out in `main.tpl.tex` (page and table-of-contents entry). Uncomment those three lines to restore them. There is no Acknowledgments page. The Chapter 1 roadmap is three sentences in `main.tpl.tex`.

## 4. Wording changes (complete list; no findings, numbers or claims changed)

1. Cross-references "Section N" now read "Chapter N" (chapters replaced sections).
2. Run-in paragraph labels (e.g. "Benchmark.") are now Title Case level-2 headings without the period (e.g. "Benchmark").
3. The run-in label "Conclusion." was dropped. Chapter 8 is "Future Work and Conclusion": a Future Work section, then a Conclusion section.
4. Chapter 7 title "Discussion and Limitations" became "Discussion". Limitations is one of its sections.
5. Appendix titles in Title Case.
6. Chapter 6, one sentence: the em-dash pair around "training on the model's own answers ... marked correct" became commas.
7. Title page: the draft/presentation line, the repo URL and the draft date replaced by the sample's title-page block with "December 2026".
8. "this paper" became "this thesis" (6 places: Fig. 1 caption, Chapter 2 glossary pointer, Chapter 3 opening, Appendix A training paragraph, glossary entries IFFC and NFC).
9. Appendix C reproducibility: `llama-server --jinja` is typeset as `-{}-jinja` so the two hyphens print as written (not as an en dash).
10. Placement only (no wording change): Fig. 2 (failure anatomy) moved from before to after the Chapter 4 paragraph that first cites it ("Figure 2 breaks down the failed tasks ..."), so the figure follows its first reference.

The readability pass in section 7 adds short chapter openings and closers and replaces some blocks. Those sentences do not add findings.

## 5. Build

`cd thesis && make dist` (latexmk + Ghostscript full font embed) writes `Thesis_CSULA_Format.pdf`.
`build_from_paper.py` regenerates `main.tex` from `main.tpl.tex` plus line ranges of `paper/main.tex`.

## 6. Content refresh from the current paper

After the original restructure, `paper/main.tex` gained the shorter-tool-list results (PR #113) and the PR #109 wording (December 2026 on the draft title, which this thesis already used; LoRA cited as the ICLR 2022 paper). `build_from_paper.py` line ranges were updated and `main.tex` regenerated. Prose and numbers are copied from the paper. The thesis date stays December 2026. Floats stay `[H]`, so the paper's McNemar-table `[tb]` placement is not copied. The ladder figure is `paper/figures/ladder_n200.pdf`, regenerated by `paper/figures/src/make_refocus_figs.py` (top bar 88/200).

## 7. Readability and focus rewrite

`paper/main.tex` is still copied by line range. `build_from_paper.py` then applies thesis-only replacement blocks, cuts, and moves, and writes `main.tex`. Running the script reproduces `main.tex`. Do not edit `main.tex` by hand.

The story the replacements tell: the model fails at deciding the next step, the harness helps, and forcing valid syntax and fine-tuning do not. Reader-facing prose uses one naming pair, native API and JSON text. Run ids, commit hashes, pull-request numbers, `scores.id` values, script names, and task ids stay in the Appendix C scoreboard and the reproducibility note. The body does not use them.

What moved:
- The glossary, to just after the lists of tables and figures.
- The `[STATE]` and `[OODA]` examples, into Chapter 3. Appendix A points back.
- The guided-template tie, into the Chapter 5 section on JSON text.
- The candidate-cause section, to Appendix B under Training History, as a subsection.
- The shorter-list gate protocol and the token and timing counts, to Appendix C, section "Shorter Tool List Checks".
- The machine, the llama.cpp commits, and Metal, out of Chapter 3 and into the reproducibility note.
- The ladder figure, to just after the shorter tool list is introduced.
- The 20-task gate figure, to just after the paragraph that cites it.

What was cut from the thesis (each number remains in `paper/main.tex`):
- Result counts in the three Chapter 1 contributions, and the page-long roadmap.
- The conclusion's counts. The conclusion is the same story as the abstract, without the counts.
- Run ids 80--85, the task ids, and the pooled row (`p = 0.001`) from the shorter-list section and from Table 1.
- The first-token timing check (166 ms, 334 ms, 204 ms, 473 ms).
- The asides "2,121 versus 596" and "lowered it to 26/200".
- The Figure 2 clause "not labelled for this revision", and the failed-task counts that existed only to explain that clause.
- The `evaluate_multi_turn.py` mention, the pull-request line in Chapter 5, and the phrases "not in the repository" and "not in the notes used for this thesis".
- Five of the nine limitation bullets. The four that remain are scope, tuning on these tasks, run-to-run movement, and the gate.
- The 10-task tripwire subsection. It was already only in Appendix B. The section pass said to cut it, so the subsection, its glossary line, and the Chapter 3 pointer are gone. Self-distillation counts stay in Appendix B.
- Glossary entries BoN and GST.

Repeated results are stated once in the body: the 77, then 71 and 72 scores in "How Sure Are We?"; the 33 tasks to 17 comparison in the shorter-list section; and `p = 0.023` in that same "How Sure Are We?" section. The scoreboard and the McNemar table still show the underlying counts.

Chapter 2, Chapter 6, and Chapter 7 end with a one-sentence closer. Statistics and "How Sure Are We?" are in plain language. "Rebuilt mix", "gate", and "null test" are defined at the start of Chapter 6. Section headings in Chapter 5 no longer carry percentages. Appendix A headings do not repeat the chapter headings. The old "B.1" heading is an ordinary section, "Training History". Acknowledgments stay commented out. The two committee-member placeholders stay. The date stays December 2026.

## 8. Second readability pass

The Figure 1 loop is an under-call: the brake call never ran. Chapter 4 says so, and defines a state mismatch as tools that did run but left the wrong final state. The abstract names 30.5% as the JSON-text baseline and says the 38.5% gain is not significant after correcting for multiple comparisons. Statistics keeps the Wilson and McNemar definitions and two sentences; the family list and the sign test are in Appendix B, "Planned Comparisons". The null-test ranges are described in words in Chapter 6 and printed exactly under Training History. Chapter 5 states the about-30% context cut. The ladder caption no longer repeats the pair counts. Tuning leakage points at Limitations. `thesis/figures/ladder_n200.pdf` is the paper ladder with the same counts and the parenthetical log names removed from the two bar labels. Table 5 uses plain condition names. Runs, score rows, commits, and logs are a table in the reproducibility section, without task ids. Appendix sections are in the table of contents. The glossary is on its own page after the lists. Fine-tuning setup text sits under Training History, so that heading is not empty and Appendix A does not end on a nearly blank page.
