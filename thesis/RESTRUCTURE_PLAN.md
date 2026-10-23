# Restructure plan: draft thesis into the CSULA format of the chair's sample

Source draft: `paper/main.tex` (left untouched). Format model: the chair's sample thesis
(Mohammed Mustafa Kuko, M.S. Computer Science, Cal State LA, December 2019; chair
Dr. Mohammad Pourhomayoun). Output: a separate document, `thesis/main.tex` built to
`thesis/Thesis_CSULA_Format.pdf`. The original `paper/main.tex` and the root
`Thesis_Paper.pdf` are not modified.

## 1. Format taken from the sample

| Item | Sample | Thesis implementation |
|:--|:--|:--|
| Page | US Letter; margins 1.25 in left/right, 1 in top/bottom | `geometry` with the same margins |
| Font | Times New Roman 12 pt | `mathptmx` (Times) 12 pt |
| Spacing | Double spaced body, 0.5 in first-line indent (also first paragraph) | `setspace` `\doublespacing`, `indentfirst`, 0.5 in indent; floats single spaced |
| Page numbers | Centered bottom; front matter lower roman, title page counted as i but not printed; arabic from Chapter 1 | Same |
| Front matter order | Title (i), Copyright (ii), Approval (iii), Abstract (iv), Acknowledgments (v), Table of Contents, List of Tables, List of Figures | Same order |
| Title page | ALL-CAPS title / A Thesis / Presented to / The Faculty of the Department of Computer Science / California State University, Los Angeles / In Partial Fulfillment / of the Requirements for the Degree / Master of Science / in / Computer Science / By / name / date | Same lines, date "December 2026" |
| Copyright | "© year / name / ALL RIGHTS RESERVED", bottom of page | "© 2026 / Sanskar Thapa / ALL RIGHTS RESERVED" |
| Approval page | "The thesis of X is approved.", chair, 3 members, Department Chair, university, date | Same; chair "Dr. Mohammad Pourhomayoun, Committee Chair"; other names are placeholders |
| Abstract page | ABSTRACT heading, title, "By", name, text | Same |
| Front-matter headings | Centered, all caps, not bold | Same |
| Chapter openers | New page; "CHAPTER N" centered bold, title centered bold | Same |
| Level-2 headings | Flush left, bold, Title Case, unnumbered | Same (listed in TOC) |
| TOC | Front-matter entries with dot leaders; "Chapter" label; "1. Introduction" with level-2 entries indented; References | Same |
| LOT / LOF | "Table" / "Figure" label; numbered entries with page numbers | Same; entries use each caption's first sentence |
| Numbering | Continuous: Table 1..n, Fig. 1..n | Same (`\counterwithout`) |
| Captions | Figures below, small, italic "Fig. N."; tables above, centered, "TABLE N." | Same |
| References | "REFERENCES", numbered [n] in order of citation | `unsrtnat` numbered, order of citation |
| Appendices | None in the sample | Kept after References (draft content), as Appendix A-C |

## 2. Section mapping (draft to thesis)

| Draft (`paper/main.tex`) | Thesis (`thesis/main.tex`) | Change |
|:--|:--|:--|
| `\title`, `\author`, `\date{October 2026}`, title block line "Master's thesis draft, presentation end of November 2026" and repo URL | Title page, copyright page | Reformatted to the sample's title page; date "December 2026"; draft line and URL dropped from the title page |
| (none) | Approval page | New, chair from sample, other names placeholders |
| Abstract | ABSTRACT (p. iv) | Same text; heading block as in sample |
| (none) | ACKNOWLEDGMENTS | Placeholder |
| (none) | TABLE OF CONTENTS, LIST OF TABLES, LIST OF FIGURES | Generated |
| 1 Introduction (incl. Figure 1 vignette) | Chapter 1 Introduction | Same text; organization paragraph placeholder added at the end (the sample's Chapter 1 ends with one) |
| 2 Background and Related Work | Chapter 2 Background and Related Work | Run-in `\paragraph` labels become level-2 headings |
| 3 Setup and Methods | Chapter 3 Setup and Methods | Same |
| 4 Where Small Models Fail | Chapter 4 Where Small Models Fail | Same |
| 5 What Helps: Harness-Side Control | Chapter 5 What Helps: Harness-Side Control | Same |
| 6 Fine-Tuning Did Not Help | Chapter 6 Fine-Tuning Did Not Help | Same, one em-dash pair changed to commas |
| 7 Discussion and Limitations | Chapter 7 Discussion (Limitations is a level-2 heading inside it) | Chapter title shortened |
| 8 Future Work and Conclusion | Chapter 8 Conclusion, followed by a level-2 "Future Work" section | Title shortened; conclusion paragraph moved before Future Work |
| References | REFERENCES | Same bibliography (`paper/references.bib`) |
| Appendix A Intervention details | Appendix A Intervention Details | Title Case |
| Appendix B Failure labels, exploratory tests and training history (B.1 Training history) | Appendix B Failure Labels, Exploratory Tests and Training History (section Training History) | Title Case |
| Appendix C Full scoreboard, reproducibility and glossary | Appendix C Full Scoreboard, Reproducibility and Glossary | Title Case |

Figures and tables: Fig. 1 (loop vignette), Fig. 2 (failure anatomy), Fig. 3 (ladder),
Fig. 4 (20-task gate); Table 1 (McNemar), Table 2 (failure labels), Table 3 (exploratory
McNemar), Table 4 (scoreboard). Same contents and captions; placed at paragraph breaks.

## 3. Placeholders (all marked `[PLACEHOLDER: ...]` in bold)

1. Approval page: Committee member name (1 of 2)
2. Approval page: Committee member name (2 of 2)
3. Approval page: Department Chair name
4. Acknowledgments, to be written by Sanskar
5. Chapter 1 closing thesis organization paragraph (chapter-by-chapter roadmap), to be written by Sanskar

## 4. Wording changes (complete list; no findings, numbers or claims changed)

1. Cross-references "Section N" now read "Chapter N" (chapters replaced sections).
2. Run-in paragraph labels (e.g. "Benchmark.") are now Title Case level-2 headings without the period (e.g. "Benchmark").
3. The run-in label "Conclusion." was dropped (its paragraph now sits directly under the Chapter 8 title).
4. Chapter 7 title "Discussion and Limitations" became "Discussion"; Chapter 8 "Future Work and Conclusion" became "Conclusion" with "Future Work" as a section after the conclusion paragraph.
5. Appendix titles in Title Case.
6. Chapter 6, one sentence: the em-dash pair around "training on the model's own answers ... marked correct" became commas.
7. Title page: draft line "Master's thesis draft, presentation end of November 2026", the repo URL and "October 2026" replaced by the sample's title-page block with "December 2026".

No other connective sentences were added.

## 5. Build

`cd thesis && make dist` (latexmk + Ghostscript full font embed) writes `Thesis_CSULA_Format.pdf`.
`build_from_paper.py` regenerates `main.tex` from `main.tpl.tex` plus line ranges of `paper/main.tex`.
