# Vendored paper-writing skills and agents

Installed 2026-10-18 for work on `paper/main.tex`. Entry point: `.claude/skills/thesis-paper/SKILL.md` (this repo's workflow; it overrides the upstream skills where they conflict).

| Path | Upstream | Commit | License |
|---|---|---|---|
| `skills/ml-paper-writing/` (SKILL.md + `references/`) | [Orchestra-Research/AI-Research-SKILLs](https://github.com/Orchestra-Research/AI-Research-SKILLs) `20-ml-paper-writing/ml-paper-writing` v1.2.0 | `773a529` | MIT (`skills/ml-paper-writing/LICENSE`) |
| `skills/academic/` + `agents/*.md` (12 agents) | [andrehuang/academic-writing-agents](https://github.com/andrehuang/academic-writing-agents) v2.0.0 | `d4d9d3a` | MIT (`skills/academic/LICENSE`) |

## Local changes
- Orchestra venue templates (`templates/`, ~1 MB of NeurIPS/ICML/ICLR/ACL/AAAI/COLM files) not copied: the thesis uses `article` class.
- Agents: the upstream author's hard-coded `/Users/owl/.claude/principles/academic-writing.md` repointed to `.claude/skills/academic/academic-writing.md`.
- `skills/academic/SKILL.md`: review findings persist to `docs/reviews/` instead of `.review/` (outside `paper/`, so the Drive PDF sync is not triggered; readable by Codex/Antigravity); "new specialist" suggestions point to repo `.claude/agents/`.

To update: re-copy from upstream at a newer commit, re-apply the changes above, update the commit column.
