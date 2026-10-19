# Vendored paper-writing skills and agents

Installed 2026-10-18 for work on `paper/main.tex`. Entry point: `.agents/skills/thesis-paper/SKILL.md` (this repo's workflow; it overrides the upstream skills where they conflict). Skills live under `.agents/skills/` so Cursor, Codex, and Antigravity load them. Reviewer agents stay in `.claude/agents/` (Cursor loads that directory). Claude Code is not used for paper work as of 2026-10-18.

| Path | Upstream | Commit | License |
|---|---|---|---|
| `.agents/skills/ml-paper-writing/` (SKILL.md + `references/`) | [Orchestra-Research/AI-Research-SKILLs](https://github.com/Orchestra-Research/AI-Research-SKILLs) `20-ml-paper-writing/ml-paper-writing` v1.2.0 | `773a529` | MIT (`.agents/skills/ml-paper-writing/LICENSE`) |
| `.agents/skills/academic/` + `.claude/agents/*.md` (12 agents) | [andrehuang/academic-writing-agents](https://github.com/andrehuang/academic-writing-agents) v2.0.0 | `d4d9d3a` | MIT (`.agents/skills/academic/LICENSE`) |

## Local changes
- Orchestra venue templates (`templates/`, ~1 MB of NeurIPS/ICML/ICLR/ACL/AAAI/COLM files) not copied: the thesis uses `article` class.
- Agents: the upstream author's hard-coded `/Users/owl/.claude/principles/academic-writing.md` repointed to `.agents/skills/academic/academic-writing.md`.
- `.agents/skills/academic/SKILL.md`: review findings persist to `docs/reviews/` instead of `.review/` (outside `paper/`, so the Drive PDF sync is not triggered; readable by Codex/Antigravity); "new specialist" suggestions point to repo `.claude/agents/`. In this move, the only additional edit to that file is `.claude/CLAUDE.md` → `AGENTS.md`.
- Agent frontmatter: `model: opus` → `model: inherit`. Read-only reviewers, plus any agent whose tools are only Read, Glob, and Grep, set `readonly: true`. The `tools:` field is kept.

To update: re-copy from upstream at a newer commit, re-apply the changes above, update the commit column.
