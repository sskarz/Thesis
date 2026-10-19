# Agent notes

Paper work (`paper/main.tex`, its figures, and `references.bib`) starts at [`.agents/skills/thesis-paper/SKILL.md`](.agents/skills/thesis-paper/SKILL.md). Follow that file's hard rules. History, headline numbers, and sources of truth: [`docs/PAPER_WRITING_PROCESS.md`](docs/PAPER_WRITING_PROCESS.md). Reviewer roles live in [`.claude/agents/`](.claude/agents/) (`<role>.md`).

Claude Code is not used for paper work as of 2026-10-18. This workflow runs in Cursor, Codex, and Antigravity.

- Cursor loads skills from `.agents/skills/`, `.cursor/skills/`, and `.claude/skills/`, and subagents from `.cursor/agents/` and `.claude/agents/` ([skills](https://cursor.com/docs/context/skills), [subagents](https://cursor.com/docs/context/subagents)).
- Codex loads skills from `.agents/skills/` and reads this file ([skills](https://developers.openai.com/codex/skills)).
- Antigravity loads skills from `.agents/skills/` and reads this file ([skills](https://antigravity.google/docs/skills/)).

A subagent running a reviewer role reads that role file and the thesis-paper skill as its prompt.

Sanskar picks which fixes are applied and decides all merges.

Where the vendored ml-paper-writing says "Claude should be proactive" or tells you to deliver drafts without waiting, the thesis-paper rules override it.
