# ADR-0004: Two-tier agent hierarchy for parallel idea exploration

- Status: Accepted (user decision)
- Date: 2026-11-21

## Context

There are roughly 20 hours until the Sunday 10:00 deadline. Many compilation strategies are
plausible, and the cheapest way to choose between them is to try several in parallel and compare
scores. Strong models are best spent on design and judgment, cheap fast models on writing code.

## Decision

- **Orchestrator** (the main opencode session, Claude Opus): owns `main`, writes idea briefs, starts
  branches, compares `results/*.json`, runs strict gates, merges winners and reports to the user
  after every round.
- **Planner** (agent `pq-planner`, `claude-agent/opus#high` (Opus 5.5)): one per idea, in its own git
  worktree (`/home/hackathon/challenge-wt/<idea>`, branch `idea/<idea>`). It may search the papers
  MCP, writes `docs/ideas/<idea>.md` before any code, delegates implementation, reviews it, iterates
  on bench results and writes the verdict.
- **Implementer** (agent `pq-implementer`, `opencode-go/deepseek-v4.1-flash#high`): writes and
  debugs code in the given worktree to the planner's spec and must report bench numbers.
- At most 4 branches run at a time (8 cores; each bench run uses 2 workers).
- Agent definitions live in `.opencode/agents/` in this repo, **copied** (not symlinked: the config watcher does not follow symlinks, so edits were not picked up) into
  `/home/hackathon/.opencode/agents/` so the main session discovers them.

## Consequences

- Ideas are isolated and comparable. Merge conflicts are limited to what the orchestrator merges.
- Each compiler is a single file (ADR 0006), so parallel ideas rarely touch the same file.
- Planners must cite real arXiv ids when they borrow a technique.
