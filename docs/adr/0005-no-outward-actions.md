# ADR-0005: No submissions, publishing or pushes without explicit user approval

- Status: Accepted (user decision)
- Date: 2026-11-21

## Context

Leaderboard submissions, publishing aqora workspaces, and pushing to the team GitHub repository are
outward-facing and hard to undo. Scoring only needs the local harness.

## Decision

No agent may submit to the leaderboard, create or publish aqora workspaces, run aqora GraphQL
mutations, or `git push`, unless the user explicitly says so for that specific action. Local commits
and worktrees are fine.

## Consequences

- `bench/package.py` only writes a local notebook (`submission/<idea>.py`). Uploading it is a
  separate, user-approved step.
- The brief and vendored validator stay local until the user approves a push.
