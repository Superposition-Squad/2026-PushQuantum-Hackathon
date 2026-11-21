# Idea design docs

One file per idea branch: `docs/ideas/<idea>.md`. The planner writes it **before** any code and
updates it after every bench iteration. Keep it short.

## Template

```markdown
# Idea: <name>

- Branch: idea/<name>  ·  Worktree: /home/hackathon/challenge-wt/<name>
- Base: <commit or compiler it starts from>
- Status: planned | in progress | merged | rejected | parked

## Hypothesis
What changes, and why it should lower move distance and/or depth (with a rough estimate).

## Design
Placement, move scheduling, gate synthesis and optimisation. How it stays valid: move rules, the
transversal-CZ rule (ADR 0007), equivalence (ADR 0003). Sources: arXiv ids if borrowed.

## Results
| iteration | commit | score | dist % | depth % | notes |
|---|---|---|---|---|---|

## Verdict
Merged / rejected / parked, and why. What should carry over to other ideas.
```
