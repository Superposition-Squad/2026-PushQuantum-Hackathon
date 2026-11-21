# ADR-0001: Score locally with the official validator

- Status: Accepted
- Date: 2026-11-21

## Context

The leaderboard notebook scores a published aqora workspace and only keeps the best score. Each
submission is a manual, outward-facing action, and we must not submit without the user's OK. We have
the complete validator source (`pushquantum_validator`), the 36 metric and 13 sanity benchmarks, and
the naive baseline from the public `planqc/pushquantum-2026-leaderboard` workspace.

## Decision

Vendor the validator unchanged into `vendor/pushquantum_validator/` and build `bench/` around its
public functions (`run_compiler` with `network=False`, `verify_movement`, `verify_equivalence`,
`compute_metrics`) instead of re-implementing them. `bench/bench.py` computes the exact leaderboard
formula. `bench/package.py` turns a compiler into a notebook and scores it through the same import
path the leaderboard uses.

Before trusting the harness, it must reproduce `naive_baseline.json` with an independently written
naive compiler. Status: done, exact on all 49 circuits and all six metrics.

## Consequences

- One command gives the official score in seconds (fast mode), so agents can iterate without aqora.
- If planqc updates the validator or benchmarks, re-download the leaderboard workspace and re-vendor.
- Fast mode skips equivalence. A compiler is only mergeable after `--mode strict` passes.
