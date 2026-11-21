# Docs index

- [scoring.md](scoring.md): the leaderboard metric, validity checks, architecture geometry, move
  rules, benchmark structure.
- [harness.md](harness.md): `bench/bench.py`, `bench/compare.py`, `bench/package.py`, modes and
  outputs.
- [viz.md](viz.md): `bench/viz.py`, the step-by-step visualiser (atom view + wire diagram) for a
  compiled circuit.
- [bounds.md](bounds.md): per-circuit lower bounds and headroom (`bench/bounds.py`).
- [literature.md](literature.md): techniques from the papers corpus mapped to our metric.
- [ideas/](ideas/): one design doc per idea branch, with results and verdict.
- [results-log.md](results-log.md): round-by-round leaderboard of our compilers (orchestrator).

## Decisions (ADRs)

| ADR | Decision |
|---|---|
| [0001](adr/0001-local-harness.md) | Score locally with the vendored official validator; the harness reproduces the naive baseline exactly |
| [0002](adr/0002-cz-only.md) | Gate set rx, rz, cz only (no rzz) |
| [0003](adr/0003-true-equivalence.md) | True equivalence at every size; strict mode adds checks above 19 qubits |
| [0004](adr/0004-agent-hierarchy.md) | Opus orchestrator → Opus planners → DeepSeek Flash implementers, 4 parallel worktrees |
| [0005](adr/0005-no-outward-actions.md) | No submission, publishing or push without the user's explicit approval |
| [0006](adr/0006-single-file-compilers.md) | Each compiler is one self-contained file (stdlib, qiskit, numpy, scipy) |
| [0007](adr/0007-transversal-cz.md) | Respect transversal CZ: every occupied pair gets the same number of pulses between moves |
| [0008](adr/0008-portfolio.md) | Portfolio: run all components, keep the cheapest valid result per circuit |
| [0011](adr/0011-validator-v5.md) | Validator v5.0.0 is final: `idle` rule, no rzz, only 20 circuits scored (18 iHVA + 12q HEA); bench defaults to that set |
| [0010](adr/0010-zone-span.md) | A multi-row move stays within one zone (pickup rows and target rows) |

The original brief, template notebook, dataset and leaderboard notebook are archived in
[`../planqc-pushquantum-2026/`](../planqc-pushquantum-2026/README.md).
