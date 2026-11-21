# ADR-0011: Validator v5.0.0 is final; score the 20-circuit leaderboard set

- Status: Accepted (user decision, 2026-11-21 ~20:50 UTC)
- Date: 2026-11-21

## Context

planqc updated the leaderboard workspace `planqc/pushquantum-2026-leaderboard` five times on
2026-11-21. v5.0.0 is announced as final.

| version | UTC | change |
|---|---|---|
| 1.0.0 | 18:26 | `_validation.py`: **`idle` rule** (in every two-qubit gate layer, i.e. between two moves, every atom in the entangling zone must take part in a two-qubit gate) and **single-zone moves** (all picked-up atoms in one zone, all drop-off sites in one zone). `verify.py`: `run_compiler` lost its `network` argument; `metric_benchmarks(reduced=...)` added |
| 2.0.0 | 19:40 | identical to 1.0.0 |
| 3.0.0 | 19:44 | leaderboard notebook: dropped `network=False` (API fix) |
| 4.0.0 | 20:04 | `architectures/default.json`: entangling zone operations `["cz"]` (**rzz removed**) |
| 5.0.0 | 20:09 | leaderboard notebook: `score_compiler(..., reduced=True)` → **only 20 circuits are scored**: the 18 iHVA circuits (`metric_circuit_07–24`) and the first two HEA circuits (12q p1/p3, `metric_circuit_01–02`) |

Unchanged throughout: the score formula, `naive_baseline.json`, the benchmark circuits, and the
equivalence check (above 19 qubits still only interaction pairs).

## Decision

- `vendor/pushquantum_validator/` is byte-identical to v5.0.0.
- `bench/bench.py`, `bench/compare.py`, the portfolio's internal cost and the visualiser ranking use
  the **v5 leaderboard set** by default (naive totals D = 163 364.3, depth = 13 364). `--full-set`
  gives the old 36-circuit score (written to `results/<name>-full36.json`).
- The `idle` rule is mandatory (enforced by the vendored validator in every bench mode). It
  supersedes the lone-atom allowance of ADR-0007; ADR-0007's equal-pulse-count check stays.
  ADR-0010 (row-based zone span) stays; it implies v1's single-zone moves.

## Consequences

- iHVA is ~90 % of the scored naive distance; QAOA and 20/50q HEA no longer count.
- Exchange rate on the scored set: 1 depth unit = 3.06 distance units (was 2.92).
- Designs that parked waiting atoms in the zone (zonedwell conveyor, hubwalk/systolic hub walks,
  part of hubstream) must keep waiting atoms in storage.
- Portfolio under v5 (hubstream, rectbatch, gates): **+82.01 %**, 20/20 valid in full mode.
