# Idea: portfolio

- Branch: idea/portfolio  ·  Worktree: /home/hackathon/challenge-wt/portfolio
- Base: main 7265da3 (components: hubstream, zonedwell, rectbatch, gates; naive as fallback)
- Status: done (recommended for submission)

## Hypothesis
No single round-1 compiler wins everywhere. The per-circuit best of the four committed results is
+86.24 %, against +81.51 % for the best single compiler. The scorer calls one `compile_circuit` per
circuit, so a compiler that runs all four and returns the cheapest *valid* output should score the
best-of number, up to rectbatch's run-to-run noise.

## Design
Details are in ADR 0008. In short:

- `bench/build_portfolio.py [components...]` generates `compilers/portfolio/compiler.py`. Each
  component's source is embedded as one string literal per line (indentation-proof for
  `bench/package.py`), and its `if __name__ == "__main__"` block is stripped. The component is
  exec'd into its own module `pq_portfolio_<name>`, registered in `sys.modules` so that dataclasses
  and `from __future__ import annotations` work. Module-level imports are checked against the
  sandbox allow-list at build time.
- The validator's `errors.py`, `spec.py` and `_validation.py` are concatenated with the package
  imports removed. `default.json` is inlined, and the logger is renamed and silenced. The script
  fails the build if those anchors change.
- Per circuit, each component runs on `circuit.copy()` with all exceptions caught. Each output then
  goes through the gate-set check, the official replay (trusted spec), the transversal-CZ check
  (`physics_check` copy) and the interaction-pair check (validator logic, all sizes). Up to
  14 qubits, a statevector guard also runs (fixed random product state, fidelity ≥ 1 − 1e-6). The
  cost is `0.8·D/276726.91 + 0.2·P/23676`. The lowest cost wins, and ties go to the earlier
  component. Naive runs only if nothing is valid.
- Time guard: once max(wall, CPU) > 150 s, the remaining components are skipped.
- Debugging: `portfolio_compile(circuit, spec)` returns `(output, report)`, and the env var
  `PORTFOLIO_LOG=<file>` appends one JSON line per circuit. The scorer's sandbox clears the
  environment, so normal runs never write anything. `bench/portfolio_report.py` prints the
  per-circuit pick table, the pick counts, the slowest compile time and the predicted score.
- Negative tests (done in-process on 12q QAOA): a perturbed `rz` angle and a dropped `cz` are
  rejected by the statevector guard, and injected `rzz` and `h` gates by the gate-set check.

## Results
| iteration | commit | score | dist % | depth % | notes |
|---|---|---|---|---|---|
| 1: 4 components + naive | 56b6520 | **+86.24** | +87.93 | +79.49 | full 36/36, sanity full 13/13, `package.py --score` 36/36 +86.24 |

Per family: HEA dist +95.31 / depth +35.48; iHVA +87.38 / +79.91; QAOA +88.34 / +79.74.

Picks (`bench/portfolio_report.py`): rectbatch 12 (all HEA, all `tree_warm`), hubstream 12 (12/20q
iHVA stagger/tree, 12q QAOA), zonedwell 12 (50q iHVA stagger/tree, 20/50q QAOA), gates 0. No
component output was rejected on any circuit. Predicted score from the internal costs is +86.2403 %,
which matches bench, so the internal replay and metric agree with the official ones.

Run-to-run noise: on circuit 13 (20q iHVA stagger p1), rectbatch scored 485/67 in round 1 and
504.7/67 here. That is its wall-clock search under load. The pick flipped to hubstream (430.6/88)
and the total did not change.

Timings (in-process, 2 workers, machine load ~13): the slowest metric circuit (50q iHVA stagger p3)
took 31.3 s wall and 12.4 s CPU. Most of that is rectbatch's budgeted search, about 20 s on every
50q circuit. Sanity circuit 13 (70 qubits) took 39.6 s wall and 33.1 s CPU in-process, but 128 s
in bench under heavy load; gates (17.7 s) is the slowest component there. All of these are under
the 300 s limit, and the 150 s guard bounds the total.

## Verdict
Works. Recommend it as the submission (ADR 0008 draft). Rebuild with
`bench/build_portfolio.py && bench/package.py compilers/portfolio/compiler.py --score` after any
component merges.

Next steps:
1. Make rectbatch deterministic. It is the only noise source, and its ~20 s wall-clock budget is
   the largest share of compile time.
2. Improve components where they nearly win. Exchange rate: 1 depth unit = 2.92 distance units.
   - 20q iHVA: rectbatch has the better depth, hubstream the better distance.
   - 50q `tree_warm`: hubstream has the better distance (751), rectbatch the better depth (116).
3. gates never wins. It can be dropped (`build_portfolio.py hubstream zonedwell rectbatch`) if
   compile time becomes a problem.
