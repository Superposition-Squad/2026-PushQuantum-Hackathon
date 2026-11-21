# Idea: portfolio

- Branch: idea/portfolio  ·  Worktree: /home/hackathon/challenge-wt/portfolio
- Base: main 7265da3 (components: hubstream, zonedwell, rectbatch, gates; naive as fallback)
- Status: in progress

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

## Verdict
(pending)
