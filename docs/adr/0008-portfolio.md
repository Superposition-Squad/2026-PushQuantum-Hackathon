# ADR-0008: Submit a portfolio compiler (best valid result per circuit)

- Status: Proposed (round 2, planner `portfolio`)
- Date: 2026-11-21

## Context

Round 1 produced four valid compilers, and each wins different circuit families (rectbatch: HEA and
`*_warm`; hubstream: 12/20q iHVA, 12q QAOA; zonedwell: 50q iHVA/QAOA, 20q QAOA). Best-of-four is
+86.2 %, while the best single compiler is +81.5 %. The scorer imports one `compile_circuit` per
circuit, with 300 s CPU and 4 GiB, and has no notion of which compiler suits which circuit.

## Decision

The submission is `compilers/portfolio/compiler.py`. **It is generated** by
`bench/build_portfolio.py [components...]` and is never edited by hand:

- Each component's `compiler.py` is embedded verbatim as string literals (one literal per line, so
  the marimo packager's re-indentation leaves it alone). At import time each one is `exec`'d into its
  own module object (`pq_portfolio_<name>`, registered in `sys.modules`), so clashing names such as
  `Move` and `Architecture` stay isolated. A component that fails to load is skipped.
- For each circuit, every component runs on a copy of the input. Each output is checked inside the
  compiler:
  1. gate set `rx, rz, cz, move, measure, barrier` (ADR 0002), with float-convertible params;
  2. the official replay: an embedded copy of the validator's `errors.py`, `spec.py` and
     `_validation.py`, with the default architecture inlined, always run on the trusted spec;
  3. transversal CZ (a copy of bench's `physics_check`, ADR 0007);
  4. the interaction-pair set, exactly as the validator's `_check_interactions` (required pairs from
     `transpile(..., optimization_level=1)`, allowed pairs from the reference), at every size;
  5. up to 14 qubits, a statevector guard: both circuits are applied to one fixed random product
     state, and the fidelity must be 1 up to 1e-6.
- The cost is computed exactly as the score: `0.8·D/276726.91 + 0.2·P/23676`, where D is the
  replay's per-move maximum distance and P is the depth with moves and barriers stripped. The
  valid output with the lowest cost is returned. Ties go to the earlier component in build order.
  `naive` runs only if no component is valid.
- A time guard skips the remaining components once 150 s of wall or CPU time has been spent (it
  never skips the fallback). Locally the slowest circuit takes about 30 s in total.

## Consequences

- The score is the per-circuit best of the components, so improving a component on the circuits it
  already wins (or nearly wins) pays off directly. Rebuilding is one command, after a component merges.
- Equivalence (ADR 0003) is inherited: the portfolio only ever returns an output that a component
  produced, and every component must pass `bench --mode strict` on its own before it is added. The
  internal guards add cheap evidence; they do not replace QCEC.
- Every component must be deterministic, or the pick (and so the score) becomes noisy. rectbatch's
  placement search is wall-clock-budgeted, measured from its own start, so other components cannot
  starve it. Its result can still vary with machine load until its planner makes it deterministic.
- Compile time is the sum of the components' times, so the CPU budget (300 s) needs watching as
  components get slower.
- The generated file is large (~220 KiB), and its components are readable only through the embedded
  strings. For review, read `compilers/<name>/compiler.py` at the build commit. The file header records that
  commit and each embedded source's sha256.
