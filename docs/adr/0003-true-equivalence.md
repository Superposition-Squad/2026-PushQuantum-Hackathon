# ADR-0003: Compiled circuits must be truly equivalent at every size

- Status: Accepted (user decision)
- Date: 2026-11-21

## Context

The official validator checks full equivalence with QCEC only up to 19 qubits. Above that it only
compares which qubit pairs interact, so a wrong compilation of a 20- or 50-qubit circuit could pass.
The 20- and 50-qubit circuits carry most of the score.

## Decision

Every transformation must be semantics-preserving (equal up to global phase) for every input
circuit. Exploiting the >19-qubit gap is not allowed. `bench --mode strict` adds evidence the
official check lacks:

- ≤ 22 qubits: statevector comparison on three random product input states.
- 50 qubits: QCEC (ZX, simulation and alternating checkers) with a 120 s timeout. `not_equivalent`
  fails. A timeout is reported as inconclusive, and a merge then needs an argument in the idea doc
  for why each transformation is correct.

## Consequences

- Strict mode is slower and runs only as a merge gate, not on every iteration.
- Gate-level optimizations should be ones that are correct by construction (commutation rules,
  exact resynthesis, cancellation) and are tested on the ≤ 19-qubit circuits, where QCEC is exact.
