# ADR-0007: Respect transversal CZ (physics check beyond the validator)

- Status: Partly superseded by ADR-0011 (validator v1.0.0 `idle` rule forbids lone atoms in the zone during a gate layer); the equal-pulse-count check stays
- Date: 2026-11-21

## Context

The brief says the CZ pulse "affect[s] all neighboring qubit pairs in the entangling zone
transversally", while single atoms only see the identity. The validator checks each `cz` gate on its
own and never simulates the pulse. A compiler could leave two atoms parked on an interaction pair
while other pairs receive a CZ. That passes validation but would apply unwanted CZs on hardware, so
it isn't really equivalent (the spirit of ADR 0003).

## Decision

Atom positions are fixed between two consecutive `move` instructions. Within each such segment,
every occupied interaction pair must receive the same number of `cz` gates (the number of pulses),
and no `cz` may act on an unoccupied pair. `bench` enforces this in every mode (`PhysicsError`). It
can be disabled with `--no-physics` for experiments only.

## Consequences

- Reuse strategies (keeping atoms in the entangling zone across layers) must either separate atoms
  that should not interact (different pairs, or a lone atom on a pair) or give them a matching CZ.
- The naive compiler passes trivially (one pair at a time).
- If the user rejects this rule, drop the check. Compilers that pass it remain valid either way.
