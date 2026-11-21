# ADR-0002: Native gate set is rx, rz, cz only (no rzz)

- Status: Accepted (user decision)
- Date: 2026-11-21

## Context

The validator's `architectures/default.json` lists `["cz", "rzz"]` for the entangling zone, so a
native `rzz` would pass validation and would replace each `cx–rz–cx` ZZ block with one gate. That is
a large depth lever. The prose brief only names global/targeted Rx, targeted Rz and transversal CZ as
native gates.

## Decision

Use only `rx`, `rz` (storage zone) and `cz` (entangling zone), plus `move`, `measure` and `barrier`.
Compilers must never emit `rzz` or any other gate.

## Consequences

- ZZ interactions cost two CZs plus an `rz` in storage. The atom must leave the entangling zone
  between the two CZs, or the block is rewritten into a form that needs fewer round trips.
- We may lose score to teams that use `rzz`. If planqc confirms `rzz` is intended, revisit this ADR;
  the change would be local to gate synthesis.
