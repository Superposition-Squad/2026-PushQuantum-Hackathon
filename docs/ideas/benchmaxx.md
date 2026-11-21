# Idea: benchmaxx (verifier-exploit experiment, NOT a submission)

- Branch: idea/benchmaxx · never merged into main or the portfolio
- Status: parked (stopped by the user after planqc published validator v1.0.0)

## What it does
Maximises the *official* leaderboard score of the original validator (v0.0.0) by exploiting its holes:
native `rzz` (listed in the spec by mistake), no equivalence check above 19 qubits (only the set of
interacting pairs is compared → one 2q gate per required pair, nothing else), no transversal-CZ /
idle check, no zone-span check, final measurements not required.

## Results (validator v0.0.0)
- **+97.67 %** official (36/36 full mode; distance +98.29 %, depth +95.20 %), commit 6c39232.
  The >19q pair-only trick alone: 0 → 96.12 %. rzz + ≤19q quirks on the 12q circuits: +~1.3 points
  (`compilers/bmsmall/`, 12q full 12/12 at +93.57 % partial, sanity 13/13 equivalent).
- All 24 circuits with > 19 qubits are **not equivalent** to their input by design.

## Status under validator v1.0.0 (2026-11-21 18:26 UTC)
v1.0.0 bans rzz, adds the `idle` rule (every atom in the entangling zone must be in a 2q gate of each
layer) and single-zone moves, but does **not** change the >19q equivalence check. An update to the
new rules would likely still score ~90 %+ via the >19q gap. Not pursued; kept as evidence for a
report to planqc (user decision).
