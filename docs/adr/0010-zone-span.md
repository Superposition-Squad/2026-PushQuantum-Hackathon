# ADR-0010: A multi-row move must stay within one zone

- Status: Accepted (user decision, 2026-11-21 ~19:40 UTC)
- Date: 2026-11-21

## Context

The user stated that selecting several rows that span both the storage and the entangling area in
one move is not allowed. The official validator does not check this (it only replays rectangles on
the grid), so compilers could legally score with such moves on the leaderboard. `zonedwell`'s
conveyor (partner column sliding across the zone boundary as one multi-row move) depended on it.

## Decision

For every `move` that selects **two or more rows**:

- all picked-up rows (`rows`) must lie in one zone (storage rows 0–9, or entangling rows 10–19), and
- all target rows (`new_rows`) must lie in one zone.

A single-row move may still cross the boundary (e.g. row 9 → row 10). Picking rows 8–9 (storage)
and dropping them on rows 10–11 (entangling) is allowed.

`bench/bench.py` enforces this in every mode (`ZoneSpanError`; `--no-zonespan` for experiments
only), and the portfolio's embedded checker rejects violating component outputs.

## Consequences

- On introduction: zonedwell failed 34/36 metric circuits; hubstream, rectbatch, gates passed. The
  portfolio fell from +87.83 % to **+82.23 %** (valid, rebuilt from the compliant results).
- The conveyor has to be redesigned: slide within the entangling zone only, and enter/exit with
  single-zone moves (single-row crossings, batched).
- Interpretation is conservative (targets restricted too). If planqc confirms only pickup rows
  matter, relax the target half.
