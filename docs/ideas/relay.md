# Idea: relay — solo2 plus a relayed hub fan-out

- Branch: idea/relay · Worktree: /home/hackathon/challenge-wt/relay · File: `compilers/relay/compiler.py`
- Base: fork of `compilers/solo2` (see [solo2.md](solo2.md) and [solo.md](solo.md) for the rest of the design).
  Validator v5.0.0.
- Status: valid end-to-end; results below.

## Hypothesis

`docs/bounds.md` assumes the hub needs two sequential cz pulses per hub block. It does not. A hub batch
is `F · ∏ rx_p(θ_p) · F` with the fan-out `F = ∏_p cz(hub, p)`, and `F` only depends on the **parity**
of the partners. Parity can be collected along partner–partner pairs, in parallel, in one storage band:

    cz(hub, l) · cz(hub, r)  =  P · cz(hub, r) · P,      P = rx_r(−π/2) · cz(l, r) · rx_r(π/2)   (a controlled-Y)

More generally, with relay edges child → parent forming a forest (children first),
`F = P† · ∏_roots cz(hub, root) · P`. Proof sketch: `P† Z_parent P = Z_child Z_parent`, `Z_child` is
unchanged, and `cz(hub, root)` is a function of `Z_hub, Z_root` only. Checked numerically on 6 qubits
against the original iHVA blocks before any compiler work.

So the hub only meets one root per group; the pair pulses of one level are disjoint and go into one
rigid band lift. Expected: half (pairs) or a quarter (paths of four) of the hub pulses per pass.

## Design

**Groups.** Pairs `(l, r)` (one level) or paths of four `l0–r0–R–l1` (two levels: `l0→r0`, `l1→R`,
then `r0→R`). They sit side by side in a storage band: level 1 lifts the band on (even, odd) pair
slots, level 2 lifts the inner atoms shifted by one column, so no relocation between the levels.
Above 19 qubits the groups are found by a randomised greedy path packing on the input's pairs;
at 19 qubits or fewer any pair may be used (the validator then runs full equivalence only).

**Validator pair rule (important).** Above 19 qubits every pair of the input must interact at least
once in the compiled circuit. A relay leaf never touches the hub, so:
- the roots **alternate between the two passes** of a batch (pair: the other atom; path: the other
  inner atom), so both atoms of a pair, and both inner atoms of a path, meet the hub;
- the end atoms of a path are only allowed if they already met the hub or still have a hub block to
  come (`Machine.met`, `Executor.hub_left`); single-pass batches (θ = ±π/2 blocks) likewise;
- every candidate is rejected unless all hub partners met the hub, and the emitted circuit is checked
  again. The first version without this failed 5 of 18 circuits in `--mode full`.

**Placement, chosen per batch by simulation** against the plain strategies (rotate, direct, compact,
gravity), cost = distance + 3.06 × depth change:
- `1`, `2`: one or two levels on a compact template in the bottom rows (two paths and one pair per row);
- `-1`: one level, pairs brought together by the band-round machinery (best rectangle by simulation).
  This is what helps at 20 qubits.

**Hub service.** The hub's sites for the partners of row 9 come from a small dynamic programme over
the ten hub sites and both sweep directions (also used by the plain strategies).

**Basis changes.** The parent's `rx(π/2)` absorbs the trailing diagonal of its buffer
(`V = rx(π/2) · D⁻¹`; any `V` with `V† Z V` in the XY plane gives a valid controlled-W, and the same
`V` is reused for the uncompute). Buffers are flushed lazily (a trailing `rz` stays pending).

**Shorter Euler sequences (independent of relays).** `_zxz` now uses
`Rz(φ) Rx(θ) Rz(λ) = Rz(φ−π) Rx(−θ) Rz(λ+π)` when that has fewer gates. solo2 emitted every negative
`rx` as three gates.

**Search.** As solo2, plus relay variants for the top configurations (relay off, level subsets,
depth weight) and policies that turn small hub groups into batches. Identical configurations are
no longer run twice. `SEARCH_WALL = 180 s` is a safety net for slow machines only.

## Results (v5 leaderboard set)

**+92.01 %** (`--mode full`, 20/20; solo2 +91.56 %): distance 11 568 → 11 770 (92.92 % → 92.80 %),
depth 1 856 → 1 486 (86.11 % → 88.88 %). Slowest compile 97 s (50q stagger p3; solo2 45 s).

| circuit | solo2 dist / depth | relay dist / depth |
|---|---|---|
| 12q HEA p1 | 10.1 / 10 | 10.1 / 9 |
| 12q HEA p3 | 33.2 / 22 | 33.2 / 21 |
| 12q stagger p1 | 114.1 / 37 | 119.1 / 37 |
| 12q stagger p3 | 380.1 / 103 | 401.6 / 95 |
| 12q tree p1 | 138.1 / 37 | 138.1 / 39 |
| 12q tree p3 | 402.9 / 103 | 408.1 / 105 |
| 12q tree_warm p1, p3 | 42.4 / 15 | 42.4 / 13 |
| 20q stagger p1 | 208.5 / 54 | 205.9 / 43 |
| 20q stagger p3 | 750.7 / 151 | 762.8 / 117 |
| 20q tree p1 | 249.4 / 58 | 249.0 / 54 |
| 20q tree p3 | 812.3 / 181 | 850.2 / 152 |
| 20q tree_warm p1, p3 | 80.8 / 21 | 80.4 / 20 |
| 50q stagger p1 | 734.9 / 115 | 736.2 / 84 |
| 50q stagger p3 | 2631.9 / 337 | 2646.8 / 194 |
| 50q tree p1 | 1071.3 / 125 | 1124.9 / 95 |
| 50q tree p3 | 3265.6 / 369 | 3327.3 / 295 |
| 50q tree_warm p1, p3 | 259.4 / 41 | 255.4 / 40 |

Three 12q circuits are slightly worse than solo2 (the shorter Euler sequences change the form
choices; about 0.01 points in total). A best-of with solo2 would recover that at the price of
solo2's compile time.

| iteration | score | notes |
|---|---|---|
| one level, pairs via band rounds, no pair-rule handling | invalid | 5/18 fail the pair check above 19 qubits |
| two levels on a template, alternating roots, coverage tracking | +91.89 (full) | first valid version |
| Euler fix, diagonal-absorbing basis change, lazy flushes | +91.97 (fast) | |
| relay variants in the search, band-round placement as an option | +92.01 (full) | this commit |

## Where it helps and where it does not

- The gain is almost all **depth**; distance is roughly flat. Relay batches save hub serves but pay
  for band lifts and for relocating partners into the band (about 110–130 per 50q batch).
- Break-even is about 15 partners per batch. The 12-qubit circuits never use relays.
- `tree` circuits release hub blocks in forced small groups (1, 1, 1, 7, 1, 7, 1, 31 per layer), so
  only the group of 31 is relayed.
- p = 1 circuits can only use one level (every partner has a single hub block, so each must be a
  root once). p = 3 circuits use two levels in most batches.

## Measured and rejected (do not retry without a new angle)

- **Clifford frame for the hub (Z2 gauge).** Conjugating by a fan-out makes all hub terms of one role
  single-qubit and leaves non-hub blocks unchanged, but the roles alternate per layer and the frame
  switch is a weight-n Clifford rotation (at least 2n − 4 cz, gossip bound). No pulse saving.
- **Chain layout for consecutive rounds.** 51 % of next-round blocks on 50q stagger p3 pair atoms
  that were both in the previous round, so pairs were laid out along chains to make the next round's
  pairs adjacent. Reordering a band costs more than the next round saves (e.g. +93 to save 19).
- **Second assignment pass pulling next-round partners together**, **relay pairs biased to the next
  block**, **position-aware choice among equal groupings**, **all-or-nothing dummy pairs**: each
  moves single circuits by ±5 % in both directions. The last two are kept as search variants.
- **Best-first search over single-option changes, 3× the runs:** not better than the two-stage search.
- **1.5× candidate budget:** +0.01 to +0.02.

Round relocation is still 43–50 % of the 50q distance and remains the open problem.

## Validity evidence

All on the committed file:

- `bench --mode full`: 20/20 (QCEC equivalent on all eight 12-qubit circuits, pair check above).
- `bench --set sanity --mode full`: 13/13.
- `bench --mode strict --filter 20q_ihva`: 6/6 (statevector on three random product inputs each).
- Exact Clifford tableaux on all six 50-qubit circuits, two angle draws each
  (`docs/ideas/relay_cliff_check.py`): 12/12 equal as compiled, and 12/12 equal with `--generic`
  (special-angle classification forced off, so the two-pass relayed path runs: e.g. 1 106 cz instead
  of 882 on 50q stagger p3).
- Deterministic: two compiles of 20q stagger p3 and of 50q tree p1 give identical circuits.
- Transversal-CZ and zone-span checks are part of every bench mode (ADR 0007, ADR 0010).

## Verdict

Keep. What carries over: the relay identity (it invalidates the hub term of `docs/bounds.md`), the
pair-coverage bookkeeping it needs above 19 qubits, the Euler fix, and the hub-site dynamic programme.
Next: the relocation into the relay band and before ordinary rounds.
