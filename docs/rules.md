# Rules: what is allowed and what is not

This is the binding rule set for every compiler in this repo. **Validator update 2026-11-21 20:12:**
rzz now rejected, CZs blocking (transversal), pickup and drop-off cannot cross zones; ancillas
announced to become allowed. Only submissions after 20:12 count. It combines the brief, the official
validator and what planqc told us directly. **Where the validator is more permissive than the rules,
the rules win.** planqc confirmed that using validator gaps is not allowed, even though they are not
enforced.

> Our vendored copy in `vendor/pushquantum_validator/` is still the version **before** 20:12 (it
> still accepts `rzz`, does not check transversal CZ, and accepts rectangles spanning both zones).
> Until the new validator is vendored, everything marked "since 20:12" is enforced only by our own
> harness (`bench/bench.py` transversal check, ADR 0007) and by our compilers' design.

Legend:
- Source: **B** = brief, **V** = validator, **P** = planqc said so in person.
- "V checks?": ✓ = the validator rejects violations; ✗ = it does not, so **our harness must check it**.

## 1. Gates and zones

| # | Rule | Source | V checks? |
|---|---|---|---|
| G1 | `rx`, `rz` (any angle) only on atoms in the **storage** zone. Single-qubit targeting is allowed | B, V | ✓ |
| G2 | `cz` only on an **interaction pair**: entangling zone, same row, columns (2k, 2k+1) | B, V | ✓ |
| G3 | **No other gate in the entangling zone.** In particular no `rzz` | P, V (since 2026-11-21 20:12) | ✓ |
| G4 | `measure` and `barrier` are allowed anywhere | V | ✓ |
| G5 | Output gate set: `rx`, `rz`, `cz`, `move`, `measure`, `barrier`. Nothing else | B (ADR 0002) | partly |

## 2. CZ pulses are transversal

| # | Rule | Source | V checks? |
|---|---|---|---|
| T1 | A CZ pulse hits **every occupied interaction pair** in the entangling zone at once (CZs are blocking) | B, V (since 20:12) | ✓ |
| T2 | So between two consecutive moves, **every occupied pair gets the same number of `cz`** (= the number of pulses). You may not write a `cz` for pair A while pair B sits occupied without one | B (ADR 0007) | ✓ since 20:12; our harness checks it in every mode |
| T3 | **The `cz` lines between two moves describe the pulses of that segment.** One pulse = one `cz` line for *every* occupied pair (written consecutively, no move in between). k pulses = exactly k `cz` per occupied pair. (Confirmed reading, 2026-11-21: `cz(A); cz(B)` with A and B both occupied is *one* pulse, not two) | B, P | ✓ since 20:12 |
| T4 | A **lone atom** on a pair (its sibling site empty) is fine during a pulse: it gets identity | B | ✓ (nothing to check) |

## 3. Moves (one `Move` = one AOD pair = one rectangle)

| # | Rule | Source | V checks? |
|---|---|---|---|
| M1 | Picked-up rows and columns are each **contiguous and ascending** | B, V | ✓ |
| M2 | Target rows and columns are **ascending**: order is preserved, no permutation of rows or columns | B, V | ✓ |
| M3 | No more target lines than picked-up lines. Empty picked-up lines may be dropped; the occupied ones map one-to-one | V | ✓ |
| M4 | **Every atom inside the picked rectangle moves** and must be listed. No ghosts (an atom left behind inside the rectangle makes the move illegal) | B, V | ✓ |
| M5 | **The source rectangle lies entirely in one zone, and the target rectangle lies entirely in one zone.** They may be different zones (storage → entangling or back), but no rectangle may span both | P, V (since 20:12) | ✓ since 20:12 (not in our vendored copy; `compilers/optimizer` enforces it in its own move checker) |
| M6 | **Non-proportional spacing is allowed**: rows and columns may be stretched or compressed, as long as M2 and M5 hold. The validator only warns, and the warning doubles `time`, which is not scored | B, P | (warning) |
| M7 | **Dropping into a region that already holds atoms is allowed**, as long as **no occupied site receives a new atom**. Atoms that stay put may sit inside the target rectangle | P, V | ✓ (occupied site) |
| M8 | Every target site must lie inside a zone | V | ✓ |
| M9 | A move is atomic: pickup, travel and deposit are not interleaved with other moves | B | (n/a, sequential) |
| M10 | **Moving atoms may pass over unrelated atoms.** Only the atoms inside the **source rectangle** are picked up (all of them, M4). Atoms the move travels over or lands next to stay where they are | P | ✓ (nothing to check) |

### Parallel moves (several AOD pairs at once)

These rules only matter for the brief's distance metric. They don't affect the leaderboard, see §6.

| # | Rule | Source | V checks? |
|---|---|---|---|
| P1 | No **ghost regions**: where one pair's rows cross another pair's columns outside an active region, there must be no atom, during pickup, travel and deposit. This is the only case where passing over atoms matters (M10 covers a single AOD pair) | B | partly (step grouping) |
| P2 | During travel, no pair's active region may overlap another AOD | B | partly |
| P3 | The next set of parallel moves starts only after the current set has finished | B | ✓ |

## 4. Equivalence and qubits

| # | Rule | Source | V checks? |
|---|---|---|---|
| E1 | The compiled circuit (moves and barriers stripped, final measurements removed) must be **equivalent to the input up to a global phase, at every size** | B (ADR 0003) | ✓ only ≤ 19q; **✗ above** (only the pair set is checked) |
| E2 | **Ancillas are allowed by the rules, but the validator rejects a different qubit count above 19q.** So in practice: same qubit count as the input, no ancilla tricks (not worth it at ≤ 19q alone) | P, V | ✓ (> 19q) |
| E3 | Compiled interaction pairs ⊆ the input's pairs, and every required pair is present | V | ✓ (> 19q) |
| E4 | Initial loading: every qubit on a distinct storage site. Loading happens only once, at the start | B, V | ✓ |

## 5. Geometry facts (not rules, but easy to get wrong)

- **Storage:** physical (x, y) = (col, row), spacing 1.0, rows 0–9.
- **Entangling:** rows 10–19, physical y = row, physical x = 2·(col // 2) + 0.3·(col % 2). So the columns sit at x = 0, 0.3, 2, 2.3, ….
- **The zones touch in the scorer:** storage row 9 → entangling row 10 = 1.0, the same as any row step. At an odd column the crossing costs √(1 + 0.7²) ≈ 1.22.
- Always compute distances from the spec with the validator's formula; never hard-code them.

## 6. Scoring facts (not rules)

- **Leaderboard distance** = Σ over every `move` of its farthest atom's path. Parallelism does not reduce it.
- **The brief's distance** = per set of parallel moves, only the longest counts. We report it as a second figure, computed from the validator trace's parallel steps.
- **Depth** = `depth()` of the circuit with moves and barriers removed. Measurements are included.
- **Score** = 0.8 × % less distance + 0.2 × % less depth, both relative to naive. One depth unit is worth 2.92 distance units.
- **Limits:** 300 s per circuit, sandboxed, no network, 4 GiB. The sandbox passes a minimal environment (environment variables do not reach the compiler).

## 6b. Two profiles: one compiler, two settings

| | **A: Submission** (default) | **B: Challenge best effort** |
|---|---|---|
| Constraints | validator ∩ planqc rules: everything above, so **always passes the validator** | planqc rules only. Ancillas allowed (E2); parallel laser pairs per P1–P3 |
| Objective | leaderboard: Σ over every `move` + depth, score formula | brief: per set of parallel moves only the longest counts, plus depth |
| Checked by | official validator + our harness checklist (§8) | our own checker: rules §1–§4, brief parallel rules P1–P3, equivalence (QCEC ≤ 19q, statevector/Clifford above, ancillas allowed) |
| Used for | leaderboard submission | pitch and jury: "true" metric, what the hardware could do |

- Build this as **one compiler with a profile switch** (constraints + objective), not two codebases.
- **Always** report the brief metric for profile A's output too: it's cheap to compute from the move list.
- Priority: A first. B only once A is frozen, or as reporting only (optimizing for B is a different objective).

## 7. Still unclear (ask planqc; until then, choose the stricter option)

- Ranking: leaderboard plus pitch, possibly a jury (answered). Still open: are there hidden benchmarks?
- Scorer environment: is the 300 s per circuit, CPU or wall clock, on what hardware? Which package versions? Is a seeded or time-limited search acceptable?
- Is Sunday 10:00 the final submission deadline?
- Where do the problem-circuit function, the encoding documentation and the ansatz description go?
- Exact counting in the new validator: how it groups `cz` lines between moves (T3), and its exact reading of M5. Verify as soon as the new validator is vendored.

## 8. Harness checklist (everything the validator does not catch)

Every candidate output must pass all of these before it can enter the portfolio. Since 20:12 the validator covers 1–3 as well; keep our checks anyway as a second line:

1. G3: no gate other than `cz` on an atom in the entangling zone.
2. T1–T3: transversal CZ, i.e. equal `cz` count on every occupied pair between two moves (`bench/bench.py`, every mode).
3. M5: source and target rectangles of every move each lie in one zone (**TODO: add to `bench/bench.py`**; for now a scratch check, all of optimizer / rectbatch / hubstream pass, zonedwell does not).
4. E1 above 19q: our own equivalence evidence (statevector ≤ 22q, Clifford/QCEC check at 50q).
5. Then the official replay and metrics.
