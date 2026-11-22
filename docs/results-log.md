# Results log (orchestrator)

Official score = 36 metric circuits, `bench/bench.py --mode full` or stricter. Naive = 0.00 %.

## Round 0: harness (2026-11-21 ~12:40 UTC)

| compiler | score | dist % | depth % | validity | notes |
|---|---|---|---|---|---|
| naive | +0.00 | +0.00 | +0.00 | full ✓ | reproduces `naive_baseline.json` exactly (49/49 circuits, all metrics) |

## Round 1 (started 12:55 UTC, time box 15:30 UTC)

Worktrees `/home/hackathon/challenge-wt/<idea>`, branches `idea/<idea>` from `0ea558b`.

| idea | literature # | lane |
|---|---|---|
| gates | 1, 2, 3, 11 | exact resynthesis, hub-as-control, commutation-aware CZ rounds, `schedule_rounds()` |
| hubstream | 5, 6, 2 | park the hub in the zone, stream partners from row 9 at cost ≈ 1.0 |
| rectbatch | 6, 7, 10 | generic rectangle batching of each round, placement optimisation |
| zonedwell | 4, 8, 9 | keep atoms in the zone, in-zone re-pairing, batch staging |

| idea | score | dist % | depth % | validity | commit | verdict |
|---|---|---|---|---|---|---|
| gates | +72.87 | +71.72 | +77.47 | full ✓ 36+13; strict 33/36 so far, no failures | c82ea11 | gate layer is the reference (commutation DAG, hub-as-control, ZXZ merge, in-zone retention); `gatesb` variant +66.39 (batching, HEA dist +75.7) |
| rectbatch | +77.54 | +77.99 | +75.74 | full ✓ 36+13; strict 20q ✓ | 953450e | validator-exact move checker, visitor-first lifting, placement search. **Wall-clock-budgeted search → ±0.2 noise; must be made deterministic before merging** |
| hubstream | +81.51 | +83.80 | +72.33 | full ✓ 36+13; strict running | 87210ad | hub parking (hub moves 2× per circuit), keep atoms in zone when in-between 1q = identity, exact θ=0 elision, reserved hub pair. 50q still ~1.3 atoms/move. **Round-1 leader** |
| zonedwell | +80.79 | +81.68 | +77.25 | full ✓ 36+13; strict 33/36 ✓ + exact Clifford-tableau check on the inconclusive 50q | e0c88c4 | conveyor: hub parked, partners slide one column (1.0 per hub cz); 2×2 1q buffering; QAOA dist −87.6 |

### Round 1 per-circuit comparison (distance/depth, best = lowest weighted cost)

| circuit | file | naive D/P | hubstream | zonedwell | rectbatch | gates | best |
|---|---|---|---|---|---|---|---|
| 01 | hea/bus_stop_12q_hea_linear_p1.qasm | 244/19 | 27/10 | 149/15 | **15/12** | 36/12 | rectbatch |
| 02 | hea/bus_stop_12q_hea_linear_p3.qasm | 732/43 | 123/22 | 412/31 | **44/28** | 107/28 | rectbatch |
| 03 | hea/bus_stop_20q_hea_linear_p1.qasm | 389/19 | 43/10 | 173/15 | **12/12** | 51/12 | rectbatch |
| 04 | hea/bus_stop_20q_hea_linear_p3.qasm | 1168/43 | 207/22 | 482/31 | **36/28** | 154/28 | rectbatch |
| 05 | hea/bus_stop_50q_hea_linear_p1.qasm | 886/19 | 177/10 | 258/15 | **45/12** | 584/12 | rectbatch |
| 06 | hea/bus_stop_50q_hea_linear_p3.qasm | 2659/43 | 726/22 | 735/31 | **134/28** | 1752/28 | rectbatch |
| 07 | ihva/bus_stop_12q_ihva_stagger_p1.qasm | 1957/191 | **181/53** | 516/57 | 249/43 | 317/45 | hubstream |
| 08 | ihva/bus_stop_12q_ihva_stagger_p3.qasm | 5870/501 | **596/140** | 1538/155 | 819/117 | 913/116 | hubstream |
| 09 | ihva/bus_stop_12q_ihva_tree_p1.qasm | 1957/175 | **181/72** | 563/71 | 308/57 | 324/67 | hubstream |
| 10 | ihva/bus_stop_12q_ihva_tree_p3.qasm | 5870/528 | **612/184** | 1615/179 | 866/147 | 983/174 | hubstream |
| 11 | ihva/bus_stop_12q_ihva_tree_warm_p1.qasm | 1957/175 | 86/30 | 563/65 | **89/26** | 301/57 | rectbatch |
| 12 | ihva/bus_stop_12q_ihva_tree_warm_p3.qasm | 5870/528 | 86/30 | 1615/157 | **89/26** | 938/136 | rectbatch |
| 13 | ihva/bus_stop_20q_ihva_stagger_p1.qasm | 3274/327 | 431/88 | 659/83 | **485/67** | 651/54 | rectbatch |
| 14 | ihva/bus_stop_20q_ihva_stagger_p3.qasm | 9821/841 | **1263/236** | 1877/219 | 1591/189 | 1843/168 | hubstream |
| 15 | ihva/bus_stop_20q_ihva_tree_p1.qasm | 3274/247 | **441/100** | 697/91 | 503/85 | 589/73 | hubstream |
| 16 | ihva/bus_stop_20q_ihva_tree_p3.qasm | 9821/795 | **1302/270** | 2036/228 | 1575/225 | 1788/193 | hubstream |
| 17 | ihva/bus_stop_20q_ihva_tree_warm_p1.qasm | 3274/247 | 202/69 | 697/87 | **208/53** | 571/63 | rectbatch |
| 18 | ihva/bus_stop_20q_ihva_tree_warm_p3.qasm | 9821/795 | 202/69 | 2036/208 | **208/53** | 1619/160 | rectbatch |
| 19 | ihva/bus_stop_50q_ihva_stagger_p1.qasm | 8302/837 | 1971/207 | **1485/135** | 2977/159 | 3737/116 | zonedwell |
| 20 | ihva/bus_stop_50q_ihva_stagger_p3.qasm | 24907/2139 | 5579/576 | **4452/396** | 8990/461 | 9773/343 | zonedwell |
| 21 | ihva/bus_stop_50q_ihva_tree_p1.qasm | 8302/555 | 1938/223 | **1909/162** | 2714/187 | 2821/147 | zonedwell |
| 22 | ihva/bus_stop_50q_ihva_tree_p3.qasm | 24907/1933 | 5857/644 | **5428/447** | 8512/533 | 8594/405 | zonedwell |
| 23 | ihva/bus_stop_50q_ihva_tree_warm_p1.qasm | 8302/555 | 751/153 | 1909/151 | **808/116** | 2601/143 | rectbatch |
| 24 | ihva/bus_stop_50q_ihva_tree_warm_p3.qasm | 24907/1933 | 751/153 | 5428/394 | **808/116** | 7340/382 | rectbatch |
| 25 | qaoa/bus_stop_12q_qaoa_p1.qasm | 1957/206 | **235/71** | 326/52 | 301/70 | 314/78 | hubstream |
| 26 | qaoa/bus_stop_12q_qaoa_p3.qasm | 5870/550 | **644/171** | 977/140 | 858/148 | 940/164 | hubstream |
| 27 | qaoa/bus_stop_12q_qaoa_warm_p1.qasm | 1957/212 | **235/72** | 326/52 | 301/70 | 314/78 | hubstream |
| 28 | qaoa/bus_stop_12q_qaoa_warm_p3.qasm | 5870/568 | **644/174** | 977/142 | 858/148 | 943/171 | hubstream |
| 29 | qaoa/bus_stop_20q_qaoa_p1.qasm | 3274/326 | 524/105 | **389/67** | 492/118 | 593/100 | zonedwell |
| 30 | qaoa/bus_stop_20q_qaoa_p3.qasm | 9821/910 | 1262/269 | **1166/185** | 1571/244 | 1744/209 | zonedwell |
| 31 | qaoa/bus_stop_20q_qaoa_warm_p1.qasm | 3274/332 | 524/112 | **389/67** | 492/118 | 593/112 | zonedwell |
| 32 | qaoa/bus_stop_20q_qaoa_warm_p3.qasm | 9821/928 | 1262/280 | **1166/185** | 1571/244 | 1729/223 | zonedwell |
| 33 | qaoa/bus_stop_50q_qaoa_p1.qasm | 8302/791 | 2021/274 | **970/140** | 2668/298 | 2913/178 | zonedwell |
| 34 | qaoa/bus_stop_50q_qaoa_p3.qasm | 24907/2275 | 5861/677 | **2911/396** | 8525/604 | 8566/459 | zonedwell |
| 35 | qaoa/bus_stop_50q_qaoa_warm_p1.qasm | 8302/797 | 2021/273 | **970/140** | 2653/298 | 2913/178 | zonedwell |
| 36 | qaoa/bus_stop_50q_qaoa_warm_p3.qasm | 24907/2293 | 5861/679 | **2911/396** | 8526/604 | 8312/422 | zonedwell |

Portfolio of these four: 86.24 %. Exchange rate: 1 depth unit = 2.92 distance units.

All four branches merged into main (2609b0a). Decision: round 2 builds a **portfolio** compiler (best valid result per circuit) and keeps improving the components where they win.

## Round 2 (15:25–18:45 UTC, incl. a 40 min Opus usage-limit pause)

| idea | score | dist % | depth % | validity | commit | notes |
|---|---|---|---|---|---|---|
| zonedwell r2 | +87.32 | +89.02 | +80.51 | full ✓ 36+13; 20q statevector ✓; 50q Clifford-tableau exact ×6; QCEC ✓ ×7 | 25f752c | compact lanes, θ=0 elision, relaxed hub detection; fixed latent conveyor bug (e884bb5) |
| hubstream r2 | +82.01 | +83.69 | +75.28 | full ✓ 36+13; 12q QCEC, 20q statevector ✓ | 0c9961a | gates' commutation DAG front end + deterministic per-circuit variant grid |
| rectbatch r2 | +76.98 | +77.11 | +76.47 | full ✓ 36+13; deterministic (2 identical runs) | 8b1b536 | fixed work budget; winners unchanged; in-zone parking measured −40 %, off |
| portfolio (zonedwell, hubstream, rectbatch) | **+87.83** | +89.56 | +80.90 | full ✓ 36/36 metric + 13/13 sanity | (this commit) | gates dropped (wins nothing); picks: zonedwell 24, hubstream 6, rectbatch 6 |

Bounds (docs/bounds.md): ceiling 96.9 %. Phase B (hubwalk, systolic) running until 21:30 UTC.

## Zone-span rule introduced (ADR 0010, ~19:45 UTC)

| compiler | under ADR 0010 | notes |
|---|---|---|
| zonedwell r2 | **invalid** (34/36 circuits violate) | conveyor slides across the zone boundary in multi-row moves; fix in progress |
| hubstream r2 | +82.01, 36/36 ✓ | unaffected |
| rectbatch r2 | +76.98, 36/36 ✓ | unaffected |
| gates | +72.87, 36/36 ✓ | unaffected |
| portfolio (zonedwell, hubstream, rectbatch, gates; checker enforces ADR 0010) | **+82.23**, full ✓ 36/36 + sanity 13/13 | was +87.83 before the rule |

zonedwell's own 50q strict note (round 2): 10/12 50q circuits strict-checked before the time limit (QCEC proved 7, 3 inconclusive), plus the 2 unreached; all 5 non-proven pass the exact Clifford-tableau check.

## ~19:30–20:00 UTC: systolic, hubwalk, and validator v1.0.0

| compiler | score (validator v0.0.0 + ADR 0007 + ADR 0010) | under official v1.0.0 | commit |
|---|---|---|---|
| systolic | +88.95 | **invalid** (`idle` on 36/36) | 1d7aeeb |
| hubwalk | +88.91 | not yet checked; hub walk parks partners in the zone → expected `idle` failures | fc00cb6 |
| portfolio (systolic, hubstream, rectbatch, gates) | **+89.18**, full ✓ 36/36 + sanity 13/13 | **invalid** (`idle` on 29/36) | this commit |
| rectbatch / gates | +76.98 / +72.87 | valid 36/36 | – |
| hubstream | +82.01 | `idle` on 15/36 | – |

planqc's **validator v1.0.0** (18:26 UTC) adds: the `idle` rule (every atom in the entangling zone must
take part in a two-qubit gate of each gate layer), single-zone pick-up/drop-off, and bans `rzz`. The
>19q equivalence check is unchanged. The committed portfolio is therefore **not** valid on the new
leaderboard; re-vendoring v1.0.0 and an `idle`-compliant rebuild are next.

benchmaxx (exploit experiment, branch idea/benchmaxx only): +97.67 % on v0.0.0; parked.

## Validator v1.0.0 vendored (~20:10 UTC)

vendor/ now holds planqc's v1.0.0 (`_validation.py`, `errors.py`, `verify.py` updated; `idle` rule,
single-zone moves). bench calls `run_compiler_process(..., network=False)` directly (v1.0.0 dropped the
`network` argument). Naive still reproduces the baseline exactly.

| compiler | v1.0.0 |
|---|---|
| portfolio (hubstream, rectbatch, gates; embedded v1.0.0 replay) | **+79.19**, 36/36 valid (bench full + official v1.0.0 replay) |

systolic and hubwalk are being redesigned for the `idle` rule.

## v5 (final validator), ~22:30 UTC

| compiler | v5 leaderboard set (20 circuits) | validity | commit |
|---|---|---|---|
| systolic (idle redesign: homes in storage, only active pairs in the zone, hub stays, row rotation) | +85.34 | full 20/20, sanity 13/13, v1.0.0 replay clean on 49, Clifford ✓ 50q iHVA | 3ae6025 |
| optimizer (teammate Anton Stolz, `origin/idea/optimizer` 713e5a5) | +80.99 | fast 20/20 | not merged; adds only +0.01 to the portfolio (wins 07, 09) |
| **portfolio (systolic, hubstream, rectbatch, gates)** | **+85.50** | full 20/20 | this commit |
| hubwalk r3 (idle redesign) | +81.56 | full 20/20, sanity 13/13, 20q strict, 50q Clifford ✓ | c9e86df (idea/hubwalk) — **stopped**: wins none of the 20 vs portfolio +85.50 (same storage-home design as systolic, less tuned) |

## idea/optimizer: pulsebeam + postopt (v5 set), ~22:45 CET

`compilers/optimizer` was renamed to `compilers/pulsebeam` (the "optimizer" / "optimizer-optimizer"
entries above are its old state 713e5a5).

| compiler | v5 leaderboard set (20) | notes |
|---|---|---|
| pulsebeam (idle pulse planner, beam search, line layout, block drops) | +81.96 | fast 20/20; 12q iHVA wins 6/8 vs systolic; 12q full (QCEC) 8/8 |
| systolic_po (systolic + `compilers/postopt.py`) | **+86.30** | fast 20/20; postopt = generic gap re-planner, depth unchanged |
| best-of estimate (systolic, systolic_po, pulsebeam, portfolio) | +86.58 | `compilers/portfolio_po` full run pending |

`compilers/postopt.py`: between two CZ layers only start positions, end positions and the 1q gates
in between matter; each gap is re-planned (block drop of the zone, gates, grouped moves to the end
positions) and kept only if cheaper; validator-rule replay before returning. `bench/make_postopt.py
<name>` builds `compilers/<name>_po/` (component + postopt + wrapper).

`compilers/pulsebeam` is now `compilers/optimizer-optimizer` (team naming); `portfolio_po` full run:
+86.63 on the v5 set (20/20 valid, full mode); sanity_circuit_12 times out (postopt on large
circuits), fix pending before it replaces the official portfolio.

## ~23:05 UTC: blockingcz (partner) and solo

| compiler | v5 leaderboard set | validity | commit |
|---|---|---|---|
| blockingcz (partner's notebook, extracted verbatim: S-conjugation, blocking-CZ matchings, 2-leaf hub sweep) | +90.48 | full 20/20, slowest compile 23 s | 84c1d6f |
| solo (pq-solo, Opus 5.5, implements itself) | +90.11 (fast), bf8bc79 +89.94 full | full 20/20 (bf8bc79) | ed8ef89 |
| **portfolio (blockingcz, solo, systolic_po)** | **+90.81** | full 20/20, slowest compile 65 s | this commit |
portfolio_po rebuilt after the optimizer-optimizer size cutoff: **+86.63, full 33/33** (20 scored +
13 sanity, sanity_circuit_12 no longer times out). Deeper postopt variants (beam / local search over
per-layer configurations with free pair and storage sites) did not beat the gap-wise re-plan: the
remaining waste is systolic's fixed, scattered storage homes (single-atom drops/lifts), which a
per-gap optimizer cannot remove without re-planning the storage choreography over many layers.

## systolic2 (fork of systolic, matching-round movement), ~00:10 CET

Phase breakdown of systolic on 50q stagger p3 (5385): matching rounds 79 % (lift 1471, home 1384,
mid trip 794 + 610; 55 rounds of ~5.3 pairs, ~77 per round in ~22 moves), hub groups 21 % (1126).
A round would cost ~5 if its pairs sat side by side in row 9 (lift, drop, lift, drop at ~1.2), so the
headroom is in sorting atoms into the working row, not in the hub stream.

| change | v5 set |
|---|---|
| systolic2 = + order-preserving (DP) placement candidate, block mid-drop when the working row is free | +85.41 (systolic 85.34) |
| systolic2_po | +86.35 (systolic_po 86.30) |
| TOP_HOME = 8 (row 9 kept free as working row) | worse (50q stagger p3 5366 -> 6740: deeper lifts, block mid rarely fits) |

**~23:20 UTC: blockingcz removed** from the project at the user's request (code, results and viz pages). Portfolio rebuilt from solo + systolic_po + rectbatch: **+90.65 %**, full 20/20, slowest compile 46 s. It stays in git history only.

## ~23:55 UTC: solo special-angle blocks

| compiler | v5 leaderboard set | validity | commit |
|---|---|---|---|
| solo (θ ≡ ±π/2 blocks = one cz + local Cliffords; θ ≡ π local Paulis) | **+91.25** | full 20/20 (re-checked by the orchestrator), sanity 13/13, strict 20q 6/6, Clifford tableaux 6/6 50q | 682d68b / docs 6b4d85f |
| portfolio (solo, systolic_po, rectbatch) | +91.25 (solo wins all 20; others are fallbacks) | full 20/20, slowest compile 53 s | this commit |

Submission candidates: `submission/portfolio.py` (with fallbacks) and `submission/solo.py` (solo alone, faster).

## Solo hub rotations and phase lowering

Solo and the packaged portfolio both scored **+91.4753%** locally in full mode, passing all 20
scored circuits. From +91.2483%, total depth dropped from 2004 to 1829 and movement distance rose
from 11747.1 to 11818.3. The 50q tree p3 depth dropped from 418 to 367 at the same distance.

Solo passed 13/13 full sanity circuits and 6/6 strict 20q statevector checks. The six 50q Clifford
variants matched exact tableaux with special-angle synthesis enabled and with it forced off.
The longest compile observed during the parallel runs was 149.5 s for solo and 106.4 s for the
packaged portfolio.

## Solo storage-aware phase retiming and boundary moves

Solo and the packaged portfolio both scored **+91.5548%** locally in full mode, passing all 20
scored circuits. The gain over +91.4753% was 0.0795 percentage points, with total movement
distance down from 11818.3 to 11711.0 and total depth down from 1829 to 1811. The 50q tree p3
improved on both metrics, from 3278.4 to 3248.0 distance and from 367 to 362 depth.

Solo passed 13/13 full sanity circuits and 6/6 strict 20q statevector checks. The six 50q Clifford
variants matched exact tableaux in both synthesis modes. Two 50q solo compiles hit the 300 s
limit during overlapping experiments. The two-worker full rerun passed, with a slowest compile
of 75.9 s. The packaged portfolio passed its parallel run with a slowest compile of 216.3 s.

## relay (fork of solo2, relayed hub fan-out), 2026-11-22 ~04:00 CET

| compiler | v5 leaderboard set | validity | commit |
|---|---|---|---|
| relay (hub fan-out relayed over partner pairs / paths of four, shorter Euler sequences, hub-site DP) | **+92.01** (distance 92.80 %, depth 88.88 %) | full 20/20, sanity 13/13, strict 20q 6/6, Clifford tableaux 50q 12/12 as compiled and 12/12 generic; notebook `submission/relay.py` scores the same | idea/relay |

The gain over solo2 (+91.56) is depth (1 856 → 1 486); distance is flat (11 568 → 11 770). The hub
term of `docs/bounds.md` (two sequential hub pulses per hub block) no longer holds: see
`docs/ideas/relay.md`. Slowest compile 97 s.

## l3abstract (L3 as formal optimisation; peephole pass; solo_wide), 2026-11-22

| compiler | v5 leaderboard set | validity | commit |
|---|---|---|---|
| solo_l3 (solo @b9a2a15 + l3 peephole pass) | +91.63 (distance 11 497.3, depth 1829) | full 20/20, sanity 13/13 | idea/l3abstract |
| solo_wide_l3 (solo @b9a2a15 fork with band-width search + peephole) | **+91.72** (distance 11 368.6, depth 1815) | full 20/20, sanity 13/13; slowest compile 88 s (3 jobs) | idea/l3abstract |

| **relay_l3** (relay + l3 peephole pass) | **+92.21** (distance 11 362.3, depth 1486) | full 20/20, sanity 13/13; slowest compile 165 s (3 jobs) | idea/l3abstract |

The peephole pass is compiler-agnostic (`python l3/make_l3.py <name>` builds `<name>_l3`). On relay it
cuts distance 3.5 % (11 769.6 → 11 362.3) at unchanged depth, 18/20 circuits improved. Details, dead ends and parked steps:
`docs/ideas/l3abstract.md`.

## relay_best_l3: new leader (+92.45), 2026-11-22

| compiler | v5 leaderboard set | validity | commit |
|---|---|---|---|
| **relay_best_l3** = relay_plus2 + rollout finalists + l3 peephole (P1-P10) | **+92.45** (distance 11 113.3, depth 1411) | full 20/20, sanity 13/13 | idea/l3abstract |
| relay_best_l3 with slow-machine guards (ROLL_WALL 150 s, EMIT_WALL 200 s, L3 pass skipped after 230 s), measured under heavy load | +92.39 (distance 11 230.5, depth 1410) | full 20/20 | idea/l3abstract |
| relay_l3 (relay + l3 peephole P1-P10, guarded) | +92.25 (distance 11 283.5, depth 1486) | full 20/20, sanity 13/13 | idea/l3abstract |
| relay_plus (relay + Codex's later solo work, 3-way merged) | +92.16 (fast) | 20/20 | idea/l3abstract |
| relay_plus2 (+ WIDE / HUB_PARK_NEAR variants) | +92.19 (fast) | 20/20 | idea/l3abstract |

Slowest compile 228 s (3 jobs, laptop under load). The guards only fire on slow machines; the
+92.39 run had a background mining job competing for the CPU.

