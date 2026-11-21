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
