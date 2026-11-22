# Idea: L3 as a formal optimisation problem (l3abstract)

- Branch: `idea/l3abstract` · Worktree: `wt-l3abstract/` · Code: `l3/` (experiments, not a compiler yet)
- Status: in progress. Everything below is measured on solo's v5 outputs (cached in `l3/cache/solo/`).

## Goal

Treat the movement layer (L3) as a pure constrained optimisation problem: no quantum content, the
gates and their per-qubit order are fixed, only positions and moves are chosen. Then translate it to
known solver families instead of hand-writing strategies.

## The formal problem

Input (`l3/model.py`, `extract`): `pulses[k]` = pairs that must sit on interaction pairs at pulse k
(nobody else in the zone: idle rule); `seg[q][i]` = 1q gates of atom q between its CZs (need storage).
Derived: `B_k` = atoms in pulses k-1 and k that owe 1q gates before k (the only *binding* storage
visits; every other `s` is satisfied automatically because a non-pulsed atom is in storage).

Decision: configurations `X_k` (atom → site) at every pulse, and the moves between them.

    minimise   Σ_k g_k(X_{k-1}, X_k)
    s.t.       X_k: members of pulse k on pairs next to their partner, everyone else in storage
    g_k        = cheapest legal move sequence X_{k-1} → X_k in which every atom of B_k touches storage

A move: contiguous rectangle, every atom inside moves, monotone row map × monotone column map,
source and target each in one zone; cost = longest single-atom path. Score is 80 % distance.

This is a chain-structured problem (shortest path through layered configurations) with
- outer problem: assignment / layout over X (local search, SA, LNS, DP with beam);
- inner problem g_k: set partitioning of transfers into legal rectangle moves (clique cover /
  maximal independent sets of a conflict graph), often through intermediate waypoints.

Exact translation: time-expanded MILP (`l3/milp.py`, HiGHS via `scipy.optimize.milp`). One-hot
row/col per atom per state; per slot a picked contiguous row/col interval with integer monotone maps;
atoms inside the rectangle are moved by the maps (ghost rule built in); single-zone source/target;
occupancy; pulse and storage-visit constraints; slot cost ≥ polygonal (16-direction) norm of every
picked atom's displacement (exact cost recomputed afterwards).

Infrastructure: `Emitter` realises any plan and checks every rule; `check.py` runs the official
replay + zone-span + transversal-CZ checks; `signature` proves per-qubit gate sequences unchanged.

## Results so far

| experiment | file | result (solo total D = 11 818 on the 20 scored circuits) |
|---|---|---|
| where solo's distance goes | `l3/movekinds.py`, `l3/breakdown.py` | storage→storage shuffles 52 % (mostly single-atom moves, avg 2.2); 50q rounds 72 % of 50q distance; solo's band is tall (5 rows × 4 cols → 5.05 per band move) because storage rows 5–9 are packed |
| regroup shuffles with fixed targets (bound) | inline | at most ~10 % of shuffle cost → dead end |
| per-gap re-planning with fixed endpoints (postopt on solo) | `compilers/postopt.py` | 11 818 → 11 777 (−0.35 %) |
| zone-resident greedy planners ("reuse") | `l3/planner.py`, `l3/planner2.py` | 1.5–4× worse than solo: lifts/drops not batched (1.1–2 atoms per move, 3–4 per move) |
| greedy inner solver on solo's own endpoints | `l3/chain.py`, `l3/innercheck.py` | 2× worse than solo: good gaps route through intermediate configurations |
| exact MILP of a whole gap | `l3/milp.py` | ≤ 5 moves / ≤ 10 atoms solve in seconds; 6+ slots with 11 atoms: no feasible solution in 120 s |
| **MIP-LNS** (exact re-solve of sliding 3-move windows) | `l3/milp_lns.py` | 12q tree p1: 138.1 → 128.7 (−6.8 %), validated; full set: see below |

Patterns found by the MILP, turned into peephole operators (`l3/peep.py`, MILP-free, seconds):

| op | pattern (example from solo's plans) |
|---|---|
| P1 merge | two moves of one gap that together are legal (e.g. two diagonal single-atom moves forming a rigid segment: 1.41 + 1.41 → 1.41) |
| P2/P2b trim | a dip took a whole row down and up although only one atom owed gates: move only that atom, straight down (1.22 + 1.22 → 1.04 + 1.04). Must drop all unneeded riders at once |
| P3 waypoint | re-choose an atom's intermediate site |
| P4 split | take an atom out of a group so that it can merge elsewhere |
| P5 ride-along | split a long single-atom move so that one half rides with another translation move (4.0 → 2.0 + shared 2.0) |
| P5b staging absorbed | instead of a staging move before a block lift, ride along with the block from the original site and fix up in the zone (1.41 → 1.04) |
| P6 general ride-along | short pre-move to a waypoint from which the atom joins another move with non-uniform row/col maps (MILP pattern on 20q tree p1) |
| P8 compose | (MIP-LNS on relay_l3) a rigid drop followed by a shuffle of some of the same atoms: the drop's column map takes them straight to their final sites (several atoms jointly, maps stay monotone) |
| P9 stack-lift | two lifts from different storage rows become one lift into two zone rows plus a cheap zone shift (2.24 + 2.63 → 2.63 + 1.04) |
| P10 round-trip retarget | a rigid dip into rows 7–8 while rows 8–9 are free: shift both moves of the round trip (3.08 + 3.08 → 2.12 + 2.12 for 20 atoms) |
| P6b map-preimage ride-along | the waypoint is the preimage of the target under the other move's row/col maps; may also join an *earlier* move |
| (not built) passing lane | reorder atoms in a storage row by a detour through the empty zone (MILP found it on tree_warm, ~0.6) |

Peephole on solo's 20 plans: **11 818.3 → 11 497.4 (−2.7 %, ≈ +0.16 points)** (P1–P6), every plan validated
(official replay, zone span, transversal CZ, unchanged per-qubit gate signature). Packaged as
`compilers/solo_l3/` by `l3/make_l3.py` (solo + embedded pass, falls back to solo on any error).

MIP-LNS vs peephole (3-move windows, 5 s per window, HiGHS): on the 12q/20q plans, MIP-LNS alone
gains 2–5 %, about as much as the peephole, but takes 100–450 s per circuit (each tiny window
needs 3–5 s: the big-M relaxation is weak; symmetry breaking did not help). Running the peephole
on the MIP-LNS output roughly doubles the peephole's gain (12q stagger p1: solo 126.3, peep 117.2,
LNS+peep 115.1; 20q tree p1: 271.0 / 262.4 / 258.3). The rest is fine-grained (non-uniform
ride-along through a waypoint, passing lane). Method that worked: **solve small windows exactly
offline, read off the rewrite patterns, distil them into fast operators.**

Compile time: the pass adds ≤ 25 s at 50q (deterministic cap of 250 000 replays, no wall clock).
Under heavy machine load the bench measured 267 s for 50q stagger p3 (solo + pass); check
standalone before relying on it (hard limit 300 s).

Why the 50q rounds stay expensive: solo's band is 5 storage rows tall (rows 5–9 packed with 50
atoms), and a 5-row stack needs ≥ 5 per lift / dip / return (rows map monotonically). A 2-row
band would cost ~2.1 per band move (~12 less per round, 52 rounds at 50q p3), but needs a
different storage layout over time — an outer-problem change no local optimiser of solo's plan can
make. The zone-resident ("reuse") planners failed for the same reason: storage near the zone was
full, so dips and newcomers travelled 4–9 per move, one atom at a time.

## Where the ideas come from (attribution)

Paper summaries are from `docs/literature.md` (arXiv ids there) plus memory; not re-verified.

| idea used here | source |
|---|---|
| exact solver-based layout synthesis on a discretised state space; it does not scale, so solve windows | OLSQ-DPQA (2306.03487); our full MILP hits the same wall |
| inner problem = grouping compatible moves (conflict graph, maximal independent sets) | Enola (2405.15095); also MQT routing-aware placement (2505.22715) |
| keep atoms in the entangling zone when the next stage needs them ("reuse"); matching-based placement | ZAC (2411.11784); our `planner2` is a greedy version (failed on batching) |
| storage used actively, joint scheduling/placement | PowerMove (2411.12263) |
| order atoms so monotone AOD maps exist (cross-minimisation) | Arctic (2405.06183) |
| windowed exact re-optimisation of an incumbent (MIP-based large neighbourhood search) | standard OR technique (LNS, Shaw 1998; MIP-LNS / RINS-style neighbourhoods) |

Differences that matter for us: our cost is Σ max-distance per move (parallelism worthless), our
rectangles must be contiguous with every atom inside moving, and the idle rule forbids parking in
the zone, so ZAC's reuse only applies between consecutive pulses.

Also tried, not worth it: **P7 stay** (ZAC reuse as an operator: a free atom skips its storage
trip and gets one zone move) −0.15 to −0.25 % only, because such atoms ride in rigid band moves and
staying makes them ghosts (off by default, `stay_op=True`); **re-ranking solo's top-5 candidates
after the peephole** (`l3/exp/topn.py`): candidates are mostly the same plan, the winner never
changes.

Official bench (`bench.py --mode full`): **solo_l3 +91.63 % vs solo +91.48 %**, 20/20 valid,
distance 11 497.3, depth 1829 (unchanged).

### Outer problem: solo's band choice is myopic (`compilers/solo_wide/`)

Solo picks each round's storage band greedily (Hungarian estimate incl. 4·h band cost and
evictions, then simulates the best 6 plans *for this round only*). Narrow, tall bands win locally
but leave tall stacks behind. Forcing a minimum band width (`l3/exp/solo_wide.py`, `WIDE_MIN`)
over all circuits: w2 +91.50, w3 +91.51, w4 +91.56 (solo +91.475); per-circuit best +91.58.
`solo_wide` adds the width as a candidate option (top-3 configurations re-run with WIDE 2/3/4,
then solo's normal refinement): +91.553, never worse than solo, faster on 50q stagger p3 (34 s vs
49 s standalone). Principled version (not built): lookahead in the band choice (score a band by
this round + the next round's cost).

| compiler | score (full, 20/20) | distance | depth | max compile |
|---|---|---|---|---|
| solo | +91.48 | 11 818.3 | 1829 | |
| solo_l3 (solo + peephole) | +91.63 | 11 497.3 | 1829 | 119 s (3 jobs) |
| **solo_wide_l3** (band-width search + peephole) | **+91.72** | 11 368.6 | 1815 | 88 s (3 jobs) |

| **relay_l3** (relay from main + peephole) | **+92.21** | 11 362.3 | 1486 | 165 s (3 jobs) |

All pass `--set sanity --mode full` (13/13). On relay the pass cuts distance 3.5 % (relay +92.01).

Other per-round knobs (`l3/exp/knobs.py`, global overrides of solo_wide): EVICT 0.75 / 3.0 →
+91.525 / +91.550, SHIFTS (0,) / (0,1) / (0..3) → +91.525 / +91.530 / +91.563 (base +91.553). No
global setting is clearly better, but the **per-circuit best over these settings is +91.634**
(before the peephole): diversity helps, calibration does not. Adding them only as variants of the
top-3 configurations gives +0.003 (reverted). Capturing the +0.08 needs whole searches per setting
(≈ 3× compile time, ~150 s at 50q) — over the 60 s guideline, so not built.

### Collecting findings on top of relay (`compilers/relay_plus*`)

relay forks solo2, which forked solo at 682d68b (+91.25), before Codex's later solo work. `relay_plus`
= 3-way merge (base solo@682d68b, ours relay, theirs solo with Codex's uncommitted hub waves): b9a2a15
hub rotations + phase lowering, d85a571 phase retiming + boundary moves, hub-wave scheduling + local
resynthesis. 9 conflicts resolved by hand (`_zxz` keeps relay's shorter Euler form as a tuple; hub-form
options of both kept, relay's as default; finalists emitted with `_finalize_phases` and
`_fold_boundary_moves`; `_band_layouts` restored and `P` re-defined after the merge).
**relay_plus: +92.16 (fast, 20/20) vs relay +92.01**, depth 1486 → 1419, distance 11 770 → 11 666.
`relay_plus2` adds WIDE (band width) and HUB_PARK_NEAR (fanframe) as search variants.

### Second mining round: MIP-LNS on relay_l3 (after the peephole)

`l3/mine.py` records every improving window rewrite. On relay_l3's 12q plans the exact windows still
find ~2 % (stagger p1 112.9 → 110.8, stagger p3 377.4 → 370.4, tree p1 126.8 → 124.1, tree p3 390.9 →
380.9; HEA and warm already optimal), on 20q tree p3 −3.0 % (36 rewrites). Distilled into P8, P9, P10,
P6b: a second peephole pass on relay_l3's plans gives **11 362.3 → 11 253.7 (−1.0 %)**, ≤ 25 s at 50q.

### Results on top of relay

| compiler | score (full, 20/20) | distance | depth |
|---|---|---|---|
| relay | +92.01 | 11 769.6 | 1486 |
| relay_l3 (P1-P10, guarded) | +92.25 | 11 283.5 | 1486 |
| relay_plus2 (fast) | +92.19 | 11 626.4 | 1415 |
| **relay_best_l3** | **+92.45** (+92.39 guarded under load) | 11 113.3 | 1411 |

Long, resumable mining on relay_best_l3's plans: `l3/long_mine.sh` (checkpoints in
`l3/out/mine_relay_best_l3/`, restart the script to resume).

### Lookahead for the band choice (rollout, approximate DP; `l3/exp/rollout.py`)

At every round try the K best band plans, simulate this round plus the next L steps of the recorded
schedule with the greedy base policy on a clone, commit the best (replay with rank 0 reproduces the
search exactly). K=4, L=5: 50q tree p1 −3.06 %, 20q tree p3 −0.43 %, 20q stagger p3 +0.08 %, 50q
stagger p3 +0.55 %, 50q tree p3 +0.88 % (finite horizon misleads). Full horizon (exact base-policy
completion, guaranteed no worse): 0 changes on 12q stagger p3 and 20q tree p3, −0.29 % on 20q stagger
p3. **The greedy band choice is at a local optimum against single deviations** (agrees with fanframe's
beam search, ≤ 2.4 %). Kept only as a guarded extra finalist (`relay_plus_roll`).

## Conclusions

1. The formalisation is exact and useful as a specification, an oracle (Emitter + official
   replay) and for exact small windows. A full exact model does not scale (same wall as
   OLSQ-DPQA): 6 move slots with 11 atoms already finds no feasible point in 120 s.
2. On solo's plans, everything *local* is worth 3–5 %: peephole −2.7 % (deployable), MIP-LNS +
   peephole ≈ −4 to −5 % on 12q/20q (offline only, too slow).
3. The large theoretical headroom (zone-resident plans estimated at 4 000–5 300 vs 11 800) needs a
   new outer layout strategy: keep rows 8–9 free for dips and staging, wide 2-row bands, ZAC-style
   reuse with matching-based slot choice. Greedy versions failed on batching, so this is a
   compiler-sized project, not a pass.

## Next steps

- Possible step (parked, user decision 2026-11-22): run solo_wide's whole search once per knob
  setting (WIDE, SHIFTS (0,), SHIFTS (0..3), EVICT 3.0) and keep the best per circuit: ≈ +0.08
  before the peephole, but ≈ 3× compile time (~150 s at 50q, over the 60 s guideline).
- Apply the peephole pass to the newer compilers on main (relay +92.01, solo2): it is
  compiler-agnostic (`python l3/make_l3.py relay`), and to solo_wide's fork base after d85a571.
- Passing-lane operator; MIP-LNS with warm start (scipy's private `_Highs.setSolution`) for larger
  windows offline, to mine more operators.
- Outer problem: SA / matching over storage layout with the peephole-improved inner cost, or a
  wide-band + reuse planner (ZAC) using the MILP to choose slot assignments per round.
