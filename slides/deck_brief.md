# Slide brief: planqc compiler track (relay_best_l3, +92.45 %)

Text-only brief for building the slide deck. Everything below is final content: titles, on-slide
text, speaker notes and the figure to use. Numbers are measured with the repo's harness
(`bench/bench.py`, official validator v5, full mode) unless marked otherwise.

## Instructions for the slide builder

- 10 content slides + title. If time is short, keep slides 1, 3, 4, 5, 8, 9.
- Tone: technical and precise. Every claim on a slide has a number or a mechanism behind it.
- Keep on-slide text short (the bullets below); the speaker notes carry the detail.
- Figures (all in the repo):
  - `slides/anim/relay_anim.html` (interactive) / `relay_anim.mp4` / `relay_anim.gif`: the compiler
    executing a real 20-qubit circuit, 6 chapters (timestamps below). Google Slides: MP4 via
    Insert → Video → Drive, or the GIF via Insert → Image.
  - `docs/img/viz_diff_split.png`: the visualiser's split view in diff mode.
- Colours used in the animation (reuse them): hub gold `#e0a100`, atom blue `#2f6db5`, CZ orange
  `#e8762d`, 1q gates light blue `#2f8fd8`, moves purple `#7c4dc4`.

---

## Slide 0: Title

**Compiling iHVA circuits for a zoned neutral-atom machine**
Superposition Squad · PushQuantum Hackathon 2026 · planqc track

Subtitle line: relay_best_l3: +92.45 % (full validation, 20/20 scored circuits)

---

## Slide 1: The machine and the score

On slide:
- Storage zone 10 × 10: `rx`, `rz` only. Entangling zone 10 × 10: `cz` only, on interaction pairs
  (columns 2k, 2k+1, 0.3 apart; pairs 2.0 apart). Storage row 9 ↔ zone row 10: distance 1.0.
- A CZ pulse is transversal: every occupied pair gets a CZ. **Idle rule:** every atom in the zone
  must be in a CZ at every pulse, no parking.
- A move = one AOD rectangle: contiguous rows × columns, **every atom inside moves**, row and column
  maps monotone (order preserved, spacing may stretch), source and target each in one zone.
- **Cost of a move = its longest single-atom path**, however many atoms it carries.
- Score = 0.8 · distance reduction + 0.2 · depth reduction vs the naive compiler, summed over 20
  circuits (18 iHVA at 12/20/50 qubits, 2 HEA). 1 depth unit ≈ 3.06 distance units.

Speaker notes: The metric rewards carrying many atoms in one short rigid move; parallel moves are
worth nothing. The naive compiler pays two long trips per CZ (163 364 total distance, 13 364 depth on
the scored set). Equivalence is checked exactly (QCEC up to 19 qubits, interaction-pair check above).

---

## Slide 2: The circuits: one hub against everyone

On slide:
- iHVA = layers of two-qubit Pauli rotations exp(−iθ/2 Z⊗Y) on a graph: a ring with random chords
  (degree 3–6) plus one **hub** qubit that interacts with all others.
- The hub forces 2(n−1) serial CZs per layer with a naive decomposition (98 at 50 qubits).
- The non-hub graph needs several matchings per layer (up to 6 at 50 qubits); consecutive matchings share no structure,
  so atoms must be re-paired between rounds.

Speaker notes: These two facts drive everything: the hub chain sets the depth and the number of
pulses; re-pairing between rounds sets most of the distance. In the best plans, storage re-sorting
is about half of all distance (measured: 52 % for solo on the scored set).

---

## Slide 3: Three layers, three questions

On slide (diagram: three boxes left to right):
- **L1 Gate form:** which native gates realise each block?
- **L2 Schedule:** which pairs pulse together, in which order?
- **L3 Movement:** where are the atoms, and which rectangle moves get them there?
- Every improvement in this talk is attributable to exactly one layer.

Speaker notes: L1 fixes the gate list and its depth; L2 fixes the pulse sequence; L3 only changes
distance (depth is measured with moves removed). Keeping the layers separate let several people and
agents work in parallel on different layers and merge the results.

---

## Slide 4: L1: blocks without basis changes, and the relay

On slide:
- **S-frame:** conjugating by S turns every Z⊗Y block into Z⊗X = `CZ · rx(θ) · CZ`. Only the
  rotor needs storage between the two pulses; no basis change between blocks.
- Special angles: θ = ±π/2 → one CZ + local Cliffords; θ = π → local Paulis; θ = 0 dropped.
- **Relayed hub fan-out (relay):** a hub batch is F · ∏ rx(θ_p) · F with F = ∏ cz(hub, p).
  F depends only on the partners' **parity**:
  cz(hub, l) · cz(hub, r) = P · cz(hub, r) · P, with P = rx_r(−π/2) · cz(l, r) · rx_r(π/2) (a controlled-Y, P² = 1).
  Partner pairs (or paths of four) fold their parity into one root, in parallel; **the hub meets one
  root per group** → half (or a quarter) of the hub's serial pulses.
- Roots alternate between the two passes so every input pair still interacts (validator rule above
  19 qubits). Euler sequences use the shorter of two equivalent forms.

Figure: animation chapters 2 and 3 (0:19 and 0:24), or a still of the relay frame: 7 partner pairs
pulsing in parallel while the hub waits in storage.

Speaker notes: This is the step from solo2 (+91.56) to relay (+92.01): total depth 1 856 → 1 486 on
the scored set. The identity was checked numerically on 6 qubits before any compiler work, and every
compiled circuit passes the official equivalence check.

---

## Slide 5: L2 + L3: rounds, hub batches and the storage band

On slide:
- **Schedule (L2):** exact commutation DAG of the Pauli strings. Each step is a **round** (a
  matching of ready blocks) or a **hub batch** (cz #1 for every partner, one rx layer, cz #2 in
  reverse). Online policies + a multi-start search over configurations, cost = distance + 3.06 · depth.
- **Dynamic storage band (L3):** no fixed homes. Per round: choose a storage rectangle, assign pairs
  to its slots with a Hungarian assignment (distance + eviction penalty + lookahead), relocate, then
  **lift the whole band rigidly**: lift, pulse, rotors dip, lift, pulse, drop ≈ 4 moves for all pairs.
- Hub batches: hub in zone row 10, partners lifted from storage row 9, rows rotate through row 9;
  hub sites chosen by a small dynamic programme.

Figure: animation chapter 1 (0:01): a 10-pair band round.

Speaker notes: The rigid band is why moves are cheap per CZ: one 2-row band move costs ~2.1 for up to
10 pairs. The hub stream is close to its floor (~3.2 per hub pulse: drop, hop, lift).

---

## Slide 6: L3 as a formal optimisation problem

On slide:
- Variables: atom configuration X_k at every pulse k. Constraints: pulse pairs on interaction
  pairs, everyone else in storage, atoms that owe 1q gates touch storage between consecutive pulses.
- Objective: min Σ_k g_k(X_{k−1}, X_k), where g_k = cheapest legal rectangle-move sequence between
  two configurations (a set-partitioning problem with order constraints).
- Exact model: time-expanded MILP (one-hot positions, picked row/column intervals, monotone integer
  maps, ghost rule built in, single-zone, occupancy, polygonal Euclidean norm), solved with HiGHS.
- Full instances do not scale (same wall as solver-based layout synthesis in the literature), so
  the exact model is used on **sliding windows of 3 moves** over a complete plan (MIP-based large
  neighbourhood search): −4 to −7 % distance on 12-qubit plans.

Speaker notes: The point of the formal model is to get ground truth for what is locally possible,
not to run it in the compiler. Each window rewrite is replayed with the validator's rules; gates and
their per-qubit order never change, so depth and equivalence are preserved by construction.

---

## Slide 7: From exact solutions to a fast pass

On slide:
- We recorded every window the MILP improved and turned the recurring patterns into **peephole
  operators** (no solver, seconds per circuit):
  - merge moves that form one rigid rectangle · trim dips to the atoms that actually need storage
  - ride along: split a long move so half of it shares another move's rectangle
  - compose a drop with the following shuffle (re-targeted column map) · stack two lifts into one
    two-row lift · shift a rigid dip to free rows closer to the zone
- Every candidate rewrite is replayed exactly (legal moves, same end configuration, required
  storage visits); kept only if cheaper.
- Compiler-agnostic: wraps any compiler. On relay: distance 11 770 → 11 284 (−4.1 %), depth unchanged.

Speaker notes: Two mining rounds: first on solo's plans, then again on the already-optimised relay
plans, which surfaced four new operators (compose, stack-lift, round-trip retarget, map-preimage
ride-along). The pass adds ≤ 25 s at 50 qubits.

---

## Slide 8: How we worked: iterate, measure, merge

On slide:
- Parallel lines of work on separate branches/worktrees (team members and AI coding agents), one
  shared harness: the **official validator** + transversal-CZ and zone-span checks, fast/full/strict
  modes, results per circuit.
- Loop: profile where distance goes → hypothesis → experiment → full validation (20 scored +
  13 sanity circuits) → merge.
- **Merging ideas across lines:** relay forked earlier than later solo improvements; a 3-way
  merge brought them together (relay_plus +92.16), then search variants and the peephole pass on top.
- **Visualiser** (our own tool, same validator trace): atom view + circuit view in sync, play/step/
  scrub, every move as origin and target. **Split view with diff mode:** two compilations aligned on
  their CZ pulses; instructions only in one plan in red/green; click to jump both to the
  equivalent state. We used it to see *why* a plan is cheaper, not only *that* it is.

Figure: `docs/img/viz_diff_split.png`.

Speaker notes: Most ideas were judged by watching both plans side by side: e.g. the tall storage
bands at 50 qubits (5 rows → ≥ 5 per band move) and dips that took whole rows down when one atom
needed storage. Those observations turned into the band-width search option and the dip-trim
operator.

---

## Slide 9: Results

On slide (table):

| compiler | what changed | score |
|---|---|---|
| naive | reference | 0.00 |
| systolic | storage homes + hub stream | +85.34 |
| blockingcz | S-frame, matchings, 2-leaf hub sweep | +90.48 |
| solo | dynamic storage band, special angles | +91.48 |
| solo2 | hub as rotor, gather lifts | +91.56 |
| relay | relayed hub fan-out (L1) | +92.01 |
| relay_l3 | + peephole pass (L3) | +92.25 |
| **relay_best_l3** | + later solo improvements, search variants, rollout, peephole | **+92.45** |

- relay_best_l3: distance 11 113 (naive 163 364, −93.2 %), depth 1 411 (naive 13 364, −89.4 %),
  20/20 valid in full mode, 13/13 sanity circuits.
- Example (animation): 20-qubit iHVA, 106 moves, distance 199 vs 3 274 naive (−93.9 %).

Speaker notes: Compile time ≤ ~200 s per 50-qubit circuit (hard limit 300 s), with wall-clock
guards that skip the optional stages on slow machines; the score with guards active on an idle
machine is identical.

---

## Slide 10: What limits the next step

On slide:
- Local search is saturated: exact windows, peephole and rollout lookahead (full horizon: 0 changes
  on most circuits) agree that the greedy band choice is locally optimal.
- The remaining distance is re-sorting atoms between unrelated matchings (random chords) under
  order-preserving rectangle moves.
- Next levers: joint L2 + L3 search (choose the next round by its real movement cost) and a storage
  layout that keeps rows near the zone free for wide, shallow bands.

Speaker notes: Shallow bands would cost ~2.1 instead of ≥ 5 per band move at 50 qubits; the open
question is the cost of keeping storage arranged for them.

---

## Animation chapters (for slides 4, 5 and live narration)

| time | chapter | layers | what to say |
|---|---|---|---|
| 0:01 | Band rounds | L1 L2 L3 | ten pairs lifted as one band; rotors dip for rx(θ); second pulse |
| 0:19 | Relay: collect parity | L1 | seven partner pairs pulse in parallel; the hub waits in storage |
| 0:24 | Hub meets only the roots | L1 L3 | eight hub pulses instead of sixteen |
| 0:42 | Uncompute, rotate, recompute | L1 | pairs undo the relay; partners get rx(θ); relay rebuilt |
| 0:49 | Second pass | L1 L3 | other atom of each pair is the root; every input pair interacts |
| 1:09 | Last round | L2 L3 | remaining blocks as one more band round |
| 1:24 | End card | — | Σ distance 199 vs naive 3 274 (−93.9 %) |

## Sources in the repo

- Numbers: `docs/results-log.md`, `results/*.json`. Ideas and measurements: `docs/ideas/relay.md`,
  `docs/ideas/solo.md`, `docs/ideas/solo2.md`, `docs/ideas/l3abstract.md`, `docs/overview.md`.
- Related work used for orientation (`docs/literature.md`): solver-based layout synthesis for
  DPQA (OLSQ-DPQA, 2306.03487), Enola (2405.15095), ZAC (2411.11784), PowerMove (2411.12263).
