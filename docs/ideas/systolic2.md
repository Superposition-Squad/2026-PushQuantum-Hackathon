# Idea: systolic2 (staging-row choreography for matching rounds)

> **Update ~01:30 CET:** `compilers/solo` (teammate) implements essentially this plan (dynamic band,
> no fixed homes, rigid rectangle lifts per round, multi-start search) and reaches +90.11; see
> [solo.md](solo.md). Further work should build on solo / blockingcz rather than on systolic2.

- Code: `compilers/systolic2/compiler.py` (fork of `compilers/systolic`, v5 storage machine).
- Status: plan + first measurements (2026-11-22 ~00:30 CET). Current systolic2: **+85.41** on the
  v5 set (systolic +85.34), `systolic2_po` **+86.35** (systolic_po +86.30).
- Rules: validator v5.0.0 (idle rule, single-zone moves, no rzz), see [../rules.md](../rules.md).

## 1. Where the distance goes (measured)

Phase breakdown of the winning systolic plan, 50q iHVA stagger p3 (D = 5385):

| phase | distance | share |
|---|---|---|
| matching rounds: lift from homes | 1471 | |
| matching rounds: back home | 1384 | |
| matching rounds: mid `rx` trip (targets out + back) | 794 + 610 | |
| **matching rounds total** (55 rounds, 294 pairs, ~5.3 pairs/round) | **4259** | **79 %** |
| hub groups (hub stays in the zone, partners from row 9, row rotation) | 1126 | 21 % |

50q tree p3 (6230) and 20q tree p3 (1191) look the same (rounds 60–70 %, hub 30–40 %).
Tool: `phase.py` (scratch; to be moved to `bench/systolic_phases.py`, see M1).

- A matching round costs **~77 distance in ~22 moves** (lift ~27 in ~6.5 moves, home ~25,
  mid trip ~25). Nearly every move carries one atom: homes are scattered over rows 5–9, so the
  round's atoms never form one rectangle.
- **The ideal round** under the idle rule: the round's pairs sit side by side in storage row 9 at
  columns (2k, 2k+1); lift row 9 → zone row 10, cz #1, drop the row back, `rx` on the targets,
  lift, cz #2, drop. Four single-row moves of 1.0–1.22 ≈ **5 per round**.
- 55 rounds × (77 − 5) ≈ 4000 of the 4259 is, in principle, sorting overhead. Even recovering
  half of it on the 50q circuits (61 % of the scored naive distance) is worth ~+2–3 points.

## 2. What has been tried (and why it was not enough)

| change | result | lesson |
|---|---|---|
| `compilers/postopt.py` v1 (gap-wise re-plan, block drops) | systolic 85.34 → 86.30 | grouping inside a gap helps, homes stay fixed |
| postopt v2: beam / local search over per-layer configurations | no gain | per-gap choices are myopic; alternatives must return to the fixed homes |
| systolic2: order-preserving (DP) pair-site assignment | 5385 → 5366 | the assignment is not the bottleneck |
| systolic2: row 9 kept free (`TOP_HOME = 8`) | 5366 → 6740 (worse) | every lift one row deeper; rounds > 5 pairs do not fit one row |
| systolic2: narrower home rows for n ≤ 30 (`STORE_W` 8/6/4) | ±0 (one 12q circuit 206 → 201) | small registers already sit next to the zone |
| pulsebeam (generic beam planner, block drops) | wins 6/8 12q circuits, loses 20q/50q | no global storage choreography |

The common cause: **fixed, scattered homes**. The fix has to change *where atoms wait between
rounds*, with lookahead over the coming rounds.

## 3. Design

Keep from systolic: the parser, the block grouping and matching schedule (`plan_groups`,
`_greedy_matchings`), the hub stream (`hub_group` with row rotation), the batcher
(`_plan_moves`, validator-exact), `_State` (asserts the idle rule).

Replace: the storage choreography of matching rounds.

### 3.1 Storage as a deck, not as homes

- Row 9 is the **stage**: the round being executed sits there, its pairs on aligned column pairs.
- Rows 8, 7, … are the **buffer**, ordered by *next use* (atoms needed soon sit high).
- After a round the atoms drop back into the stage row (the sites they left; straight down, one
  move), then the stage is **switched**: leaving atoms sink into the buffer, the next round's
  atoms rise into the stage at their target columns.
- Rounds larger than 5 pairs use two stage rows (rows 9, 8 → zone rows 10, 11 as one 2-row move,
  ~2.1) or are split into sub-rounds (allowed: the CZs of a round act on disjoint pairs).
- The mid `rx` trip becomes "drop the zone row back into the stage, `rx`, lift" (≈ 2.4 instead
  of ~25), because the sites directly below are exactly the ones the lift vacated.

### 3.2 Switching primitives (all legal single moves, validator semantics)

1. **Row sink / rise:** a contiguous segment of row 9 moves down one row (onto free sites) or a
   segment of row 8 moves up (1.0 each, many atoms per move).
2. **Stretch / compress:** a segment keeps its order but changes spacing (monotone column map),
   e.g. to open aligned gaps for incoming atoms.
3. **Interleave:** two segments in different rows map into one row with monotone column maps
   (two moves); this is how a new pairing is formed from two ordered buffers ("zip").
4. **Single-atom fetch:** fallback for stragglers (cost = depth + sideways distance).

The batcher already checks ghosts, ordering and occupancy for any of these.

### 3.3 Planner: beam search over deck states with lookahead

- The round sequence is fixed (systolic's schedule), so every beam entry is at the same round
  index: entries are directly comparable, **no remaining-work penalty**.
- State = positions of all atoms (+ executor state for pending 1q gates).
- Expansion per round: candidate stage layouts (which column pair each block gets; which atoms
  sink where), generated by a few deterministic heuristics (keep order, minimal displacement,
  "next-use" buffer order) plus a small random set with a fixed seed.
- Transition cost = exact batched cost of switch + round (lift, drop, lift, drop).
- Lookahead term: estimated switch cost of the next k rounds (k = 2–3): for each atom of those
  rounds, its distance to the stage and whether it can join a rectangle with its round-mates.
  This is the piece postopt v2 lacked.
- Deterministic budgets (beam width, candidates per round) scaled by register size; target
  ≤ 60 s per circuit on top of systolic's own search.

### 3.4 Hub stream integration

- Simplest first: before a hub group, restore the layout the hub stream expects (its rows,
  rotation); measure what that costs.
- Better: teach `hub_group` to take partners from the current deck (partners launch from row 9;
  rotation becomes "bring the next partners into the stage").

## 4. Milestones (each with a measurable checkpoint)

| | step | checkpoint |
|---|---|---|
| M1 | move the phase breakdown into `bench/systolic_phases.py` (winning candidate only) | per-phase numbers for any systolic-like compiler in one command |
| M2 | **free-sorting bound:** execute the round sequence assuming switching costs 0 | upper bound on the gain; if it is < ~20 % on 50q, stop here |
| M3 | greedy switching (primitives 1, 2, 4; no lookahead) | beats systolic2 on 50q stagger p1 / tree p1 |
| M4 | beam + lookahead (3.3) | 50q stagger p3 ≤ 4000 (systolic 5385) |
| M5 | hub-stream integration (3.4), full validation, postopt on top, portfolio rebuild | v5 set ≥ 87.5 (portfolio_po today 86.63), full 33/33, strict on 20q/50q |

## 5. Evaluation protocol

- Fast loop: single circuits (50q stagger p1, 50q tree p1, 20q tree p3, 12q tree p3), a few
  seconds each; phase breakdown after every change.
- Background: full v5 set (`bench/bench.py`), full mode for validity, strict before merging.
- Equivalence: the gate sequence and its per-qubit order are taken from systolic unchanged; only
  positions and moves change. `_State` asserts the idle rule; the validator replays every output.
- Depth should stay equal to systolic's (same gates); if switching interleaves differently, depth
  is part of the cost (× 3.06 on the v5 set).

## 6. Risks

- The switching cost may eat most of the gain (the free-sorting bound in M2 answers this early).
- iHVA ordering constraints fix the round sequence; reordering rounds is out of scope here (it
  touches gate order and needs the commutation analysis).
- Compile time: systolic already spends up to ~15 s per 50q circuit on layout search; the beam has
  to fit next to it (deterministic budgets, no wall clock).
- Rounds > 5 pairs (two stage rows) are where most of the complexity is; start with ≤ 5-pair
  rounds and split larger ones.
