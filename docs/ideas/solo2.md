# Idea: solo2 — solo plus hub-as-rotor rounds, gather lifts and more start variants

- Branch: idea/solo2 · Worktree: /home/hackathon/challenge-wt/solo2 · File: `compilers/solo2/compiler.py`
- Fork of `compilers/solo` (see [solo.md](solo.md) for the design). Validator v5.0.0.
- Result: **+91.56 %** (full, 20/20; solo +91.25): distance 11 747 → 11 568, depth 2 004 → 1 856.

## Where solo's cost goes (profile, read before trying more)

| circuit | D | storage relocation before rounds | rigid band lifts | hub stream |
|---|---|---|---|---|
| 50q stagger p3 | 2664 | 44 % | 16 % | 40 % |
| 50q tree p3 | 3278 | 52 % | 23 % | 25 % |
| 20q tree p3 | 854 | 43 % | 25 % | 32 % |
| 12q tree p3 | 414 | 56 % | 44 % | – |

Relocation moves carry 1.1–1.4 atoms each (≈ 2.3 long; two thirds member gathering, one third
evictions). The hub stream costs 4.2 per pulse (partner up/down 2.2, hub hops 1.2, rotation 0.6).

## What changed

1. **Hub as the rotor in rounds (`HUBN`, the gain).** A hub-as-target block inside a round no longer
   forces the swapped form with a basis change on the hub; the form chooser may take the normal form
   `cz · rx_hub(θ) · cz`. The hub drops with the band anyway, and its wire loses the role-switch
   gates, which sat on the critical path (12q tree p3: depth 141 → 103, 50q tree p3: 418 → 369).
   `HUBN = False` stays as a search variant.
2. **Gather lifts (`_gather_plan`, `_exec_gather`; rounds of ≤ 5 pairs).** Instead of relocating
   pairs into a band and lifting it rigidly, each ghost-free run of members in a storage row goes
   straight to its pair slots in zone row 10 with one move (monotone column map), and the first drop
   puts the pairs compactly on a free storage band. Chosen per round against the best rigid plan on
   a cloned machine. Small: about −0.3 % cost.
3. **More start variants** for the top configurations: critical-path weight in the matching key
   (`HW`), eviction targets pulled towards the evicted atom's next partner (`EVPULL`), a
   simulated-annealing locality layout (`sa_layout`). Each is within ±1 % alone; as extra candidates
   about −0.4 % together. The 50q p = 3 circuits get four variants and no SA layout (compile time).
4. The hub's trailing `rz` of a batch basis change stays pending through the batch (`HUBLAZY`): ±0.

## Measured and rejected (do not retry without a new angle)

- **Riders on the hub stream** (ready non-hub pairs lifted together with a hub partner): at batch
  time 0–1 ready pairs per batch sit adjacent in a hub row, so riders need the same relocation as a
  round and only save lifts.
- **Pipelining rounds** (cz #2 of one round sharing a lift with cz #1 of the next): the number of
  cycles already equals the DAG critical path (rounds × 2 = pipelined cycles on every circuit).
- **Locality layout as the only layout:** ±1–2 %; the configuration is scrambled after a few rounds.
- **Four times the seed budget, dummy pairs on any pair for ≤ 19 qubits:** no change.
- **Hub stream geometry:** conveyor rows, two-row service and hub-rotor batches (one pass, but +1
  depth per hub block) all cost at least the current 3.4 + rotation per pulse.
- **Order-based (column-remap) rigid lifts in dense storage:** a 50q band has no empty columns, and
  a stretched lift is paid four times per round; only the first lift may stretch (that is the gather
  lift above).

Depth facts: a hub block inside a round costs 3 on the critical path (cz, mid `rx`, cz on the rotor
wire), in a batch 2. That is inherent to consecutive pulses of one pair.

## Validity evidence

`bench --mode full` 20/20 (QCEC equivalent on all 12q), `--set sanity --mode full` 13/13,
`--mode strict --filter 20q` 6/6 (statevector ×3), Clifford tableaux on the 50q circuits
(`docs/ideas/hubwalk_cliff_check.py`, as compiled and with the special-angle classification forced
off): all six equal, two angle draws each. Slowest compile about 45 s (50q stagger p3).
