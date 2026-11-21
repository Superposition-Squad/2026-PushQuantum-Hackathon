# The input of the movement layer (L3), made visible

What every movement planner (systolic's storage machine, optimizer-optimizer, postopt) has to
realise, extracted from a compiled circuit by `bench/l3_input.py <compiler> <qasm> [--summary]`.
Everything not listed here (where atoms wait, which moves happen) is free.

- **Pulse list:** pulse k = the pairs that must sit side by side in the entangling zone at that
  pulse; nobody else may be in the zone then (idle rule). `a^s` = atom a has 1q gates due since its
  previous pulse, so it must visit storage in between.
- **Timelines:** one row per atom, one column per pulse: partner, `S` = storage visit needed before
  that pulse, `.` = not pulsed (must not be in the zone).

## Scale (schedule from systolic, v5)

| circuit | pulses | pairs / pulse | pulses with one pair | CZs needing a storage visit right before |
|---|---|---|---|---|
| 12q iHVA tree p1 | 40 | 1.45 | 30 | 83 % |
| 20q iHVA tree p3 | 174 | 1.72 | 130 | 72 % |
| 50q iHVA stagger p3 | 404 | 2.18 | 334 | 67 % |
| 50q iHVA tree p3 | 504 | 1.75 | 378 | 72 % |

Observations:
1. The schedule is very serial: most pulses carry one pair (the hub's chain: 2(n−1) CZs per layer),
   the non-hub pairs come in a few large pulses (10–24 pairs at 50q). Every pulse costs at least
   one lift and one drop.
2. Two thirds or more of all CZs need a storage visit of one atom right before them (iHVA's Y-basis
   changes around every block), so "stay in the zone" is rarely possible.
3. A block shows up as two pulses of the same pair, the second with `^s` on the target (its `rx`).
   Blocks on the hub nest (e.g. 11–7 at P13 and P16 around 11–1 at P14/P15).

## Full example: 12q iHVA tree p1 (schedule from systolic)

```
# ihva/bus_stop_12q_ihva_tree_p1.qasm (schedule from systolic): 12 atoms, 40 pulses, 58 CZ, 1.45 pairs/pulse, 83% of CZs need a storage visit right before
pairs per pulse: {1: 30, 2: 4, 3: 4, 4: 2}

## A. Pulse list
P01: 8-0^s  2^s-1^s  4-5^s  10-6^s
P02: 8^s-0  2-1^s  4^s-5  10^s-6
P03: 2^s-3^s  8^s-7^s  10^s-9^s
P04: 2^s-3  8-7^s  10-9^s
P05: 8^s-9^s
P06: 8^s-9
P07: 6^s-4^s
P08: 6^s-4
P09: 3-4^s  6-5^s
P10: 3-4^s  6^s-5
P11: 11^s-10^s
P12: 11-10^s
P13: 11^s-7^s
P14: 11-1
P15: 11-1^s
P16: 11-7
P17: 1^s-7^s  3-10^s
P18: 1-7^s  3-10^s
P19: 11^s-2^s
P20: 11-2^s
P21: 11^s-5^s
P22: 11-6
P23: 11-0
P24: 11-9
P25: 11-9^s
P26: 11-0
P27: 11-6
P28: 11-5
P29: 0^s-1^s  9^s-5^s  7^s-6^s
P30: 0-1^s  9^s-5  7^s-6
P31: 0-2^s
P32: 0-2^s
P33: 0^s-10^s
P34: 0^s-10
P35: 11^s-8^s
P36: 11-8^s
P37: 11^s-3^s
P38: 11-4
P39: 11-4^s
P40: 11-3

## B. Atom timelines
atom   1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40
   0  S8  8  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . 11  .  . 11  .  . S1  1  2  2S10S10  .  .  .  .  .  .
   1  S2 S2  .  .  .  .  .  .  .  .  .  .  . 11S11  . S7  7  .  .  .  .  .  .  .  .  .  . S0 S0  .  .  .  .  .  .  .  .  .  .
   2  S1  1 S3 S3  .  .  .  .  .  .  .  .  .  .  .  .  .  .S11S11  .  .  .  .  .  .  .  .  .  . S0 S0  .  .  .  .  .  .  .  .
   3   .  . S2  2  .  .  .  .  4  4  .  .  .  .  .  . 10 10  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .S11  .  . 11
   4   5 S5  .  .  .  . S6  6 S3 S3  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . 11S11  .
   5  S4  4  .  .  .  .  .  . S6  6  .  .  .  .  .  .  .  .  .  .S11  .  .  .  .  .  . 11 S9  9  .  .  .  .  .  .  .  .  .  .
   6 S10 10  .  .  .  . S4 S4  5 S5  .  .  .  .  .  .  .  .  .  .  . 11  .  .  .  . 11  . S7  7  .  .  .  .  .  .  .  .  .  .
   7   .  . S8 S8  .  .  .  .  .  .  .  .S11  .  . 11 S1 S1  .  .  .  .  .  .  .  .  .  . S6 S6  .  .  .  .  .  .  .  .  .  .
   8   0 S0 S7  7 S9 S9  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .S11S11  .  .  .  .
   9   .  .S10S10 S8  8  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . 11S11  .  .  . S5 S5  .  .  .  .  .  .  .  .  .  .
  10   6 S6 S9  9  .  .  .  .  .  .S11S11  .  .  .  . S3 S3  .  .  .  .  .  .  .  .  .  .  .  .  .  . S0  0  .  .  .  .  .  .
  11   .  .  .  .  .  .  .  .  .  .S10 10 S7  1  1  7  .  . S2  2 S5  6  0  9  9  0  6  5  .  .  .  .  .  . S8  8 S3  4  4  3
```
