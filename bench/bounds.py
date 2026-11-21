"""Per-circuit lower bounds on move distance and depth, and headroom vs our best results.

Usage: uv run python bench/bounds.py [--results portfolio] [--md docs/bounds.md]

Bounds are deliberately simple and *provable for our rules*, assuming each two-qubit block is
synthesised with its minimal CZ count (2 for a generic ZZ/ZY rotation, 1 for a CX, 0 for an exact
identity such as rz(0) inside cx-rz-cx). They are not tight; they show where the points are.

Distance lower bound (per circuit), D_lb = max over qubits q of c_q, where c_q is the number of CZ
pulses q must take part in:
  * consecutive CZs on q with *different partners* need a partner change between the pulses;
  * the nearest site from which a new partner (or a new pair for q) can be reached is ≥ 1.0 away:
    sites are 0.3 apart only inside a pair (the slot q or its partner already occupies), 1.0
    vertically and ≥ 1.7 horizontally between pairs, and 1.0 across the storage/zone boundary;
  * so ≥ c_q − 1 partner changes of ≥ 1.0 each, plus ≥ 1.0 to bring q into the zone at all.
  Two consecutive CZs on the same pair with nothing in between cancel, so they are not counted as
  separate pulses only if adjacent; here every block's two CZs are separated by a 1q rotation on one
  side, which needs storage (≥ 1.0 out of the zone for that atom, its partner then changes or
  waits). The bound uses c_q for the busiest qubit only, so it ignores everything else.
Depth lower bound: max over q of (c_q) — every CZ is one layer on q (1q gates ignored).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

from qiskit import QuantumCircuit  # noqa: E402

from pushquantum_validator.verify import benchmark_label, metric_benchmarks, naive_baseline  # noqa: E402

ZERO = 1e-9


def _is_zero_angle(theta: float) -> bool:
    r = math.remainder(float(theta), 2 * math.pi)
    return abs(r) < ZERO


def cz_pulses(circuit: QuantumCircuit) -> tuple[dict[int, int], dict]:
    """c_q per qubit, with cx–rz(θ)–cx blocks counted as 2 CZs (0 if θ ≡ 0 mod 2π) and every other
    cx / cz as 1 CZ. Returns (counts, info)."""
    index = {b: i for i, b in enumerate(circuit.qubits)}
    ops = [(inst.operation.name, [index[q] for q in inst.qubits], list(inst.operation.params)) for inst in circuit.data]
    wire = defaultdict(list)  # qubit -> list of op indices
    for k, (_, qs, _) in enumerate(ops):
        for q in qs:
            wire[q].append(k)
    pos = {(k, q): i for q, ks in wire.items() for i, k in enumerate(ks)}

    def nxt(k, q):
        i = pos[(k, q)] + 1
        return wire[q][i] if i < len(wire[q]) else None

    counts = defaultdict(int)
    used = set()
    blocks = zero_blocks = single = 0
    pairs = set()
    for k, (name, qs, params) in enumerate(ops):
        if k in used or name not in ("cx", "cz", "rzz", "cp", "crz", "swap", "ecr"):
            continue
        if name == "cx":
            c, t = qs
            m = nxt(k, t)
            if m is not None and ops[m][0] == "rz" and ops[m][1] == [t]:
                k2 = nxt(m, t)
                if k2 is not None and ops[k2][0] == "cx" and ops[k2][1] == [c, t] and nxt(k, c) == k2:
                    used.update((k, m, k2))
                    blocks += 1
                    if _is_zero_angle(ops[m][2][0]):
                        zero_blocks += 1
                        continue
                    counts[c] += 2
                    counts[t] += 2
                    pairs.add(frozenset(qs))
                    continue
        n = {"swap": 3, "rzz": 2, "crz": 2, "cp": 2}.get(name, 1)
        for q in qs:
            counts[q] += n
        pairs.add(frozenset(qs))
        single += 1
    for name, qs, _ in ops:  # 3+ qubit gates: count conservatively as 1 per involved qubit
        if len(qs) > 2:
            for q in qs:
                counts[q] += 1
    deg = defaultdict(int)
    for p in pairs:
        for q in p:
            deg[q] += 1
    hub = max(deg, key=deg.get) if deg else None
    return dict(counts), {"blocks": blocks, "zero_blocks": zero_blocks, "other_2q": single,
                          "hub": hub, "hub_degree": deg.get(hub, 0), "edges": len(pairs)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="portfolio", help="results/<name>.json to compare against")
    ap.add_argument("--md", default=None, help="also write a markdown report here")
    args = ap.parse_args(argv)
    naive = naive_baseline()
    res_file = ROOT / "results" / f"{args.results}.json"
    best = {r["label"]: r for r in json.loads(res_file.read_text())["records"]} if res_file.is_file() else {}

    rows = []
    for path in metric_benchmarks():
        label = benchmark_label(path)
        qc = QuantumCircuit.from_qasm_file(str(path))
        counts, info = cz_pulses(qc)
        cmax = max(counts.values()) if counts else 0
        rows.append({"label": label, "file": str(path.relative_to(path.parents[1])), "n": qc.num_qubits,
                     "d_lb": float(cmax), "p_lb": cmax, **info,
                     "naive_d": naive[label]["metrics"]["move_distance"], "naive_p": naive[label]["metrics"]["depth"],
                     "best_d": best.get(label, {}).get("metrics", {}).get("move_distance"),
                     "best_p": best.get(label, {}).get("metrics", {}).get("depth")})

    ND = sum(r["naive_d"] for r in rows)
    NP = sum(r["naive_p"] for r in rows)
    w = lambda d, p: 0.8 * d / ND + 0.2 * p / NP  # noqa: E731
    ub = 100 * (1 - sum(w(r["d_lb"], r["p_lb"]) for r in rows))
    have = all(r["best_d"] is not None for r in rows)
    cur = 100 * (1 - sum(w(r["best_d"], r["best_p"]) for r in rows)) if have else None

    out = []
    out.append(f"# Lower bounds and headroom (generated by `bench/bounds.py`, vs `results/{args.results}.json`)\n")
    out.append("Bound per circuit: D ≥ max_q c_q and depth ≥ max_q c_q, c_q = CZ pulses of the busiest qubit "
               "(2 per nonzero ZZ/ZY block, 0 for θ ≡ 0, 1 per other cx). See the module docstring for the argument. "
               "Points = this circuit's share of the score lost vs perfect (100 × weighted cost).\n")
    if have:
        out.append(f"**Score upper bound from these bounds: {ub:.2f} %. Current ({args.results}): {cur:.2f} %. "
                   f"Gap: {ub - cur:.2f} points.**\n")
    out.append("| circuit | file | n | hub deg | blocks (θ=0) | D_lb | best D | naive D | depth_lb | best depth | points lost now | points at bound |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: -(w(r["best_d"], r["best_p"]) if have else r["naive_d"])):
        lost = 100 * w(r["best_d"], r["best_p"]) if have else float("nan")
        out.append(f"| {r['label'][-2:]} | {r['file']} | {r['n']} | {r['hub_degree']} | {r['blocks']} ({r['zero_blocks']}) "
                   f"| {r['d_lb']:.0f} | {r['best_d'] if r['best_d'] is None else round(r['best_d'])} | {r['naive_d']:.0f} "
                   f"| {r['p_lb']} | {r['best_p']} | {lost:.3f} | {100 * w(r['d_lb'], r['p_lb']):.3f} |")
    text = "\n".join(out) + "\n"
    print(text)
    if args.md:
        Path(args.md).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
