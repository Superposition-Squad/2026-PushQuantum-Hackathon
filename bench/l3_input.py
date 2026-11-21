#!/usr/bin/env python3
"""Print the input of the movement layer (L3) for a compiled circuit: the pulse schedule and the
storage obligations, extracted from any compiler's output.

    python bench/l3_input.py <compiler> <metric qasm path relative to metric_benchmarks/> [--summary]

A. Pulse list: pulse k = the pairs that must sit side by side in the zone at that pulse (nobody else
   may be in the zone then: idle rule). `a^s` = atom a has 1q gates due since its previous pulse, so
   it must visit storage in between.
B. Atom timelines: one row per atom, one column per pulse: partner number, `S` = storage visit
   needed before that pulse, `.` = not pulsed (must not be in the zone at that pulse).
Everything else (where atoms wait, which moves) is what the movement layer is free to choose.
"""
import importlib.util
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
logging.disable(logging.WARNING)
from qiskit import QuantumCircuit  # noqa: E402

from pushquantum_validator.spec import load_trusted_spec  # noqa: E402


def _load(path):
    spec = importlib.util.spec_from_file_location("m" + str(abs(hash(str(path)))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


po = _load(ROOT / "compilers" / "postopt.py")
_PoSt = po._PoSt


def _po_parse(circuit, loading, geo):
    """Schedule of a compiled circuit: CZ layers (instruction lists), the 1q gates of each gap
    (gap k = before layer k; gap K = after the last layer), measurements, and the positions of
    every atom at every layer and at the end (validator semantics)."""
    qi = {b: i for i, b in enumerate(circuit.qubits)}
    st = _PoSt(geo, loading)
    layers, gaps, confs, meas = [], [[]], [], []
    cur = None
    for ins in circuit.data:
        op = ins.operation
        qs = [qi[q] for q in ins.qubits]
        if op.name == "cz":
            if cur is None:
                cur = []
                confs.append(dict(st.pos))
            cur.append(ins)
            continue
        if op.name == "move":
            if cur is not None:
                layers.append(cur)
                gaps.append([])
                cur = None
            rows, cols, nr, nc = op.params
            occ_rows = [y for y in rows if any(st.pos[q][1] == y for q in qs)]
            occ_cols = [x for x in cols if any(st.pos[q][0] == x for q in qs)]
            if len(nr) == len(rows):
                occ_rows = list(rows)
            if len(nc) == len(cols):
                occ_cols = list(cols)
            rm, cm = dict(zip(occ_rows, nr)), dict(zip(occ_cols, nc))
            st.apply([(q, (cm[st.pos[q][0]], rm[st.pos[q][1]])) for q in qs])
            continue
        if op.name == "measure":
            meas.append(ins)
        elif op.name in ("rx", "rz"):
            gaps[-1].append((ins, qs))  # a 1q gate inside a layer acts on a storage atom: next gap
        elif op.name == "barrier":
            continue
        else:
            raise ValueError("unexpected gate " + op.name)
    if cur is not None:
        layers.append(cur)
        gaps.append([])
    return layers, gaps, confs, dict(st.pos), meas, qi




def main():
    comp_name, qasm = sys.argv[1], sys.argv[2]
    summary = "--summary" in sys.argv
    comp = _load(ROOT / "compilers" / comp_name / "compiler.py")
    qc = QuantumCircuit.from_qasm_file(str(ROOT / "vendor/pushquantum_validator/metric_benchmarks" / qasm))
    out, arch = comp.compile_circuit(qc, load_trusted_spec())
    geo = po._PoGeo(load_trusted_spec())
    layers, gaps, confs, final, meas, qi = _po_parse(out, arch.initial_loading, geo)
    n = qc.num_qubits
    pairs = [[tuple(qi[q] for q in ins.qubits) for ins in L] for L in layers]
    owe = [{q for _, qs in g for q in qs} for g in gaps]
    K = len(layers)
    ncz = sum(len(p) for p in pairs)
    with_s = sum(1 for k in range(K) for a, b in pairs[k] if a in owe[k] or b in owe[k])
    sizes = {}
    for p in pairs:
        sizes[len(p)] = sizes.get(len(p), 0) + 1
    print(f"# {qasm} (schedule from {comp_name}): {n} atoms, {K} pulses, {ncz} CZ, "
          f"{ncz / max(1, K):.2f} pairs/pulse, {with_s / max(1, ncz):.0%} of CZs need a storage visit right before")
    print("pairs per pulse:", dict(sorted(sizes.items())))
    if summary:
        return
    print()
    print("## A. Pulse list")
    for k in range(K):
        cells = []
        for a, b in sorted(pairs[k], key=min):
            cells.append(f"{a}{'^s' if a in owe[k] else ''}-{b}{'^s' if b in owe[k] else ''}")
        print(f"P{k + 1:02d}: " + "  ".join(cells))
    print()
    print("## B. Atom timelines")
    w = 3 if n < 100 else 4
    print("atom " + "".join(f"{k + 1:>{w}}" for k in range(K)))
    for q in range(n):
        row = []
        for k in range(K):
            part = next((b if a == q else a for a, b in pairs[k] if q in (a, b)), None)
            cell = "." if part is None else ("S" if q in owe[k] else "") + str(part)
            row.append(f"{cell:>{w}}")
        print(f"{q:>4} " + "".join(row))


if __name__ == "__main__":
    main()
