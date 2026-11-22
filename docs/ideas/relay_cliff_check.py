"""Exact Clifford-tableau equivalence check for the relay compiler at any size.

Like hubwalk_cliff_check.py (angles replaced by Clifford angles, stabilizer tableaux compared up to
global phase), plus `--generic`: the compiler's special-angle classification is forced off, so the
generic two-pass path (and with it the relayed fan-out) is exercised on the Clifford angles too.

usage (repo root): python docs/ideas/relay_cliff_check.py compilers/relay/compiler.py [--generic] ihva/bus_stop_50q_ihva_tree_p3.qasm ...
"""
import importlib.util
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor"))
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Clifford, Operator  # noqa: E402

from pushquantum_validator.verify import strip_moves  # noqa: E402

args = sys.argv[1:]
generic = "--generic" in args
args = [a for a in args if a != "--generic"]
spec = importlib.util.spec_from_file_location("cc", args[0])
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
if generic:
    cc.classify = lambda th: ("G", 0)


def tableau(qc):
    n = qc.num_qubits
    C = Clifford(QuantumCircuit(n))
    idx = {b: i for i, b in enumerate(qc.qubits)}
    run = {}

    def flush(q):
        nonlocal C
        if q in run:
            C = C.compose(Clifford.from_matrix(run.pop(q)), qargs=[q])

    for ins in qc.data:
        op = ins.operation
        qs = [idx[b] for b in ins.qubits]
        if op.name in ("measure", "barrier", "move"):
            continue
        if len(qs) == 1:
            run[qs[0]] = Operator(op).data @ run.get(qs[0], np.eye(2))
        else:
            for q in qs:
                flush(q)
            c = QuantumCircuit(2)
            c.append(op, [0, 1])
            C = C.compose(Clifford(c), qargs=qs)
    for q in list(run):
        flush(q)
    return C


def cliffordize_qasm(txt, seed):
    rng = random.Random(seed)

    def sub(m):
        try:
            val = float(eval(m.group(2).replace("pi", str(np.pi))))
        except Exception:  # noqa: BLE001
            val = 1.0
        k = val / (np.pi / 2)
        if abs(k - round(k)) < 1e-9:  # keep 0, +-pi/2, pi ... as they are
            return m.group(0)
        return f"{m.group(1)}({rng.choice([1, 2, 3])}*pi/2)"

    return re.sub(r"\b(rz|rx|ry|p|u1)\(([^)]*)\)", sub, txt)


ok = True
for rel in args[1:]:
    txt = open(ROOT / "vendor/pushquantum_validator/metric_benchmarks" / rel).read()
    for seed in (1, 2):
        ref = QuantumCircuit.from_qasm_str(cliffordize_qasm(txt, seed))
        out, _arch = cc.compile_circuit(ref.copy(), None)
        ncz = sum(1 for i in out.data if i.operation.name == "cz")
        eq = tableau(strip_moves(out)) == tableau(ref)
        ok &= eq
        print(f"{rel:45s} seed {seed} generic={generic}: {ncz} cz, Clifford tableaux equal: {eq}", flush=True)
print("ALL EQUAL" if ok else "MISMATCH")
