"""Exact Clifford-tableau equivalence check at any size (hubwalk validity evidence, docs/ideas/hubwalk.md).

Every rotation angle of the benchmark is replaced by a Clifford angle (k*pi/2, k drawn
deterministically per gate from {1,2,3}; existing multiples of pi/2 such as iHVA's +-pi/2 are kept,
zero angles are kept zero), the circuit is compiled, moves/barriers/measures are stripped, and the
stabilizer tableaux (up to global phase) of compiled vs reference are compared.

usage (repo root): python docs/ideas/hubwalk_cliff_check.py compilers/hubwalk/compiler.py qaoa/bus_stop_50q_qaoa_p3.qasm ...
"""
import sys, re, importlib.util, random
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor"))
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, Operator
from pushquantum_validator.verify import strip_moves

spec = importlib.util.spec_from_file_location("cc", sys.argv[1])
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)


def tableau(qc):
    n = qc.num_qubits
    C = Clifford(QuantumCircuit(n))
    idx = {b: i for i, b in enumerate(qc.qubits)}
    run = {}

    def flush(q):
        nonlocal C
        if q in run:
            m = run.pop(q)
            C = C.compose(Clifford.from_matrix(m), qargs=[q])

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
        expr = m.group(2)
        try:
            val = float(eval(expr.replace("pi", str(np.pi))))
        except Exception:
            val = 1.0
        k = val / (np.pi / 2)
        if abs(k - round(k)) < 1e-9:  # keep 0, +-pi/2, pi ... as they are
            return m.group(0)
        return f"{m.group(1)}({rng.choice([1, 2, 3])}*pi/2)"

    return re.sub(r"\b(rz|rx|ry|p|u1)\(([^)]*)\)", sub, txt)


ok = True
for rel in sys.argv[2:]:
    txt = open(ROOT / "vendor/pushquantum_validator/metric_benchmarks" / rel).read()
    for seed in (1, 2):
        ref = QuantumCircuit.from_qasm_str(cliffordize_qasm(txt, seed))
        out, _arch = cc.compile_circuit(ref.copy(), None)
        a = tableau(strip_moves(out))
        b = tableau(ref)
        eq = a == b
        ok &= eq
        print(f"{rel:45s} seed {seed}: Clifford tableaux equal: {eq}", flush=True)
print("ALL EQUAL" if ok else "MISMATCH")
