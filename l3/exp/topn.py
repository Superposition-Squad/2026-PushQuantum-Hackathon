"""Does re-ranking solo's top-N candidates after the peephole pick a better plan?"""
import sys, importlib.util, logging, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'vendor'))
logging.disable(logging.WARNING)
import model as M, chain as C, peep as P
from check import check
from qiskit import QuantumCircuit
from pushquantum_validator.spec import load_trusted_spec
spec = importlib.util.spec_from_file_location('st', str(Path(__file__).parent / 'solo_topn.py')); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
DW = S.DEPTH_W
def peep_circ(qc, home):
    sch = M.extract(qc); X, g = C.incumbent(qc, home, sch); B = C.binding(sch)
    for k in range(sch.K + 1):
        need = P.gap_need(sch, k, X[k], X[k + 1], B); old = C.gap_cost(X[k], g[k])
        mv, c = P.optimise_gap(X[k], g[k], need, X[k + 1])
        if c is not None and c < old - 1e-9: g[k] = mv
    out, load, d = C.assemble(sch, home, g); return out
for f in sys.argv[2:]:
    S.TOPN = int(sys.argv[1]); S.TOPN_OUT.clear()
    qc = QuantumCircuit.from_qasm_file(f); t = time.time()
    S.compile_circuit(qc, load_trusted_spec())
    rows = []
    for sc, cq, home in S.TOPN_OUT:
        cq = S._lower_z_phases(cq); cq.global_phase = 0
        a = check(cq, home); po = peep_circ(cq, home); b = check(po, home)
        rows.append((round(a['distance'] + DW * a['depth'], 1), round(b['distance'] + DW * b['depth'], 1), round(a['distance'], 1), round(b['distance'], 1)))
    print(Path(f).stem[9:], f'{time.time()-t:.0f}s', rows, flush=True)
