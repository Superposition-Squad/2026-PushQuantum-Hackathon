"""Sweep WIDE_MIN over the 20 scored circuits; print distance/depth per circuit and the score."""
import concurrent.futures as cf, sys, json
from pathlib import Path
VB = Path(__file__).resolve().parents[2] / 'vendor/pushquantum_validator/metric_benchmarks'
FILES = sorted(VB.glob('ihva/*.qasm')) + [VB / 'hea/bus_stop_12q_hea_linear_p1.qasm', VB / 'hea/bus_stop_12q_hea_linear_p3.qasm']
def one(a):
    w, f = a
    import importlib.util, logging; logging.disable(logging.WARNING)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'vendor'))
    from check import check
    from qiskit import QuantumCircuit
    from pushquantum_validator.spec import load_trusted_spec
    src = str(Path(__file__).parent / 'solo_wide.py') if w > 0 else str(Path(__file__).resolve().parents[2] / 'compilers/solo_wide/compiler.py')
    spec = importlib.util.spec_from_file_location('sw', src); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
    if w > 0: S.WIDE_MIN = w
    out, arch = S.compile_circuit(QuantumCircuit.from_qasm_file(str(f)), load_trusted_spec())
    r = check(out, arch.initial_loading)
    return w, f.stem, r['distance'], r['depth']
if __name__ == '__main__':
    ws = [int(x) for x in sys.argv[1].split(',')]
    jobs = [(w, f) for f in sorted(FILES, key=lambda f: -int(f.name.split('_')[2][:-1])) for w in ws]
    res = {}
    with cf.ProcessPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 6) as ex:
        for w, stem, d, dep in ex.map(one, jobs):
            res.setdefault(w, {})[stem] = (d, dep); print(w, stem, round(d, 1), dep, flush=True)
    json.dump({str(w): v for w, v in res.items()}, open(Path(__file__).parent / ('wide_all_%s.json' % sys.argv[1].replace(',', '_')), 'w'))
    for w, v in sorted(res.items()):
        D = sum(x[0] for x in v.values()); P = sum(x[1] for x in v.values())
        print('WIDE_MIN', w, 'D', round(D, 1), 'depth', P, 'score', round(80 * (163364.298 - D) / 163364.298 + 20 * (13364 - P) / 13364, 4))
