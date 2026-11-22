import sys, importlib.util, logging, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'vendor'))
logging.disable(logging.WARNING)
from check import check
from qiskit import QuantumCircuit
from pushquantum_validator.spec import load_trusted_spec
spec = importlib.util.spec_from_file_location('sw', str(Path(__file__).parent / 'solo_wide.py')); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
S.WIDE_MIN = int(sys.argv[1])
for f in sys.argv[2:]:
    qc = QuantumCircuit.from_qasm_file(f); t = time.time()
    out, arch = S.compile_circuit(qc, load_trusted_spec())
    r = check(out, arch.initial_loading)
    print(Path(f).stem[9:], 'WIDE_MIN', S.WIDE_MIN, round(r['distance'], 1), r['depth'], f'{time.time()-t:.0f}s', flush=True)
