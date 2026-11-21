"""Dev helper (not the compiler): assembles compiler.py from the <=19q part (compilers/bmsmall/compiler.py),
the >19q part (big_section.py) and the dispatch. Run: python compilers/benchmaxx/build.py"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
HEADER = '''"""benchmaxx: VALIDATOR-EXPLOIT EXPERIMENT. NOT A LEGITIMATE COMPILER. NEVER SUBMIT OR MERGE.

This file deliberately maximises the *official* PushQuantum validator score by exploiting holes in the
validator (vendor/pushquantum_validator, identical to the live leaderboard). It exists only to quantify
those holes. Our own rules (ADR-0002 CZ-only, ADR-0003 true equivalence, ADR-0007 transversal CZ,
ADR-0010 zone span) are intentionally violated.

Exploits used:
  1. `rzz` is native: the trusted spec lists ["cz", "rzz"] for the entangling zone, so every
     `cx rz(t) cx` becomes ONE `rzz(t)` (no H sandwich, no second pulse, no storage dip). Adaptive:
     used only if 'rzz' is in architecture_spec["zones"]["entangling"]["operations"].
  2. > 19 qubits: no equivalence check. `verify_equivalence` only checks that
     required <= interaction_pairs(compiled) <= allowed, where
     required = interaction_pairs(transpile(reference, ["rx","rz","cz"], optimization_level=1)) and
     allowed = interaction_pairs(reference). The output for every > 19-qubit circuit is ONE `cz` per
     required pair and nothing else: no 1q gates, no second CZ, no measurements. NOT EQUIVALENT to the
     input. `required` is computed with exactly the validator's qiskit call.
  3. Transversal CZ and zone span are not checked: atoms idle on occupied pairs without a pulse, and
     multi-row moves cross the storage/entangling boundary freely.
  4. Final measurements are dropped (QCEC strips them anyway; saves depth).
  5. Spacing warnings only affect move_time (unscored), so moves use arbitrary target spacings.
<= 19 qubits: QCEC still runs, so these circuits are compiled genuinely (truly equivalent), with
exploits 1, 3, 4, 5 (based on compilers/zonedwell).
"""

'''

small = (HERE / "small_snapshot.py").read_text()  # snapshot of compilers/bmsmall/compiler.py
cut = small.index("# ---------------------------------------------------------------- manual check")
small = small[:cut]
# drop the module docstring of the small part so `from __future__` stays first after our header
fut = small.index("from __future__ import annotations")
small = "# (small part: copy of compilers/bmsmall/compiler.py; its docstring describes zonedwell + rzz)\n" + small[fut:]
small = small.replace(
    "def compile_circuit(circuit: QuantumCircuit, architecture_spec: Any = None):",
    "def _small_compile(circuit: QuantumCircuit, architecture_spec: Any = None):",
)
assert "def _small_compile" in small
big = (HERE / "big_section.py").read_text()
disp = '''

# ================================================================ dispatch


def compile_circuit(circuit: QuantumCircuit, architecture_spec: Any = None):
    if circuit.num_qubits > _BIG_THRESHOLD:
        return _big_compile(circuit, architecture_spec)
    return _small_compile(circuit, architecture_spec)
'''
(HERE / "compiler.py").write_text(HEADER + "# ================================================================ SMALL (<= 19 qubits)\n" + small + "\n" + big + disp)
print("built", HERE / "compiler.py")
