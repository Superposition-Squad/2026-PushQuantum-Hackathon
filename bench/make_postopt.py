#!/usr/bin/env python3
"""Generate compilers/<name>_po/compiler.py = compilers/<name>/compiler.py + compilers/postopt.py +
a wrapper that post-optimizes every compile_circuit result (single self-contained file, ADR 0006).

Usage: python bench/make_postopt.py <name> [<name> ...]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PO = (ROOT / "compilers" / "postopt.py").read_text()
PO_BODY = "\n".join(line for line in PO.splitlines() if not line.startswith("from __future__"))
WRAP = '''

# ---------------------------------------------------------------- post-optimizer wrapper (bench/make_postopt.py)
_po_base_compile_circuit = compile_circuit


def compile_circuit(circuit, architecture_spec):
    out, arch = _po_base_compile_circuit(circuit, architecture_spec)
    try:
        out = postoptimize(out, arch.initial_loading, architecture_spec)
    except Exception:  # noqa: BLE001 - keep the component's own (valid) result
        pass
    return out, arch
'''

for name in sys.argv[1:]:
    src = (ROOT / "compilers" / name / "compiler.py").read_text()
    out = ROOT / "compilers" / f"{name}_po" / "compiler.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(src.rstrip() + "\n\n\n# ================================================================ postopt (compilers/postopt.py)\n"
                   + PO_BODY + WRAP)
    print("wrote", out)
