"""Package a single-file compiler into a marimo notebook that the planqc leaderboard can score.

Usage: uv run python bench/package.py compilers/<idea>/compiler.py [-o submission/<idea>.py] [--score]

The leaderboard imports the notebook and only runs its setup cell plus top-level definitions, in a
sandbox without network, where only marimo, qiskit, numpy and scipy (plus the standard library) are
available. So the whole compiler goes into the `with app.setup:` block. `--score` then runs the full
official check on the generated notebook, exactly as the leaderboard calls it (`<notebook>:compile_circuit`).

This never uploads anything; publishing/submitting on aqora needs the user's explicit OK.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_THIRD_PARTY = {"qiskit", "numpy", "scipy", "marimo"}

HEADER = '''# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "qiskit==2.6.2",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
'''

FOOTER = '''

@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(r"""
    # {title}

    Compiler for the planqc zoned neutral-atom track (PushQuantum 2026). The whole compiler lives in
    the setup cell above so that the leaderboard, which only runs the setup cell and top-level
    definitions, can import `compile_circuit`. Design notes: see the team repository `docs/`.
    """)
    return


@app.cell
def _():
    from qiskit import QuantumCircuit as _QC

    _ghz = _QC(3)
    _ghz.h(0)
    _ghz.cx(0, 1)
    _ghz.cx(1, 2)
    _ghz.measure_all()
    _compiled, _architecture = compile_circuit(_ghz, None)
    _compiled.draw("text")
    return


if __name__ == "__main__":
    app.run()
'''


def check_imports(source: str) -> list[str]:
    """Imports that will not exist in the scoring sandbox."""
    bad = []
    for node in ast.walk(ast.parse(source)):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        for name in names:
            top = name.split(".")[0]
            if top not in sys.stdlib_module_names and top not in ALLOWED_THIRD_PARTY:
                bad.append(name)
    return bad


def build(compiler: Path, title: str) -> str:
    source = compiler.read_text(encoding="utf-8")
    tree = ast.parse(source)
    if not any(isinstance(n, ast.FunctionDef) and n.name == "compile_circuit" for n in tree.body):
        raise SystemExit(f"{compiler}: no top-level compile_circuit")
    if bad := check_imports(source):
        raise SystemExit(f"{compiler}: imports not available in the scoring sandbox: {bad}")
    lines = source.splitlines()
    # drop `from __future__` imports (must be first in a file) and the module docstring
    drop = set()
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.module == "__future__":
            drop.update(range(n.lineno - 1, n.end_lineno))
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(getattr(tree.body[0], "value", None), ast.Constant):
        drop.update(range(tree.body[0].lineno - 1, tree.body[0].end_lineno))
    body = "\n".join(l for i, l in enumerate(lines) if i not in drop).strip("\n")
    notebook = HEADER + textwrap.indent(body, "    ", lambda l: l.strip() != "") + "\n" + FOOTER.replace("{title}", title)
    ast.parse(notebook)
    return notebook


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("compiler")
    ap.add_argument("-o", "--output")
    ap.add_argument("--title", default=None)
    ap.add_argument("--score", action="store_true", help="run bench --mode full on the generated notebook")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args(argv)
    compiler = Path(args.compiler).resolve()
    idea = compiler.parent.name
    out = Path(args.output) if args.output else ROOT / "submission" / f"{idea}.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(compiler, args.title or f"planqc compiler: {idea}"), encoding="utf-8")
    print(f"wrote {out}")
    if args.score:
        cmd = [sys.executable, str(ROOT / "bench" / "bench.py"), f"{out.resolve()}:compile_circuit",
               "--mode", "full", "--jobs", str(args.jobs), "--name", f"notebook-{idea}"]
        return subprocess.call(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
