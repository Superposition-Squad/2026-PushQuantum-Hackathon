"""Generate `compilers/portfolio/compiler.py`: run several compilers, keep the best *valid* result.

Usage (from the repo root):
    <python> bench/build_portfolio.py [component ...] [-o compilers/portfolio/compiler.py]

Default components: hubstream zonedwell rectbatch gates (naive is always appended as the last-resort
fallback). Each component is `compilers/<name>/compiler.py`; its source is embedded verbatim as a
string literal (one literal per line, so the marimo packager's re-indentation cannot touch it) and
exec'd at import time into its own module object, because the components define clashing names
(`Move`, `Architecture`, helpers).

The generated file also embeds the official validator's replay (`vendor/pushquantum_validator/
errors.py`, `spec.py`, `_validation.py`, imports adapted, default architecture JSON inlined) and a
copy of bench's transversal-CZ `physics_check` and zone-span check (ADR 0010), so every component result is replayed exactly as
the scorer would replay it and scored with the exact metric formula before one is chosen.

The output is a single self-contained file (ADR 0006): stdlib + qiskit + numpy imports only.
Rebuild after a component changes; this script is the only thing that should write that file.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "pushquantum_validator"
DEFAULT_COMPONENTS = ("hubstream", "zonedwell", "rectbatch", "gates")
FALLBACK = "naive"
ALLOWED_THIRD_PARTY = {"qiskit", "numpy", "scipy"}


# ------------------------------------------------------------------------------------ helpers


def _literal_lines(text: str, indent: str = "    ") -> str:
    """`text` as a parenthesised concatenation of one string literal per source line."""
    lines = text.splitlines(keepends=True)
    if not lines:
        return "''"
    return "(\n" + "".join(f"{indent}{line!r}\n" for line in lines) + ")"


def _nodes_outside_functions(tree):
    """Every node except those inside function bodies (imports there only run if called)."""
    stack = [tree]
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                stack.append(child)


def _check_imports(source: str, where: str) -> None:
    tree = ast.parse(source)
    inner = set(map(id, _nodes_outside_functions(tree)))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        elif isinstance(node, ast.ImportFrom):
            raise SystemExit(f"{where}: relative import")
        for name in names:
            top = name.split(".")[0]
            if top != "__future__" and top not in sys.stdlib_module_names and top not in ALLOWED_THIRD_PARTY:
                if id(node) not in inner:
                    print(f"warning: {where}: import {name!r} inside a function (line {node.lineno}); "
                          "fine unless that function is called", file=sys.stderr)
                    continue
                raise SystemExit(f"{where}: import {name!r} is not available in the scoring sandbox")


def _strip_main_block(source: str) -> str:
    """Drop top-level `if __name__ == "__main__":` blocks (developer helpers, may import vendor code)."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    drop = set()
    for node in tree.body:
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"):
            drop.update(range(node.lineno - 1, node.end_lineno))
    return "".join(l for i, l in enumerate(lines) if i not in drop)


def component_source(name: str) -> str:
    path = ROOT / "compilers" / name / "compiler.py"
    source = _strip_main_block(path.read_text(encoding="utf-8"))
    tree = ast.parse(source)
    if not any(isinstance(n, ast.FunctionDef) and n.name == "compile_circuit" for n in tree.body):
        raise SystemExit(f"{path}: no top-level compile_circuit")
    _check_imports(source, str(path))
    return source


def validator_source() -> str:
    """errors.py + spec.py + _validation.py as one module, package imports removed, default
    architecture inlined, logger renamed and silenced (warnings must not print)."""
    parts = []
    for fname in ("errors.py", "spec.py", "_validation.py"):
        text = (VENDOR / fname).read_text(encoding="utf-8")
        out = []
        for line in text.splitlines():
            if line.startswith("from __future__ import"):
                continue
            if line.startswith("from pushquantum_validator."):
                continue
            out.append(line)
        parts.append(f"# ---- vendor/pushquantum_validator/{fname}\n" + "\n".join(out))
    src = "from __future__ import annotations\n\n" + "\n\n".join(parts) + "\n"
    spec_json = (VENDOR / "architectures" / "default.json").read_text(encoding="utf-8")
    old = 'return resources.files("pushquantum_validator").joinpath("architectures/default.json").read_text(encoding="utf-8")'
    if src.count(old) != 1:
        raise SystemExit("validator spec.py changed: cannot inline the default architecture")
    src = src.replace(old, f"return {spec_json!r}")
    old_logger = 'logger = logging.getLogger("pushquantum_validator")'
    if src.count(old_logger) != 1:
        raise SystemExit("validator _validation.py changed: logger line not found")
    src = src.replace(old_logger, 'logger = logging.getLogger("portfolio_validator")\n'
                                  'logger.addHandler(logging.NullHandler())\nlogger.propagate = False')
    for needed in ("class ValidationArchitecture", "def load_trusted_spec", "class ArchitectureError"):
        if needed not in src:
            raise SystemExit(f"validator source changed: {needed!r} missing")
    _check_imports(src, "embedded validator")
    compile(src, "<validator>", "exec", dont_inherit=True)
    return src


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"


# ------------------------------------------------------------------------------------ template

RUNTIME = r'''

# ============================================================================== portfolio runtime

import importlib.machinery as _importlib_machinery

_PORTFOLIO_PREFIX = "pq_portfolio_"


def _load_embedded(name, source):
    """Exec `source` into a fresh module registered as `pq_portfolio_<name>` (dataclasses and
    `from __future__ import annotations` need the module in sys.modules). Returns it or the error."""
    modname = _PORTFOLIO_PREFIX + name
    module = types.ModuleType(modname)
    module.__file__ = f"<portfolio:{name}>"
    module.__spec__ = _importlib_machinery.ModuleSpec(modname, None)
    sys.modules[modname] = module
    try:
        exec(compile(source, f"<portfolio:{name}>", "exec", dont_inherit=True), module.__dict__)
    except BaseException as error:  # noqa: BLE001 - a broken component must not break the portfolio
        sys.modules.pop(modname, None)
        return error
    return module


_V = _load_embedded("validator", _VALIDATOR_SOURCE)
if isinstance(_V, BaseException):  # cannot happen unless the build is broken; fail loudly
    raise _V

_COMPONENT_MODULES = {}
_LOAD_ERRORS = {}
for _name in _COMPONENTS + (_FALLBACK,):
    _m = _load_embedded(_name, _SOURCES[_name])
    if isinstance(_m, BaseException):
        _LOAD_ERRORS[_name] = f"{type(_m).__name__}: {_m}"
    else:
        _COMPONENT_MODULES[_name] = _m
del _name, _m

_ALLOWED_GATES = frozenset({"rx", "rz", "cz", "move", "measure", "barrier"})
_IDENTITIES = frozenset({"move", "barrier"})


class PortfolioReject(Exception):
    """A component result failed one of the portfolio's own checks."""


def _strip_moves(circuit):
    stripped = circuit.copy_empty_like()
    for instruction in circuit.data:
        if instruction.operation.name not in _IDENTITIES:
            stripped._append(instruction)
    return stripped


def _physics_check(compiled, loading, trace):
    """Copy of bench.py `physics_check` (ADR 0007, transversal CZ)."""
    ent = _V.load_trusted_spec()["zones"]["entangling"]
    (x0, x1), (y0, y1) = ent["x"], ent["y"]
    move_at = {rec.index: rec.positions for rec in trace}
    index = {bit: i for i, bit in enumerate(compiled.qubits)}
    pos = dict(loading)

    def occupied_pairs():
        sites = {s: q for q, s in pos.items() if x0 <= s[0] <= x1 and y0 <= s[1] <= y1}
        pairs = set()
        for (x, y), q in sites.items():
            if (x - x0) % 2 == 0 and (x + 1, y) in sites:
                pairs.add(frozenset((q, sites[(x + 1, y)])))
        return pairs

    counts = {}

    def close(at):
        if counts:
            occ = occupied_pairs()
            ks = {counts.get(p, 0) for p in occ}
            if len(ks) > 1 or set(counts) - occ:
                raise PortfolioReject(f"transversal CZ violated in the segment ending at instruction {at}")
        counts.clear()

    for i, inst in enumerate(compiled.data):
        name = inst.operation.name
        if name == "move":
            close(i)
            pos.update(move_at.get(i, {}))
        elif name == "cz":
            p = frozenset(index[q] for q in inst.qubits)
            counts[p] = counts.get(p, 0) + 1
    close(len(compiled.data))


def _zonespan_check(trace):
    """Copy of bench.py `zonespan_check` (ADR 0010): a move selecting >= 2 rows keeps its picked-up
    rows within one zone and its target rows within one zone."""
    zones = _V.load_trusted_spec()["zones"]

    def zone_of(row):
        for name, z in zones.items():
            if z["y"][0] <= row <= z["y"][1]:
                return name
        return None

    for rec in trace:
        mv = rec.move
        for lines in (mv.rows, mv.new_rows):
            if len(lines) >= 2 and len({zone_of(r) for r in lines}) > 1:
                raise PortfolioReject(f"zone span violated at instruction {rec.index}: rows {tuple(lines)}")


def _interaction_pairs(circuit):
    index = {bit: i for i, bit in enumerate(circuit.qubits)}
    return {
        frozenset(pair)
        for instruction in circuit.data
        if len(instruction.qubits) > 1 and instruction.operation.name not in _IDENTITIES
        for pair in itertools.combinations([index[q] for q in instruction.qubits], 2)
    }


def _reference_pairs(reference):
    """(required, allowed) interaction pairs, exactly as the validator's `_check_interactions`."""
    allowed = _interaction_pairs(reference)
    required = _interaction_pairs(transpile(reference, basis_gates=["rx", "rz", "cz"], optimization_level=1))
    return required, allowed


def _statevector_guard(gates, reference, max_qubits):
    """Cheap equivalence evidence for small circuits: both applied to one fixed random product state
    must agree up to global phase. Returns a status string; raises PortfolioReject on a mismatch.
    Anything it cannot simulate (e.g. mid-circuit measurements) is skipped, never rejected."""
    if reference.num_qubits > max_qubits:
        return "skipped"
    try:
        import numpy as _np
        from qiskit.quantum_info import Statevector
        a = gates.remove_final_measurements(inplace=False)
        b = reference.remove_final_measurements(inplace=False)
        rng = _np.random.default_rng(20261121)
        prep = QuantumCircuit(reference.num_qubits)
        for q in range(reference.num_qubits):
            prep.u(*rng.uniform(0, 2 * math.pi, 3), q)
        sa = Statevector(prep).evolve(a)
        sb = Statevector(prep).evolve(b)
        fid = abs(sa.inner(sb)) ** 2
    except Exception as error:  # noqa: BLE001
        return f"skipped ({type(error).__name__})"
    if fid < 1 - 1e-6:
        raise PortfolioReject(f"statevector guard: fidelity {fid:.6f} < 1")
    return "ok"


def _evaluate(reference, output, ref_pairs):
    """Validate one component output like the scorer + our ADRs; returns (cost, D, P)."""
    if not (isinstance(output, tuple) and len(output) == 2):
        raise PortfolioReject(f"compile_circuit returned {type(output).__name__}, not a 2-tuple")
    compiled, architecture = output
    if not isinstance(compiled, QuantumCircuit):
        raise PortfolioReject(f"compiled circuit is a {type(compiled).__name__}")
    if compiled.num_qubits != reference.num_qubits:
        raise PortfolioReject(f"{compiled.num_qubits} qubits, reference has {reference.num_qubits}")
    for instruction in compiled.data:
        op = instruction.operation
        if op.name not in _ALLOWED_GATES:
            raise PortfolioReject(f"gate {op.name!r} outside rx/rz/cz/move/measure/barrier")
        if op.name != "move":
            for p in op.params:
                float(p)  # the scorer's child serialises params with float()
    raw = architecture.initial_loading
    loading = {int(q): (int(s[0]), int(s[1])) for q, s in raw.items()}
    replay = _V.ValidationArchitecture.from_loading(loading, None)  # always the trusted spec, like the scorer
    trace = replay.run(compiled, log_file=os.devnull)
    _zonespan_check(trace)
    _physics_check(compiled, loading, trace)
    gates = _strip_moves(compiled)
    required, allowed = ref_pairs
    got = _interaction_pairs(gates)
    if required - got or got - allowed:
        raise PortfolioReject(f"interaction pairs differ ({len(required - got)} missing, {len(got - allowed)} extra)")
    distance = sum(record.distance for record in trace)
    depth = gates.depth()
    cost = _W_DISTANCE * distance / _NAIVE_DISTANCE + _W_DEPTH * depth / _NAIVE_DEPTH
    return cost, distance, depth, gates


def _clock():
    return max(time.monotonic(), 0.0), time.process_time()


def portfolio_compile(circuit, architecture_spec=None):
    """Runs every component, validates each output, returns `(best_output, report)`.

    report: {"chosen": name, "components": {name: {...}}, "total_s": float, "cpu_s": float}."""
    w0, c0 = _clock()
    report = {"chosen": None, "components": {}, "load_errors": dict(_LOAD_ERRORS)}
    ref_pairs = _reference_pairs(circuit)
    best = None  # (cost, order, name, output)
    for order, name in enumerate(_COMPONENTS + (_FALLBACK,)):
        if name == _FALLBACK and best is not None:
            break
        entry = report["components"].setdefault(name, {})
        module = _COMPONENT_MODULES.get(name)
        if module is None:
            entry["status"] = "load error: " + _LOAD_ERRORS.get(name, "?")
            continue
        w, c = _clock()
        spent = max(w - w0, c - c0)
        if name != _FALLBACK and best is not None and spent > _BUDGET_S:
            entry["status"] = f"skipped (budget: {spent:.1f}s spent)"
            continue
        try:
            output = module.compile_circuit(circuit.copy(), architecture_spec)
            entry["compile_s"] = round(time.monotonic() - w, 3)
            cost, distance, depth, gates = _evaluate(circuit, output, ref_pairs)
            entry["guard"] = _statevector_guard(gates, circuit, _SV_MAX_QUBITS)
        except BaseException as error:  # noqa: BLE001 - includes SystemExit/RecursionError from a component
            if isinstance(error, KeyboardInterrupt):
                raise
            entry.setdefault("compile_s", round(time.monotonic() - w, 3))
            entry["status"] = f"rejected: {type(error).__name__}: {str(error)[:200]}"
            continue
        entry.update(status="valid", cost=cost, distance=round(distance, 6), depth=depth,
                     total_s=round(time.monotonic() - w, 3))
        if best is None or cost < best[0]:  # strict: earlier component wins ties
            best = (cost, order, name, output)
    if best is None:
        raise RuntimeError(f"portfolio: no component produced a valid result: {report['components']}")
    report["chosen"] = best[2]
    w, c = _clock()
    report["total_s"] = round(w - w0, 3)
    report["cpu_s"] = round(c - c0, 3)
    log = os.environ.get("PORTFOLIO_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"qubits": circuit.num_qubits, "name": circuit.name, **report}, default=str) + "\n")
    return best[3], report


def compile_circuit(circuit, architecture_spec=None):
    return portfolio_compile(circuit, architecture_spec)[0]
'''


def build(components: list[str], budget: float, sv_max_qubits: int) -> str:
    names = list(dict.fromkeys(components))
    if FALLBACK in names:
        names.remove(FALLBACK)
    sources = {n: component_source(n) for n in names + [FALLBACK]}
    vsrc = validator_source()
    head = _git("rev-parse", "--short", "HEAD")
    info = []
    for n in names + [FALLBACK]:
        digest = hashlib.sha256(sources[n].encode()).hexdigest()[:12]
        info.append(f"#   {n:<12} compilers/{n}/compiler.py  sha256:{digest}  ({sources[n].count(chr(10))} lines)")
    out = [
        '"""Portfolio compiler: GENERATED by bench/build_portfolio.py, do not edit by hand.',
        "",
        "Runs every embedded component compiler on the circuit, replays each output with an embedded copy",
        "of the official validator, checks transversal CZ (ADR 0007), the interaction-pair set, and (small",
        "circuits) a statevector guard, then returns the valid output with the lowest exact score cost",
        "0.8*D/D_naive + 0.2*P/P_naive. Ties go to the earlier component; naive is the last-resort fallback.",
        '"""',
        "",
        f"# built from commit {head} with components: {' '.join(names)} (+ {FALLBACK} fallback)",
        *info,
        "",
        "import itertools",
        "import json",
        "import math",
        "import os",
        "import sys",
        "import time",
        "import types",
        "",
        "from qiskit import QuantumCircuit, transpile",
        "",
        f"_COMPONENTS = {tuple(names)!r}",
        f"_FALLBACK = {FALLBACK!r}",
        f"_BUDGET_S = {float(budget)!r}  # skip remaining components once this much wall/CPU time is spent",
        f"_SV_MAX_QUBITS = {int(sv_max_qubits)!r}",
        "_W_DISTANCE, _W_DEPTH = 0.8, 0.2",
        "_NAIVE_DISTANCE, _NAIVE_DEPTH = 163364.3, 13364  # v5 leaderboard set (docs/adr/0011)",
        "",
        f"_VALIDATOR_SOURCE = {_literal_lines(vsrc)}",
        "",
        "_SOURCES = {}",
    ]
    for n in names + [FALLBACK]:
        out.append(f"\n# {'=' * 90} component: {n}\n_SOURCES[{n!r}] = {_literal_lines(sources[n])}")
    text = "\n".join(out) + "\n" + RUNTIME
    ast.parse(text)
    _check_imports(text, "generated portfolio")
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("components", nargs="*", default=list(DEFAULT_COMPONENTS))
    ap.add_argument("-o", "--output", default=str(ROOT / "compilers" / "portfolio" / "compiler.py"))
    ap.add_argument("--budget", type=float, default=150.0,
                    help="seconds (max of wall and CPU) after which remaining components are skipped")
    ap.add_argument("--sv-max-qubits", type=int, default=14, help="statevector guard up to this many qubits")
    args = ap.parse_args(argv)
    text = build(args.components, args.budget, args.sv_max_qubits)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text) // 1024} KiB, components: {' '.join(args.components)} + {FALLBACK})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
