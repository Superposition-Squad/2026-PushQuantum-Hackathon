"""Gate-level resynthesis + commutation-aware CZ-round scheduling for the zoned neutral-atom machine.

The compiler is organised in three separable layers:

1. **Gate layer** (`lower_to_items`, `expand_rot2`, `merge_1q_runs`, `build_commutation_dag`,
   `schedule_rounds`): turns an arbitrary input circuit into an exactly equivalent sequence of
   *phases*, alternating a 1q phase (`rx`/`rz` on atoms sitting in storage) and a CZ round (a set of
   disjoint pairs that are entangled between two moves).  Nothing in this layer knows about
   geometry, so a different movement planner can consume `schedule_rounds` unchanged.
2. **Placement layer** (`initial_loading`): where each qubit lives in the storage zone.
3. **Movement layer** (`plan_round_moves`, `merge_move_records`): per CZ round, bring the pairs to
   interaction sites with as few / as short `Move`s as possible and bring them home again.

Every rewrite is exact algebra (Pauli conjugation, diagonal commutation, Euler decomposition), so
the output is equivalent up to global phase at every circuit size (ADR 0003).  Only `rx`, `rz`, `cz`,
`move`, `measure` are emitted (ADR 0002).  Within a round every doubly occupied interaction pair
receives exactly one `cz`, so the transversal-CZ rule (ADR 0007) holds trivially.
"""

import bisect
import itertools
import math
from collections.abc import Sequence
from numbers import Integral
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction
from qiskit.quantum_info import Operator

Site = tuple[int, int]


# ---------------------------------------------------------------- template plumbing (keep as is)


def _as_lines(values, name):
    items = [values] if isinstance(values, Integral) else list(values)
    if not items:
        raise ValueError(f"`{name}` must not be empty.")
    result = tuple(int(v) for v in items)
    if len(set(result)) != len(result):
        raise ValueError(f"`{name}` must not contain duplicates, got {result!r}.")
    return result


def _check_lines(start, end, name):
    if len(end) > len(start):
        raise ValueError(f"`new_{name}` cannot have more lines than `{name}`.")
    if start != tuple(range(start[0], start[0] + len(start))):
        raise ValueError(f"Picked-up {name} {start} must be contiguous and ascending.")
    if any(b <= a for a, b in itertools.pairwise(end)):
        raise ValueError(f"Target {name} {end} must be ascending.")


class Move(Instruction):
    """Params are `(rows, cols, new_rows, new_cols)`, tuples of ints; rows are y, columns are x."""

    def __init__(self, rows, cols, new_rows, new_cols, num_qubits):
        names = ("rows", "cols", "new_rows", "new_cols")
        lines = [_as_lines(v, n) for v, n in zip((rows, cols, new_rows, new_cols), names, strict=True)]
        _check_lines(lines[0], lines[2], "rows")
        _check_lines(lines[1], lines[3], "cols")
        super().__init__("move", num_qubits, 0, lines)

    def validate_parameter(self, parameter):
        return parameter

    def _define(self):
        self.definition = QuantumCircuit(self.num_qubits, name=self.name)


class Architecture:
    """Only `initial_loading` is read by the validator (qubit -> (x, y) site in storage)."""

    def __init__(self, initial_loading: dict[int, Site], spec: Any = None) -> None:
        self.initial_loading = dict(initial_loading)
        self.spec = spec


def _move(circuit: QuantumCircuit, qubits: Sequence[int], rows, cols, new_rows, new_cols) -> None:
    circuit.append(Move(rows, cols, new_rows, new_cols, len(qubits)), list(qubits))


# ---------------------------------------------------------------- architecture geometry

STORAGE_X = (0, 9)
STORAGE_Y = (0, 9)
ENT_X = (0, 9)
ENT_Y = (10, 19)


def _geometry(spec: Any):
    """(storage, entangling) bounds as ((x0, x1), (y0, y1)) each, from the spec when it is a dict."""
    storage = (STORAGE_X, STORAGE_Y)
    ent = (ENT_X, ENT_Y)
    if isinstance(spec, dict):
        zones = spec.get("zones")
        if isinstance(zones, dict):
            if isinstance(zones.get("storage"), dict):
                z = zones["storage"]
                storage = (tuple(z["x"]), tuple(z["y"]))
            if isinstance(zones.get("entangling"), dict):
                z = zones["entangling"]
                ent = (tuple(z["x"]), tuple(z["y"]))
    return storage, ent


def _phys(site: Site, ent) -> tuple[float, float]:
    """Physical (x, y) of a site. The entangling zone squeezes pair partners 0.3 apart."""
    x, y = site
    (ex0, ex1), (ey0, ey1) = ent
    if ex0 <= x <= ex1 and ey0 <= y <= ey1:
        k = x - ex0
        return (ex0 + 2 * (k // 2) + 0.3 * (k % 2), float(y))
    return (float(x), float(y))


def _dist(a: Site, b: Site, ent) -> float:
    (xa, ya), (xb, yb) = _phys(a, ent), _phys(b, ent)
    return math.hypot(xa - xb, ya - yb)


# ---------------------------------------------------------------- 1q algebra

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
_PAULI = {"X": _X, "Y": _Y, "Z": _Z}


def _rx_mat(t: float) -> np.ndarray:
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _rz_mat(t: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j * t), 0], [0, np.exp(0.5j * t)]], dtype=complex)


# C with C Z C^dagger = P, so that exp(-i t/2 P) = C exp(-i t/2 Z) C^dagger.
_BASIS_CHANGE = {"Z": _I2, "X": _H, "Y": _rx_mat(-math.pi / 2)}


def _is_identity(u: np.ndarray) -> bool:
    """True if u is a global phase times the identity."""
    if abs(u[0, 0]) < 1e-9:
        return False
    v = u / u[0, 0]
    return bool(np.allclose(v, _I2, atol=1e-9))


def _commuting_pauli(u: np.ndarray):
    """Which Pauli u commutes with ('Z' first, so diagonal gates are classified as Z), else None."""
    for name in ("Z", "X", "Y"):
        p = _PAULI[name]
        if np.allclose(u @ p, p @ u, atol=1e-9):
            return name
    return None


def euler_zxz(u: np.ndarray) -> tuple[list[tuple[str, float]], float]:
    """Decompose a 2x2 unitary into circuit-order [rz(d), rx(p), rz(l)] plus a global phase.

    Returns (gates, phase) with `u = exp(i*phase) Rz(l) Rx(p) Rz(d)` as matrices; near-zero
    rotations are dropped."""
    det = u[0, 0] * u[1, 1] - u[0, 1] * u[1, 0]
    phase = 0.5 * float(np.angle(det))
    su = u * np.exp(-1j * phase)  # in SU(2)
    # Rz(l) Rx(p) Rz(d) = [[c e^{-i(l+d)/2}, -i s e^{-i(l-d)/2}],
    #                      [-i s e^{ i(l-d)/2},  c e^{ i(l+d)/2}]]   with c=cos(p/2), s=sin(p/2)
    c = min(1.0, abs(su[0, 0]))
    s = min(1.0, abs(su[0, 1]))
    p = 2 * math.atan2(s, c)
    if s < 1e-9:  # pure Rz
        lpd = -2 * float(np.angle(su[0, 0]))
        l, d = lpd, 0.0
    elif c < 1e-9:  # Rx(pi) sandwiched
        lmd = -2 * float(np.angle(su[0, 1] / -1j))
        l, d = lmd, 0.0
    else:
        a = -2 * float(np.angle(su[0, 0]))          # l + d
        b = -2 * float(np.angle(su[0, 1] / -1j))    # l - d
        l, d = 0.5 * (a + b), 0.5 * (a - b)
    gates: list[tuple[str, float]] = []
    for name, angle in (("rz", d), ("rx", p), ("rz", l)):
        if abs(angle) > 1e-12:
            gates.append((name, angle))
    return gates, phase


# ---------------------------------------------------------------- items (the gate-level IR)


class Item:
    """One scheduling unit.

    kind: "1q" (matrix `mat` on qubits[0]), "cz" (qubits[0], qubits[1]),
          "rot2" (exp(-i theta/2 P0 (x) P1), `paulis`), "meas" (qubits[0] -> `clbit`).
    `basis[q]` is the Pauli whose rotations commute with this item on qubit q (None = nothing does).
    """

    __slots__ = ("kind", "qubits", "mat", "theta", "paulis", "clbit", "basis")

    def __init__(self, kind, qubits, mat=None, theta=0.0, paulis=None, clbit=None):
        self.kind = kind
        self.qubits = tuple(qubits)
        self.mat = mat
        self.theta = theta
        self.paulis = paulis
        self.clbit = clbit
        if kind == "1q":
            self.basis = {self.qubits[0]: _commuting_pauli(mat)}
        elif kind == "cz":
            self.basis = {self.qubits[0]: "Z", self.qubits[1]: "Z"}
        elif kind == "rot2":
            self.basis = {self.qubits[0]: paulis[0], self.qubits[1]: paulis[1]}
        else:
            self.basis = {q: None for q in self.qubits}

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"Item({self.kind}, {self.qubits})"


_PAULI_ROT_GATES = {"rzz": ("Z", "Z"), "rxx": ("X", "X"), "ryy": ("Y", "Y")}
_CX_NAMES = ("cx", "cnot")


def _native_plan(op, width: int):
    """Transpile one k-qubit gate into rx/rz/cz and *verify* the result against its unitary.

    `optimization_level >= 2` runs `ElidePermutations`, which replaces a `swap` by a relabelling
    recorded in the layout: taking its `data` at face value silently drops the swap.  So we try the
    low levels in order and keep the first synthesis whose operator matches the gate (up to global
    phase).  Nothing here may be relaxed — ADR 0003 requires true equivalence at every size.
    """
    target = None
    if width <= 5:
        target = np.asarray(Operator(op).data, dtype=complex)
    last_error = None
    for level in (1, 0):
        scratch = QuantumCircuit(width)
        scratch.append(op, range(width))
        native = transpile(scratch, basis_gates=["rx", "rz", "cz"], optimization_level=level)
        nidx = {b: k for k, b in enumerate(native.qubits)}
        steps = [(ins.operation.name,
                  tuple(nidx[b] for b in ins.qubits),
                  tuple(float(p) for p in ins.operation.params))
                 for ins in native.data if ins.operation.name != "barrier"]
        plan = (steps, float(native.global_phase))
        if target is None:
            return plan
        check = QuantumCircuit(width)
        check.global_phase = plan[1]
        for sname, sqs, sparams in steps:
            if sname == "cz":
                check.cz(sqs[0], sqs[1])
            elif sname == "rx":
                check.rx(sparams[0], sqs[0])
            elif sname == "rz":
                check.rz(sparams[0], sqs[0])
            else:
                check = None
                break
        if check is None:
            last_error = f"unexpected gate in synthesis of {op.name}"
            continue
        got = np.asarray(Operator(check).data, dtype=complex)
        overlap = abs(np.trace(got.conj().T @ target)) / target.shape[0]
        if overlap > 1 - 1e-9:
            return plan
        last_error = f"synthesis of {op.name} at level {level} is not equivalent (overlap {overlap:.6f})"
    raise ValueError(f"cannot synthesise {op.name} exactly: {last_error}")


def _next_on(positions, q, i):
    lst = positions[q]
    k = bisect.bisect_right(lst, i)
    return lst[k] if k < len(lst) else None


def lower_to_items(circuit: QuantumCircuit) -> tuple[list[Item], float]:
    """Lower an arbitrary circuit to a flat list of items, exactly. Returns (items, global phase).

    Recognised patterns:
      * `cx(c,t); rz(theta) t; cx(c,t)`  ->  rot2(c, t, theta, Z, Z)        (QAOA / iHVA cores)
      * `rzz/rxx/ryy(theta)`             ->  rot2(..., P, P)
      * `cx`, `cz`                       ->  native
      * any 1q gate                      ->  its exact 2x2 matrix
      * anything else                    ->  per-gate `transpile` to rx/rz/cz (cached)
    """
    index = {bit: i for i, bit in enumerate(circuit.qubits)}
    cindex = {bit: i for i, bit in enumerate(circuit.clbits)}
    ops = []
    positions: dict[int, list[int]] = {q: [] for q in range(circuit.num_qubits)}
    for ins in circuit.data:
        qs = tuple(index[b] for b in ins.qubits)
        ops.append((ins.operation.name, qs, ins))
        for q in qs:
            positions[q].append(len(ops) - 1)

    phase = float(circuit.global_phase)
    items: list[Item] = []
    consumed: set[int] = set()
    cache: dict[Any, Any] = {}

    for i, (name, qs, ins) in enumerate(ops):
        if i in consumed:
            continue
        op = ins.operation
        if name == "barrier":
            continue
        if name == "measure":
            items.append(Item("meas", qs, clbit=cindex[ins.clbits[0]]))
            continue
        if name in _CX_NAMES and len(qs) == 2:
            c, t = qs
            j = _next_on(positions, t, i)
            if j is not None and ops[j][0] == "rz" and ops[j][1] == (t,):
                k1 = _next_on(positions, t, j)
                k2 = _next_on(positions, c, i)
                if k1 is not None and k1 == k2 and ops[k1][0] in _CX_NAMES and ops[k1][1] == (c, t):
                    theta = float(ops[j][2].operation.params[0])
                    items.append(Item("rot2", (c, t), theta=theta, paulis=("Z", "Z")))
                    consumed.add(j)
                    consumed.add(k1)
                    continue
            items.append(Item("1q", (t,), mat=_H))
            items.append(Item("cz", (c, t)))
            items.append(Item("1q", (t,), mat=_H))
            continue
        if name in _PAULI_ROT_GATES and len(qs) == 2:
            theta = float(op.params[0])
            items.append(Item("rot2", qs, theta=theta, paulis=_PAULI_ROT_GATES[name]))
            continue
        if name == "cz" and len(qs) == 2:
            items.append(Item("cz", qs))
            continue
        if len(qs) == 1:
            items.append(Item("1q", qs, mat=np.asarray(Operator(op).data, dtype=complex)))
            continue
        # generic fallback: transpile this one gate into the native basis.
        key = (name, tuple(float(p) for p in op.params) if op.params else (), len(qs))
        plan = cache.get(key)
        if plan is None:
            plan = _native_plan(op, len(qs))
            cache[key] = plan
        steps, gphase = plan
        phase += gphase
        for sname, sqs, sparams in steps:
            if sname == "cz":
                items.append(Item("cz", (qs[sqs[0]], qs[sqs[1]])))
            elif sname == "rx":
                items.append(Item("1q", (qs[sqs[0]],), mat=_rx_mat(sparams[0])))
            elif sname == "rz":
                items.append(Item("1q", (qs[sqs[0]],), mat=_rz_mat(sparams[0])))
            elif sname == "barrier":
                continue
            else:  # pragma: no cover - transpile is asked for rx/rz/cz only
                raise ValueError(f"unexpected gate {sname} from fallback transpilation")
    return items, phase


def two_qubit_degree(items: list[Item], num_qubits: int) -> list[int]:
    deg = [0] * num_qubits
    for it in items:
        if it.kind in ("cz", "rot2"):
            for q in it.qubits:
                deg[q] += 1
    return deg


def expand_rot2(items: list[Item], deg: list[int]) -> list[Item]:
    """Expand every `rot2` into 1q gates + 2 CZ, putting the inter-CZ rotation on the LOW-degree side.

    exp(-i t/2 Pu (x) Pv) = (Cu (x) Cv) exp(-i t/2 Z (x) Z) (Cu (x) Cv)^dagger   with C Z C^d = P
    exp(-i t/2 Z (x) Z)   = H_m . CZ . rx(t)_m . CZ . H_m     for either m in {u, v}

    Choosing m = the qubit with fewer two-qubit gates means the hub / reference qubit sees only CZs,
    which lets all its CZs commute (ADR 0003-safe: this is exact algebra).
    """
    out: list[Item] = []
    for it in items:
        if it.kind != "rot2":
            out.append(it)
            continue
        u, v = it.qubits
        pu, pv = it.paulis
        cu, cv = _BASIS_CHANGE[pu], _BASIS_CHANGE[pv]
        mid_on_v = deg[v] <= deg[u]
        if mid_on_v:
            pre_u, post_u = cu.conj().T, cu
            pre_v, post_v = _H @ cv.conj().T, cv @ _H
            mid_q = v
        else:
            pre_u, post_u = _H @ cu.conj().T, cu @ _H
            pre_v, post_v = cv.conj().T, cv
            mid_q = u
        for q, m in ((u, pre_u), (v, pre_v)):
            if not _is_identity(m):
                out.append(Item("1q", (q,), mat=m))
        out.append(Item("cz", (u, v)))
        out.append(Item("1q", (mid_q,), mat=_rx_mat(it.theta)))
        out.append(Item("cz", (u, v)))
        for q, m in ((u, post_u), (v, post_v)):
            if not _is_identity(m):
                out.append(Item("1q", (q,), mat=m))
    return out


def merge_1q_runs(items: list[Item], num_qubits: int) -> tuple[list[Item], float]:
    """Merge maximal runs of consecutive 1q items on the same wire; drop identities.

    Two 1q items adjacent in a qubit's wire order can be merged even if unrelated items on other
    qubits sit between them, so this is exact. It is what cancels the iHVA `rx(pi/2) rx(-pi/2)`
    pairs and the `H H` of back-to-back `cx`s, freeing the hub's CZ chain to commute.
    """
    wire: dict[int, list[int]] = {q: [] for q in range(num_qubits)}
    for idx, it in enumerate(items):
        for q in it.qubits:
            wire[q].append(idx)
    dead = [False] * len(items)
    phase = 0.0
    for q in range(num_qubits):
        run: list[int] = []
        for idx in wire[q] + [None]:
            if idx is not None and items[idx].kind == "1q":
                run.append(idx)
                continue
            if len(run) >= 1:
                mat = _I2
                for r in run:
                    mat = items[r].mat @ mat
                keep = run[-1]
                for r in run[:-1]:
                    dead[r] = True
                if _is_identity(mat):
                    dead[keep] = True
                    if abs(mat[0, 0]) > 1e-9:
                        phase += float(np.angle(mat[0, 0]))
                else:
                    items[keep] = Item("1q", (q,), mat=mat)
            run = []
    return [it for it, d in zip(items, dead, strict=True) if not d], phase


def build_commutation_dag(items: list[Item], num_qubits: int):
    """Commutation-set construction. Returns (successors, indegree).

    Soundness: every item's generator is a sum of Pauli strings S with S(q) in {I, basis(q)}.  If two
    items agree on the basis of every shared qubit they anticommute on zero qubits, hence commute.
    """
    n = len(items)
    succ: list[list[int]] = [[] for _ in range(n)]
    indeg = [0] * n
    cur: dict[int, list[int]] = {q: [] for q in range(num_qubits)}
    prev: dict[int, list[int]] = {q: [] for q in range(num_qubits)}
    cur_basis: dict[int, Any] = {q: object() for q in range(num_qubits)}
    for idx, it in enumerate(items):
        for q in it.qubits:
            b = it.basis[q]
            if cur[q] and b is not None and cur_basis[q] == b:
                for p in prev[q]:
                    succ[p].append(idx)
                    indeg[idx] += 1
                cur[q].append(idx)
            else:
                for p in cur[q]:
                    succ[p].append(idx)
                    indeg[idx] += 1
                prev[q] = cur[q]
                cur[q] = [idx]
                cur_basis[q] = b
    return succ, indeg


def schedule_rounds(items: list[Item], num_qubits: int) -> list[tuple[str, list[int]]]:
    """ASAP list schedule of the commutation DAG into alternating phases.

    Returns a list of `("1q", [item indices])` and `("cz", [item indices])` phases.  A `"cz"` phase
    (a *round*) is always a set of pairwise disjoint pairs, so it can be executed between two moves.
    CZs are picked by a maximal matching in order of decreasing criticality (longest remaining
    CZ-path), which keeps the hub busy in every round and gives ~2*max_degree rounds per layer
    (edge-colouring behaviour, cf. arXiv:2405.15095).
    """
    succ, indeg = build_commutation_dag(items, num_qubits)
    n = len(items)
    crit = [0] * n
    for idx in range(n - 1, -1, -1):
        best = 0
        for s in succ[idx]:
            if crit[s] > best:
                best = crit[s]
        crit[idx] = best + (1 if items[idx].kind == "cz" else 0)

    remaining = list(indeg)
    ready = [idx for idx in range(n) if remaining[idx] == 0]
    phases: list[tuple[str, list[int]]] = []
    done = 0

    def release(idx, bucket_1q, bucket_cz):
        for s in succ[idx]:
            remaining[s] -= 1
            if remaining[s] == 0:
                (bucket_cz if items[s].kind == "cz" else bucket_1q).append(s)

    ready_1q = [i for i in ready if items[i].kind != "cz"]
    ready_cz = [i for i in ready if items[i].kind == "cz"]

    while done < n:
        emitted: list[int] = []
        queue = ready_1q
        ready_1q = []
        while queue:
            idx = queue.pop()
            emitted.append(idx)
            done += 1
            release(idx, queue, ready_cz)
        if emitted:
            emitted.sort()
            phases.append(("1q", emitted))
        if done >= n:
            break
        ready_cz.sort(key=lambda i: -crit[i])
        used: set[int] = set()
        chosen: list[int] = []
        rest: list[int] = []
        for idx in ready_cz:
            u, v = items[idx].qubits
            if u in used or v in used:
                rest.append(idx)
                continue
            used.add(u)
            used.add(v)
            chosen.append(idx)
        if not chosen:
            raise RuntimeError("scheduler deadlock")
        ready_cz = rest
        phases.append(("cz", chosen))
        for idx in chosen:
            done += 1
            release(idx, ready_1q, ready_cz)
    return phases


# ---------------------------------------------------------------- placement


def initial_loading(num_qubits: int, storage, order=None) -> dict[int, Site]:
    """Lay `order` out in boustrophedon ("snake") order, filling the rows closest to the
    entangling zone first.

    Two properties matter:
      * a qubit early in `order` is close to the zone, so sorting `order` by descending two-qubit
        degree makes the atoms that travel most often travel least (the reference/hub qubit lands in
        the bottom row, 1.0 from the zone, instead of 3.0 on average);
      * consecutive entries of `order` are physically adjacent, including across row boundaries, so
        linear-chain circuits (HEA) pair up inside one storage row and their lifts merge.
    """
    (x0, x1), (y0, y1) = storage
    width = x1 - x0 + 1
    if order is None:
        order = list(range(num_qubits))
    loading = {}
    for i, q in enumerate(order):
        band, offset = divmod(i, width)
        row = y1 - band
        col = (x0 + offset) if band % 2 == 0 else (x1 - offset)
        if row < y0:
            raise ValueError("circuit does not fit in the storage zone")
        loading[q] = (col, row)
    return loading


def placement_order(deg: list[int], num_qubits: int) -> list[int]:
    """Busiest qubits first (ties in circuit order, which keeps chain locality for HEA)."""
    return sorted(range(num_qubits), key=lambda q: (-deg[q], q))


# ---------------------------------------------------------------- movement


class _MoveRecord:
    """A planned rectangle move: `qubits` sit on `rows` x `cols` and go to `new_rows` x `new_cols`."""

    __slots__ = ("qubits", "rows", "cols", "new_rows", "new_cols", "src", "dst")

    def __init__(self, qubits, rows, cols, new_rows, new_cols, src, dst):
        self.qubits = list(qubits)
        self.rows = tuple(rows)
        self.cols = tuple(cols)
        self.new_rows = tuple(new_rows)
        self.new_cols = tuple(new_cols)
        self.src = dict(src)  # qubit -> site before
        self.dst = dict(dst)  # qubit -> site after


def _single_moves(assignment, pos) -> list[_MoveRecord]:
    """One single-atom move per atom (never has ghosts)."""
    out = []
    for q, target in assignment:
        (x, y) = pos[q]
        (nx, ny) = target
        out.append(_MoveRecord([q], (y,), (x,), (ny,), (nx,), {q: (x, y)}, {q: (nx, ny)}))
    return out


def merge_move_records(records: list[_MoveRecord], occupancy: dict[Site, int],
                       landing: dict[Site, int] | None = None) -> list[_MoveRecord]:
    """Greedily merge single-row moves that share (source row -> target row) into one rectangle.

    Legal iff the picked-up column span contains no atom that is not part of the merged move and the
    target columns are strictly ascending in source-column order.  A merged move costs the *max*
    displacement of its atoms instead of the sum, which is where distance is saved.
    """
    groups: dict[tuple[int, int], list[_MoveRecord]] = {}
    order: list[tuple[int, int]] = []
    for rec in records:
        if len(rec.rows) != 1 or len(rec.new_rows) != 1:
            key = ("raw", id(rec))
            groups[key] = [rec]
            order.append(key)
            continue
        key = (rec.rows[0], rec.new_rows[0])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(rec)

    merged: list[_MoveRecord] = []
    for key in order:
        bucket = groups[key]
        if len(bucket) == 1 or not isinstance(key[0], int):
            merged.extend(bucket)
            continue
        row, new_row = key
        bucket.sort(key=lambda r: r.cols[0])
        i = 0
        while i < len(bucket):
            run = [bucket[i]]
            j = i + 1
            while j < len(bucket):
                trial = run + [bucket[j]]
                if _mergeable(trial, row, occupancy, landing):
                    run = trial
                    j += 1
                else:
                    break
            if len(run) == 1:
                merged.append(run[0])
            else:
                cols = tuple(range(run[0].cols[0], run[-1].cols[-1] + 1))
                new_cols = tuple(r.new_cols[0] for r in run)
                qubits = [q for r in run for q in r.qubits]
                src = {}
                dst = {}
                for r in run:
                    src.update(r.src)
                    dst.update(r.dst)
                merged.append(_MoveRecord(qubits, (row,), cols, (new_row,), new_cols, src, dst))
            i = j
    return merged


def _mergeable(run: list[_MoveRecord], row: int, occupancy: dict[Site, int],
               landing: dict[Site, int] | None = None) -> bool:
    if any(len(r.qubits) != 1 for r in run):
        return False
    cols = [r.cols[0] for r in run]
    new_cols = [r.new_cols[0] for r in run]
    if any(b <= a for a, b in itertools.pairwise(cols)):
        return False
    if any(b <= a for a, b in itertools.pairwise(new_cols)):
        return False
    moving = {r.qubits[0] for r in run}
    for c in range(cols[0], cols[-1] + 1):
        q = occupancy.get((c, row))
        if q is not None and q not in moving:
            return False
        if landing is not None:
            q = landing.get((c, row))
            if q is not None and q not in moving:
                return False
    return True


def _apply(circuit: QuantumCircuit, records, pos, occupancy, ent) -> float:
    """Emit the moves, update positions/occupancy, return the summed distance (for diagnostics)."""
    total = 0.0
    for rec in records:
        for q in rec.qubits:
            del occupancy[rec.src[q]]
        for q in rec.qubits:
            occupancy[rec.dst[q]] = q
            pos[q] = rec.dst[q]
        total += max(_dist(rec.src[q], rec.dst[q], ent) for q in rec.qubits)
        _move(circuit, rec.qubits, rec.rows, rec.cols, rec.new_rows, rec.new_cols)
    return total


def _in_zone(site: Site, ent) -> bool:
    (ex0, ex1), (ey0, ey1) = ent
    x, y = site
    return ex0 <= x <= ex1 and ey0 <= y <= ey1


def _interaction_sites(ent):
    (ex0, ex1), (ey0, ey1) = ent
    return [(row, c) for row in range(ey0, ey1 + 1) for c in range(ex0, ex1, 2)]


def _slots(site, ent):
    """The two atom sites of the interaction pair whose left slot is at column `site[1]`."""
    row, c = site
    return (c, row), (c + 1, row)


def _pair_of_site(site, ent):
    """The interaction pair (row, left column) that the atom site `site` belongs to."""
    (ex0, _ex1), (_ey0, _ey1) = ent
    x, y = site
    return (y, ex0 + 2 * ((x - ex0) // 2))


def plan_round_sites(pairs, pos, occupancy, ent, future=None, home=None):
    """Assign this round's pairs to interaction sites.

    Returns `(targets, ordered)` with `targets[q]` the atom site qubit q must occupy and `ordered` a
    list of `(a, b, site)` in emission order.  Each pair takes the free site of lowest cost, trying
    both slot orientations; an atom that stayed in the entangling zone since the previous round
    usually keeps the site it already sits on, because staying there costs it nothing.

    The cost of a site is `max(storage lifts) + sum(moves inside the zone) + eps*(total)`: lifts out
    of storage can be merged into a single rectangle move (whose distance is the max over its
    atoms), while an atom that is already in the zone always needs a move of its own.  With
    `future`/`home` given, the mean distance from the homes of an atom's upcoming partners to the
    opposite slot is added, which parks long-lived atoms (the hub) in a central column.
    """
    sites = _interaction_sites(ent)
    free = dict.fromkeys(sites, True)
    targets: dict[int, Site] = {}
    ordered = []
    # Pairs holding an atom that stayed in the entangling zone go first, so that they can claim the
    # site it already sits on, which costs that atom nothing.
    def parked_here(p):
        return any(_in_zone(pos[q], ent) for q in p)

    queue = sorted(pairs, key=lambda p: (not parked_here(p),
                                         (pos[p[0]][0] + pos[p[1]][0]) / 2))
    for u, v in queue:
        best = None
        for site in sites:
            if not free[site]:
                continue
            left, right = _slots(site, ent)
            for a, b in ((u, v), (v, u)):
                # Every target must be empty or already hold that very atom, so the moves of a
                # round never collide and can be emitted in any order.
                if occupancy.get(left, a) is not a or occupancy.get(right, b) is not b:
                    continue
                da, db = _dist(pos[a], left, ent), _dist(pos[b], right, ent)
                # lifts out of storage can be merged into one rectangle (cost = max), while an atom
                # that is already in the zone needs a move of its own (cost adds up)
                lift = [d for q, d in ((a, da), (b, db)) if not _in_zone(pos[q], ent)]
                solo = [d for q, d in ((a, da), (b, db)) if _in_zone(pos[q], ent)]
                cost = (max(lift) if lift else 0.0) + sum(solo) + 0.001 * (da + db)
                if future is not None:
                    # An atom that stays parked here for several more rounds makes every one of its
                    # upcoming partners travel to the opposite slot, so steer its site towards the
                    # middle of where those partners live (the hub ends up in a central column).
                    for x, opposite in ((a, right), (b, left)):
                        later = future(x)
                        if later:
                            cost += sum(_dist(home[w], opposite, ent) for w in later) / len(later)
                if best is None or cost < best[0] - 1e-12:
                    best = (cost, site, a, b)
        if best is None:
            raise RuntimeError("no free interaction site")
        _cost, site, a, b = best
        left, right = _slots(site, ent)
        free[site] = False
        targets[a] = left
        targets[b] = right
        ordered.append((a, b, site))
    return targets, ordered


def choose_keeps(ordered, next_pairs, blocked, home, targets, ent):
    """Which atoms stay in the entangling zone until the next round (ZAC-style idle reuse).

    An atom may stay only if it takes part in the next round and has **no** 1q gate in between (1q
    gates need storage), which is exactly the information `schedule_rounds` produces.  At most one
    atom per interaction pair stays, so a staying atom always has its partner slot free, and at most
    one atom of each *next* pair stays, so the next round can pin that pair to its site without a
    conflict.  A stay is only taken when it is actually cheaper: staying saves the atom's two trips
    but forces its next partner to travel to this site instead of to the best meeting point.
    """
    if not next_pairs:
        return set()
    next_pair_of: dict[int, Any] = {}
    for p in next_pairs:
        for q in p:
            next_pair_of[q] = p
    sites = _interaction_sites(ent)

    def gain(q):
        """Distance saved by leaving q where it is, minus what its next partner pays extra."""
        pair = next_pair_of[q]
        w = pair[0] if pair[1] == q else pair[1]
        site = _pair_of_site(targets[q], ent)
        left, right = _slots(site, ent)
        partner_slot = right if targets[q] == left else left
        stay = 2 * _dist(home[w], partner_slot, ent)
        apart = None
        for s in sites:
            l2, r2 = _slots(s, ent)
            for a, b in ((q, w), (w, q)):
                hb = home[b]
                cost = 2 * (_dist(home[a], l2, ent) + _dist(hb, r2, ent))
                if apart is None or cost < apart:
                    apart = cost
        return (apart or 0.0) - stay

    candidates = []
    for a, b, _site in ordered:
        options = [q for q in (a, b) if q in next_pair_of and q not in blocked]
        options = [(gain(q), q) for q in options]
        options = [o for o in options if o[0] > 1e-9]
        if not options:
            continue
        candidates.append(max(options))
    keeps: set[int] = set()
    taken: set[Any] = set()
    for _g, q in sorted(candidates, reverse=True):
        pair = frozenset(next_pair_of[q])
        if pair in taken:
            continue
        taken.add(pair)
        keeps.add(q)
    return keeps


# ---------------------------------------------------------------- top level


def compile_circuit(circuit: QuantumCircuit, architecture_spec: Any):
    storage, ent = _geometry(architecture_spec)
    n = circuit.num_qubits

    items, phase = lower_to_items(circuit)
    deg = two_qubit_degree(items, n)
    items = expand_rot2(items, deg)
    items, extra = merge_1q_runs(items, n)
    phase += extra
    phases = schedule_rounds(items, n)

    cz_deg = two_qubit_degree(items, n)
    loading = initial_loading(n, storage, placement_order(cz_deg, n))
    pos = dict(loading)
    occupancy = {site: q for q, site in loading.items()}

    out = circuit.copy_empty_like()
    out.global_phase = 0.0
    qubits = list(out.qubits)
    clbits = list(out.clbits)

    # for every round: the pairs of the next round and which qubits get a 1q gate in between
    round_at = [i for i, (kind, _) in enumerate(phases) if kind == "cz"]
    next_pairs: list[list] = []
    blocked_between: list[set[int]] = []
    for k, i in enumerate(round_at):
        if k + 1 < len(round_at):
            j = round_at[k + 1]
            next_pairs.append([items[x].qubits for x in phases[j][1]])
            touched: set[int] = set()
            for mid in range(i + 1, j):
                if phases[mid][0] == "1q":
                    for x in phases[mid][1]:
                        touched.update(items[x].qubits)
            blocked_between.append(touched)
        else:
            next_pairs.append([])
            blocked_between.append(set())
    partner_in = []
    for i in round_at:
        table = {}
        for x in phases[i][1]:
            u, v = items[x].qubits
            table[u] = v
            table[v] = u
        partner_in.append(table)

    def future_partners(round_index, qubit, cap=16):
        """Partners of `qubit` in the rounds it can stay parked for, from `round_index` on."""
        out = []
        j = round_index
        while len(out) < cap and j + 1 < len(round_at):
            if qubit in blocked_between[j] or qubit not in partner_in[j + 1]:
                break
            out.append(partner_in[j + 1][qubit])
            j += 1
        return out

    round_no = 0

    for kind, idxs in phases:
        if kind == "1q":
            # per qubit: merge the phase's gates into <= 3 rz/rx, split at measurements
            per_qubit: dict[int, list[Item]] = {}
            for idx in idxs:
                it = items[idx]
                per_qubit.setdefault(it.qubits[0], []).append(it)
            for q, seq in per_qubit.items():
                mat = _I2
                for it in seq:
                    if it.kind == "meas":
                        mat = _flush_1q(out, qubits, q, mat)
                        out.measure(qubits[q], clbits[it.clbit])
                    else:
                        mat = it.mat @ mat
                _flush_1q(out, qubits, q, mat)
            continue

        pairs = [items[idx].qubits for idx in idxs]
        targets, ordered = plan_round_sites(
            pairs, pos, occupancy, ent,
            future=lambda q, r=round_no: future_partners(r, q), home=loading)
        trips = [(q, t) for q, t in targets.items() if pos[q] != t]
        landing = {t: q for q, t in trips}
        up = merge_move_records(_single_moves(trips, pos), occupancy, landing)
        _apply(out, up, pos, occupancy, ent)
        for a, b, _site in ordered:
            out.cz(qubits[a], qubits[b])
        keeps = choose_keeps(ordered, next_pairs[round_no], blocked_between[round_no],
                             loading, targets, ent)
        leaving = [(q, loading[q]) for q in targets if q not in keeps]
        leaving.sort()
        down = merge_move_records(_single_moves(leaving, pos), occupancy,
                                  {t: q for q, t in leaving})
        _apply(out, down, pos, occupancy, ent)
        round_no += 1

    out.global_phase = float(out.global_phase) + float(phase)
    return out, Architecture(loading, spec=architecture_spec)


def _flush_1q(circuit: QuantumCircuit, qubits, q: int, mat: np.ndarray) -> np.ndarray:
    if _is_identity(mat):
        if abs(mat[0, 0]) > 1e-9:
            circuit.global_phase = float(circuit.global_phase) + float(np.angle(mat[0, 0]))
        return _I2
    gates, gphase = euler_zxz(mat)
    circuit.global_phase = float(circuit.global_phase) + gphase
    for name, angle in gates:
        if name == "rz":
            circuit.rz(angle, qubits[q])
        else:
            circuit.rx(angle, qubits[q])
    return _I2
