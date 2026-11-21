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
            scratch = QuantumCircuit(len(qs))
            scratch.append(op, range(len(qs)))
            native = transpile(scratch, basis_gates=["rx", "rz", "cz"], optimization_level=2)
            nidx = {b: k for k, b in enumerate(native.qubits)}
            plan = ([(ins2.operation.name,
                      tuple(nidx[b] for b in ins2.qubits),
                      tuple(float(p) for p in ins2.operation.params))
                     for ins2 in native.data], float(native.global_phase))
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


def _rec_cost(rec, ent) -> float:
    return max(_dist(rec.src[q], rec.dst[q], ent) for q in rec.qubits)


def _ent_shape(ent):
    """(first x, first y, last y, interaction pairs per row)."""
    (ex0, ex1), (ey0, ey1) = ent
    return ex0, ey0, ey1, (ex1 - ex0 + 1) // 2


def _in_zone(site, storage, ent) -> bool:
    (sx0, sx1), (sy0, sy1) = storage
    (ex0, ex1), (ey0, ey1) = ent
    x, y = site
    return (sx0 <= x <= sx1 and sy0 <= y <= sy1) or (ex0 <= x <= ex1 and ey0 <= y <= ey1)


def _has_ghost(row, lo, hi, occupancy, allowed) -> bool:
    """True if an atom that is not part of the move sits on `row` between columns `lo` and `hi`."""
    for c in range(lo, hi + 1):
        q = occupancy.get((c, row))
        if q is not None and q not in allowed:
            return True
    return False


def _partition_group(row, atoms, occupancy):
    """Split `[(qubit, src_col, dst_col)]` (sorted by src_col) into legal single-row runs.

    A run is one rectangle move: its picked columns span [first, last], no atom outside the run sits
    in that span (no ghost), and its target columns ascend with the source columns.
    """
    runs: list[list[tuple[int, int, int]]] = []
    cur: list[tuple[int, int, int]] = []
    for atom in atoms:
        if cur:
            _q, src, dst = atom
            _pq, psrc, pdst = cur[-1]
            if dst <= pdst or _has_ghost(row, psrc + 1, src - 1, occupancy, {a[0] for a in cur}):
                runs.append(cur)
                cur = []
        cur.append(atom)
    if cur:
        runs.append(cur)
    return runs


def _ent_shape_rows() -> int:
    return 5


def _group_cost(row, dst_row, atoms, occupancy, partner=None) -> float:
    """Cost estimate for one (source row, target row) group: max distance per move.

    With `partner`, the estimate assumes that a whole span can ride in one move (extra atoms come
    along on their own interaction pair), which is what keeps the atoms of one storage row together.
    """
    dy = abs(dst_row - row)
    if partner is None:
        return sum(
            math.hypot(max(abs(a[2] - a[1]) for a in run), dy)
            for run in _partition_group(row, atoms, occupancy)
        )
    total = 0.0
    i = 0
    while i < len(atoms):
        inside = {a[0] for a in atoms}
        j = i + 1
        best = None
        while j <= len(atoms):
            seg = atoms[i:j]
            lo, hi = seg[0][1], seg[-1][1]
            span = [c for c in range(lo, hi + 1) if (c, row) in occupancy]
            pairs = sum(1 for a in seg if partner.get(a[0]) in {b[0] for b in seg})
            if len(span) - pairs > _ent_shape_rows():
                break
            adjacent = True
            for k in range(1, len(seg)):
                if seg[k][2] <= seg[k - 1][2]:
                    adjacent = False
                    break
            if not adjacent:
                break
            best = (math.hypot(max(abs(a[2] - a[1]) for a in seg), dy), j)
            j += 1
        if best is None:
            best = (math.hypot(abs(atoms[i][2] - atoms[i][1]), dy), i + 1)
        total += best[0]
        i = best[1]
        del inside
    return total


def _pair_order(pairs, pos):
    """Order the round's pairs so that greedy slot assignment packs one storage row at a time."""
    def key(pair):
        u, v = pair
        su, sv = pos[u], pos[v]
        return (0 if su[1] == sv[1] else 1, -max(su[1], sv[1]), min(su[0], sv[0]), max(su[0], sv[0]))

    return sorted((tuple(pair) for pair in pairs), key=key)


def _assign_round_targets(pairs, pos, occupancy, ent, reserve_cross=False, partner=None):
    """Choose a target interaction site for every atom of the round.

    Each pair gets an exclusive slot `(row, 2k)`, and slots are filled nearest-to-storage first, so a
    pair costs one hop and the atoms of one storage row stay in one entangling row (one merged move).
    With `reserve_cross`, the pairs that span two storage rows stay out of the entangling rows that
    one storage row already fills, which keeps that row's lift mergeable into one rectangle.
    """
    ex0, ey0, ey1, per_row = _ent_shape(ent)
    if per_row < 1:
        raise RuntimeError("no interaction pairs in the entangling zone")
    need = max(1, -(-len(pairs) // per_row))
    span = [ey0 + i for i in range(min(need + 1, ey1 - ey0 + 1))]
    groups: dict[tuple[int, int], list] = {}
    used: set = set()
    assign: dict[int, Site] = {}
    ordered = _pair_order(pairs, pos)
    if reserve_cross:
        phases = [[p for p in ordered if pos[p[0]][1] == pos[p[1]][1]],
                  [p for p in ordered if pos[p[0]][1] != pos[p[1]][1]]]
    else:
        phases = [ordered]
    for phase in phases:
        if not phase:
            continue
        pack_rows = {t for (_r, t) in groups}
        for pos_in_phase, (u, v) in enumerate(phase):
            candidates = [(t, k) for t in span for k in range(per_row)]
            if reserve_cross and phase is phases[-1]:
                free = [c for c in candidates if c[0] not in pack_rows and c not in used]
                if len(free) >= len(phase) - pos_in_phase:
                    candidates = free
            best = None
            for t, k in candidates:
                if (t, k) in used:
                    continue
                c0 = ex0 + 2 * k
                for a, b in ((u, v), (v, u)):
                    trial = {a: (c0, t), b: (c0 + 1, t)}
                    delta = 0.0
                    touched: dict[tuple[int, int], dict] = {}
                    for q, target in trial.items():
                        touched.setdefault((pos[q][1], t), {})[q] = target
                    for (row, dst_row), added in touched.items():
                        old = groups.get((row, dst_row), [])
                        new = sorted(old + [(q, pos[q][0], added[q][0]) for q in added], key=lambda z: z[1])
                        delta += (_group_cost(row, dst_row, new, occupancy)
                                  - _group_cost(row, dst_row, old, occupancy))
                    if best is None or delta < best[0] - 1e-12:
                        best = (delta, t, k, dict(trial))
            if best is None:
                raise RuntimeError("no free interaction site")
            _delta, t, k, trial = best
            used.add((t, k))
            assign.update(trial)
            for q, target in trial.items():
                key = (pos[q][1], t)
                groups.setdefault(key, []).append((q, pos[q][0], target[0]))
                groups[key].sort(key=lambda z: z[1])
    return assign


def _gap_fill(srcs, lo, hi, free_slots, ex0):
    """Give each passenger in `srcs` a distinct free pair slot strictly between columns lo and hi.

    Returns (max |dx|, [(target column, slot)]) or None.  A passenger occupies a whole interaction
    pair on its own, so no unwanted CZ can hit it.
    """
    m = len(srcs)
    if m == 0:
        return 0.0, []
    cols = []
    for k in free_slots:
        c = ex0 + 2 * k
        if lo < c < hi:
            cols.append((c, k))
    cols.sort()
    if len(cols) < m:
        return None
    n = len(cols)
    inf = 99.0
    dp = [[inf] * n for _ in range(m)]
    back = [[-1] * n for _ in range(m)]
    for j in range(n):
        dp[0][j] = abs(cols[j][0] - srcs[0])
    for i in range(1, m):
        best, arg = inf, -1
        for j in range(n):
            if j > 0 and dp[i - 1][j - 1] < best:
                best, arg = dp[i - 1][j - 1], j - 1
            if arg >= 0:
                cost = max(best, abs(cols[j][0] - srcs[i]))
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    back[i][j] = arg
    j = min(range(n), key=lambda z: dp[m - 1][z])
    value = dp[m - 1][j]
    if value >= inf:
        return None
    out = []
    for i in range(m - 1, -1, -1):
        out.append(cols[j])
        j = back[i][j]
    out.reverse()
    return value, out


def _run_move(run, row, dst_row, occupancy, assignment, booked, ex0, ex1):
    """Build one rectangle move covering `run` (consecutive participants) and its passengers.

    Every atom between the first and last picked column has to come along; the ones that are not part
    of the round ride along on their own interaction pair, so the move costs its max distance only.
    Returns (record, used_slots) or None when the geometry forbids it.
    """
    lo, hi = run[0][1], run[-1][1]
    target_of = {q: dst for q, _src, dst in run}
    cols = [c for c in range(lo, hi + 1) if (c, row) in occupancy]
    if len(cols) < len(run):
        return None
    free = set(range(ex0 // 2, ex0 // 2 + (ex1 - ex0 + 1) // 2)) - booked
    used: set = set()
    dx = 0.0
    new_cols = []
    dst = {}
    prev_dst = ex0 - 1
    pending: list[int] = []
    for c in cols:
        q = occupancy[(c, row)]
        if q in target_of:
            tgt = target_of[q]
            if pending:
                filled = _gap_fill(pending, prev_dst, tgt, free - used, ex0)
                if filled is None:
                    return None
                add_dx, placed = filled
                dx = max(dx, add_dx)
                for src, (col, slot) in zip(pending, placed):
                    used.add(slot)
                    dst[occupancy[(src, row)]] = (col, dst_row)
                # keep the passenger targets in column order
                new_cols.extend(col for col, _slot in placed)
                pending = []
            if tgt <= prev_dst:
                return None
            new_cols.append(tgt)
            prev_dst = tgt
            slot = (tgt - ex0) // 2
            used.add(slot)
            dst[q] = (tgt, dst_row)
        else:
            if q in assignment:
                return None  # belongs to another target row of this round
            pending.append(c)
    if pending:
        filled = _gap_fill(pending, prev_dst, ex1 + 1, free - used, ex0)
        if filled is None:
            return None
        add_dx, placed = filled
        dx = max(dx, add_dx)
        for src, (col, slot) in zip(pending, placed):
            used.add(slot)
            if not new_cols or col > new_cols[-1]:
                new_cols.append(col)
            else:
                return None
            dst[occupancy[(src, row)]] = (col, dst_row)
    for c, col in zip(cols, new_cols):
        dx = max(dx, abs(col - c))
    if len(new_cols) != len(cols) or any(b <= a for a, b in itertools.pairwise(new_cols)):
        return None
    src = {occupancy[(c, row)]: (c, row) for c in cols}
    qubits = sorted(src)
    record = _MoveRecord(qubits, (row,), tuple(range(lo, hi + 1)), (dst_row,), tuple(new_cols), src, dst)
    return record, used


def _plan_group(row, dst_row, atoms, occupancy, assignment, booked, ex0, ex1, ent):
    """Partition one (source row, target row) group into moves, allowing passengers to fill the span."""
    m = len(atoms)
    inf = math.inf
    best = [inf] * (m + 1)
    best[0] = 0.0
    pick: list = [None] * (m + 1)
    for i in range(1, m + 1):
        for j in range(i):
            got = _run_move(atoms[j:i], row, dst_row, occupancy, assignment, booked, ex0, ex1)
            if got is None:
                continue
            rec, _used = got
            cost = best[j] + _rec_cost(rec, ent)
            if cost < best[i] - 1e-12:
                best[i] = cost
                pick[i] = j
    if pick[m] is None:
        return None
    out = []
    e = m
    while e > 0:
        j = pick[e]
        got = _run_move(atoms[j:e], row, dst_row, occupancy, assignment, booked, ex0, ex1)
        rec, used = got
        booked |= used
        out.append(rec)
        e = j
    out.reverse()
    return out


def _trip_records_group(row, dst_row, atoms, occupancy):
    """Ghost-safe fallback: split a group into runs that pick up only participating atoms."""
    out = []
    for run in _partition_group(row, sorted(atoms, key=lambda z: z[1]), occupancy):
        cols = tuple(range(run[0][1], run[-1][1] + 1))
        new_cols = tuple(a[2] for a in run)
        src = {q: (s, row) for q, s, _ in run}
        dst = {q: (d, dst_row) for q, _, d in run}
        out.append(_MoveRecord([a[0] for a in run], (row,), cols, (dst_row,), new_cols, src, dst))
    return out


def _assign_with_rows(pairs, pos, occupancy, ent, forced=None):
    """Slot assignment where `forced[i]` pins pair i to one entangling row (None = nearest free)."""
    ex0, ey0, ey1, per_row = _ent_shape(ent)
    need = max(1, -(-len(pairs) // per_row))
    span = [ey0 + i for i in range(min(need + 2, ey1 - ey0 + 1))]
    used: set = set()
    assign: dict[int, Site] = {}
    groups: dict[tuple[int, int], list] = {}
    partner = {}
    for u, v in pairs:
        partner[u] = v
        partner[v] = u
    for i, (u, v) in enumerate(_pair_order(pairs, pos)):
        rows = [forced[i]] if forced and forced[i] is not None else span
        best = None
        for t in rows:
            for k in range(per_row):
                if (t, k) in used:
                    continue
                c0 = ex0 + 2 * k
                for a, b in ((u, v), (v, u)):
                    trial = {a: (c0, t), b: (c0 + 1, t)}
                    delta = 0.0
                    touched: dict[tuple[int, int], dict] = {}
                    for q, target in trial.items():
                        touched.setdefault((pos[q][1], t), {})[q] = target
                    for (row, dst_row), added in touched.items():
                        old = groups.get((row, dst_row), [])
                        new = sorted(old + [(q, pos[q][0], added[q][0]) for q in added], key=lambda z: z[1])
                        delta += (_group_cost(row, dst_row, new, occupancy)
                                  - _group_cost(row, dst_row, old, occupancy))
                    if best is None or delta < best[0] - 1e-12:
                        best = (delta, t, k, dict(trial))
        if best is None:
            raise RuntimeError("no free interaction site")
        _delta, t, k, trial = best
        used.add((t, k))
        assign.update(trial)
        for q, target in trial.items():
            key = (pos[q][1], t)
            groups.setdefault(key, []).append((q, pos[q][0], target[0]))
            groups[key].sort(key=lambda z: z[1])
    return assign


def _trip_records(assignment, pos, occupancy):
    """Cover an assignment with single-row rectangle moves: one per (source row, target row) run."""
    groups: dict[tuple[int, int], list] = {}
    for q, target in assignment.items():
        groups.setdefault((pos[q][1], target[1]), []).append((q, pos[q][0], target[0]))
    records: list[_MoveRecord] = []
    for (row, dst_row), atoms in sorted(groups.items()):
        records.extend(_trip_records_group(row, dst_row, atoms, occupancy))
    return records


def _rec_sig(rec):
    return (tuple(sorted(rec.src[q][0] for q in rec.qubits)), rec.new_cols)


def _block_record(records, occupancy, storage, ent, upward, site_owner=None, partner=None):
    """Merge consecutive single-row records that share one column map into a rigid block move.

    All rows shift by the same amount, so every atom travels the same distance: a five-row lift costs
    one hop of five instead of hops of one, two, three, four and five.  `upward` trips may choose the
    target rows (nearest to storage first); the way home must land on the rows the atoms live on.
    """
    first = records[0]
    a, b = first.rows[0], records[-1].rows[0]
    if b - a + 1 != len(records) or any(rec.rows[0] != a + i for i, rec in enumerate(records)):
        return None
    col_map: dict[int, int] = {}
    row_of: dict[int, int] = {}
    moving: set = set()
    for rec in records:
        row = rec.rows[0]
        for q in rec.qubits:
            src = rec.src[q]
            row_of[q] = row
            moving.add(q)
            if col_map.setdefault(src[0], rec.dst[q][0]) != rec.dst[q][0]:
                return None
    src_cols = sorted(col_map)
    tgt_cols = [col_map[c] for c in src_cols]
    if any(y <= x for x, y in itertools.pairwise(tgt_cols)):
        return None
    if partner is not None:
        for q in moving:
            other = partner.get(q)
            if other is not None and (other not in moving or row_of[other] != row_of[q]):
                return None
    lo, hi = src_cols[0], src_cols[-1]
    for row in range(a, b + 1):
        if _has_ghost(row, lo, hi, occupancy, moving):
            return None
    if partner is not None:
        # a shared column map pairs the same source columns in every row: those must be round pairs
        for i, c in enumerate(src_cols[:-1]):
            nxt = src_cols[i + 1]
            if col_map[nxt] != col_map[c] + 1 or col_map[c] % 2:
                continue
            for row in range(a, b + 1):
                x, y = occupancy.get((c, row)), occupancy.get((nxt, row))
                if x is not None and y is not None and partner.get(x) != y:
                    return None
    dst_row_of = {rec.rows[0]: rec.new_rows[0] for rec in records}
    _ex0, ey0, ey1, _per_row = _ent_shape(ent)
    (sx0, sx1), (sy0, sy1) = storage
    if upward:
        candidates = [dy for dy in range(ey0 - a, ey1 - b + 1)]
        lo_row, hi_row = ey0, ey1
    else:
        dy = dst_row_of[a] - a
        if any(dst_row_of[row] != a + dy + (row - a) for row in range(a, b + 1)):
            return None
        candidates = [dy]
        lo_row, hi_row = sy0, sy1
    for dy in candidates:
        rows_used = tuple(range(a + dy, b + dy + 1))
        if min(rows_used) < lo_row or max(rows_used) > hi_row:
            continue
        src = {}
        dst = {}
        ok = True
        for rec in records:
            row = rec.rows[0]
            target_row = row + dy
            for q in rec.qubits:
                source = rec.src[q]
                target = (col_map[source[0]], target_row)
                occupant = occupancy.get(target)
                if occupant is not None and occupant not in moving:
                    ok = False
                    break
                if site_owner is not None and site_owner.get(target, q) not in moving:
                    ok = False
                    break
                if not _in_zone(target, storage, ent):
                    ok = False
                    break
                src[q] = source
                dst[q] = target
            if not ok:
                break
        if ok:
            return _MoveRecord(sorted(src), tuple(range(a, b + 1)), tuple(range(lo, hi + 1)),
                               rows_used, tuple(tgt_cols), src, dst)
    return None


def _dp_plan(rows, by_row, occupancy, storage, ent, upward, site_owner, partner):
    """DP over a cluster of consecutive rows: cheapest mix of per-row moves and rigid blocks."""
    m = len(rows)
    best = [math.inf] * (m + 1)
    best[0] = 0.0
    pick: list = [None] * (m + 1)
    for e in range(1, m + 1):
        recs = by_row[rows[e - 1]]
        cost = sum(_rec_cost(r, ent) for r in recs)
        if best[e - 1] + cost < best[e] - 1e-12:
            best[e] = best[e - 1] + cost
            pick[e] = (e - 1, [(r, [r]) for r in recs])
        for s in range(e - 1):
            block = [r for row in rows[s:e] for r in by_row[row]]
            rec = _block_record(block, occupancy, storage, ent, upward, site_owner, partner)
            if rec is None:
                continue
            cost = _rec_cost(rec, ent)
            if best[s] + cost < best[e] - 1e-12:
                best[e] = best[s] + cost
                pick[e] = (s, [(rec, block)])
    out: list = []
    e = m
    while e > 0:
        s, entries = pick[e]
        out = entries + out
        e = s
    return out


def _block_plan(assignment, pos, occupancy, storage, ent, upward, pairs=None):
    """Plan with rigid multi-row blocks (good when several rows pair up in the same columns)."""
    site_owner = {site: q for q, site in assignment.items()}
    partner = {}
    if upward and pairs is not None:
        for u, v in pairs:
            partner[u] = v
            partner[v] = u
    base = _trip_records(assignment, pos, occupancy)
    by_row: dict[int, list] = {}
    for rec in base:
        by_row.setdefault(rec.rows[0], []).append(rec)
    rows = sorted(by_row)
    clusters: list[list[int]] = []
    for r in rows:
        if clusters and r == clusters[-1][-1] + 1:
            clusters[-1].append(r)
        else:
            clusters.append([r])
    plan: list = []
    for cluster in clusters:
        if len(cluster) == 1:
            plan.extend((rec, [rec]) for rec in by_row[cluster[0]])
        else:
            plan.extend(_dp_plan(cluster, by_row, occupancy, storage, ent, upward, site_owner, partner))
    return plan


def _plan_cost(plan, ent) -> float:
    return sum(_rec_cost(rec, ent) for rec, _fallback in plan)


def _passenger_plan(assignment, pos, occupancy, storage, ent, pairs):
    """Plan with passengers: lift a whole row span in one move, parking the extra atoms on pairs."""
    ex0, _ey0, _ey1, per_row = _ent_shape(ent)
    booked_by_row: dict[int, set] = {}
    groups: dict[tuple[int, int], list] = {}
    for q, target in assignment.items():
        booked_by_row.setdefault(target[1], set()).add((target[0] - ex0) // 2)
        groups.setdefault((pos[q][1], target[1]), []).append((q, pos[q][0], target[0]))
    plan: list = []
    for (row, dst_row), atoms in sorted(groups.items()):
        atoms.sort(key=lambda z: z[1])
        booked = booked_by_row.setdefault(dst_row, set())
        move = _plan_group(row, dst_row, atoms, occupancy, assignment, booked, ex0,
                           ex0 + 2 * per_row - 1, ent)
        if move is None:
            for rec in _trip_records_group(row, dst_row, atoms, occupancy):
                plan.append((rec, [rec]))
                booked |= {(c - ex0) // 2 for c in rec.new_cols}
        else:
            for rec in move:
                plan.append((rec, [rec]))
            booked |= {(c - ex0) // 2 for rec in move for c in rec.new_cols}
    return plan


def _assign_components(pairs, pos, occupancy, ent):
    """Assign whole storage-row components to one entangling row each.

    Storage rows joined by a pair that spans them must share an entangling row (both atoms of a pair
    have to meet), so they form a component; every component takes its own entangling row, which
    keeps the atoms of one storage row together and lets a single rectangle carry them all.
    """
    ex0, ey0, ey1, per_row = _ent_shape(ent)
    rows = sorted({pos[q][1] for pair in pairs for q in pair})
    parent = {r: r for r in rows}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v in pairs:
        ru, rv = find(pos[u][1]), find(pos[v][1])
        if ru != rv:
            parent[rv] = ru
    comps: dict[int, list] = {}
    for pair in _pair_order(pairs, pos):
        comps.setdefault(find(pos[pair[0]][1]), []).append(pair)
    free = {(t, k) for t in range(ey0, ey1 + 1) for k in range(per_row)}
    assign: dict[int, Site] = {}
    for comp_pairs in comps.values():
        need = len(comp_pairs)
        choice = None
        for t in range(ey0, ey1 + 1):
            slots = [k for k in range(per_row) if (t, k) in free]
            if len(slots) >= need:
                choice = (t, slots[:need])
                break
        if choice is None:
            return None
        t, slots = choice
        for k, (u, v) in zip(slots, comp_pairs):
            free.discard((t, k))
            c0 = ex0 + 2 * k
            if (pos[u][0], pos[v][0]) <= (pos[v][0], pos[u][0]):
                assign[u] = (c0, t)
                assign[v] = (c0 + 1, t)
            else:
                assign[v] = (c0, t)
                assign[u] = (c0 + 1, t)
    return assign


def _evaluate(assign, pos, occupancy, storage, ent, pairs):
    plan = _passenger_plan(assign, pos, occupancy, storage, ent, pairs)
    return _plan_cost(plan, ent), plan


def _choose_up_plan(pairs, pos, occupancy, storage, ent):
    """Site assignment and move plan are chosen jointly: build candidate plans, keep the cheapest.

    Candidates are the greedy assignment (two flavours) and, when a storage row's atoms are spread
    over several entangling rows, a coordinate-descent pass that pulls each storage row onto one row
    (a single rectangle then carries the whole row, costing its max distance instead of the sum).
    """
    partner = {}
    for u, v in pairs:
        partner[u] = v
        partner[v] = u
    best = None
    seen = set()
    starts = []
    for reserve_cross in (False, True):
        starts.append(_assign_round_targets(pairs, pos, occupancy, ent, reserve_cross, partner))
    component = _assign_components(pairs, pos, occupancy, ent)
    if component is not None:
        starts.append(component)
    for assign in starts:
        key = tuple(sorted(assign.items()))
        if key in seen:
            continue
        seen.add(key)
        cost, plan = _evaluate(assign, pos, occupancy, storage, ent, pairs)
        if best is None or cost < best[0] - 1e-9:
            best = (cost, dict(assign), plan)
    assign = best[1]
    order = _pair_order(pairs, pos)
    if len(order) > 1:
        rows_of: dict[int, set] = {}
        for q, site in assign.items():
            rows_of.setdefault(pos[q][1], set()).add(site[1])
        spread = [r for r, ts in rows_of.items() if len(ts) > 1]
        if spread:
            forced = [assign[q][1] for pair in order for q in pair[:1]]
            for _round in range(2):
                improved = False
                for row in spread:
                    idxs = [i for i, pair in enumerate(order) if pos[pair[0]][1] == row or pos[pair[1]][1] == row]
                    if not idxs:
                        continue
                    current = rows_of.get(row, set())
                    options = sorted(current) + [t for t in range((_ent_shape(ent))[1], (_ent_shape(ent))[1] + 5)
                                                 if t not in current]
                    for t in options:
                        trial = list(forced)
                        for i in idxs:
                            trial[i] = t
                        try:
                            cand = _assign_with_rows(pairs, pos, occupancy, ent, trial)
                        except RuntimeError:
                            continue
                        cost, plan = _evaluate(cand, pos, occupancy, storage, ent, pairs)
                        if cost < best[0] - 1e-9:
                            best = (cost, dict(cand), plan)
                            forced = trial
                            assign = cand
                            rows_of = {}
                            for q, site in assign.items():
                                rows_of.setdefault(pos[q][1], set()).add(site[1])
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
    block = _block_plan(best[1], pos, occupancy, storage, ent, True, pairs)
    block_cost = _plan_cost(block, ent)
    if block_cost < best[0] - 1e-9:
        return best[1], block
    return best[1], best[2]


def _emit_step(circuit, rec, step, pos, occupancy, ent):
    """Commit a move that the replay accepted, using the replay's own site mapping."""
    new_pos, new_occ = step
    total = 0.0
    for q in rec.qubits:
        total = max(total, _dist(pos[q], new_pos[q], ent))
        pos[q] = new_pos[q]
    occupancy.clear()
    occupancy.update(new_occ)
    _move(circuit, rec.qubits, rec.rows, rec.cols, rec.new_rows, rec.new_cols)
    return total


def _emit_trip(circuit, plan, pos, occupancy, storage, ent):
    """Emit a planned trip; any move that the replay rejects is replaced by single-atom moves."""
    total = 0.0
    for rec, fallback in plan:
        step = _sim_move(rec, pos, occupancy, storage, ent)
        if step is not None:
            total += _emit_step(circuit, rec, step, pos, occupancy, ent)
            continue
        for sub in fallback:
            step = _sim_move(sub, pos, occupancy, storage, ent)
            if step is not None:
                total += _emit_step(circuit, sub, step, pos, occupancy, ent)
                continue
            for single in _single_moves([(q, sub.dst[q]) for q in sub.qubits], pos):
                step = _sim_move(single, pos, occupancy, storage, ent)
                if step is None:
                    raise RuntimeError("compiler produced an illegal single-atom move")
                total += _emit_step(circuit, single, step, pos, occupancy, ent)
    return total


def _move_sites(rows, cols, new_rows, new_cols, occupied):
    """Mirror the validator's site mapping, so a candidate move can be checked before it is emitted."""
    if len(new_rows) < len(rows) or len(new_cols) < len(cols):
        selected = [(x, y) for (x, y) in occupied if x in cols and y in rows]
        used_rows = {y for _, y in selected}
        used_cols = {x for x, _ in selected}
        rows = tuple(y for y in rows if y in used_rows)
        cols = tuple(x for x in cols if x in used_cols)
        if len(rows) != len(new_rows) or len(cols) != len(new_cols):
            return None
    row_map = dict(zip(rows, new_rows, strict=True))
    col_map = dict(zip(cols, new_cols, strict=True))
    return {(x, y): (col_map[x], row_map[y]) for y in rows for x in cols}


def _sim_move(rec, pos, occupancy, storage, ent):
    """Return the (positions, occupancy) after `rec`, or None when the move would be illegal."""
    if len(rec.qubits) != len(set(rec.qubits)):
        return None
    mapping = _move_sites(rec.rows, rec.cols, rec.new_rows, rec.new_cols, occupancy)
    if mapping is None:
        return None
    picked = {occupancy[s]: s for s in mapping if s in occupancy}
    listed = set(rec.qubits)
    if picked.keys() - listed or listed - picked.keys():
        return None
    if any(pos[q] != picked[q] for q in listed):
        return None
    new_pos = dict(pos)
    new_occ = dict(occupancy)
    for q, source in picked.items():
        target = mapping[source]
        if not _in_zone(target, storage, ent):
            return None
        occupant = occupancy.get(target)
        if occupant is not None and occupant not in picked:
            return None
        del new_occ[source]
    for q, source in picked.items():
        new_occ[mapping[source]] = q
        new_pos[q] = mapping[source]
    return new_pos, new_occ


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
        start = dict(pos)
        assignment, up = _choose_up_plan(pairs, pos, occupancy, storage, ent)
        _emit_trip(out, up, pos, occupancy, storage, ent)
        for a, b in pairs:
            out.cz(qubits[a], qubits[b])
        (ex0, ex1), (ey0, ey1) = ent
        home = {q: start[q] for q, (x, y) in pos.items() if ex0 <= x <= ex1 and ey0 <= y <= ey1}
        down = _passenger_plan(home, pos, occupancy, storage, ent, pairs)
        if _plan_cost(down, ent) > _plan_cost(_block_plan(home, pos, occupancy, storage, ent, False), ent):
            down = _block_plan(home, pos, occupancy, storage, ent, False)
        _emit_trip(out, down, pos, occupancy, storage, ent)

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
