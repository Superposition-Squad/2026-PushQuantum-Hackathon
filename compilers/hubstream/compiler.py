"""hubstream: round-based zoned compilation that parks the hub qubit in the entangling zone.

Design (see `docs/ideas/hubstream.md`):

* The circuit is parsed into a list of *items*: single-qubit unitaries, ZZ rotations
  ``exp(-i theta/2 Z_u Z_v)`` (from ``cx-rz-cx`` triples and ``rzz``), plain ``cz`` and measurements.
* Every ZZ rotation is synthesised as ``H_s . CZ(u,v) . RX(theta)_s . CZ(u,v) . H_s`` with the
  rotation side ``s`` chosen to be the non-hub endpoint, so the hub's wire carries only CZs and the
  hub can stay parked in the entangling zone across a whole layer.
* Per-qubit *wires* of ``("g", matrix)`` / ``("cz", task)`` / ``("meas", clbit)`` entries are
  scheduled into rounds (ASAP list scheduling on the wire order).  Per-qubit order is never
  changed and every rewrite is exact, so the output is truly equivalent at every size.
* One round = lower (zone atoms that are not parked) / single-qubit gates (merged per wire) / lift /
  one CZ pulse / park.  A round holds at most one pulse, so the transversal-CZ rule (ADR 0007)
  holds by construction: a parked atom sits alone on its interaction pair.

The file is self-contained: stdlib, numpy and qiskit only.
"""

from __future__ import annotations

import bisect
import itertools
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from numbers import Integral
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction
from qiskit.circuit.library import CZGate, Measure, RXGate, RZGate
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


# ---------------------------------------------------------------- geometry (hard-coded)

STORAGE_X = (0, 9)
STORAGE_Y = (0, 9)
ENT_X = (0, 9)
ENT_Y = (10, 19)
PAIR_DISTANCE = 0.3
MAX_PAIRS = 4
"""At most this many interaction pairs per round (fits in one entangling row beside the hub)."""
PARK_WINDOW = 1
"""A non-hub atom stays in the zone only if its next task is at most this many rounds away."""

DAG_MAX_PAIRS = (2, 3, 4)
DAG_MAX_PAIRS_LARGE = (2,)
LARGE_CIRCUIT = 40
"""Round caps tried for the commutation-DAG front end; big circuits try fewer (compile time)."""
HUB_END_PAIR = True
HUB_END_MIN_QUBITS = 20
"""The end pair only pays off above this size (12q/20q measured worse with it)."""
"""Park the hub on the rightmost pair of entangling row 10 (hub in column 9, partner in column 8)."""
HUB_PAIR_LEFT = (8, 10)
SQRT1_2 = 1.0 / math.sqrt(2.0)
H_GATE = np.array([[SQRT1_2, SQRT1_2], [SQRT1_2, -SQRT1_2]], dtype=complex)
_STORAGE = [(x, y) for y in range(STORAGE_Y[0], STORAGE_Y[1] + 1) for x in range(STORAGE_X[0], STORAGE_X[1] + 1)]
_PAIRS = [
    ((2 * k, row), (2 * k + 1, row), row)
    for row in range(ENT_Y[0], ENT_Y[1] + 1)
    for k in range((ENT_X[1] - ENT_X[0] + 1) // 2)
]


def _physical(site: Site) -> tuple[float, float]:
    """Physical position of a grid site (storage spacing 1, entangling columns paired 0.3 apart)."""
    x, y = site
    if y >= ENT_Y[0]:
        return (2.0 * (x // 2) + PAIR_DISTANCE * (x % 2), float(y))
    return (float(x), float(y))


_SITES = list(_STORAGE) + [slot for pair in _PAIRS for slot in pair[:2]]
_PHYS = {site: _physical(site) for site in _SITES}
"""Physical position of every site of the machine."""


def _dist(a: Site, b: Site) -> float:
    (ax, ay), (bx, by) = _PHYS[a], _PHYS[b]
    return math.hypot(ax - bx, ay - by)


def _storage_by_closeness(site: Site) -> list[Site]:
    """Every storage site, nearest to `site` first (cached)."""
    cached = _CLOSENESS.get(site)
    if cached is None:
        px, py = _PHYS[site]
        cached = sorted(_STORAGE, key=lambda s: ((_PHYS[s][0] - px) ** 2 + (_PHYS[s][1] - py) ** 2, s[1], s[0]))
        _CLOSENESS[site] = cached
    return cached


_CLOSENESS: dict[Site, list[Site]] = {}


def _in_ent(site: Site) -> bool:
    return ENT_X[0] <= site[0] <= ENT_X[1] and ENT_Y[0] <= site[1] <= ENT_Y[1]


def _paired(a: Site, b: Site) -> bool:
    return (
        a[1] == b[1]
        and abs(a[0] - b[0]) == 1
        and min(a[0], b[0]) % 2 == 0
        and _in_ent(a)
        and _in_ent(b)
    )


# ---------------------------------------------------------------- single-qubit helpers


def _is_identity(u, tol: float = 1e-9) -> bool:
    """True if `u` is a scalar multiple of the identity (i.e. no gate is needed)."""
    if u is None:
        return True
    return bool(abs(u[0, 1]) < tol and abs(u[1, 0]) < tol and abs(u[0, 0] - u[1, 1]) < tol)


def _wrap(angle: float) -> float:
    """Bring an angle into (-pi, pi]; an exact 0 modulo 2*pi becomes 0.0 (the gate is skipped)."""
    if abs(angle) < 1e-12:
        return 0.0
    remainder = math.remainder(angle, 2.0 * math.pi)
    return 0.0 if abs(remainder) < 1e-12 else remainder


def _zxz(u):
    """Angles ``(a, b, c)`` with ``u = e^{i phase} rz(a) rx(b) rz(c)``."""
    det = u[0, 0] * u[1, 1] - u[0, 1] * u[1, 0]
    v = u / np.sqrt(det)  # det 1; the branch of the square root only adds a global phase
    alpha, beta = complex(v[0, 0]), complex(v[0, 1])
    arg_alpha = math.atan2(alpha.imag, alpha.real)
    if abs(beta) < 1e-12:
        return _wrap(-2.0 * arg_alpha), 0.0, 0.0
    arg_beta = math.atan2(beta.imag, beta.real)
    b = 2.0 * math.atan2(abs(beta), abs(alpha))
    if abs(alpha) < 1e-12:  # b = pi; (a + c) is free, pick a = 0
        return 0.0, _wrap(b), _wrap(math.pi + 2.0 * arg_beta)
    total = -2.0 * arg_alpha  # a + c
    spread = math.pi + 2.0 * arg_beta  # c - a
    return _wrap((total - spread) / 2.0), _wrap(b), _wrap((total + spread) / 2.0)


def _rz_matrix(angle: float):
    return np.array([[math.cos(angle / 2) - 1j * math.sin(angle / 2), 0.0],
                     [0.0, math.cos(angle / 2) + 1j * math.sin(angle / 2)]], dtype=complex)


def _emit_1q_defer(out: QuantumCircuit, q: int, u) -> float:
    """Emit a run except its trailing `rz`, which is returned so it can be merged into the next run.

    A trailing `rz` commutes with the `cz` that comes next on the same wire, so folding it into the
    following run is exact and saves one gate per run.
    """
    a, b, c = _zxz(np.asarray(u, dtype=complex))
    qubit = out.qubits[q]
    if c:
        out.append(RZGate(c), [qubit])
    if b:
        out.append(RXGate(b), [qubit])
    return a


# ---------------------------------------------------------------- move replay self-check


def _sim_move(occ: dict[Site, int], qubits, rows, cols, new_rows, new_cols) -> dict[Site, Site]:
    """Re-derive a move's site mapping exactly like the validator and assert it is legal."""
    rows, cols = tuple(rows), tuple(cols)
    new_rows, new_cols = tuple(new_rows), tuple(new_cols)
    if len(new_rows) < len(rows) or len(new_cols) < len(cols):
        selected = [(x, y) for (x, y) in occ if x in cols and y in rows]
        rows = tuple(r for r in rows if r in {y for _, y in selected})
        cols = tuple(c for c in cols if c in {x for x, _ in selected})
        if len(rows) != len(new_rows) or len(cols) != len(new_cols):
            raise RuntimeError(f"hubstream: squeezed move does not match its targets ({rows}, {cols})")
    row_map = dict(zip(rows, new_rows, strict=True))
    col_map = dict(zip(cols, new_cols, strict=True))
    mapping = {(x, y): (col_map[x], row_map[y]) for y in rows for x in cols}
    picked = {occ[site]: site for site in mapping if site in occ}
    if set(picked) != set(qubits):
        raise RuntimeError(f"hubstream: move qubits {sorted(qubits)} != picked up {sorted(picked)}")
    for qubit, source in picked.items():
        target = mapping[source]
        if target in occ and occ[target] not in picked:
            raise RuntimeError(f"hubstream: deposit of qubit {qubit} on occupied site {target}")
    return mapping


# ---------------------------------------------------------------- 1. parse


_ALLOWED = frozenset({"barrier", "measure", "cx", "cz", "rzz"})


def _needs_transpile(circuit: QuantumCircuit) -> bool:
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name in _ALLOWED or operation.num_qubits == 1:
            continue
        return True
    return False


def parse_items(circuit: QuantumCircuit) -> list:
    """Flatten the circuit into exact items (see the module docstring)."""
    if _needs_transpile(circuit):
        # optimisation_level MUST stay <= 1: from level 2 qiskit's ElidePermutations pass turns
        # `swap`s into a `final_layout` permutation, and reading `.data` afterwards would silently
        # drop them (wrong circuit). Level 1 keeps every swap visible as `rx/rz/cz` gates.
        circuit = transpile(circuit, basis_gates=["rx", "rz", "cz"], optimization_level=1)
    index = {bit: i for i, bit in enumerate(circuit.qubits)}
    cindex = {bit: i for i, bit in enumerate(circuit.clbits)}
    data = list(circuit.data)
    items: list = []
    i = 0
    total = len(data)
    while i < total:
        instruction = data[i]
        operation = instruction.operation
        name = operation.name
        qubits = [index[q] for q in instruction.qubits]
        if name == "barrier":
            i += 1
            continue
        if name == "measure":
            items.append(("meas", qubits[0], cindex[instruction.clbits[0]]))
            i += 1
            continue
        if name == "rzz":
            items.append(("zz", qubits[0], qubits[1], float(operation.params[0])))
            i += 1
            continue
        if name == "cz":
            items.append(("cz", qubits[0], qubits[1]))
            i += 1
            continue
        if name == "cx":
            if i + 2 < total:
                first, middle, last = data[i], data[i + 1], data[i + 2]
                if (
                    first.operation.name == "cx"
                    and last.operation.name == "cx"
                    and middle.operation.name == "rz"
                    and [index[q] for q in first.qubits] == qubits
                    and [index[q] for q in last.qubits] == qubits
                    and [index[q] for q in middle.qubits] == [qubits[1]]
                ):
                    items.append(("zz", qubits[0], qubits[1], float(middle.operation.params[0])))
                    i += 3
                    continue
            items.append(("u1", qubits[1], H_GATE))
            items.append(("cz", qubits[0], qubits[1]))
            items.append(("u1", qubits[1], H_GATE))
            i += 1
            continue
        if operation.num_qubits == 1:
            items.append(("u1", qubits[0], np.asarray(operation.to_matrix(), dtype=complex)))
            i += 1
            continue
        raise ValueError(f"hubstream: unsupported instruction {name!r} on {len(qubits)} qubits")
    return items


# ---------------------------------------------------------------- 2. wires and tasks


def choose_hub(items, n: int):
    """The highest-degree qubit of the interaction graph, if it really is a hub; else None."""
    if n < 6:
        return None
    neighbours: dict[int, set[int]] = defaultdict(set)
    for item in items:
        if item[0] == "zz" and abs(math.remainder(item[3], 2.0 * math.pi)) < 1e-12:
            continue
        if item[0] in ("zz", "cz"):
            neighbours[item[1]].add(item[2])
            neighbours[item[2]].add(item[1])
    if not neighbours:
        return None
    hub = max(neighbours, key=lambda q: (len(neighbours[q]), -q))
    if len(neighbours[hub]) >= 0.4 * (n - 1):
        return hub
    return None


def build_wires(items, n: int, hub):
    """Per-qubit wires of entries, plus the flat task list (task id -> (u, v))."""
    wires: list[list] = [[] for _ in range(n)]
    tasks: list[tuple[int, int]] = []
    for item in items:
        kind = item[0]
        if kind == "u1":
            wires[item[1]].append(("g", item[2]))
        elif kind == "meas":
            wires[item[1]].append(("meas", item[2]))
        elif kind == "cz":
            task = len(tasks)
            tasks.append((item[1], item[2]))
            wires[item[1]].append(("cz", task))
            wires[item[2]].append(("cz", task))
        elif kind == "zz":
            _, u, v, theta = item
            angle = _wrap(theta)
            if not angle:
                continue  # exp(0) = identity: nothing to synthesise
            if hub is not None and u == hub:
                side, other = v, u
            elif hub is not None and v == hub:
                side, other = u, v
            else:
                side, other = v, u
            first, second = len(tasks), len(tasks) + 1
            tasks.append((u, v))
            tasks.append((u, v))
            wires[side].append(("g", H_GATE))
            wires[side].append(("cz", first))
            wires[side].append(("g", np.array(
                [
                    [math.cos(angle / 2.0), -1j * math.sin(angle / 2.0)],
                    [-1j * math.sin(angle / 2.0), math.cos(angle / 2.0)],
                ],
                dtype=complex,
            )))
            wires[side].append(("cz", second))
            wires[side].append(("g", H_GATE))
            wires[other].append(("cz", first))
            wires[other].append(("cz", second))
        else:  # pragma: no cover
            raise ValueError(f"hubstream: bad item {item!r}")
    return wires, tasks


# ---------------------------------------------------------------- 3. scheduler


def _next_cz_index(wire, start: int) -> int:
    i = start
    while i < len(wire) and wire[i][0] != "cz":
        i += 1
    return i


def schedule_rounds(wires, tasks, hub, max_pairs: int = MAX_PAIRS) -> list[list[int]]:
    """ASAP list scheduling: each round takes ready tasks with disjoint qubits, hub tasks first."""
    n = len(wires)
    at = [_next_cz_index(wire, 0) for wire in wires]
    done = [False] * len(tasks)
    remaining = len(tasks)
    rounds: list[list[int]] = []
    while remaining:
        pointers: dict[int, list[int]] = defaultdict(list)
        for q in range(n):
            i = at[q]
            if i < len(wires[q]):
                entry = wires[q][i]
                if entry[0] == "cz":
                    pointers[entry[1]].append(q)
        ready = [task for task, qs in pointers.items() if len(qs) == 2 and not done[task]]
        ready.sort(key=lambda task: (0 if (hub is not None and hub in tasks[task]) else 1, task))
        chosen: list[int] = []
        used: set[int] = set()
        for task in ready:
            u, v = tasks[task]
            if u in used or v in used:
                continue
            chosen.append(task)
            used.add(u)
            used.add(v)
            if len(chosen) >= max_pairs:
                break
        if not chosen:
            raise RuntimeError("hubstream: scheduler deadlock (no ready task)")
        for task in chosen:
            done[task] = True
            remaining -= 1
            for q in tasks[task]:
                at[q] = _next_cz_index(wires[q], at[q] + 1)
        rounds.append(sorted(chosen))
    return rounds


# ---------------------------------------------------------------- 4. transfer planner


def plan_transfer(moves: dict[int, tuple[Site, Site]], occ: dict[Site, int]) -> list[tuple]:
    """Group (source -> destination) relocations into legal, merged `Move` descriptors."""
    groups: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for qubit, (source, target) in moves.items():
        groups[(source[1], target[1])].append((source[0], target[0], qubit))
    runs: list[tuple] = []
    for (source_row, target_row), entries in groups.items():
        entries.sort()
        current: list[tuple[int, int, int]] = []
        for source_col, target_col, qubit in entries:
            if current and (
                target_col <= current[-1][1]
                or _ghost_between(occ, source_row, current[-1][0], source_col)
            ):
                runs.append(_run(source_row, target_row, current))
                current = []
            current.append((source_col, target_col, qubit))
        if current:
            runs.append(_run(source_row, target_row, current))
    runs.sort(key=lambda run: (run[1][0], run[4][0], run[2][0]))
    merged: list[tuple] = []
    for run in runs:
        if merged:
            first = merged[-1]
            if (
                first[2] == run[2]
                and first[4] == run[4]
                and run[1][0] == first[1][-1] + 1
                and run[3][0] == first[3][-1] + 1
            ):
                merged[-1] = (first[0] + run[0], first[1] + run[1], first[2], first[3] + run[3], first[4])
                continue
        merged.append(run)
    return merged


def _ghost_between(occ, row: int, col_a: int, col_b: int) -> bool:
    """True if any atom (moving or not) sits on the source row between the two columns."""
    for col in range(col_a + 1, col_b):
        if (col, row) in occ:
            return True
    return False


def _run(source_row: int, target_row: int, entries: list[tuple[int, int, int]]) -> tuple:
    return (
        [q for _, _, q in entries],
        (source_row,),
        tuple(range(entries[0][0], entries[-1][0] + 1)),
        (target_row,),
        tuple(target for _, target, _ in entries),
    )


# ---------------------------------------------------------------- 5. emit


def _nearest_free(free: set[Site], site: Site) -> Site:
    """The free storage site closest to `site` (ties: closer column, lower row)."""
    for candidate in _storage_by_closeness(site):
        if candidate in free:
            return candidate
    raise RuntimeError("hubstream: no free storage site left")


def _match_columns(source_cols: list[int], free_cols: list[int]) -> list[int]:
    """Pick `len(source_cols)` ascending columns out of `free_cols` minimising the total shift."""
    k, m = len(source_cols), len(free_cols)
    if m < k:
        raise RuntimeError("hubstream: not enough free columns")
    inf = float("inf")
    best = [[inf] * (m + 1) for _ in range(k + 1)]
    back = [[False] * (m + 1) for _ in range(k + 1)]
    for j in range(m + 1):
        best[0][j] = 0
    for i in range(1, k + 1):
        for j in range(1, m + 1):
            skip = best[i][j - 1]
            take = best[i - 1][j - 1] + abs(free_cols[j - 1] - source_cols[i - 1])
            if take <= skip:
                best[i][j] = take
                back[i][j] = True
            else:
                best[i][j] = skip
    picked: list[int] = []
    i, j = k, m
    while i > 0:
        if back[i][j]:
            picked.append(free_cols[j - 1])
            i -= 1
            j -= 1
        else:
            j -= 1
    picked.reverse()
    return picked


def _pair_options(task, pos, hub, left, right):
    """``(cost, slot for u, slot for v)`` options for putting `task` on the pair (`left`, `right`)."""
    u, v = task
    if hub is not None and (u == hub or v == hub):
        if u == hub:  # the hub always takes the right (odd) slot of its pair
            return [(max(_dist(pos[u], right), _dist(pos[v], left)), right, left)]
        return [(max(_dist(pos[u], left), _dist(pos[v], right)), left, right)]
    return [
        (max(_dist(pos[u], left), _dist(pos[v], right)), left, right),
        (max(_dist(pos[u], right), _dist(pos[v], left)), right, left),
    ]


def _best_pair(task: tuple[int, int], pos: dict[int, Site], reserved: set[Site], hub, hub_end: bool = False):
    """Cheapest free interaction pair for `task`: ``(cost, slot for u, slot for v)``.

    Pairs are taken from the lowest entangling row that still has a free pair: keeping a whole
    round in one row lets the lower and lift phases become one wide `move` each.
    """
    u, v = task
    best = None
    best_row = None
    hub_task = hub is not None and hub in task
    for left, right, row in _PAIRS:
        if left in reserved or right in reserved:
            continue
        if hub_end and hub is not None and not hub_task and left == HUB_PAIR_LEFT:
            continue  # the end pair of row 10 is reserved for the hub: it never splits a row-10 move
        if best_row is not None and row > best_row:
            continue
        for cost, slot_u, slot_v in _pair_options(task, pos, hub, left, right):
            preferred = 0 if (hub_end and hub_task and left == HUB_PAIR_LEFT) else 1
            key = (preferred, row, cost, left[0])
            if best is None or key < best[0]:
                best = (key, (cost, slot_u, slot_v))
                best_row = row
    if best is None:
        raise RuntimeError("hubstream: no free interaction pair left in this round")
    return best[1]


def _remaining_cz(wire, start: int) -> int:
    return sum(1 for entry in wire[start:] if entry[0] == "cz")


def emit(n: int, wires, tasks, rounds, hub, circuit: QuantumCircuit, loading: dict[int, Site]) -> QuantumCircuit:
    out = circuit.copy_empty_like()
    hub_end = HUB_END_PAIR and n > HUB_END_MIN_QUBITS
    pos: dict[int, Site] = dict(loading)
    occ: dict[Site, int] = {site: qubit for qubit, site in pos.items()}
    pointer = [0] * n
    parked: set[int] = set()
    deferred: dict[int, float] = {}

    def flush_deferred(q: int) -> None:
        angle = deferred.get(q, 0.0)
        if angle:
            out.append(RZGate(angle), [out.qubits[q]])
            deferred[q] = 0.0

    def flush(q: int) -> None:
        wire = wires[q]
        while pointer[q] < len(wire):
            entry = wire[pointer[q]]
            if entry[0] == "g":
                matrix = entry[1]
                pointer[q] += 1
                while pointer[q] < len(wire) and wire[pointer[q]][0] == "g":
                    matrix = wire[pointer[q]][1] @ matrix
                    pointer[q] += 1
                if deferred.get(q):
                    matrix = matrix @ _rz_matrix(deferred[q])
                    deferred[q] = 0.0
                deferred[q] = _emit_1q_defer(out, q, matrix)
            elif entry[0] == "meas":
                flush_deferred(q)
                out.append(Measure(), [out.qubits[q]], [out.clbits[entry[1]]])
                pointer[q] += 1
            else:
                return

    def do_moves(moves: dict[int, tuple[Site, Site]]) -> None:
        if not moves:
            return
        for qubits, rows, cols, new_rows, new_cols in plan_transfer(moves, occ):
            mapping = _sim_move(occ, qubits, rows, cols, new_rows, new_cols)
            sources = {occ[site]: site for site in mapping if site in occ}
            for qubit in qubits:
                del occ[sources[qubit]]
            for qubit in qubits:
                site = mapping[sources[qubit]]
                occ[site] = qubit
                pos[qubit] = site
            _move(out, qubits, rows, cols, new_rows, new_cols)

    def lower(atoms: list[int]) -> None:
        if not atoms:
            return
        free = {site for site in _STORAGE if site not in occ}
        by_row: dict[int, list[int]] = defaultdict(list)
        for qubit in atoms:
            by_row[pos[qubit][1]].append(qubit)
        moves: dict[int, tuple[Site, Site]] = {}
        # pack each source row into one target row: the lower phase then needs one move per row
        for source_row in sorted(by_row):
            qubits = sorted(by_row[source_row], key=lambda q: (pos[q][0]))
            need = len(qubits)
            preferred = Counter(_nearest_free(free, pos[q])[1] for q in qubits).most_common(1)[0][0]
            rows = sorted(
                (row for row in range(STORAGE_Y[0], STORAGE_Y[1] + 1)
                 if sum(1 for col in range(10) if (col, row) in free) >= need),
                key=lambda row: (row != preferred, abs(row - source_row), -row),
            )
            if rows:
                target_row = rows[0]
            else:  # scattered fallback: closest free site per atom, from the deepest first
                target_row = None
            if target_row is None:
                for qubit in reversed(qubits):
                    target = _nearest_free(free, pos[qubit])
                    free.discard(target)
                    moves[qubit] = (pos[qubit], target)
                continue
            columns = [col for col in range(10) if (col, target_row) in free]
            if len(columns) < need:
                for qubit in reversed(qubits):
                    target = _nearest_free(free, pos[qubit])
                    free.discard(target)
                    moves[qubit] = (pos[qubit], target)
                continue
            picked = _match_columns([pos[q][0] for q in qubits], columns)
            for qubit, col in zip(qubits, picked):
                free.discard((col, target_row))
                moves[qubit] = (pos[qubit], (col, target_row))
        do_moves(moves)

    def zone_atoms() -> list[int]:
        return [q for q in range(n) if _in_ent(pos[q])]

    def pending(q: int):
        """(pending single-qubit matrix, index of the next cz entry or None) from the pointer."""
        wire = wires[q]
        i = pointer[q]
        matrix = None
        while i < len(wire) and wire[i][0] == "g":
            matrix = wire[i][1] if matrix is None else wire[i][1] @ matrix
            i += 1
        has_cz = i < len(wire) and wire[i][0] == "cz"
        # a deferred `rz` commutes with every `cz` in between, so it may wait: it does not stop us
        # from parking the atom (it is emitted with the next run that needs storage).
        return matrix, (i if has_cz else None)

    round_of = {}
    for index, round_tasks in enumerate(rounds):
        for task in round_tasks:
            round_of[task] = index

    for round_index, round_tasks in enumerate(rounds):
        # --- 1. assign an interaction pair to every task of this round
        slot_of: dict[int, Site] = {}
        reserved = {pos[q] for q in parked}
        waiting: list[int] = []
        for task in round_tasks:
            u, v = tasks[task]
            if u in parked and v in parked:
                slot_of[u], slot_of[v] = pos[u], pos[v]
            elif u in parked or v in parked:
                keep, move = (u, v) if u in parked else (v, u)
                partner_slot = (pos[keep][0] - 1 if pos[keep][0] % 2 else pos[keep][0] + 1, pos[keep][1])
                slot_of[keep], slot_of[move] = pos[keep], partner_slot
                reserved.add(partner_slot)
            else:
                waiting.append(task)
        # hardest task first: the one whose cheapest pair is worst gets first pick
        for task in sorted(waiting, key=lambda t: (-_best_pair(tasks[t], pos, reserved, hub, hub_end)[0], t)):
            _, slot_u, slot_v = _best_pair(tasks[task], pos, reserved, hub, hub_end)
            u, v = tasks[task]
            slot_of[u], slot_of[v] = slot_u, slot_v
            reserved.add(slot_u)
            reserved.add(slot_v)

        # --- 2. lower every zone atom that does not stay parked for this round
        lower([q for q in zone_atoms() if q not in parked])

        # --- 3. single-qubit gates (and measurements) for everything now in storage
        in_zone = {q for q in range(n) if _in_ent(pos[q])}
        for q in range(n):
            if q in in_zone:
                if q not in parked:
                    continue
                matrix, _ = pending(q)
                if not _is_identity(matrix):
                    raise RuntimeError(f"hubstream: parked atom {q} needs single-qubit gates")
            flush(q)

        # --- 4. lift the participants into their assigned slots
        do_moves({q: (pos[q], slot) for q, slot in slot_of.items() if pos[q] != slot})

        # --- 5. one CZ pulse per pair of the round
        for task in round_tasks:
            u, v = tasks[task]
            if pointer[u] >= len(wires[u]) or wires[u][pointer[u]] != ("cz", task):
                raise RuntimeError(f"hubstream: wire {u} is not at task {task}")
            if pointer[v] >= len(wires[v]) or wires[v][pointer[v]] != ("cz", task):
                raise RuntimeError(f"hubstream: wire {v} is not at task {task}")
            if not _paired(pos[u], pos[v]):
                raise RuntimeError(f"hubstream: task {task} atoms {pos[u]}, {pos[v]} are not an interaction pair")
            out.append(CZGate(), [out.qubits[u], out.qubits[v]])
            pointer[u] += 1
            pointer[v] += 1

        # --- 6. park rule: stay iff the pending run is trivial and another task follows soon
        candidates: dict[int, tuple[int, int]] = {}
        for q in zone_atoms():
            matrix, next_cz = pending(q)
            if next_cz is None or not _is_identity(matrix):
                continue
            task = wires[q][next_cz][1]
            # keeping an atom in the zone occupies a pair: only worth it when the pair is reused
            # immediately (or for the hub, which never leaves the zone)
            if q != hub and round_of.get(task, 1 << 30) > round_index + PARK_WINDOW:
                continue
            candidates[q] = (task, _remaining_cz(wires[q], next_cz))

        def preference(q: int):
            return (0 if q == hub else 1, -candidates[q][1], q)

        by_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
        for q in candidates:
            by_pair[(pos[q][0] // 2, pos[q][1])].append(q)
        kept: list[int] = []
        for group in by_pair.values():
            group.sort(key=preference)
            if (
                len(group) == 2
                and candidates[group[0]][0] == candidates[group[1]][0]
                and _paired(pos[group[0]], pos[group[1]])
            ):
                kept.extend(group)  # both need the same next task on this pair: both may stay
            else:
                kept.append(group[0])
        claims: dict[int, list[int]] = defaultdict(list)
        for q in kept:
            claims[candidates[q][0]].append(q)
        parked = set()
        for group in claims.values():
            if len(group) == 2 and _paired(pos[group[0]], pos[group[1]]):
                parked.update(group)
            else:
                group.sort(key=preference)
                parked.add(group[0])

    # --- tail: bring every remaining zone atom back to storage, then flush all wires
    lower(zone_atoms())
    for q in range(n):
        flush(q)
        flush_deferred(q)
    return out


# ================================================================ gates front end (copied)
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


def _is_phase_identity(u: np.ndarray) -> bool:
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
            if not _is_phase_identity(m):
                out.append(Item("1q", (q,), mat=m))
        out.append(Item("cz", (u, v)))
        out.append(Item("1q", (mid_q,), mat=_rx_mat(it.theta)))
        out.append(Item("cz", (u, v)))
        for q, m in ((u, post_u), (v, post_v)):
            if not _is_phase_identity(m):
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
                if _is_phase_identity(mat):
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


# ================================================================ DAG scheduler + front end


def choose_hub_dag(items, n: int):
    """Highest-degree qubit of the interaction graph, if it really is a hub; else None."""
    if n < 6:
        return None
    neighbours: dict[int, set[int]] = defaultdict(set)
    for item in items:
        if item.kind == "cz":
            u, v = item.qubits
            neighbours[u].add(v)
            neighbours[v].add(u)
    if not neighbours:
        return None
    hub = max(neighbours, key=lambda q: (len(neighbours[q]), -q))
    return hub if len(neighbours[hub]) >= 0.4 * (n - 1) else None


def schedule_rounds_dag(items, num_qubits: int, hub, max_pairs: int, key_mode: int = 0):
    """Commutation-aware list schedule: drain all ready 1q work, then one capped CZ round.

    Same commutation DAG and criticality as `gates`, but a CZ round is (a) capped at `max_pairs` and
    (b) selected with a hub / parked-atom preference so the hub keeps its interaction pair:
    key = (hub first, -criticality, warm first, index) for `key_mode` 0 and
    (hub first, warm first, -criticality, index) for 1.
    """
    succ, indeg = build_commutation_dag(items, num_qubits)
    n = len(items)
    crit = [0] * n
    for idx in range(n - 1, -1, -1):
        best = 0
        for other in succ[idx]:
            if crit[other] > best:
                best = crit[other]
        crit[idx] = best + (1 if items[idx].kind == "cz" else 0)

    remaining = list(indeg)
    ready = [idx for idx in range(n) if remaining[idx] == 0]
    phases: list[tuple[str, list[int]]] = []
    done = 0
    prev_round: set[int] = set()
    touched: set[int] = set()  # qubits with a 1q item emitted since the last CZ round

    def release(idx, bucket_1q, bucket_cz):
        for other in succ[idx]:
            remaining[other] -= 1
            if remaining[other] == 0:
                (bucket_cz if items[other].kind == "cz" else bucket_1q).append(other)

    ready_1q = [idx for idx in ready if items[idx].kind != "cz"]
    ready_cz = [idx for idx in ready if items[idx].kind == "cz"]

    def warm(idx) -> bool:
        """Could stay parked: it was in the previous round and has had no 1q gate since."""
        u, v = items[idx].qubits
        return (u in prev_round and u not in touched) or (v in prev_round and v not in touched)

    def key(idx):
        hub_first = 0 if (hub is not None and hub in items[idx].qubits) else 1
        is_warm = 0 if warm(idx) else 1
        if key_mode == 0:
            return (hub_first, -crit[idx], is_warm, idx)
        return (hub_first, is_warm, -crit[idx], idx)

    while done < n:
        emitted: list[int] = []
        queue = ready_1q
        ready_1q = []
        while queue:
            idx = queue.pop()
            emitted.append(idx)
            done += 1
            touched.update(items[idx].qubits)
            release(idx, queue, ready_cz)
        if emitted:
            emitted.sort()
            phases.append(("1q", emitted))
        if done >= n:
            break
        ready_cz.sort(key=key)
        used: set[int] = set()
        chosen: list[int] = []
        rest: list[int] = []
        for idx in ready_cz:
            if len(chosen) >= max_pairs:
                rest.append(idx)
                continue
            u, v = items[idx].qubits
            if u in used or v in used:
                rest.append(idx)
                continue
            used.add(u)
            used.add(v)
            chosen.append(idx)
        if not chosen:
            raise RuntimeError("hubstream: DAG scheduler deadlock")
        ready_cz = rest
        phases.append(("cz", chosen))
        prev_round = set(used)
        touched = set()
        for idx in chosen:
            done += 1
            release(idx, ready_1q, ready_cz)
    return phases


def _dag_frontend(circuit: QuantumCircuit, n: int, max_pairs: int, key_mode: int):
    """gates' front end -> (wires, tasks, rounds, hub) in the layout `emit` consumes."""
    items, _phase = lower_to_items(circuit)
    # theta == 0 modulo 2*pi is the identity: drop it (exact up to global phase)
    items = [it for it in items
             if not (it.kind == "rot2" and abs(math.remainder(it.theta, 2.0 * math.pi)) < 1e-12)]
    degree = two_qubit_degree(items, n)
    items = expand_rot2(items, degree)
    items, _extra = merge_1q_runs(items, n)
    hub = choose_hub_dag(items, n)
    phases = schedule_rounds_dag(items, n, hub, max_pairs, key_mode)

    wires: list[list] = [[] for _ in range(n)]
    tasks: list[tuple[int, int]] = []
    task_of_item: dict[int, int] = {}
    for kind, indices in phases:
        for idx in sorted(indices):  # DAG edges go low -> high index: this is a topological order
            item = items[idx]
            if item.kind == "1q":
                wires[item.qubits[0]].append(("g", item.mat))
            elif item.kind == "meas":
                wires[item.qubits[0]].append(("meas", item.clbit))
            elif item.kind == "cz":
                task = len(tasks)
                tasks.append(item.qubits)
                wires[item.qubits[0]].append(("cz", task))
                wires[item.qubits[1]].append(("cz", task))
                task_of_item[idx] = task
            else:  # pragma: no cover - expand_rot2 removed every rot2
                raise ValueError(f"hubstream: unexpected item {item.kind}")
    rounds = [[task_of_item[idx] for idx in sorted(indices)]
              for kind, indices in phases if kind == "cz"]
    return wires, tasks, rounds, hub


def _move_distance(circuit: QuantumCircuit, loading: dict[int, Site]) -> float:
    """Sum over the circuit's `move`s of the longest physical displacement (the scored distance)."""
    index = {bit: i for i, bit in enumerate(circuit.qubits)}
    pos = dict(loading)
    occ = {site: q for q, site in pos.items()}
    total = 0.0
    for instruction in circuit.data:
        if instruction.operation.name != "move":
            continue
        qubits = [index[q] for q in instruction.qubits]
        rows, cols, new_rows, new_cols = instruction.operation.params
        mapping = _sim_move(occ, qubits, rows, cols, new_rows, new_cols)
        sources = {occ[site]: site for site in mapping if site in occ}
        best = 0.0
        for qubit in qubits:
            best = max(best, _dist(sources[qubit], mapping[sources[qubit]]))
        total += best
        for qubit in qubits:
            del occ[sources[qubit]]
        for qubit in qubits:
            site = mapping[sources[qubit]]
            occ[site] = qubit
            pos[qubit] = site
    return total


def _gates_depth(circuit: QuantumCircuit) -> int:
    """Depth of the circuit with moves and barriers removed (the scored depth)."""
    stripped = circuit.copy_empty_like()
    for instruction in circuit.data:
        if instruction.operation.name not in ("move", "barrier"):
            stripped._append(instruction)
    return stripped.depth()

# ---------------------------------------------------------------- entry point


def _first_use_order(wires, round_of, n: int) -> list[int]:
    """Qubits sorted by the round of their first CZ (later / never used ones last)."""

    def first_use(q: int) -> int:
        for entry in wires[q]:
            if entry[0] == "cz":
                return round_of.get(entry[1], 1 << 30)
        return 1 << 30

    return sorted(range(n), key=lambda q: (first_use(q), q))


def _loading_by_first_use(wires, round_of, n: int, hub=None) -> dict[int, Site]:
    order = _first_use_order(wires, round_of, n)
    sites = sorted(_STORAGE, key=lambda site: (9 - site[1], site[0]))
    loading = {q: sites[i] for i, q in enumerate(order)}
    if HUB_END_PAIR and n > HUB_END_MIN_QUBITS and hub is not None:
        # the hub's home goes right under its parked slot (9, 10): swap it with whoever is at (9, 9)
        target = (9, 9)
        other = next((q for q, site in loading.items() if site == target), None)
        if other is not None:
            loading[other] = loading[hub]
        loading[hub] = target
    return loading


def _build_wire_frontend(circuit: QuantumCircuit, n: int):
    """Original front end: exact cx-rz-cx / rzz parsing + ASAP wire scheduling."""
    items = parse_items(circuit)
    hub = choose_hub(items, n)
    wires, tasks = build_wires(items, n, hub)
    rounds = schedule_rounds(wires, tasks, hub) if tasks else []
    return wires, tasks, rounds, hub


def _compile_with(circuit, front, n, loading_hook):
    wires, tasks, rounds, hub = front
    round_of = {}
    for index, round_tasks in enumerate(rounds):
        for task in round_tasks:
            round_of[task] = index
    loading = loading_hook(wires, round_of, n, hub)
    out = emit(n, wires, tasks, rounds, hub, circuit, loading)
    return out, loading


WEIGHT = 2.92
"""Scoring exchange rate: one depth unit costs as much as this many distance units."""


def compile_circuit(circuit: QuantumCircuit, architecture_spec: Any):
    n = circuit.num_qubits
    candidates = []
    errors = []
    # Both front ends are exact; a circuit is compiled by each and the cheaper one (distance +
    # 2.92 * depth, the portfolio's exchange rate) is kept.  No file-name special cases.
    candidates_spec = [("wire", lambda: _build_wire_frontend(circuit, n))]
    for cap in (DAG_MAX_PAIRS if n <= LARGE_CIRCUIT else DAG_MAX_PAIRS_LARGE):
        candidates_spec.append((f"dag{cap}", lambda cap=cap: _dag_frontend(circuit, n, cap, 0)))
    for tag, build in candidates_spec:
        try:
            front = build()
            out, loading = _compile_with(circuit, front, n, _loading_by_first_use)
            cost = _move_distance(out, loading) + WEIGHT * _gates_depth(out)
            candidates.append((cost, tag, out, loading))
        except Exception as error:  # a front end that cannot handle this circuit is simply skipped
            errors.append(f"{tag}: {type(error).__name__}: {error}")
    if not candidates:
        raise RuntimeError("hubstream: no front end could compile the circuit: " + "; ".join(errors))
    candidates.sort(key=lambda c: (c[0], c[1]))
    _cost, _tag, out, loading = candidates[0]
    return out, Architecture(loading, spec=architecture_spec)
