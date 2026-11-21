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

import itertools
import math
from collections import defaultdict
from numbers import Integral
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction
from qiskit.circuit.library import CZGate, Measure, RXGate, RZGate

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


def _move(circuit: QuantumCircuit, qubits, rows, cols, new_rows, new_cols) -> None:
    circuit.append(Move(rows, cols, new_rows, new_cols, len(qubits)), list(qubits))


# ---------------------------------------------------------------- geometry (hard-coded)

STORAGE_X = (0, 9)
STORAGE_Y = (0, 9)
ENT_X = (0, 9)
ENT_Y = (10, 19)
PAIR_DISTANCE = 0.3
MAX_PAIRS = 25
"""At most this many interaction pairs per round (five entangling rows)."""

SQRT1_2 = 1.0 / math.sqrt(2.0)
H_GATE = np.array([[SQRT1_2, SQRT1_2], [SQRT1_2, -SQRT1_2]], dtype=complex)
_STORAGE = [(x, y) for y in range(STORAGE_Y[0], STORAGE_Y[1] + 1) for x in range(STORAGE_X[0], STORAGE_X[1] + 1)]
_PAIRS = [
    ((2 * k, row), (2 * k + 1, row), row)
    for row in range(ENT_Y[0], ENT_Y[1] + 1)
    for k in range((ENT_X[1] - ENT_X[0] + 1) // 2)
]


def _phys(site: Site) -> tuple[float, float]:
    """Physical position of a grid site (storage spacing 1, entangling columns paired 0.3 apart)."""
    x, y = site
    if y >= ENT_Y[0]:
        return (2.0 * (x // 2) + PAIR_DISTANCE * (x % 2), float(y))
    return (float(x), float(y))


def _dist(a: Site, b: Site) -> float:
    (ax, ay), (bx, by) = _phys(a), _phys(b)
    return math.hypot(ax - bx, ay - by)


def _in_storage(site: Site) -> bool:
    return STORAGE_X[0] <= site[0] <= STORAGE_X[1] and STORAGE_Y[0] <= site[1] <= STORAGE_Y[1]


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


def _emit_1q(out: QuantumCircuit, q: int, u) -> None:
    """Emit one merged single-qubit run as at most three `rz`/`rx` gates.

    ``_zxz`` gives ``u = rz(a) rx(b) rz(c)``, so the gates go out in reverse: ``rz(c), rx(b), rz(a)``.
    """
    if _is_identity(u):
        return
    a, b, c = _zxz(np.asarray(u, dtype=complex))
    qubit = out.qubits[q]
    if c:
        out.append(RZGate(c), [qubit])
    if b:
        out.append(RXGate(b), [qubit])
    if a:
        out.append(RZGate(a), [qubit])


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


def _nearest_free(free: list[Site], site: Site) -> Site:
    best = free[0]
    best_key = None
    for candidate in free:
        key = (_dist(site, candidate), abs(candidate[0] - site[0]), candidate[1], candidate[0])
        if best_key is None or key < best_key:
            best_key, best = key, candidate
    return best


def _best_pair(task: tuple[int, int], pos: dict[int, Site], reserved: set[Site], hub):
    """Cheapest free interaction pair for `task`: ``(cost, slot for u, slot for v)``."""
    u, v = task
    hub_in_task = hub is not None and (u == hub or v == hub)
    best = None
    for left, right, row in _PAIRS:
        if left in reserved or right in reserved:
            continue
        if hub_in_task:
            # the hub always takes the right (odd, cheap-to-lower) slot of its pair
            options = [(max(_dist(pos[u], right), _dist(pos[v], left)), right, left)] if u == hub else [
                (max(_dist(pos[u], left), _dist(pos[v], right)), left, right)
            ]
        else:
            options = [
                (max(_dist(pos[u], left), _dist(pos[v], right)), left, right),
                (max(_dist(pos[u], right), _dist(pos[v], left)), right, left),
            ]
        for cost, slot_u, slot_v in options:
            key = (cost, row, left[0])
            if best is None or key < best[0]:
                best = (key, (cost, slot_u, slot_v))
    if best is None:
        raise RuntimeError("hubstream: no free interaction pair left in this round")
    return best[1]


def _remaining_cz(wire, start: int) -> int:
    return sum(1 for entry in wire[start:] if entry[0] == "cz")


def emit(n: int, wires, tasks, rounds, hub, circuit: QuantumCircuit, loading: dict[int, Site]) -> QuantumCircuit:
    out = circuit.copy_empty_like()
    pos: dict[int, Site] = dict(loading)
    occ: dict[Site, int] = {site: qubit for qubit, site in pos.items()}
    pointer = [0] * n
    parked: set[int] = set()

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
                _emit_1q(out, q, matrix)
            elif entry[0] == "meas":
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
        free = [site for site in _STORAGE if site not in occ]
        moves: dict[int, tuple[Site, Site]] = {}
        for qubit in sorted(atoms, key=lambda q: (pos[q][1], pos[q][0])):
            target = _nearest_free(free, pos[qubit])
            free.remove(target)
            moves[qubit] = (pos[qubit], target)
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
        return matrix, (i if has_cz else None)

    round_of = {}
    for index, round_tasks in enumerate(rounds):
        for task in round_tasks:
            round_of[task] = index

    for round_index, round_tasks in enumerate(rounds):
        # --- 1. assign an interaction pair to every task of this round
        slot_of: dict[int, Site] = {}
        reserved = {pos[q] for q in parked}
        waiting: list[tuple[float, int]] = []
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
                best = _best_pair(tasks[task], pos, reserved, hub)
                waiting.append((-best[0], task))
        waiting.sort()
        for _, task in waiting:
            u, v = tasks[task]
            best = _best_pair(tasks[task], pos, reserved, hub)
            slot_of[u], slot_of[v] = best[1], best[2]
            reserved.add(best[1])
            reserved.add(best[2])

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
            if q != hub and round_of.get(task, 1 << 30) > round_index + 1:
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
    return out


# ---------------------------------------------------------------- entry point


def compile_circuit(circuit: QuantumCircuit, architecture_spec: Any):
    n = circuit.num_qubits
    items = parse_items(circuit)
    hub = choose_hub(items, n)
    wires, tasks = build_wires(items, n, hub)

    rows_used = (n + 9) // 10
    r0 = max(0, 9 - rows_used - 2)
    loading = {i: (i % 10, r0 + i // 10) for i in range(n)}

    rounds = schedule_rounds(wires, tasks, hub) if tasks else []
    out = emit(n, wires, tasks, rounds, hub, circuit, loading)
    return out, Architecture(loading, spec=architecture_spec)
