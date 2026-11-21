"""rectbatch: schedule the native circuit into rounds of disjoint `cz`, then lift each round with a
few rectangle moves and place the qubits so that the partners of a round share a storage row.

Self-contained (qiskit + stdlib) so it can be packaged into a marimo notebook unchanged.
"""

import heapq
import itertools
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction

Site = tuple[int, int]

OPT_LEVEL = 1
"""Transpile level; the validator's interaction-pair reference uses 1, and level 2 drifts angles."""

# ------------------------------------------------------------------ machine geometry (hard-coded)

STORAGE = (0, 9, 0, 9)  # x0, x1, y0, y1
ENTANGLING = (0, 9, 10, 19)
ENT_Y0 = 10
N_PAIRS_PER_ROW = 5  # interaction pairs (2k, 2k+1), k = 0..4


def _physical(site: Site) -> tuple[float, float]:
    """Physical position of a grid site: storage is a unit grid, entangling pairs are 0.3 apart."""
    x, y = site
    if y >= ENT_Y0:
        return (2.0 * (x // 2) + 0.3 * (x % 2), float(y))
    return (float(x), float(y))


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


# ------------------------------------------------------------------ 1. native circuit


def native_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """Transpile to `rx`/`rz`/`cz` (no coupling map, so qubit indices do not move); drop barriers.

    Optimisation level 2/3 is unsafe: `ElidePermutations` can return a circuit that is only
    equivalent up to a qubit permutation, and approximate synthesis drifts the angles. Only accept a
    transpilation whose layout is trivial, otherwise fall back to a cheaper level.
    """
    native = None
    for level in (OPT_LEVEL, 0):
        try:
            candidate = transpile(circuit, basis_gates=["rx", "rz", "cz"], optimization_level=level)
        except Exception:  # noqa: BLE001 - fall back to a cheaper optimisation level
            continue
        native = candidate
        if getattr(candidate, "layout", None) is None:
            break
    if native is None:
        native = circuit.copy()
    out = native.copy_empty_like()
    for instruction in native.data:
        if instruction.operation.name != "barrier":
            out.append(instruction)
    return out


def zz_orient(circuit: QuantumCircuit) -> QuantumCircuit:
    """Rewrite every `cx(a,b); rz(t) b; cx(a,b)` block into the same block with the other control.

    `cx(a,b) rz(t) b cx(a,b)` equals `cx(b,a) rz(t) a cx(b,a)`. Orienting each block so that the
    endpoint with the larger interaction degree is the control leaves the hub's wire almost free of
    single-qubit gates within a `cz` layer, so the hub can stay parked in the entangling zone.
    """
    index = {bit: i for i, bit in enumerate(circuit.qubits)}
    data = list(circuit.data)
    degree: dict[int, int] = defaultdict(int)
    seen = set()
    for instruction in data:
        if len(instruction.qubits) == 2:
            a, b = sorted(index[q] for q in instruction.qubits)
            if (a, b) not in seen:
                seen.add((a, b))
                degree[a] += 1
                degree[b] += 1
    nxt: dict[tuple[int, int], int] = {}
    last: dict[int, int] = {}
    for i in range(len(data) - 1, -1, -1):
        for qubit in data[i].qubits:
            q = index[qubit]
            nxt[(i, q)] = last.get(q)
            last[q] = i
    out = circuit.copy_empty_like()
    skip = set()
    for i, instruction in enumerate(data):
        if i in skip:
            continue
        if instruction.operation.name == "cx":
            a, b = (index[q] for q in instruction.qubits)
            j = nxt.get((i, b))
            if j is not None and data[j].operation.name == "rz" and index[data[j].qubits[0]] == b:
                k = nxt.get((j, b))
                if (
                    k is not None
                    and k == nxt.get((i, a))
                    and data[k].operation.name == "cx"
                    and [index[q] for q in data[k].qubits] == [a, b]
                ):
                    angle = data[j].operation.params[0]
                    skip |= {j, k}
                    if degree[b] > degree[a]:
                        a, b = b, a
                    out.cx(a, b)
                    out.rz(angle, b)
                    out.cx(a, b)
                    continue
        out.append(instruction)
    return out


def _native_key(native: QuantumCircuit):
    """Cheap structural signature, used to skip a candidate whose native circuit is identical."""
    return tuple((instruction.operation.name, tuple(instruction.qubits), instruction.operation.params)
                 for instruction in native.data)


# ------------------------------------------------------------------ 2. round scheduling


@dataclass
class Phase:
    """One round: 1q/measure `ops`, then one `cz` per entry of `pairs` (all pairs disjoint)."""

    ops: list[Any] = field(default_factory=list)
    pairs: list[Any] = field(default_factory=list)


def schedule_rounds(native: QuantumCircuit) -> list[Phase]:
    """ASAP layering of the per-qubit dependency DAG into rounds of a greedy maximal `cz` matching."""
    index = {bit: i for i, bit in enumerate(native.qubits)}
    cindex = {bit: i for i, bit in enumerate(native.clbits)}
    items = []
    for instruction in native.data:
        items.append(
            (
                instruction.operation,
                [index[q] for q in instruction.qubits],
                [cindex[c] for c in instruction.clbits],
            )
        )
    total = len(items)
    if total == 0:
        return []
    nq = len(index)
    fifo: list[list[int]] = [[] for _ in range(nq)]
    for i, (_, qargs, _) in enumerate(items):
        for q in qargs:
            fifo[q].append(i)
    ptr = [0] * nq
    wait = [0] * total
    for i, (_, qargs, _) in enumerate(items):
        wait[i] = sum(1 for q in qargs if fifo[q][0] != i)
    is_cz = [op.name == "cz" for op, _, _ in items]
    done = [False] * total
    ready1 = [i for i in range(total) if wait[i] == 0 and not is_cz[i]]
    ready2 = [i for i in range(total) if wait[i] == 0 and is_cz[i]]
    heapq.heapify(ready1)
    heapq.heapify(ready2)
    remaining = total
    phases: list[Phase] = []

    def advance(i: int) -> None:
        for q in items[i][1]:
            ptr[q] += 1
            if ptr[q] < len(fifo[q]):
                j = fifo[q][ptr[q]]
                if wait[j] > 0:
                    wait[j] -= 1
                    if wait[j] == 0:
                        heapq.heappush(ready2 if is_cz[j] else ready1, j)

    while remaining > 0:
        phase = Phase()
        # (a) every ready 1q gate / measure
        while ready1:
            i = heapq.heappop(ready1)
            if done[i] or wait[i] != 0:
                continue
            done[i] = True
            remaining -= 1
            phase.ops.append(i)
            advance(i)
        # (b) greedy maximal matching of the ready cz gates
        cands = []
        while ready2:
            i = heapq.heappop(ready2)
            if done[i] or wait[i] != 0:
                continue
            cands.append(i)
        used: set[int] = set()
        accepted = []
        for i in cands:
            u, v = items[i][1]
            if u in used or v in used:
                continue
            used.add(u)
            used.add(v)
            accepted.append(i)
        chosen = set(accepted)
        for i in cands:
            if i not in chosen:
                heapq.heappush(ready2, i)
        for i in accepted:
            done[i] = True
            remaining -= 1
            phase.pairs.append((items[i][1][0], items[i][1][1], i))
            advance(i)
        if not phase.ops and not phase.pairs:
            raise AssertionError("stalled scheduler: no ready instruction in a DAG")
        phases.append(phase)

    for phase in phases:
        phase.ops = [items[i] for i in phase.ops]
        phase.pairs = [(u, v, items[i][0]) for (u, v, i) in phase.pairs]
    return phases


# ------------------------------------------------------------------ 5a. legal move builder


def move_params(atoms: dict[int, tuple[Site, Site]], occupant: dict[Site, int]):
    """The validator's move rules, re-implemented. `atoms`: qubit -> (source site, target site)."""
    if not atoms:
        return None
    row_map: dict[int, int] = {}
    col_map: dict[int, int] = {}
    src_zone = set()
    dst_zone = set()
    for src, dst in atoms.values():
        sx, sy = src
        dx, dy = dst
        if not (0 <= sx <= 9 and 0 <= dx <= 9):
            return None
        if sy in row_map and row_map[sy] != dy:
            return None
        if sx in col_map and col_map[sx] != dx:
            return None
        row_map[sy] = dy
        col_map[sx] = dx
        src_zone.add(sy >= ENT_Y0)
        dst_zone.add(dy >= ENT_Y0)
    if len(src_zone) != 1 or len(dst_zone) != 1:
        return None
    src_rows = sorted(row_map)
    src_cols = sorted(col_map)
    if any(row_map[b] <= row_map[a] for a, b in itertools.pairwise(src_rows)):
        return None
    if any(col_map[b] <= col_map[a] for a, b in itertools.pairwise(src_cols)):
        return None
    rows = tuple(range(src_rows[0], src_rows[-1] + 1))
    cols = tuple(range(src_cols[0], src_cols[-1] + 1))
    y_lo, y_hi = rows[0], rows[-1]
    x_lo, x_hi = cols[0], cols[-1]
    for (x, y), q in occupant.items():
        if x_lo <= x <= x_hi and y_lo <= y <= y_hi and q not in atoms:
            return None
    for dst in (v[1] for v in atoms.values()):
        other = occupant.get(dst)
        if other is not None and other not in atoms:
            return None
    new_rows = tuple(row_map[y] for y in src_rows)
    new_cols = tuple(col_map[x] for x in src_cols)
    if len(new_rows) > len(rows) or len(new_cols) > len(cols):
        return None
    return rows, cols, new_rows, new_cols


def move_cost(atoms: dict[int, tuple[Site, Site]]) -> float:
    """Scored cost of a move: the longest physical distance travelled by one of its atoms."""
    best = 0.0
    for src, dst in atoms.values():
        d = math.dist(_physical(src), _physical(dst))
        if d > best:
            best = d
    return best


# ------------------------------------------------------------------ 3. target assignment


def _host_row_of(u, v, pos):
    yu, yv = pos[u][1], pos[v][1]
    return yu if yu <= yv else yv


def assign_targets(pairs, pos: dict[int, Site], strategy: str = "compact"):
    """Give every round pair an entangling interaction pair.

    Returns `(targets, visitors, assigned, overflow)`: `targets` maps qubit -> entangling site,
    `visitors` is the set of atoms that travel from another storage row, `assigned` the pairs that
    got a site and `overflow` the (never expected) pairs that would need an 11th entangling row.
    The atom of a cross-row pair in the smaller row is the host and sits on the even site.

    `compact` gives host row `y` the entangling row `10 + rank(y)` (monotone, so multi-row
    rectangles can merge); `flat` fills the lowest entangling rows slot by slot, which minimises the
    vertical travel when the atoms cannot merge anyway.
    """
    row_pairs: dict[int, list[Any]] = defaultdict(list)
    for pair in pairs:
        row_pairs[_host_row_of(pair[0], pair[1], pos)].append(pair)
    host_rows = sorted(row_pairs)
    by_column = {}
    for y in host_rows:
        by_column[y] = sorted(row_pairs[y], key=lambda pair: pos[pair[0]][0] if pos[pair[0]][1] == y else pos[pair[1]][0])

    groups = []  # (host row, chunk of <= 5 pairs, candidate slot lines, target entangling row)
    used_rows: set[int] = set()
    overflow: list[Any] = []
    if strategy == "flat":
        row = ENT_Y0
        taken = 0
        for y in host_rows:
            hosted = by_column[y]
            index = 0
            while index < len(hosted):
                if taken >= N_PAIRS_PER_ROW:
                    row += 1
                    taken = 0
                if row > ENTANGLING[3]:
                    overflow.extend(hosted[index:])
                    break
                capacity = N_PAIRS_PER_ROW - taken
                chunk = hosted[index : index + capacity]
                groups.append((y, chunk, tuple(range(taken, taken + len(chunk))), row))
                taken += len(chunk)
                index += len(chunk)
        extra = []
    else:
        target_of = {y: ENT_Y0 + i for i, y in enumerate(host_rows)}
        extra = []
        for y in host_rows:
            hosted = by_column[y]
            row = target_of[y]
            groups.append((y, hosted[:N_PAIRS_PER_ROW], tuple(range(N_PAIRS_PER_ROW)), row))
            used_rows.add(row)
            if len(hosted) > N_PAIRS_PER_ROW:
                extra.append((y, hosted[N_PAIRS_PER_ROW:], row))
        for y, rest, first_row in extra:
            for start in range(0, len(rest), N_PAIRS_PER_ROW):
                chunk = rest[start : start + N_PAIRS_PER_ROW]
                row = None
                for candidate in itertools.chain(range(first_row + 1, ENTANGLING[3] + 1), range(ENT_Y0, first_row + 1)):
                    if candidate not in used_rows:
                        row = candidate
                        break
                if row is None:
                    overflow.extend(chunk)
                    continue
                used_rows.add(row)
                groups.append((y, chunk, tuple(range(N_PAIRS_PER_ROW)), row))

    targets: dict[int, Site] = {}
    visitors: set[int] = set()
    assigned = []
    for y, chunk, slots, row in groups:
        if not chunk:
            continue
        host_costs = []
        visit_costs = []
        for pair in chunk:
            u, v = pair[0], pair[1]
            if pos[u][1] == pos[v][1]:
                a, b = (u, v) if pos[u][0] <= pos[v][0] else (v, u)
                hc = []
                vc = []
                for k in slots:
                    hc.append(
                        max(
                            math.dist(_physical((2 * k, row)), _physical(pos[a])),
                            math.dist(_physical((2 * k + 1, row)), _physical(pos[b])),
                        )
                    )
                    vc.append(0.0)
            else:
                host, visitor = (u, v) if pos[u][1] == y else (v, u)
                hc = [math.dist(_physical((2 * k, row)), _physical(pos[host])) for k in slots]
                vc = [math.dist(_physical((2 * k + 1, row)), _physical(pos[visitor])) for k in slots]
            host_costs.append(hc)
            visit_costs.append(vc)

        best_score = None
        best_combo = None
        m = len(chunk)
        if m == len(slots):
            best_combo = tuple(range(m))
        else:
            for combo in itertools.combinations(range(len(slots)), m):
                host_max = 0.0
                visit_sum = 0.0
                for idx, slot in enumerate(combo):
                    value = host_costs[idx][slot]
                    if value > host_max:
                        host_max = value
                    visit_sum += visit_costs[idx][slot]
                score = host_max + visit_sum
                if best_score is None or score < best_score:
                    best_score = score
                    best_combo = combo
        for idx, slot_index in enumerate(best_combo):
            k = slots[slot_index]
            u, v = chunk[idx][0], chunk[idx][1]
            if pos[u][1] == pos[v][1]:
                a, b = (u, v) if pos[u][0] <= pos[v][0] else (v, u)
            else:
                host, visitor = (u, v) if pos[u][1] == y else (v, u)
                a, b = host, visitor
                visitors.add(visitor)
            targets[a] = (2 * k, row)
            targets[b] = (2 * k + 1, row)
            assigned.append(chunk[idx])
    return targets, visitors, assigned, overflow


# ------------------------------------------------------------------ 4. move planning


def _mkey(move: dict[int, tuple[Site, Site]]):
    srcs = [src for src, _ in move.values()]
    return (min(y for _, y in srcs), min(x for x, _ in srcs), len(move))


def _merge(moves, occupant, pair_budget: int = 400):
    """Agglomerative merging: fusing two legal moves never costs more than keeping them apart."""
    moves = list(moves)
    if len(moves) < 2:
        return moves
    calls = 0
    changed = True
    while changed:
        changed = False
        moves.sort(key=_mkey)
        n = len(moves)
        best = None
        for i in range(n):
            for j in range(i + 1, n):
                if n > 8 and j > i + 3:
                    break
                a, b = moves[i], moves[j]
                if not a.keys().isdisjoint(b.keys()):
                    continue
                merged = {**a, **b}
                if move_params(merged, occupant) is None:
                    continue
                calls += 1
                saving = move_cost(a) + move_cost(b) - move_cost(merged)
                if best is None or saving > best[0] + 1e-12:
                    best = (saving, i, j)
                if calls > pair_budget:
                    break
            if calls > pair_budget:
                break
        if best is not None:
            _, i, j = best
            moves[i] = {**moves[i], **moves[j]}
            moves.pop(j)
            changed = True
    return moves


def _plan_group(atoms: dict[int, tuple[Site, Site]], occupant: dict[Site, int]):
    """Plan the moves of one sub-phase: column runs per (source row, target row), then fuse them."""
    if not atoms:
        return []
    by_row: dict[tuple[int, int], list[int]] = defaultdict(list)
    for q, (src, dst) in atoms.items():
        by_row[(src[1], dst[1])].append(q)
    moves = []
    for key in sorted(by_row):
        qubits = sorted(by_row[key], key=lambda q: atoms[q][0][0])
        current: dict[int, tuple[Site, Site]] = {}
        for q in qubits:
            if not current:
                current = {q: atoms[q]}
                continue
            candidate = {**current, q: atoms[q]}
            if move_params(candidate, occupant) is not None:
                current = candidate
            else:
                moves.append(current)
                current = {q: atoms[q]}
        if current:
            moves.append(current)
    return _merge(moves, occupant)


def _plan_segment(pairs, targets, visitors, pos: dict[int, Site]):
    """(up moves, cz list, down moves) for one segment; every atom ends where it started."""
    storage_occ = {site: q for q, site in pos.items()}

    atoms_v = {q: (pos[q], targets[q]) for q in visitors}
    occ = dict(storage_occ)
    up_v = _plan_group(atoms_v, occ)
    for q in visitors:
        occ.pop(pos[q], None)
    atoms_h = {q: (pos[q], targets[q]) for q in targets if q not in visitors}
    up_h = _plan_group(atoms_h, occ)

    ent_occ = {site: q for q, site in targets.items()}
    down_v = _plan_group({q: (targets[q], pos[q]) for q in visitors}, dict(ent_occ))
    for q in visitors:
        ent_occ.pop(targets[q], None)
    down_h = _plan_group({q: (targets[q], pos[q]) for q in targets if q not in visitors}, dict(ent_occ))

    return up_v + up_h, list(pairs), down_v + down_h


def _segment_cost(segments) -> float:
    total = 0.0
    for up, _, down in segments:
        for move in up:
            total += move_cost(move)
        for move in down:
            total += move_cost(move)
    return total


def _plan_strategy(pairs, pos: dict[int, Site], strategy: str):
    return _plan_assignment(assign_targets(pairs, pos, strategy), pos)


def _plan_assignment(assignment, pos: dict[int, Site]):
    targets, visitors, assigned, overflow = assignment
    segments = [_plan_segment(assigned, targets, visitors, pos)] if assigned else []
    if overflow:
        segments.extend(plan_segments(overflow, pos))
    return segments


def plan_segments(pairs, pos: dict[int, Site]):
    """One `(up, czs, down)` triple per segment, planned with both target strategies, best kept."""
    if not pairs:
        return []
    compact = assign_targets(pairs, pos, "compact")
    segments = _plan_assignment(compact, pos)
    if len(pairs) > 1:
        flat = assign_targets(pairs, pos, "flat")
        if flat[0] != compact[0]:
            alternative = _plan_assignment(flat, pos)
            if _segment_cost(alternative) < _segment_cost(segments) - 1e-12:
                return alternative
    return segments


def plan_moves(pairs, pos: dict[int, Site]):
    """(up moves, cz list, down moves) for one round, flattened over its segments."""
    segments = plan_segments(pairs, pos)
    up = [move for segment in segments for move in segment[0]]
    czs = [cz for segment in segments for cz in segment[1]]
    down = [move for segment in segments for move in segment[2]]
    return up, czs, down


def _round_cost(pairs, pos) -> float:
    return _segment_cost(plan_segments(pairs, pos))


def plan_cost(rounds, pos) -> float:
    return sum(_round_cost(phase.pairs, pos) for phase in rounds)


# ------------------------------------------------------------------ 6. placement


def _row_proxy(rounds, pos) -> float:
    """Cheap stand-in for the plan cost: one vertical trip per host row of a round, both ways."""
    total = 0.0
    for phase in rounds:
        rows = set()
        for pair in phase.pairs:
            u, v = pair[0], pair[1]
            yu, yv = pos[u][1], pos[v][1]
            rows.add(yu if yu <= yv else yv)
        for y in rows:
            total += ENT_Y0 - y
    return 2.0 * total


def _layout_linear(n: int):
    return {q: (q % 10, 9 - q // 10) for q in range(n)}


def _layout_boustro(n: int):
    return {q: ((q % 10) if (q // 10) % 2 == 0 else 9 - (q % 10), 9 - q // 10) for q in range(n)}


def _layout_path(n: int, weight, activity):
    if n == 0:
        return {}
    order = []
    used = set()
    current = max(range(n), key=lambda q: (activity.get(q, 0), -q))
    order.append(current)
    used.add(current)
    while len(order) < n:
        best = None
        for q in range(n):
            if q in used:
                continue
            key = (weight.get((current, q), 0), activity.get(q, 0), -q)
            if best is None or key > best[0]:
                best = (key, q)
        order.append(best[1])
        used.add(best[1])
        current = best[1]
    return {q: ((j % 10) if (j // 10) % 2 == 0 else 9 - (j % 10), 9 - j // 10) for j, q in enumerate(order)}


def _layout_pair_slots(n: int, weight, activity):
    """Pair the busiest partners into one storage row and fill rows 9 upwards."""
    ordered = sorted(weight.items(), key=lambda kv: (-kv[1], kv[0]))
    free = set(range(n))
    matched = []
    for (u, v), w in ordered:
        if u in free and v in free:
            free.discard(u)
            free.discard(v)
            matched.append(((u, v), w))
    matched.sort(key=lambda item: (-(item[1]), -activity[item[0][0]] - activity[item[0][1]], item[0]))
    pos: dict[int, Site] = {}
    slots = [((2 * k, y), (2 * k + 1, y)) for y in range(9, -1, -1) for k in range(N_PAIRS_PER_ROW)]
    for index, ((u, v), _) in enumerate(matched):
        if index >= len(slots):
            break
        a, b = slots[index]
        pos[u] = a
        pos[v] = b
    rest = sorted(free, key=lambda q: (-activity[q], q))
    taken = set(pos.values())
    remaining_sites = [(x, y) for y in range(9, -1, -1) for x in range(10) if (x, y) not in taken]
    for q, site in zip(rest, remaining_sites):
        pos[q] = site
    return pos


def initial_placement(n: int, rounds) -> dict[int, Site]:
    """Best of a few deterministic layouts, each pairing busy partners in a storage row."""
    weight: dict[tuple[int, int], int] = defaultdict(int)
    for phase in rounds:
        for u, v, _ in phase.pairs:
            key = (u, v) if u < v else (v, u)
            weight[key] += 1
    activity: dict[int, int] = defaultdict(int)
    for (u, v), w in weight.items():
        activity[u] += w
        activity[v] += w

    candidates = [
        _layout_linear(n),
        _layout_boustro(n),
        _layout_path(n, weight, activity),
        _layout_pair_slots(n, weight, activity),
    ]
    if not rounds:
        return candidates[0]
    candidates.sort(key=lambda layout: _row_proxy(rounds, layout))
    if n * len(rounds) <= 40000:
        return min(candidates, key=lambda layout: plan_cost(rounds, layout))
    return candidates[0]


def _budget(n: int, n_rounds: int) -> int:
    """Deterministic work budget of the placement search, in planned pair-rounds.

    One unit = re-planning one interaction pair of one round (~0.15-0.3 ms here). The budget is a
    pure function of the circuit size, so the same input always yields the same placement, whatever
    the machine load or whatever ran earlier in the process."""
    if n_rounds > 1000:
        return 10_000
    if n <= 16:
        return 20_000
    if n <= 24:
        return 40_000
    if n <= 50:
        return 50_000
    return 10_000


def optimise_placement(rounds, pos: dict[int, Site], budget: int) -> dict[int, Site]:
    """Incremental hill climbing on the plan cost: swap two sites, or move an atom to a free one.

    Deterministic: fixed seed and a fixed work budget (see `_budget`), no wall clock."""
    if not rounds or budget <= 0:
        return pos
    rng = random.Random(12345)
    pos = dict(pos)
    qubits = sorted(pos)
    sites = [(x, y) for y in range(9, -1, -1) for x in range(10)]
    used = set(pos.values())
    free_sites = [site for site in sites if site not in used]
    pred_of_round = [[(u, v) for (u, v, _) in phase.pairs] for phase in rounds]
    involved = [tuple(sorted({q for u, v in pr for q in (u, v)})) for pr in pred_of_round]
    pred_key = [tuple(sorted(pr)) for pr in pred_of_round]
    touch: dict[int, list[int]] = defaultdict(list)
    for i, qs in enumerate(involved):
        for q in qs:
            touch[q].append(i)
    active = [q for q in qubits if touch[q]]

    cache: dict[Any, float] = {}
    work = [0]

    def rcost(i: int, p: dict[int, Site]) -> float:
        # Keyed on the round's own atoms only: a heuristic (the ghost check also sees other atoms),
        # measured to beat an exact full-placement key because it allows far more proposals.
        key = (pred_key[i], tuple(p[q] for q in involved[i]))
        value = cache.get(key)
        if value is None:
            value = _round_cost(pred_of_round[i], p)
            cache[key] = value
            k = len(pred_of_round[i])
            work[0] += max(1, k if k <= 8 else 2 * k)  # big rounds plan super-linearly
        return value

    cost = [rcost(i, pos) for i in range(len(rounds))]
    if not active or work[0] > budget // 2:
        return dict(pos)
    site_of = {q: pos[q] for q in qubits}
    holder = {site: q for q, site in site_of.items()}
    proposals_left = 3 * budget  # guard: stop even if most proposals hit the cache

    def swap(u: int, v: int) -> None:
        keys = touch[u] + touch[v]
        target = site_of[v]
        origin = site_of[u]
        if not keys or target == origin:
            return
        unique = set(keys)
        old = 0.0
        for i in unique:
            old += cost[i]
        pos[u], pos[v] = target, origin
        site_of[u], site_of[v] = target, origin
        new = 0.0
        for i in unique:
            new += rcost(i, pos)
        if new < old - 1e-12:
            for i in unique:
                cost[i] = rcost(i, pos)
            holder[target] = u
            holder[origin] = v
        else:
            pos[u], pos[v] = origin, target
            site_of[u], site_of[v] = origin, target

    def relocate(u: int, site: Site) -> None:
        keys = touch[u]
        if not keys:
            return
        origin = site_of[u]
        unique = set(keys)
        old = 0.0
        for i in unique:
            old += cost[i]
        pos[u] = site
        site_of[u] = site
        new = 0.0
        for i in unique:
            new += rcost(i, pos)
        if new < old - 1e-12:
            for i in unique:
                cost[i] = rcost(i, pos)
            holder.pop(origin, None)
            holder[site] = u
            free_sites.remove(site)
            free_sites.append(origin)
        else:
            pos[u] = origin
            site_of[u] = origin

    while work[0] < budget and proposals_left > 0:
        proposals_left -= 1
        roll = rng.random()
        if roll < 0.3 and free_sites:
            relocate(rng.choice(active), free_sites[rng.randrange(len(free_sites))])
        elif active:
            swap(active[rng.randrange(len(active))], qubits[rng.randrange(len(qubits))])
        if len(cache) > 300000:
            cache.clear()
    return dict(pos)


# ------------------------------------------------------------------ 7. parking emission


PARK = False
"""Keep an atom already in the entangling zone for its next `cz`. Measured 40 % worse: the partner
is forced into the parked atom's slot, which is usually far, so this stays off."""


def _are_paired(a: Site, b: Site) -> bool:
    if a[1] != b[1] or a[1] < ENT_Y0:
        return False
    low, high = (a[0], b[0]) if a[0] < b[0] else (b[0], a[0])
    return high == low + 1 and low % 2 == 0


def _complement(site: Site) -> Site:
    x, y = site
    return (x + 1, y) if x % 2 == 0 else (x - 1, y)


def _occupied_pairs(occ: dict[Site, int]):
    return {frozenset((occ[s], occ[_complement(s)])) for s in occ if s[1] >= ENT_Y0 and s[0] % 2 == 0 and _complement(s) in occ}


def _park_actions(pairs, current, home, occ, ops_qubits):
    """Plan one round keeping the atoms that are already where their next `cz` needs them.

    Returns `(before, after)`: `before` are move dicts emitted before the round's 1q ops, `after`
    a list of `("move", atoms)` / `("cz", pair)` entries emitted afterwards, or `None` if the round
    cannot be planned safely (the caller then falls back to the static plan).
    """
    cur = dict(current)
    o = dict(occ)
    before = []
    after = []

    def apply(atoms):
        if move_params(atoms, o) is None:
            return False
        for q, (src, _) in atoms.items():
            o.pop(src, None)
        for q, (_, dst) in atoms.items():
            o[dst] = q
            cur[q] = dst
        return True

    pair_of = {}
    for pair in pairs:
        pair_of[pair[0]] = pair
        pair_of[pair[1]] = pair

    ready = []
    for pair in pairs:
        u, v = pair[0], pair[1]
        if u in ops_qubits or v in ops_qubits:
            continue
        if cur[u][1] >= ENT_Y0 and cur[v][1] >= ENT_Y0 and _are_paired(cur[u], cur[v]):
            ready.append(pair)
    ready_keys = {id(pair) for pair in ready}

    parked: dict[int, tuple[Any, Site]] = {}
    for pair in pairs:
        if id(pair) in ready_keys:
            continue
        for q in (pair[0], pair[1]):
            if q in parked or q in ops_qubits or cur[q][1] < ENT_Y0:
                continue
            if o.get(_complement(cur[q])) is not None:
                continue
            parked[q] = (pair, _complement(cur[q]))

    stay = set(parked)
    for pair in ready:
        stay.add(pair[0])
        stay.add(pair[1])
    dropping = {q: (cur[q], home[q]) for q in sorted(cur) if cur[q][1] >= ENT_Y0 and q not in stay}

    # lower in two sub-phases: odd entangling columns first, then even ones (rectangles stay clean)
    for parity in (1, 0):
        selection = {q: v for q, v in dropping.items() if v[0][0] % 2 == parity}
        if not selection:
            continue
        for atoms in _plan_group(selection, o):
            if not apply(atoms):
                return None
            before.append(atoms)

    static = [
        pair
        for pair in pairs
        if id(pair) not in ready_keys and pair[0] not in parked and pair[1] not in parked
    ]
    targets: dict[int, Site] = {}
    visitors: set[int] = set()
    if static:
        options = [assign_targets(static, cur, "compact"), assign_targets(static, cur, "flat")]
        best = None
        for option in options:
            option_targets, option_visitors, _, overflow = option
            if overflow:
                return None
            if any(q not in option_targets for pair in static for q in (pair[0], pair[1])):
                continue
            cost = sum(max(_trip(cur[q], option_targets[q]) for q in (pair[0], pair[1])) for pair in static)
            if best is None or cost < best[0] - 1e-12:
                best = (cost, option_targets, option_visitors)
        if best is None:
            return None
        targets.update(best[1])
        visitors.update(best[2])
    for q, (pair, other_site) in parked.items():
        targets[q] = cur[q]
        partner = pair[1] if pair[0] == q else pair[0]
        targets[partner] = other_site

    lifting = {q: (cur[q], targets[q]) for q in targets if cur[q] != targets[q]}
    for parity in (1, 0):
        selection = {q: v for q, v in lifting.items() if v[0][0] % 2 == parity}
        if not selection:
            continue
        for atoms in _plan_group(selection, o):
            if not apply(atoms):
                return None
            after.append(("move", atoms))

    for pair in pairs:
        u, v = pair[0], pair[1]
        if o.get(targets.get(u, cur[u])) != u or o.get(targets.get(v, cur[v])) != v:
            return None
        after.append(("cz", pair))
    occupied = _occupied_pairs(o)
    if not occupied <= {frozenset(pair[:2]) for pair in pairs}:
        return None
    # every interaction pair that holds two atoms gets exactly one pulse this segment
    counts = {}
    for entry in after:
        if entry[0] == "cz":
            counts[frozenset(entry[1][:2])] = counts.get(frozenset(entry[1][:2]), 0) + 1
    if any(counts.get(pair, 0) != 1 for pair in occupied):
        return None
    return before, after


def _trip(src: Site, dst: Site) -> float:
    return math.dist(_physical(src), _physical(dst))


def _static_actions(pairs, current, home, occ):
    """Today's plan: lower everything that is up, then lift/cz/lower the round from `home`."""
    cur = dict(current)
    o = dict(occ)
    before = []
    dropping = {q: (cur[q], home[q]) for q in sorted(cur) if cur[q][1] >= ENT_Y0}
    for parity in (1, 0):
        selection = {q: v for q, v in dropping.items() if v[0][0] % 2 == parity}
        if not selection:
            continue
        for atoms in _plan_group(selection, o):
            if move_params(atoms, o) is None:
                return None
            for q, (src, _) in atoms.items():
                o.pop(src, None)
            for q, (_, dst) in atoms.items():
                o[dst] = q
                cur[q] = dst
            before.append(atoms)
    after = []
    for up, czs, down in plan_segments(pairs, cur):
        for atoms in up:
            if move_params(atoms, o) is None:
                return None
            for q, (src, _) in atoms.items():
                o.pop(src, None)
            for q, (_, dst) in atoms.items():
                o[dst] = q
                cur[q] = dst
            after.append(("move", atoms))
        for pair in czs:
            after.append(("cz", pair))
        for atoms in down:
            if move_params(atoms, o) is None:
                return None
            for q, (src, _) in atoms.items():
                o.pop(src, None)
            for q, (_, dst) in atoms.items():
                o[dst] = q
                cur[q] = dst
            after.append(("move", atoms))
    return before, after


def _emit_parked(native: QuantumCircuit, rounds, home: dict[int, Site]):
    """Emit the whole circuit keeping atoms in the zone across rounds where that pays."""
    out = native.copy_empty_like()
    qbits = out.qubits
    cbits = out.clbits
    current = dict(home)
    occ = {site: q for q, site in current.items()}
    distance = 0.0

    def do_move(atoms):
        nonlocal distance
        params = move_params(atoms, occ)
        if params is None:
            raise AssertionError("parked emission planned an illegal move")
        for q, (src, _) in atoms.items():
            occ.pop(src, None)
        for q, (_, dst) in atoms.items():
            occ[dst] = q
            current[q] = dst
        _move(out, list(atoms), *params)
        distance += move_cost(atoms)

    moved = True
    for index, phase in enumerate(rounds):
        pairs = phase.pairs
        ops_qubits = set()
        for op, qargs, cargs in phase.ops:
            if op.name != "measure":
                ops_qubits.update(qargs)
        plan = None
        if PARK and pairs:
            plan = _park_actions(pairs, current, home, occ, ops_qubits)
        if plan is None:
            plan = _static_actions(pairs, current, home, occ)
            if plan is None:
                break
        before, after = plan
        if pairs and not moved and not before and not any(entry[0] == "move" for entry in after):
            # two cz segments with nothing in between: break the segment unless the pair set matches
            if not _occupied_pairs(occ) <= {frozenset(pair[:2]) for pair in pairs}:
                spare = [q for pair in pairs for q in (pair[0], pair[1]) if q not in ops_qubits and current[q][1] >= ENT_Y0]
                if spare:
                    q = spare[0]
                    site = current[q]
                    down = {q: (site, home[q])}
                    if move_params(down, occ) is not None:
                        do_move(down)
                        up = {q: (home[q], site)}
                        if move_params(up, occ) is not None:
                            do_move(up)
        for atoms in before:
            do_move(atoms)
        for op, qargs, cargs in phase.ops:
            out.append(op, [qbits[q] for q in qargs], [cbits[c] for c in cargs])
        for entry in after:
            if entry[0] == "move":
                do_move(entry[1])
                moved = True
            else:
                u, v = entry[1][0], entry[1][1]
                out.append(entry[1][2], [qbits[u], qbits[v]])
                moved = False
    return out, distance


# ------------------------------------------------------------------ 8. orchestrator


def _emit_move(out: QuantumCircuit, atoms: dict[int, tuple[Site, Site]]) -> None:
    rows, cols, new_rows, new_cols = move_params(atoms, {})
    _move(out, list(atoms), rows, cols, new_rows, new_cols)


DEPTH_WEIGHT = 2.922
"""Score-neutral exchange rate between one unit of moved distance and one unit of depth."""


def _emit_static(native: QuantumCircuit, rounds, pos):
    """Emit today's plan: lift every round's atoms, one `cz` per pair, lower them again."""
    out = native.copy_empty_like()
    qbits = out.qubits
    cbits = out.clbits
    distance = 0.0
    for phase in rounds:
        for op, qargs, cargs in phase.ops:
            out.append(op, [qbits[q] for q in qargs], [cbits[c] for c in cargs])
        for up, czs, down in plan_segments(phase.pairs, pos):
            for atoms in up:
                _emit_move(out, atoms)
                distance += move_cost(atoms)
            for u, v, op in czs:
                out.append(op, [qbits[u], qbits[v]])
            for atoms in down:
                _emit_move(out, atoms)
                distance += move_cost(atoms)
    return out, distance


def _candidate(native: QuantumCircuit, n: int, division: int = 1):
    """Plan and emit one candidate; returns the best `(objective, circuit, home placement)`."""
    rounds = schedule_rounds(native)
    pos = initial_placement(n, rounds)
    pos = optimise_placement(rounds, pos, _budget(n, len(rounds)) // division)
    depth = native.depth()
    out, distance = _emit_static(native, rounds, pos)
    best = (distance + DEPTH_WEIGHT * depth, out, pos)
    if PARK and rounds:
        try:
            parked, parked_distance = _emit_parked(native, rounds, pos)
        except AssertionError:
            return best
        objective = parked_distance + DEPTH_WEIGHT * depth
        if objective < best[0] - 1e-9:
            best = (objective, parked, pos)
    return best


def compile_circuit(circuit: QuantumCircuit, architecture_spec: Any):
    n = circuit.num_qubits
    plain = native_circuit(circuit)
    best = _candidate(plain, n)
    oriented = native_circuit(zz_orient(circuit))
    if _native_key(oriented) != _native_key(plain):
        alternative = _candidate(oriented, n, 2)
        if alternative[0] < best[0] - 1e-9:
            best = alternative
    return best[1], Architecture(best[2], spec=architecture_spec)


# ================================================================ postopt (compilers/postopt.py)
"""postopt: generic move post-optimizer for any compiler's output (validator v5 rules).

Between two CZ layers ("gaps") only three things matter: where every atom is at the first layer,
where it is at the next layer, and the 1q gates applied in between (they need the atom in
storage). Each gap is re-planned from scratch and the new plan is kept only if it is cheaper:

  1. drop: atoms in the zone that owe 1q gates (or must leave the zone) go to storage, as one block
     (uniform row shift, smallest that fits) where possible;
  2. the gap's 1q gates (per-qubit order kept);
  3. every atom moves to its end position, grouped into legal rectangles.

Gates, their per-qubit order and the CZ layers are untouched, so depth and equivalence are
unchanged. The result is replayed with the validator's rules (single-zone moves, idle rule, CZ on
pairs, 1q gates in storage) before it is returned; on any doubt the input is returned unchanged.

Self-contained (stdlib + qiskit), appended to a component by `bench/make_postopt.py`.
"""


import itertools
import math

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction


class _PoMove(Instruction):
    def __init__(self, rows, cols, new_rows, new_cols, num_qubits):
        super().__init__("move", num_qubits, 0, [tuple(rows), tuple(cols), tuple(new_rows), tuple(new_cols)])

    def validate_parameter(self, parameter):
        return parameter

    def _define(self):
        self.definition = QuantumCircuit(self.num_qubits, name=self.name)


class _PoGeo:
    def __init__(self, spec):
        zones = (spec or {}).get("zones") if isinstance(spec, dict) else None
        if not zones:
            zones = {
                "storage": {"x": [0, 9], "y": [0, 9], "spacing": 1.0, "operations": ["rx", "rz"]},
                "entangling": {"x": [0, 9], "y": [10, 19], "spacing": 1.0,
                               "pairs": {"axis": "x", "distance": 0.3}, "operations": ["cz"]},
            }
        self.zone_of, self.phys = {}, {}
        self.ent = self.sto = None
        for name, z in reversed(list(zones.items())):
            (x0, x1), (y0, y1) = z["x"], z["y"]
            sp = float(z.get("spacing", 1.0))
            pairs = z.get("pairs")
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    self.zone_of[(x, y)] = name
                    px = self._c(x, x0, sp, pairs, "x")
                    py = self._c(y, y0, sp, pairs, "y")
                    self.phys[(x, y)] = (px, py)
        for name, z in zones.items():
            if z.get("pairs"):
                self.ent = name
                self.ex0, self.ey0, self.ey1 = z["x"][0], z["y"][0], z["y"][1]
            elif "rx" in z.get("operations", []):
                self.sto = name
                self.sy0, self.sy1 = z["y"]
        self.storage_sites = [s for s, n in self.zone_of.items() if n == self.sto]

    @staticmethod
    def _c(line, start, sp, pairs, axis):
        if not pairs or pairs.get("axis") != axis:
            return line * sp
        p, s = divmod(line - start, 2)
        return start * sp + p * 2 * sp + s * float(pairs["distance"])

    def dist(self, a, b):
        (ax, ay), (bx, by) = self.phys[a], self.phys[b]
        return math.hypot(ax - bx, ay - by)

    def in_ent(self, s):
        return self.zone_of.get(s) == self.ent

    def in_sto(self, s):
        return self.zone_of.get(s) == self.sto

    def sibling(self, s):
        x, y = s
        return (x + 1, y) if (x - self.ex0) % 2 == 0 else (x - 1, y)


class _PoSt:
    def __init__(self, geo, pos):
        self.geo, self.pos = geo, dict(pos)
        self.occ = {s: q for q, s in self.pos.items()}

    def copy(self):
        st = _PoSt.__new__(_PoSt)
        st.geo, st.pos, st.occ = self.geo, dict(self.pos), dict(self.occ)
        return st

    def apply(self, group):
        """group = [(q, target)]; validator semantics; returns ((rows, cols, nr, nc, qs), dist)."""
        g = self.geo
        src = {q: self.pos[q] for q, _ in group}
        tgt = dict(group)
        xs = [s[0] for s in src.values()]
        ys = [s[1] for s in src.values()]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        mem = set(src)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                o = self.occ.get((x, y))
                if o is not None and o not in mem:
                    raise ValueError("ghost")
        rmap, cmap = {}, {}
        for q in mem:
            (sx, sy), (tx, ty) = src[q], tgt[q]
            if rmap.setdefault(sy, ty) != ty or cmap.setdefault(sx, tx) != tx:
                raise ValueError("not a product map")
        for mp in (rmap, cmap):
            ks = sorted(mp)
            if any(mp[a] >= mp[b] for a, b in itertools.pairwise(ks)):
                raise ValueError("order")
        if len({g.zone_of[s] for s in src.values()}) != 1 or len({g.zone_of.get(t) for t in tgt.values()}) != 1:
            raise ValueError("zones")
        if None in {g.zone_of.get(t) for t in tgt.values()}:
            raise ValueError("outside")
        for q in mem:
            o = self.occ.get(tgt[q])
            if o is not None and o not in mem:
                raise ValueError("occupied")
        d = max(g.dist(src[q], tgt[q]) for q in mem)
        for q in mem:
            del self.occ[src[q]]
        for q in mem:
            self.pos[q] = tgt[q]
            self.occ[tgt[q]] = q
        return (tuple(range(y0, y1 + 1)), tuple(range(x0, x1 + 1)),
                tuple(rmap[y] for y in sorted(rmap)), tuple(cmap[x] for x in sorted(cmap)), sorted(mem)), d


def _po_sgn(v):
    return (v > 0) - (v < 0)


def _po_compat(a, b):
    (_, (ax, ay), (atx, aty)), (_, (bx, by), (btx, bty)) = a, b
    return _po_sgn(ax - bx) == _po_sgn(atx - btx) and _po_sgn(ay - by) == _po_sgn(aty - bty)


def _po_group(st, transfers):
    """Grouped legal moves for transfers [(q, src, dst)] (largest displacement seeds a group)."""
    g = st.geo
    todo = {q: (q, s, d) for q, s, d in transfers if s != d}
    out = []
    guard = 0
    while todo:
        guard += 1
        if guard > 6 * len(transfers) + 30:
            raise ValueError("no progress")
        cands = sorted(todo.values(), key=lambda t: -g.dist(t[1], t[2]))
        seed = next((t for t in cands if st.occ.get(t[2]) is None), None)
        if seed is None:
            t = cands[0]
            q = st.occ[t[2]]
            if q not in todo:
                raise ValueError("target held by a stationary atom")
            pool = [s for s in g.zone_of if g.zone_of[s] == g.zone_of[st.pos[q]] and s not in st.occ]
            spot = min(pool, key=lambda s: g.dist(st.pos[q], s))
            out.append(st.apply([(q, spot)]))
            todo[q] = (q, spot, todo[q][2])
            continue
        zs, zt = g.zone_of[seed[1]], g.zone_of[seed[2]]

        def fits(t, grp):
            return g.zone_of[t[1]] == zs and g.zone_of[t[2]] == zt and all(_po_compat(t, u) for u in grp)

        grp = [seed] + [t for t in cands if t is not seed]
        grp = [seed] + [t for t in grp[1:] if fits(t, [seed])]
        keep = [seed]
        for t in grp[1:]:
            if fits(t, keep):
                keep.append(t)
        grp, excl = keep, set()
        sx, sy = seed[1]
        while True:
            mem = {u[0] for u in grp}
            xs = [u[1][0] for u in grp]
            ys = [u[1][1] for u in grp]
            ghosts = [st.occ[(x, y)] for x in range(min(xs), max(xs) + 1) for y in range(min(ys), max(ys) + 1)
                      if (x, y) in st.occ and st.occ[(x, y)] not in mem]
            held = [u for u in grp if st.occ.get(u[2]) is not None and st.occ.get(u[2]) not in mem]
            if not ghosts and not held:
                break
            pulled = False
            for q in ghosts:
                if q in todo and q not in excl and fits(todo[q], grp):
                    grp.append(todo[q])
                    pulled = True
            if pulled:
                continue
            if held:
                excl.update(u[0] for u in held)
                grp = [u for u in grp if u not in held]
                continue
            far = max(grp[1:], key=lambda u: abs(u[1][0] - sx) + abs(u[1][1] - sy))
            excl.add(far[0])
            grp.remove(far)
        out.append(st.apply([(u[0], u[2]) for u in grp]))
        for u in grp:
            del todo[u[0]]
    return out


def _po_block_drop(st, qs):
    """Targets for dropping zone atoms `qs` as one block: zone row r -> storage row r - shift (same
    column), smallest shift with all targets free. None if no shift fits."""
    g = st.geo
    if not qs:
        return {}
    for shift in range(1, g.ey1 - g.sy0 + 1):
        tg = {}
        ok = True
        for q in qs:
            x, y = st.pos[q]
            t = (x, y - shift)
            if not g.in_sto(t) or (t in st.occ and st.occ[t] not in qs):
                ok = False
                break
            tg[q] = t
        if ok and len(set(tg.values())) == len(tg):
            return tg
    return None


def _po_replan_gap(geo, start, end, gates, n):
    """Cheapest of: drop only the atoms that must leave the zone, or drop the whole zone as a block."""
    best = None
    for whole in (False, True):
        try:
            r = _po_replan_gap1(geo, start, end, gates, n, whole)
        except (ValueError, KeyError):
            r = None
        if r is not None and (best is None or r[1] < best[1]):
            best = r
    return best


def _po_replan_gap1(geo, start, end, gates, n, whole):
    """New item list for one gap, or None. gates: list of (instruction, qubits) 1q gates in order."""
    st = _PoSt(geo, start)
    owe = {q for _, qs in gates for q in qs}
    items, cost = [], 0.0
    drop = [q for q in range(n) if geo.in_ent(st.pos[q]) and (whole or q in owe or st.pos[q] != end[q])]
    tg = _po_block_drop(st, drop)
    if tg is None:
        tg = {}
        taken = set()
        for q in drop:
            x, _ = st.pos[q]
            col = [(x, y) for y in range(geo.sy1, geo.sy0 - 1, -1) if (x, y) not in st.occ and (x, y) not in taken]
            if not col:
                return None
            tg[q] = col[0]
            taken.add(col[0])
    # atoms whose end is in storage go there directly if that site is free now
    for q in drop:
        e = end[q]
        if geo.in_sto(e) and q not in owe and e not in st.occ and e not in tg.values():
            tg[q] = e
    for params, d in _po_group(st, [(q, st.pos[q], t) for q, t in tg.items()]):
        items.append(("move", params))
        cost += d
    for ins, qs in gates:
        if not all(geo.in_sto(st.pos[q]) for q in qs):
            return None
        items.append(("gate", ins))
    for params, d in _po_group(st, [(q, st.pos[q], end[q]) for q in range(n) if st.pos[q] != end[q]]):
        items.append(("move", params))
        cost += d
    if any(st.pos[q] != end[q] for q in range(n)):
        return None
    return items, cost


def _po_replay_ok(geo, loading, circuit) -> bool:
    """Validator rules: legal moves, 1q in storage, cz on pairs, idle rule, equal pulses per layer."""
    st = _PoSt(geo, loading)
    qi = {b: i for i, b in enumerate(circuit.qubits)}
    layer: dict = {}

    def close():
        if not layer:
            return True
        if len(set(layer.values())) != 1:
            return False
        active = {q for k in layer for q in k}
        for q, s in st.pos.items():
            if geo.in_ent(s) and q not in active:
                return False
        return True

    for ins in circuit.data:
        op = ins.operation
        qs = [qi[q] for q in ins.qubits]
        if op.name == "move":
            if not close():
                return False
            layer.clear()
            rows, cols, nr, nc = op.params
            occ_rows = [y for y in rows if any(st.pos[q][1] == y for q in qs)]
            occ_cols = [x for x in cols if any(st.pos[q][0] == x for q in qs)]
            if len(nr) == len(rows):
                occ_rows = list(rows)
            if len(nc) == len(cols):
                occ_cols = list(cols)
            rm, cm = dict(zip(occ_rows, nr)), dict(zip(occ_cols, nc))
            try:
                st.apply([(q, (cm[st.pos[q][0]], rm[st.pos[q][1]])) for q in qs])
            except (ValueError, KeyError):
                return False
        elif op.name in ("rx", "rz"):
            if not geo.in_sto(st.pos[qs[0]]):
                return False
        elif op.name == "cz":
            a, b = (st.pos[q] for q in qs)
            if not (geo.in_ent(a) and geo.sibling(a) == b):
                return False
            k = frozenset(qs)
            layer[k] = layer.get(k, 0) + 1
    return close()


def postoptimize(circuit: QuantumCircuit, loading: dict, spec=None) -> QuantumCircuit:
    try:
        geo = _PoGeo(spec)
        n = circuit.num_qubits
        qi = {b: i for i, b in enumerate(circuit.qubits)}
        data = list(circuit.data)
        # positions before every instruction (replay)
        st = _PoSt(geo, loading)
        layers = []  # (start_index, end_index) of each CZ layer (consecutive czs between moves)
        pos_at = []
        cur = None
        for idx, ins in enumerate(data):
            op = ins.operation
            qs = [qi[q] for q in ins.qubits]
            if op.name == "cz":
                if cur is None:
                    cur = [idx, idx]
                    pos_at.append(dict(st.pos))
                else:
                    cur[1] = idx
                continue
            if op.name == "move":
                if cur is not None:
                    layers.append(tuple(cur))
                    cur = None
                rows, cols, nr, nc = op.params
                occ_rows = [y for y in rows if any(st.pos[q][1] == y for q in qs)]
                occ_cols = [x for x in cols if any(st.pos[q][0] == x for q in qs)]
                if len(nr) == len(rows):
                    occ_rows = list(rows)
                if len(nc) == len(cols):
                    occ_cols = list(cols)
                rm, cm = dict(zip(occ_rows, nr)), dict(zip(occ_cols, nc))
                st.apply([(q, (cm[st.pos[q][0]], rm[st.pos[q][1]])) for q in qs])
        if cur is not None:
            layers.append(tuple(cur))
        if len(layers) < 2:
            return circuit
        out = circuit.copy_empty_like()
        prev_end = -1
        # gap before the first layer and after the last layer are kept as they are
        keep_prefix = data[: layers[0][0]]
        for ins in keep_prefix:
            out.append(ins)
        for k in range(len(layers)):
            s, e = layers[k]
            for ins in data[s: e + 1]:
                out.append(ins)
            if k + 1 == len(layers):
                for ins in data[e + 1:]:
                    out.append(ins)
                break
            s2 = layers[k + 1][0]
            gap = data[e + 1: s2]
            moves = [ins for ins in gap if ins.operation.name == "move"]
            others = [ins for ins in gap if ins.operation.name != "move"]
            if any(ins.operation.name not in ("rx", "rz", "barrier") for ins in others):
                for ins in gap:
                    out.append(ins)
                continue
            # original cost of the gap
            st0 = _PoSt(geo, pos_at[k])
            old = 0.0
            ok = True
            for ins in moves:
                qs = [qi[q] for q in ins.qubits]
                rows, cols, nr, nc = ins.operation.params
                occ_rows = [y for y in rows if any(st0.pos[q][1] == y for q in qs)]
                occ_cols = [x for x in cols if any(st0.pos[q][0] == x for q in qs)]
                if len(nr) == len(rows):
                    occ_rows = list(rows)
                if len(nc) == len(cols):
                    occ_cols = list(cols)
                rm, cm = dict(zip(occ_rows, nr)), dict(zip(occ_cols, nc))
                try:
                    _, d = st0.apply([(q, (cm[st0.pos[q][0]], rm[st0.pos[q][1]])) for q in qs])
                except (ValueError, KeyError):
                    ok = False
                    break
                old += d
            gates = [(ins, [qi[q] for q in ins.qubits]) for ins in others if ins.operation.name in ("rx", "rz")]
            new = _po_replan_gap(geo, pos_at[k], pos_at[k + 1], gates, n) if ok else None
            if new is None or new[1] >= old - 1e-9:
                for ins in gap:
                    out.append(ins)
                continue
            for kind, val in new[0]:
                if kind == "gate":
                    out.append(val)
                else:
                    rows, cols, nr, nc, qs = val
                    out.append(_PoMove(rows, cols, nr, nc, len(qs)), [out.qubits[q] for q in qs])
        if not _po_replay_ok(geo, loading, out):
            return circuit
        return out
    except Exception:  # noqa: BLE001 - never break a valid compilation
        return circuit

# ---------------------------------------------------------------- post-optimizer wrapper (bench/make_postopt.py)
_po_base_compile_circuit = compile_circuit


def compile_circuit(circuit, architecture_spec):
    out, arch = _po_base_compile_circuit(circuit, architecture_spec)
    try:
        out = postoptimize(out, arch.initial_loading, architecture_spec)
    except Exception:  # noqa: BLE001 - keep the component's own (valid) result
        pass
    return out, arch
