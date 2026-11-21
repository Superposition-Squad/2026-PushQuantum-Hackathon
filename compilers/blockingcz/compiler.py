"""blockingcz: partner's compiler (CZ-only, blocking-CZ / idle-rule aware, iHVA global S-conjugation,
Pauli-commutation DAG matchings, 2-leaf hub sweep). Extracted verbatim from the partner's marimo notebook
(original: original_readme.py, `compile_circuit` cell); the function is fully self-contained."""

def compile_circuit(circuit, architecture_spec=None):
    """Self-contained single-cell `compile_circuit` for copy-pasting into Marimo `readme.py`."""
    import itertools
    import json
    import math
    import random
    from collections import Counter, defaultdict
    from collections.abc import Sequence
    from numbers import Integral
    from pathlib import Path
    from typing import Any

    import numpy as np
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Instruction
    from qiskit.synthesis import OneQubitEulerDecomposer

    Site = tuple[int, int]
    ArchitectureSpec = str | Path | dict[str, Any]

    _EULER_ZXZ = OneQubitEulerDecomposer(basis="ZXZ")
    _TWO_PI = 2.0 * math.pi
    _TOL = 1e-11


    # ---------------------------------------------------------------------------
    # Architecture & Move definitions (self-contained, compatible with validator)
    # ---------------------------------------------------------------------------


    class Zone:
        def __init__(
            self,
            name: str,
            x: tuple[int, int],
            y: tuple[int, int],
            operations: frozenset[str],
            spacing: float = 1.0,
            pair_axis: str | None = None,
            pair_distance: float | None = None,
        ) -> None:
            self.name = name
            self.x = x
            self.y = y
            self.operations = operations
            self.spacing = spacing
            self.pair_axis = pair_axis
            self.pair_distance = pair_distance


    def default_spec() -> dict[str, Any]:
        return {
            "loading_zone": "storage",
            "zones": {
                "storage": {
                    "x": [0, 9],
                    "y": [0, 9],
                    "spacing": 1.0,
                    "operations": ["rx", "rz"],
                },
                "entangling": {
                    "x": [0, 9],
                    "y": [10, 19],
                    "spacing": 1.0,
                    "pairs": {"axis": "x", "distance": 0.3},
                    "operations": ["cz"],
                },
            },
        }


    def load_spec(spec: ArchitectureSpec | None = None) -> dict[str, Any]:
        if spec is None:
            return default_spec()
        if isinstance(spec, dict):
            return spec
        return json.loads(Path(spec).read_text(encoding="utf-8"))


    def parse_zone(name: str, z: dict[str, Any]) -> Zone:
        pairs = z.get("pairs")
        return Zone(
            name=name,
            x=(int(z["x"][0]), int(z["x"][1])),
            y=(int(z["y"][0]), int(z["y"][1])),
            operations=frozenset(z.get("operations", ())),
            spacing=float(z.get("spacing", 1.0)),
            pair_axis=pairs["axis"] if pairs else None,
            pair_distance=float(pairs["distance"]) if pairs else None,
        )


    class Architecture:
        """The zones of `spec` and `initial_loading` (qubit `0..n-1` -> `(x, y)` site)."""

        def __init__(self, initial_loading: dict[int, Site], spec: ArchitectureSpec | None = None) -> None:
            data = load_spec(spec)
            self.zones: dict[str, Zone] = {name: parse_zone(name, z) for name, z in data["zones"].items()}
            self.loading_zone: str = data.get("loading_zone", "storage")
            self.initial_loading: dict[int, Site] = {int(k): (int(v[0]), int(v[1])) for k, v in initial_loading.items()}


    def as_lines(values: int | Sequence[int], name: str) -> tuple[int, ...]:
        items = [values] if isinstance(values, Integral) else list(values)
        if not items:
            raise ValueError(f"`{name}` must not be empty.")
        result = tuple(int(v) for v in items)
        if len(set(result)) != len(result):
            raise ValueError(f"`{name}` must not contain duplicates, got {result!r}.")
        return result


    def check_lines(start: tuple[int, ...], end: tuple[int, ...], name: str) -> None:
        if len(end) > len(start):
            raise ValueError(f"`new_{name}` cannot have more lines than `{name}`.")
        if start != tuple(range(start[0], start[0] + len(start))):
            raise ValueError(f"Picked-up {name} {start} must be contiguous and ascending.")
        if any(b <= a for a, b in itertools.pairwise(end)):
            raise ValueError(f"Target {name} {end} must be ascending.")


    class Move(Instruction):
        """Params are `(rows, cols, new_rows, new_cols)`, tuples of ints; rows are y, columns are x."""

        def __init__(
            self,
            rows: int | Sequence[int],
            cols: int | Sequence[int],
            new_rows: int | Sequence[int],
            new_cols: int | Sequence[int],
            num_qubits: int,
        ) -> None:
            names = ("rows", "cols", "new_rows", "new_cols")
            lines = [as_lines(v, n) for v, n in zip((rows, cols, new_rows, new_cols), names, strict=True)]
            check_lines(lines[0], lines[2], "rows")
            check_lines(lines[1], lines[3], "cols")
            if num_qubits < 1:
                raise ValueError("A move must act on at least one qubit.")
            super().__init__("move", num_qubits, 0, lines)

        rows = property(lambda self: self.params[0])
        cols = property(lambda self: self.params[1])
        new_rows = property(lambda self: self.params[2])
        new_cols = property(lambda self: self.params[3])

        def validate_parameter(self, parameter: object) -> tuple[int, ...]:
            if isinstance(parameter, tuple) and all(isinstance(v, Integral) for v in parameter):
                return parameter
            raise TypeError(f"Move params must be tuples of ints, got {parameter!r}.")

        def _define(self) -> None:
            self.definition = QuantumCircuit(self.num_qubits, name=self.name)


    # ---------------------------------------------------------------------------
    # Geometry & Validator-Mirror GridState
    # ---------------------------------------------------------------------------


    def zone_contains(zone: Zone, site: Site) -> bool:
        return zone.x[0] <= site[0] <= zone.x[1] and zone.y[0] <= site[1] <= zone.y[1]


    def pair_slot(zone: Zone, line: int) -> int:
        return (line - (zone.x[0] if zone.pair_axis == "x" else zone.y[0])) % 2


    def are_paired(zone: Zone, a: Site, b: Site) -> bool:
        if zone.pair_axis is None or not (zone_contains(zone, a) and zone_contains(zone, b)):
            return False
        axis = 0 if zone.pair_axis == "x" else 1
        low, high = (a[axis], b[axis]) if a[axis] <= b[axis] else (b[axis], a[axis])
        return a[1 - axis] == b[1 - axis] and high == low + 1 and pair_slot(zone, low) == 0


    def physical_line(zone: Zone, line: int, axis: str) -> float:
        if axis != zone.pair_axis or zone.pair_distance is None:
            return line * zone.spacing
        start = zone.x[0] if axis == "x" else zone.y[0]
        pair, slot = divmod(line - start, 2)
        return start * zone.spacing + pair * 2.0 * zone.spacing + slot * zone.pair_distance


    class ArchGeometry:
        def __init__(
            self,
            zones: dict[str, Zone],
            loading_zone: str,
            storage: Zone,
            entangling: Zone,
            zone_at: dict[Site, Zone],
            phys_cache: dict[Site, tuple[float, float]],
            has_rzz: bool,
        ) -> None:
            self.zones = zones
            self.loading_zone = loading_zone
            self.storage = storage
            self.entangling = entangling
            self.zone_at = zone_at
            self.phys_cache = phys_cache
            self.has_rzz = has_rzz

        @classmethod
        def from_spec(cls, spec: ArchitectureSpec | None) -> "ArchGeometry":
            data = load_spec(spec)
            zones = {name: parse_zone(name, z) for name, z in data["zones"].items()}
            loading_zone = data.get("loading_zone", "storage")
            storage = zones.get(loading_zone) or next(z for z in zones.values() if "rx" in z.operations or "rz" in z.operations)
            entangling = zones.get("entangling") or next(z for z in zones.values() if "cz" in z.operations or "rzz" in z.operations)
            zone_at: dict[Site, Zone] = {}
            for z in reversed(list(zones.values())):
                for x in range(z.x[0], z.x[1] + 1):
                    for y in range(z.y[0], z.y[1] + 1):
                        zone_at[(x, y)] = z
            phys_cache = {
                s: (physical_line(z, s[0], "x"), physical_line(z, s[1], "y"))
                for s, z in zone_at.items()
            }
            return cls(
                zones=zones,
                loading_zone=loading_zone,
                storage=storage,
                entangling=entangling,
                zone_at=zone_at,
                phys_cache=phys_cache,
                has_rzz=False,
            )

        def phys(self, site: Site) -> tuple[float, float]:
            return self.phys_cache[site]

        def dist(self, a: Site, b: Site) -> float:
            pa, pb = self.phys_cache[a], self.phys_cache[b]
            return math.hypot(pa[0] - pb[0], pa[1] - pb[1])

        def partner_site(self, site: Site) -> Site | None:
            z = self.zone_at.get(site)
            if z is None or z.pair_axis is None:
                return None
            if z.pair_axis == "x":
                start = z.x[0]
                px = start + ((site[0] - start) ^ 1)
                cand = (px, site[1])
            else:
                start = z.y[0]
                py = start + ((site[1] - start) ^ 1)
                cand = (site[0], py)
            return cand if zone_contains(z, cand) else None


    class GridState:
        """Exact in-memory validator mirror tracking positions, occupancy, blocking CZ layers, single-zone moves, distance, and DAG depth."""

        def __init__(self, geom: ArchGeometry, initial_loading: dict[int, Site]) -> None:
            self.geom = geom
            self.initial_loading = dict(initial_loading)
            self.positions: dict[int, Site] = dict(initial_loading)
            self.occupant: dict[Site, int] = {s: q for q, s in initial_loading.items()}
            self.total_distance: float = 0.0
            self.q_depth: list[int] = [0] * len(initial_loading)
            self._layer_2q_qubits: set[int] = set()
            self._layer_2q_zones: set[str] = set()
            # ops: ("move", qubits_tuple, (rows, cols, new_rows, new_cols)) or ("gate", name, qubits_tuple, params_tuple)
            self.ops: list[tuple[Any, ...]] = []

        def clone(self) -> "GridState":
            g = GridState.__new__(GridState)
            g.geom = self.geom
            g.initial_loading = self.initial_loading
            g.positions = dict(self.positions)
            g.occupant = dict(self.occupant)
            g.total_distance = self.total_distance
            g.q_depth = list(self.q_depth)
            g._layer_2q_qubits = set(self._layer_2q_qubits)
            g._layer_2q_zones = set(self._layer_2q_zones)
            g.ops = list(self.ops)
            g._raw_spec = getattr(self, "_raw_spec", None)
            return g

        @property
        def depth(self) -> int:
            return max(self.q_depth, default=0)

        def _close_layer(self) -> None:
            """Enforce blocking 2Q gate rule: every atom in a zone where a 2Q gate ran must participate."""
            if not self._layer_2q_zones:
                return
            for zname in self._layer_2q_zones:
                z = self.geom.zones[zname]
                idle = [q for q, s in self.positions.items() if zone_contains(z, s) and q not in self._layer_2q_qubits]
                if idle:
                    raise RuntimeError(f"Idle atoms in {zname!r} during 2Q gate layer: {idle}")
            self._layer_2q_qubits.clear()
            self._layer_2q_zones.clear()

        def move(
            self,
            rows: Sequence[int] | int,
            cols: Sequence[int] | int,
            new_rows: Sequence[int] | int,
            new_cols: Sequence[int] | int,
        ) -> float:
            self._close_layer()
            r_t = as_lines(rows, "rows")
            c_t = as_lines(cols, "cols")
            nr_t = as_lines(new_rows, "new_rows")
            nc_t = as_lines(new_cols, "new_cols")
            check_lines(r_t, nr_t, "rows")
            check_lines(c_t, nc_t, "cols")

            r_set = set(r_t)
            c_set = set(c_t)
            selected = [s for s in self.occupant if s[0] in c_set and s[1] in r_set]
            if not selected:
                raise RuntimeError("Move selected 0 qubits.")
            if len(nr_t) < len(r_t):
                used_r = tuple(sorted({s[1] for s in selected}))
                if len(used_r) != len(nr_t):
                    raise RuntimeError(f"Occupied rows {used_r} != target {nr_t}")
                kept_r = used_r
            else:
                kept_r = r_t
            if len(nc_t) < len(c_t):
                used_c = tuple(sorted({s[0] for s in selected}))
                if len(used_c) != len(nc_t):
                    raise RuntimeError(f"Occupied cols {used_c} != target {nc_t}")
                kept_c = used_c
            else:
                kept_c = c_t

            rmap = dict(zip(kept_r, nr_t))
            cmap = dict(zip(kept_c, nc_t))
            mapping: dict[Site, Site] = {}
            for x in kept_c:
                for y in kept_r:
                    tgt = (cmap[x], rmap[y])
                    if tgt not in self.geom.zone_at:
                        raise RuntimeError(f"Target {tgt} outside every zone.")
                    mapping[(x, y)] = tgt

            picked_sites = [s for s in selected]
            picked_set = set(picked_sites)
            src_zones = {self.geom.zone_at[s].name for s in picked_sites}
            if len(src_zones) > 1:
                raise RuntimeError(f"Move picks up atoms from multiple zones: {src_zones}")
            dst_zones = {self.geom.zone_at[mapping[s]].name for s in picked_sites}
            if len(dst_zones) > 1:
                raise RuntimeError(f"Move drops off atoms in multiple zones: {dst_zones}")
            qubits = tuple(self.occupant[s] for s in picked_sites)
            for s in picked_sites:
                tgt = mapping[s]
                occ_tgt = self.occupant.get(tgt)
                if occ_tgt is not None and tgt not in picked_set:
                    raise RuntimeError(f"Collision at {tgt}: occupied by qubit {occ_tgt}")

            d = 0.0
            for s in picked_sites:
                d = max(d, self.geom.dist(s, mapping[s]))
            for s in picked_sites:
                del self.occupant[s]
            for q, s in zip(qubits, picked_sites):
                ns = mapping[s]
                self.positions[q] = ns
                self.occupant[ns] = q

            self.total_distance += d
            before_pairs = tuple((q, s) for q, s in zip(qubits, picked_sites))
            self.ops.append((
                "move",
                qubits,
                (
                    r_t,
                    c_t,
                    nr_t,
                    nc_t,
                    tuple(sorted({s[1] for s in picked_sites})),
                    tuple(sorted({s[0] for s in picked_sites})),
                    before_pairs,
                ),
            ))
            return d

        def move_site(self, q: int, target: Site) -> float:
            s = self.positions[q]
            if s == target:
                return 0.0
            return self.move((s[1],), (s[0],), (target[1],), (target[0],))

        def gate(self, name: str, qubits: Sequence[int], params: Sequence[float] = ()) -> None:
            if name == "rzz":
                raise RuntimeError("rzz gates are forbidden.")
            qs = tuple(int(q) for q in qubits)
            ps = tuple(float(p) for p in params)
            if name not in ("measure", "barrier"):
                for q in qs:
                    z = self.geom.zone_at[self.positions[q]]
                    if name not in z.operations:
                        raise RuntimeError(f"Gate {name} on q{q} at {self.positions[q]} not allowed in {z.name}")
                if len(qs) == 2:
                    a, b = self.positions[qs[0]], self.positions[qs[1]]
                    z = self.geom.zone_at[a]
                    if z.pair_axis is not None:
                        if not are_paired(z, a, b):
                            raise RuntimeError(f"Gate {name} on {qs} at {a}, {b} not paired!")
                    elif abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
                        raise RuntimeError(f"Gate {name} on {qs} at {a}, {b} not adjacent!")
                    self._layer_2q_qubits.update(qs)
                    self._layer_2q_zones.add(z.name)
            d = max(self.q_depth[q] for q in qs) + 1
            for q in qs:
                self.q_depth[q] = d
            self.ops.append(("gate", name, qs, ps))

        def build_circuit(
            self,
            template: QuantumCircuit,
            trailing_meas: list[tuple[int, int]],
        ) -> tuple[QuantumCircuit, Architecture]:
            self._close_layer()
            compiled = template.copy_empty_like()
            qbits = compiled.qubits
            cbits = compiled.clbits
            for op in self.ops:
                if op[0] == "move":
                    _, qs, (r, c, nr, nc, *_) = op
                    compiled.append(Move(r, c, nr, nc, len(qs)), [qbits[q] for q in qs])
                else:
                    _, name, qs, ps = op
                    if name == "rx":
                        compiled.rx(ps[0], qbits[qs[0]])
                    elif name == "rz":
                        compiled.rz(ps[0], qbits[qs[0]])
                    elif name == "cz":
                        compiled.cz(qbits[qs[0]], qbits[qs[1]])
            for q_idx, c_idx in trailing_meas:
                compiled.measure(qbits[q_idx], cbits[c_idx])
            return compiled, Architecture(self.initial_loading, spec=self._raw_spec)


    # ---------------------------------------------------------------------------
    # Stage 1: Exact Algebraic Front-End
    # ---------------------------------------------------------------------------


    def _norm_angle(theta: float) -> float:
        return ((theta + math.pi) % _TWO_PI) - math.pi


    def _is_zero_angle(theta: float) -> bool:
        return abs(_norm_angle(theta)) < _TOL


    def _mat_rx(t: float) -> np.ndarray:
        c, s = math.cos(t / 2.0), -1j * math.sin(t / 2.0)
        return np.array([[c, s], [s, c]], dtype=complex)


    def _mat_ry(t: float) -> np.ndarray:
        c, s = math.cos(t / 2.0), math.sin(t / 2.0)
        return np.array([[c, -s], [s, c]], dtype=complex)


    def _mat_rz(t: float) -> np.ndarray:
        return np.array([[np.exp(-0.5j * t), 0.0], [0.0, np.exp(0.5j * t)]], dtype=complex)


    _MAT_1Q: dict[str, Any] = {
        "id": lambda: np.eye(2, dtype=complex),
        "x": lambda: _mat_rx(math.pi),
        "y": lambda: _mat_ry(math.pi),
        "z": lambda: _mat_rz(math.pi),
        "h": lambda: np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / math.sqrt(2.0),
        "s": lambda: _mat_rz(math.pi / 2.0),
        "sdg": lambda: _mat_rz(-math.pi / 2.0),
        "t": lambda: _mat_rz(math.pi / 4.0),
        "tdg": lambda: _mat_rz(-math.pi / 4.0),
        "sx": lambda: _mat_rx(math.pi / 2.0),
        "sxdg": lambda: _mat_rx(-math.pi / 2.0),
    }


    def _gate_to_matrix(name: str, params: tuple[float, ...]) -> np.ndarray | None:
        if name in _MAT_1Q:
            return _MAT_1Q[name]()
        if name == "rx":
            return _mat_rx(params[0])
        if name == "ry":
            return _mat_ry(params[0])
        if name == "rz" or name == "p" or name == "u1":
            return _mat_rz(params[0])
        if name == "u2":
            phi, lam = params
            return np.array(
                [
                    [1.0, -np.exp(1j * lam)],
                    [np.exp(1j * phi), np.exp(1j * (phi + lam))],
                ],
                dtype=complex,
            ) / math.sqrt(2.0)
        if name in ("u3", "u"):
            th, phi, lam = params
            c, s = math.cos(th / 2.0), math.sin(th / 2.0)
            return np.array(
                [
                    [c, -np.exp(1j * lam) * s],
                    [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
                ],
                dtype=complex,
            )
        return None


    def _matrix_to_zxz(u: np.ndarray) -> list[tuple[str, float]]:
        """Decompose 2x2 unitary into minimal circuit-order list of ('rz'|'rx', angle)."""
        # Check pure diagonal (pure rz)
        if abs(u[0, 1]) < 1e-10 and abs(u[1, 0]) < 1e-10:
            ang = _norm_angle(float(np.angle(u[1, 1] / u[0, 0])))
            return [] if abs(ang) < _TOL else [("rz", ang)]
        # Use Qiskit's exact ZXZ decomposer: returns (theta, phi, lam) -> rz(lam), rx(theta), rz(phi)
        theta, phi, lam = _EULER_ZXZ.angles(u)
        lam_n = _norm_angle(float(lam))
        th_n = _norm_angle(float(theta))
        phi_n = _norm_angle(float(phi))

        # Eliminate redundant rz(±pi) via Z * rx(th) = rx(-th) * Z (up to global phase)
        if abs(abs(lam_n) - math.pi) < _TOL and abs(th_n) >= _TOL:
            lam_n = 0.0
            th_n = _norm_angle(-th_n)
            phi_n = _norm_angle(phi_n + math.pi)
        elif abs(abs(phi_n) - math.pi) < _TOL and abs(th_n) >= _TOL:
            phi_n = 0.0
            th_n = _norm_angle(-th_n)
            lam_n = _norm_angle(lam_n + math.pi)
        # Eliminate redundant double-rz around rx(±pi) via rz(phi) * X * rz(lam) = X * rz(lam - phi)
        elif abs(abs(th_n) - math.pi) < _TOL and abs(lam_n) >= _TOL and abs(phi_n) >= _TOL:
            lam_n = _norm_angle(lam_n - phi_n)
            phi_n = 0.0

        ops: list[tuple[str, float]] = []
        if abs(lam_n) >= _TOL:
            ops.append(("rz", lam_n))
        if abs(th_n) >= _TOL:
            ops.append(("rx", th_n))
        if abs(phi_n) >= _TOL:
            ops.append(("rz", phi_n))
        # If theta == 0, merge the two rz
        if len(ops) == 2 and ops[0][0] == "rz" and ops[1][0] == "rz":
            s = _norm_angle(ops[0][1] + ops[1][1])
            return [] if abs(s) < _TOL else [("rz", s)]
        return ops


    def _extract_circuit_gates(
        circuit: QuantumCircuit, recognize_virtual_rzz: bool = True
    ) -> tuple[list[tuple[str, tuple[int, ...], tuple[float, ...]]], list[tuple[int, int]]]:
        """Strip trailing measurements & barriers, decompose to internal IR {rx, rz, cz, (virtual) rzz}."""
        q_idx = {bit: i for i, bit in enumerate(circuit.qubits)}
        c_idx = {bit: i for i, bit in enumerate(circuit.clbits)}

        # Identify trailing measurements (possibly separated by barriers)
        data = list(circuit.data)
        trail_start = len(data)
        while trail_start > 0:
            nm = data[trail_start - 1].operation.name
            if nm in ("measure", "barrier"):
                trail_start -= 1
            else:
                break

        trailing_meas: list[tuple[int, int]] = []
        for inst in data[trail_start:]:
            if inst.operation.name == "measure":
                trailing_meas.append((q_idx[inst.qubits[0]], c_idx[inst.clbits[0]]))

        raw_ops: list[tuple[str, tuple[int, ...], tuple[float, ...]]] = []
        for inst in data[:trail_start]:
            op = inst.operation
            if op.name == "barrier":
                continue
            qs = tuple(q_idx[q] for q in inst.qubits)
            ps = tuple(float(p) for p in op.params)
            raw_ops.append((op.name, qs, ps))

        if recognize_virtual_rzz:
            raw_ops = _expand_and_recognize_rzz(raw_ops, circuit.num_qubits)
        else:
            raw_ops = _expand_cz_only(raw_ops, circuit.num_qubits)

        return raw_ops, trailing_meas


    def _expand_virtual_rzz_to_cz(
        ops: list[tuple[str, tuple[int, ...], tuple[float, ...]]],
        allow_pi_simplify: bool = True,
    ) -> list[tuple[str, tuple[int, ...], tuple[float, ...]]]:
        """Expand any internal virtual `rzz(t, u, v)` into native {rx, rz, cz} via H_v · CZ · RX(t)_v · CZ · H_v."""
        h_ops = _matrix_to_zxz(_MAT_1Q["h"]())
        out: list[tuple[str, tuple[int, ...], tuple[float, ...]]] = []
        for name, qs, ps in ops:
            if name != "rzz":
                out.append((name, qs, ps))
                continue
            u, v = qs
            ang = _norm_angle(ps[0])
            if _is_zero_angle(ang):
                continue
            if allow_pi_simplify and abs(abs(ang) - math.pi) < _TOL:
                out.append(("rz", (u,), (math.pi,)))
                out.append(("rz", (v,), (math.pi,)))
            elif abs(abs(ang) - math.pi / 2) < _TOL:
                s = math.pi / 2 if ang > 0 else -math.pi / 2
                out.append(("rz", (u,), (s,)))
                out.append(("rz", (v,), (s,)))
                out.append(("cz", (min(u, v), max(u, v)), ()))
            else:
                for gn, a in h_ops:
                    out.append((gn, (v,), (a,)))
                out.append(("cz", (min(u, v), max(u, v)), ()))
                out.append(("rx", (v,), (ang,)))
                out.append(("cz", (min(u, v), max(u, v)), ()))
                for gn, a in h_ops:
                    out.append((gn, (v,), (a,)))
        return out


    def _expand_cz_only(
        ops: list[tuple[str, tuple[int, ...], tuple[float, ...]]], nq: int
    ) -> list[tuple[str, tuple[int, ...], tuple[float, ...]]]:
        out: list[tuple[str, tuple[int, ...], tuple[float, ...]]] = []
        for name, qs, ps in ops:
            if name in ("rx", "rz", "cz"):
                out.append((name, qs, ps))
            elif name == "cx":
                c, t = qs
                out.extend([
                    ("rz", (t,), (-math.pi / 2,)),
                    ("rx", (t,), (-math.pi / 2,)),
                    ("cz", (c, t), ()),
                    ("rx", (t,), (math.pi / 2,)),
                    ("rz", (t,), (math.pi / 2,)),
                ])
            elif len(qs) == 1 and _gate_to_matrix(name, ps) is not None:
                for gname, ang in _matrix_to_zxz(_gate_to_matrix(name, ps)):
                    out.append((gname, qs, (ang,)))
            else:
                sub = QuantumCircuit(len(qs))
                sub.append(_build_qiskit_inst(name, ps, len(qs)), list(range(len(qs))))
                t_sub = transpile(sub, basis_gates=["rx", "rz", "cz"], optimization_level=0)
                sub_q = {b: i for i, b in enumerate(t_sub.qubits)}
                for inst in t_sub.data:
                    iqs = tuple(qs[sub_q[b]] for b in inst.qubits)
                    out.append((inst.operation.name, iqs, tuple(float(p) for p in inst.operation.params)))
        return out


    def _build_qiskit_inst(name: str, params: tuple[float, ...], num_q: int) -> Instruction:
        from qiskit.circuit.library import get_standard_gate_name_mapping
        mapping = get_standard_gate_name_mapping()
        gate = mapping[name]
        return type(gate)(*params)


    def _expand_and_recognize_rzz(
        ops: list[tuple[str, tuple[int, ...], tuple[float, ...]]], nq: int
    ) -> list[tuple[str, tuple[int, ...], tuple[float, ...]]]:
        """Expand multi-qubit gates using 1-RZZ identities, transpile exotic ones, and fuse CX-RZ-CX -> RZZ."""
        stage1: list[tuple[str, tuple[int, ...], tuple[float, ...]]] = []
        for name, qs, ps in ops:
            if name in ("rx", "rz", "cz", "rzz", "cx"):
                stage1.append((name, qs, ps))
            elif len(qs) == 1 and _gate_to_matrix(name, ps) is not None:
                for gname, ang in _matrix_to_zxz(_gate_to_matrix(name, ps)):
                    stage1.append((gname, qs, (ang,)))
            elif name in ("cp", "cu1"):
                u, v = qs
                t = ps[0]
                stage1.extend([("rz", (u,), (t / 2,)), ("rz", (v,), (t / 2,)), ("rzz", (u, v), (-t / 2,))])
            elif name == "crz":
                u, v = qs
                t = ps[0]
                stage1.extend([("rz", (v,), (t / 2,)), ("rzz", (u, v), (-t / 2,))])
            elif name == "rxx":
                u, v = qs
                t = ps[0]
                stage1.extend([
                    ("rz", (u,), (-math.pi / 2,)), ("rx", (u,), (-math.pi / 2,)),
                    ("rz", (v,), (-math.pi / 2,)), ("rx", (v,), (-math.pi / 2,)),
                    ("rzz", (u, v), (t,)),
                    ("rx", (u,), (math.pi / 2,)), ("rz", (u,), (math.pi / 2,)),
                    ("rx", (v,), (math.pi / 2,)), ("rz", (v,), (math.pi / 2,)),
                ])
            elif name == "ryy":
                u, v = qs
                t = ps[0]
                stage1.extend([
                    ("rx", (u,), (math.pi / 2,)), ("rx", (v,), (math.pi / 2,)),
                    ("rzz", (u, v), (t,)),
                    ("rx", (u,), (-math.pi / 2,)), ("rx", (v,), (-math.pi / 2,)),
                ])
            else:
                # Transpile exotic multi-qubit gate (ccx, cswap, cu3, swap, ch, crx, cry, cy) -> cx + 1q
                sub = QuantumCircuit(len(qs))
                sub.append(_build_qiskit_inst(name, ps, len(qs)), list(range(len(qs))))
                t_sub = transpile(sub, basis_gates=["rx", "rz", "cx"], optimization_level=0)
                sub_q = {b: i for i, b in enumerate(t_sub.qubits)}
                for inst in t_sub.data:
                    iqs = tuple(qs[sub_q[b]] for b in inst.qubits)
                    stage1.append((inst.operation.name, iqs, tuple(float(p) for p in inst.operation.params)))

        # Recognize cx(u,v) · [rz(p,u) / rz(t,v)] · cx(u,v) -> rzz(t, u, v)
        # Track wire-adjacent operations using per-qubit index lists!
        # Simple linear scan with wire-commutation lookahead:
        n = len(stage1)
        used = [False] * n
        recognized: list[tuple[str, tuple[int, ...], tuple[float, ...]]] = []

        # Build per-qubit occurrence lists for O(1) next-gate-on-wire lookup
        i = 0
        while i < n:
            if used[i]:
                i += 1
                continue
            name, qs, ps = stage1[i]
            if name == "cx":
                u, v = qs
                # Scan forward on wires u and v until a non-rz gate on u or v
                j = i + 1
                rz_u: list[int] = []
                rz_v: list[int] = []
                matched_cx = -1
                while j < n:
                    if used[j]:
                        j += 1
                        continue
                    gn, gq, _ = stage1[j]
                    if u in gq or v in gq:
                        if gn == "rz" and gq == (u,):
                            rz_u.append(j)
                        elif gn == "rz" and gq == (v,):
                            rz_v.append(j)
                        elif gn == "cx" and gq == (u, v):
                            matched_cx = j
                            break
                        else:
                            break
                    j += 1
                if matched_cx != -1:
                    used[i] = True
                    used[matched_cx] = True
                    for idx in rz_u:
                        used[idx] = True
                        recognized.append(stage1[idx])
                    tot_v = sum(stage1[idx][2][0] for idx in rz_v)
                    for idx in rz_v:
                        used[idx] = True
                    if not _is_zero_angle(tot_v):
                        recognized.append(("rzz", (min(u, v), max(u, v)), (_norm_angle(tot_v),)))
                    i += 1
                    continue
            used[i] = True
            recognized.append((name, qs, ps))
            i += 1

        # Convert any remaining standalone cx(c, t) -> minimal 2-gate/side CZ
        final_ops: list[tuple[str, tuple[int, ...], tuple[float, ...]]] = []
        for name, qs, ps in recognized:
            if name == "cx":
                c, t = qs
                final_ops.extend([
                    ("rz", (t,), (-math.pi / 2,)),
                    ("rx", (t,), (-math.pi / 2,)),
                    ("cz", (min(c, t), max(c, t)), ()),
                    ("rx", (t,), (math.pi / 2,)),
                    ("rz", (t,), (math.pi / 2,)),
                ])
            elif name == "rzz":
                if not _is_zero_angle(ps[0]):
                    final_ops.append(("rzz", (min(qs), max(qs)), (_norm_angle(ps[0]),)))
            elif name == "cz":
                final_ops.append(("cz", (min(qs), max(qs)), ()))
            else:
                final_ops.append((name, qs, ps))
        return final_ops


    def compress_native_ops(
        ops: list[tuple[str, tuple[int, ...], tuple[float, ...]]], nq: int
    ) -> list[tuple[str, tuple[int, ...], tuple[float, ...]]]:
        """Iterative 1Q matrix fusion + diagonal RZ pushing + same-pair 2Q fusion."""
        cur = list(ops)
        for _ in range(6):
            prev_len = len(cur)
            # 1. Fuse consecutive 1Q gates per wire
            # Sweep forward, accumulating 2x2 unitaries per qubit,
            # flushing a qubit's 1Q unitary right before any 2Q gate on that qubit (or at the end).
            u_acc: list[np.ndarray | None] = [None] * nq
            fused: list[tuple[str, tuple[int, ...], tuple[float, ...]]] = []

            def flush_q(q: int) -> None:
                if u_acc[q] is not None:
                    for gn, ang in _matrix_to_zxz(u_acc[q]):
                        fused.append((gn, (q,), (ang,)))
                    u_acc[q] = None

            for name, qs, ps in cur:
                if len(qs) == 1:
                    q = qs[0]
                    m = _mat_rx(ps[0]) if name == "rx" else _mat_rz(ps[0])
                    u_acc[q] = m if u_acc[q] is None else (m @ u_acc[q])
                else:
                    flush_q(qs[0])
                    flush_q(qs[1])
                    fused.append((name, qs, ps))
            for q in range(nq):
                flush_q(q)

            # 2. Push diagonal RZ gates forward past diagonal 2Q gates (cz / rzz) whenever a downstream 1Q block exists!
            # Walk backwards or maintain per-qubit "pending_rz" while scanning forward:
            # Actually, scanning forward: if an rz(a) sits right before a 2Q gate, we can float it in `pending_rz[q]`
            # across any number of `cz`/`rzz` gates! When we hit an `rx` on qubit `q`, we flush `pending_rz[q]` right before `rx`!
            # At the end of the circuit, flush all `pending_rz[q]`!
            pending_rz = [0.0] * nq
            pushed: list[tuple[str, tuple[int, ...], tuple[float, ...]]] = []
            for name, qs, ps in fused:
                if name == "rz":
                    pending_rz[qs[0]] = _norm_angle(pending_rz[qs[0]] + ps[0])
                elif name == "rx":
                    q = qs[0]
                    if not _is_zero_angle(pending_rz[q]):
                        pushed.append(("rz", (q,), (pending_rz[q],)))
                        pending_rz[q] = 0.0
                    pushed.append((name, qs, ps))
                else:
                    # cz / rzz commutes with rz! Keep pending_rz floating!
                    pushed.append((name, qs, ps))
            for q in range(nq):
                if not _is_zero_angle(pending_rz[q]):
                    pushed.append(("rz", (q,), (pending_rz[q],)))

            # 3. Same-pair adjacent diagonal 2Q fusion (cz/rzz)
            cur_2q: list[tuple[str, tuple[int, ...], tuple[float, ...]]] = []
            last_on_q: list[int | None] = [None] * nq
            alive: list[bool] = []
            for name, qs, ps in pushed:
                if len(qs) == 1:
                    q = qs[0]
                    last_on_q[q] = len(cur_2q)
                    cur_2q.append((name, qs, ps))
                    alive.append(True)
                else:
                    u, v = qs
                    iu, iv = last_on_q[u], last_on_q[v]
                    if iu is not None and iu == iv and alive[iu]:
                        pname, pqs, pps = cur_2q[iu]
                        if pname == "cz" and name == "cz":
                            alive[iu] = False
                            last_on_q[u] = last_on_q[v] = None
                            continue
                        if pname == "rzz" and name == "rzz":
                            ang = _norm_angle(pps[0] + ps[0])
                            if _is_zero_angle(ang):
                                alive[iu] = False
                                last_on_q[u] = last_on_q[v] = None
                            else:
                                cur_2q[iu] = ("rzz", qs, (ang,))
                            continue
                        if {pname, name} == {"cz", "rzz"}:
                            ang = pps[0] if pname == "rzz" else ps[0]
                            new_ang = _norm_angle(ang - math.pi / 2)
                            cur_2q[iu] = ("rz", (u,), (math.pi / 2,))
                            cur_2q.append(("rz", (v,), (math.pi / 2,)))
                            alive.append(True)
                            if not _is_zero_angle(new_ang):
                                idx = len(cur_2q)
                                cur_2q.append(("rzz", qs, (new_ang,)))
                                alive.append(True)
                                last_on_q[u] = last_on_q[v] = idx
                            else:
                                last_on_q[u] = last_on_q[v] = None
                            continue
                    idx = len(cur_2q)
                    cur_2q.append((name, qs, ps))
                    alive.append(True)
                    last_on_q[u] = last_on_q[v] = idx

            cur = [op for op, ok in zip(cur_2q, alive) if ok]
            if len(cur) == prev_len:
                break
        return cur


    # ---------------------------------------------------------------------------
    # Stage 2: IHVA Detection & Pauli-Commutation Wave Scheduler
    # ---------------------------------------------------------------------------


    def _try_parse_ihva(
        circuit: QuantumCircuit,
    ) -> tuple[list[tuple[str, tuple[int, ...], tuple[float, ...]]], list[tuple[int, int, float]], list[tuple[int, int]]] | None:
        """Detect if circuit is (initial 1Q layer) -> (Rzy(u,v;t) blocks) -> (trailing measurements)."""
        q_idx = {bit: i for i, bit in enumerate(circuit.qubits)}
        c_idx = {bit: i for i, bit in enumerate(circuit.clbits)}
        data = list(circuit.data)

        trail_start = len(data)
        while trail_start > 0 and data[trail_start - 1].operation.name in ("measure", "barrier"):
            trail_start -= 1
        trailing_meas = [
            (q_idx[inst.qubits[0]], c_idx[inst.clbits[0]])
            for inst in data[trail_start:]
            if inst.operation.name == "measure"
        ]

        # Collect initial 1Q prefix before first 2Q gate
        body_start = 0
        init_1q: list[tuple[str, tuple[int, ...], tuple[float, ...]]] = []
        while body_start < trail_start:
            op = data[body_start].operation
            if op.name == "barrier":
                body_start += 1
                continue
            if len(data[body_start].qubits) == 1:
                q = q_idx[data[body_start].qubits[0]]
                mat = _gate_to_matrix(op.name, tuple(float(p) for p in op.params))
                if mat is None:
                    return None
                init_1q.append((op.name, (q,), tuple(float(p) for p in op.params)))
                body_start += 1
            else:
                break

        # Check if body[body_start:trail_start] is strictly Rzy blocks:
        # rx(pi/2, v), cx(u, v), rz(t, v), cx(u, v), rx(-pi/2, v)
        # Wait: in QASM, init H is followed by rx(pi/2, v) of the first block!
        # So step back from body_start if the last 1q op was rx(pi/2)
        while body_start > 0 and data[body_start - 1].operation.name == "rx":
            body_start -= 1
            init_1q.pop()

        blocks: list[tuple[int, int, float]] = []
        i = body_start
        while i < trail_start:
            if data[i].operation.name == "barrier":
                i += 1
                continue
            if i + 4 >= trail_start:
                return None
            o0, o1, o2, o3, o4 = (data[i + k] for k in range(5))
            if not (
                o0.operation.name == "rx"
                and o1.operation.name == "cx"
                and o2.operation.name == "rz"
                and o3.operation.name == "cx"
                and o4.operation.name == "rx"
            ):
                return None
            u, v = q_idx[o1.qubits[0]], q_idx[o1.qubits[1]]
            if (
                q_idx[o0.qubits[0]] != v
                or q_idx[o2.qubits[0]] != v
                or q_idx[o3.qubits[0]] != u
                or q_idx[o3.qubits[1]] != v
                or q_idx[o4.qubits[0]] != v
            ):
                return None
            a0 = float(o0.operation.params[0])
            a2 = float(o2.operation.params[0])
            a4 = float(o4.operation.params[0])
            if abs(_norm_angle(a0 - math.pi / 2)) > 1e-9 or abs(_norm_angle(a4 + math.pi / 2)) > 1e-9:
                return None
            if not _is_zero_angle(a2):
                blocks.append((u, v, _norm_angle(a2)))
            i += 5

        if not blocks:
            return None
        return init_1q, blocks, trailing_meas


    def _schedule_ihva_waves(
        nq: int, blocks: list[tuple[int, int, float]], mode: str = "crit"
    ) -> list[list[tuple[int, int, float]]]:
        """Pauli-commutation persistent-basis wave scheduler."""
        n = len(blocks)
        if n == 0:
            return []
        preds: list[set[int]] = [set() for _ in range(n)]
        succs: list[list[int]] = [[] for _ in range(n)]
        for j in range(n):
            uj, vj, _ = blocks[j]
            for k in range(j):
                uk, vk, _ = blocks[k]
                if ((uj == vk) + (vj == uk)) % 2 == 1:
                    preds[j].add(k)
                    succs[k].append(j)

        crit = [1] * n
        for j in range(n - 1, -1, -1):
            if succs[j]:
                crit[j] = 1 + max(crit[s] for s in succs[j])

        if mode in ("asap", "alap", "hub_alap"):
            max_lvl = max(crit) - 1
            if mode == "asap":
                lvl = [0] * n
                for j in range(n):
                    if preds[j]:
                        lvl[j] = 1 + max(lvl[k] for k in preds[j])
            elif mode == "alap":
                lvl = [max_lvl - (crit[j] - 1) for j in range(n)]
            else:
                # Consolidate hub blocks to their ALAP wave while running non-hub blocks as early as possible (ASAP)!
                deg_b: Counter[int] = Counter()
                for u, v, _ in blocks:
                    deg_b[u] += 1
                    deg_b[v] += 1
                hub_b = deg_b.most_common(1)[0][0] if deg_b else -1
                alap_l = [max_lvl - (crit[j] - 1) for j in range(n)]
                lvl = [0] * n
                for j in range(n):
                    base = alap_l[j] if (blocks[j][0] == hub_b or blocks[j][1] == hub_b) else 0
                    if preds[j]:
                        base = max(base, 1 + max(lvl[k] for k in preds[j]))
                    lvl[j] = base
            groups: list[list[int]] = [[] for _ in range(max(lvl) + 1)]
            for j in range(n):
                groups[lvl[j]].append(j)
            # Verify each wave is strictly bipartite (Z-qubits disjoint from Y-qubits)
            for g in groups:
                if not g:
                    continue
                uz = {blocks[j][0] for j in g}
                uy = {blocks[j][1] for j in g}
                if uz & uy:
                    return []
            return [[blocks[j] for j in g] for g in groups if g]

        done = [False] * n
        rem_preds = [len(preds[j]) for j in range(n)]
        basis: list[int | None] = [None] * nq  # 0=Z, 1=Y
        waves: list[list[tuple[int, int, float]]] = []
        rem = n

        while rem > 0:
            def eval_basis(b: list[int]) -> list[int]:
                d = list(done)
                rp = list(rem_preds)
                executed: list[int] = []
                changed = True
                while changed:
                    changed = False
                    for j in range(n):
                        if not d[j] and rp[j] == 0:
                            u, v, _ = blocks[j]
                            if b[u] == 0 and b[v] == 1:
                                d[j] = True
                                executed.append(j)
                                for s in succs[j]:
                                    rp[s] -= 1
                                changed = True
                return executed

            ready = [j for j in range(n) if not done[j] and rem_preds[j] == 0]
            score_z = [0.0] * nq
            score_y = [0.0] * nq
            for j in ready:
                u, v, _ = blocks[j]
                w = float(crit[j] ** 2) if mode == "crit" else float(crit[j])
                score_z[u] += w
                score_y[v] += w

            b = [0 if score_z[q] >= score_y[q] else 1 for q in range(nq)]
            for q in range(nq):
                if score_z[q] == 0 and score_y[q] == 0 and basis[q] is not None:
                    b[q] = basis[q]

            def obj(ex: list[int]) -> float:
                if not ex:
                    return -1e9
                uz = {blocks[j][0] for j in ex}
                uy = {blocks[j][1] for j in ex}
                flips = sum(1 for q in uz if basis[q] == 1) + sum(1 for q in uy if basis[q] == 0)
                if mode == "crit":
                    return sum(crit[j] for j in ex) + 0.5 * len(ex) - 0.15 * flips
                return 1.5 * len(ex) + 0.4 * sum(crit[j] for j in ex) - 0.25 * flips

            cur_ex = eval_basis(b)
            cur_val = obj(cur_ex)
            for _ in range(3):
                improved = False
                for q in range(nq):
                    if score_z[q] == 0 and score_y[q] == 0:
                        continue
                    b[q] ^= 1
                    ex = eval_basis(b)
                    val = obj(ex)
                    if val > cur_val + 1e-9:
                        cur_val = val
                        cur_ex = ex
                        improved = True
                    else:
                        b[q] ^= 1
                if not improved:
                    break

            used_z = {blocks[j][0] for j in cur_ex}
            used_y = {blocks[j][1] for j in cur_ex}
            for q in range(nq):
                if q in used_z:
                    basis[q] = 0
                elif q in used_y:
                    basis[q] = 1

            for j in cur_ex:
                done[j] = True
                for s in succs[j]:
                    rem_preds[s] -= 1
            rem -= len(cur_ex)
            waves.append([blocks[j] for j in cur_ex])

        return waves


    # ---------------------------------------------------------------------------
    # Stage 4: Layout Optimization & High-Batching AOD Routing Engine
    # ---------------------------------------------------------------------------


    def _detect_hub(nq: int, edges: Sequence[tuple[int, int]]) -> int | None:
        deg: Counter[int] = Counter()
        for u, v in set(edges):
            deg[u] += 1
            deg[v] += 1
        if not deg:
            return None
        top_q, top_d = deg.most_common(1)[0]
        if nq >= 8 and top_d >= max(5, int(0.65 * (nq - 1))):
            return top_q
        return None


    def _sa_layout(
        nq: int,
        sites: list[Site],
        weighted_edges: list[tuple[int, int, float]],
        geom: ArchGeometry,
        cost_mode: str = "mesh",
        fixed_sites: dict[int, Site] | None = None,
        steps: int = 80000,
        seed: int = 42,
        restarts: int = 1,
    ) -> dict[int, Site]:
        """Simulated Annealing placement of qubits onto entangling `sites` (supports multi-restart)."""
        fixed = dict(fixed_sites or {})
        free_qs = [q for q in range(nq) if q not in fixed]
        used_sites = set(fixed.values())
        free_sites = [s for s in sites if s not in used_sites]
        assert len(free_sites) >= len(free_qs)

        # Normalize edge weights so SA temperature T=4.0 -> 1e-3 is scale-invariant across p=1 and p=3!
        agg_w: dict[tuple[int, int], float] = defaultdict(float)
        for u, v, w in weighted_edges:
            if u not in fixed and v not in fixed and u != v:
                k = (u, v) if u < v else (v, u)
                agg_w[k] += float(w)
        if agg_w:
            mean_w = sum(agg_w.values()) / len(agg_w)
            inv_mean = 1.0 / mean_w if mean_w > 1e-9 else 1.0
        else:
            inv_mean = 1.0

        adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for (u, v), w in agg_w.items():
            wn = w * inv_mean
            adj[u].append((v, wn))
            adj[v].append((u, wn))

        # Precompute site pair cost table for O(1) lookup!
        site_to_idx = {s: i for i, s in enumerate(sites)}
        m = len(sites)
        ex0 = geom.entangling.x[0]

        _SPLIT_ZONE_PAIRS = {(0, 3), (3, 0), (4, 7), (7, 4), (2, 5), (5, 2), (6, 9), (9, 6)}
        _SPLIT_ZONE_CLEAN = {
            0: (0, 5), 3: (2, 5), 4: (4, 5), 7: (6, 5), 8: (8, 5),
            1: (1, 0), 2: (3, 0), 5: (5, 0), 6: (7, 0), 9: (9, 0),
        }
        max_col_off = max((s[0] - ex0 for s in sites), default=0)

        def pair_cost(a: Site, b: Site) -> float:
            if cost_mode == "mesh3":
                ca, cb = a[0] - ex0, b[0] - ex0
                dx = abs(ca - cb)
                dy = abs(a[1] - b[1])
                same_b = ca // 2 == cb // 2
                if dx == 1 and dy == 0:
                    return 0.01 if same_b else 0.04
                if dx == 1 and dy <= 2:
                    return 0.05 + 0.03 * dy
                if (ca, cb) in _SPLIT_ZONE_PAIRS and dy <= 2:
                    return 0.08 + 0.03 * dy
                # Exact post-Phase-C Slot-0 cleanup pairing distance (exploiting the row 14/15 seam & Slot-0 adjacency!)
                if ca in _SPLIT_ZONE_CLEAN and cb in _SPLIT_ZONE_CLEAN:
                    cax, day = _SPLIT_ZONE_CLEAN[ca]
                    cbx, dby = _SPLIT_ZONE_CLEAN[cb]
                    sa_c = (ex0 + cax, a[1] + day)
                    sb_c = (ex0 + cbx, b[1] + dby)
                    pa_c = geom.partner_site(sa_c)
                    pb_c = geom.partner_site(sb_c)
                    if pa_c is not None and pb_c is not None:
                        return 0.25 + min(geom.dist(sa_c, pb_c), geom.dist(sb_c, pa_c))
                return 1.80 + geom.dist(a, b)
            if cost_mode == "mesh2_exact":
                ca, cb = a[0] - ex0, b[0] - ex0
                dx = abs(ca - cb)
                dy = abs(a[1] - b[1])
                same_b = ca // 2 == cb // 2
                if max_col_off == 1:
                    # 12q 6x2: Phase A covers dy <= 3; unpack moves col 1 -> col 3!
                    if dx == 1 and dy == 0:
                        return 0.01
                    if dx == 1 and dy <= 3:
                        return 0.04 + 0.03 * dy
                    sa_c = (ex0 + (3 if ca == 1 else 0), a[1])
                    sb_c = (ex0 + (3 if cb == 1 else 0), b[1])
                    pa_c = geom.partner_site(sa_c)
                    pb_c = geom.partner_site(sb_c)
                    if pa_c is not None and pb_c is not None:
                        return 0.20 + min(geom.dist(sa_c, pb_c), geom.dist(sb_c, pa_c))
                else:
                    # 20q 5x4: Phase A+B covers dx==1, dy<=2; unpack moves B=(1,2) -> y+5!
                    if dx == 1 and dy == 0:
                        return 0.01 if same_b else 0.04
                    if dx == 1 and dy <= 2:
                        return 0.05 + 0.03 * dy
                    sa_c = (a[0], a[1] + (5 if ca in (1, 2) else 0))
                    sb_c = (b[0], b[1] + (5 if cb in (1, 2) else 0))
                    pa_c = geom.partner_site(sa_c)
                    pb_c = geom.partner_site(sb_c)
                    if pa_c is not None and pb_c is not None:
                        return 0.20 + min(geom.dist(sa_c, pb_c), geom.dist(sb_c, pa_c))
                return 1.60 + geom.dist(a, b)
            if cost_mode == "mesh2":
                dx = abs(a[0] - b[0])
                dy = abs(a[1] - b[1])
                same_b = (a[0] - ex0) // 2 == (b[0] - ex0) // 2
                if dx == 1 and dy == 0:
                    return 0.01 if same_b else 0.04
                if dx == 1 and dy <= 2:
                    return 0.05 + 0.03 * dy
                if dx == 0 and dy == 1:
                    return 0.65
                if dx == 2 and dy == 0:
                    return 0.75
                if dx == 0 and dy == 2:
                    return 1.15
                return 1.60 + geom.dist(a, b)
            if cost_mode == "mesh":
                dx = abs(a[0] - b[0])
                dy = abs(a[1] - b[1])
                same_b = (a[0] - ex0) // 2 == (b[0] - ex0) // 2
                if dx == 1 and dy == 0:
                    return 0.02 if same_b else 0.06
                if dx == 1 and dy == 1:
                    return 0.14
                if dx == 1 and dy == 2:
                    return 0.50
                return 1.30 + geom.dist(a, b)
            if cost_mode == "slot0":
                dc = abs((a[0] - ex0) // 2 - (b[0] - ex0) // 2)
                dy = abs(a[1] - b[1])
                if dc == 1:
                    if dy <= 1:
                        return 0.10 + 0.05 * dy
                    if dy == 2:
                        return 0.25
                    return 0.50 + 0.30 * dy
                if dc == 0:
                    return 0.80 * dy
                return 1.50 * dc + 0.80 * dy
            # "compact"
            if are_paired(geom.entangling, a, b):
                return 0.0
            if (a[0] - ex0) // 2 == (b[0] - ex0) // 2:
                return 0.40 * abs(a[1] - b[1])
            return geom.dist(a, b)

        cost_mat = [[pair_cost(sites[i], sites[j]) for j in range(m)] for i in range(m)]

        if not free_qs or not weighted_edges:
            pos0: dict[int, Site] = dict(fixed)
            for q, s in zip(free_qs, free_sites):
                pos0[q] = s
            return pos0

        global_best = float("inf")
        global_best_pos: dict[int, Site] = {}
        n_free_q = len(free_qs)
        n_free_s = len(free_sites)
        decay = (1e-3 / 4.0) ** (1.0 / max(steps, 1))

        for r_i in range(max(1, restarts)):
            rng = random.Random(seed + 1009 * r_i)
            fs_perm = list(free_sites)
            if r_i > 0:
                rng.shuffle(fs_perm)
            pos: dict[int, Site] = dict(fixed)
            for q, s in zip(free_qs, fs_perm):
                pos[q] = s
            occ: dict[Site, int] = {s: q for q, s in pos.items()}

            cur = 0.0
            for u in free_qs:
                iu = site_to_idx[pos[u]]
                for v, w in adj[u]:
                    if u < v:
                        cur += w * cost_mat[iu][site_to_idx[pos[v]]]
            best = cur
            best_pos = dict(pos)
            T = 4.0

            for _ in range(steps):
                T *= decay
                u = free_qs[rng.randrange(n_free_q)]
                s1 = pos[u]
                s2 = free_sites[rng.randrange(n_free_s)]
                if s1 == s2:
                    continue
                i1, i2 = site_to_idx[s1], site_to_idx[s2]
                v = occ.get(s2)

                old_c = 0.0
                for nb, w in adj[u]:
                    old_c += w * cost_mat[i1][site_to_idx[pos[nb]]]
                if v is not None:
                    for nb, w in adj[v]:
                        if nb != u:
                            old_c += w * cost_mat[i2][site_to_idx[pos[nb]]]

                new_c = 0.0
                for nb, w in adj[u]:
                    ni = i1 if nb == v else site_to_idx[pos[nb]]
                    new_c += w * cost_mat[i2][ni]
                if v is not None:
                    for nb, w in adj[v]:
                        if nb != u:
                            new_c += w * cost_mat[i1][site_to_idx[pos[nb]]]

                dc = new_c - old_c
                if dc <= 0.0 or rng.random() < math.exp(-dc / max(T, 1e-6)):
                    cur += dc
                    pos[u] = s2
                    occ[s2] = u
                    if v is not None:
                        pos[v] = s1
                        occ[s1] = v
                    else:
                        del occ[s1]
                    if cur < best:
                        best = cur
                        best_pos = dict(pos)

            if best < global_best:
                global_best = best
                global_best_pos = best_pos

        return global_best_pos


    def _storage_row_for_ent(ent_pos: dict[int, Site], geom: ArchGeometry) -> dict[int, Site]:
        """Map entangling sites (ex0..ex0+W-1, ey0..ey0+R-1) directly to topmost storage rows."""
        sx0, sx1 = geom.storage.x
        sy0, sy1 = geom.storage.y
        ex0 = geom.entangling.x[0]
        ey0 = geom.entangling.y[0]
        used_y = sorted({y for _, y in ent_pos.values()})
        used_x = sorted({x for x, _ in ent_pos.values()})
        # Map each distinct y to top storage rows (preserving order, capped by storage height)
        # If all x,y fit directly shifted down into storage:
        max_y = max(used_y, default=ey0)
        min_y = min(used_y, default=ey0)
        span_y = max_y - min_y + 1
        if sy1 - span_y + 1 >= sy0 and all(sx0 <= (x - ex0 + sx0) <= sx1 for x in used_x):
            y_off = sy1 - max_y
            x_off = sx0 - ex0
            return {q: (x + x_off, y + y_off) for q, (x, y) in ent_pos.items()}
        # Fallback: pack sequentially into top rows of storage
        loading: dict[int, Site] = {}
        all_s = [(x, y) for y in range(sy1, sy0 - 1, -1) for x in range(sx0, sx1 + 1)]
        for q, s in zip(sorted(ent_pos), all_s):
            loading[q] = s
        return loading


    def _lift_storage_to_ent(gs: GridState, target_ent: dict[int, Site]) -> None:
        """Move all qubits from initial storage loading into `target_ent` in 1 (or minimal) moves."""
        # Check if a single rectangular move lifts everything!
        src_xs = sorted({gs.positions[q][0] for q in target_ent})
        src_ys = sorted({gs.positions[q][1] for q in target_ent})
        dst_xs = sorted({target_ent[q][0] for q in target_ent})
        dst_ys = sorted({target_ent[q][1] for q in target_ent})
        if len(src_xs) == len(dst_xs) and len(src_ys) == len(dst_ys):
            xmap = dict(zip(src_xs, dst_xs))
            ymap = dict(zip(src_ys, dst_ys))
            target_qs = set(target_ent)
            no_ghosts = not any(
                (x, y) in gs.occupant and gs.occupant[(x, y)] not in target_qs
                for x in range(src_xs[0], src_xs[-1] + 1)
                for y in range(src_ys[0], src_ys[-1] + 1)
            )
            if no_ghosts and all((xmap[gs.positions[q][0]], ymap[gs.positions[q][1]]) == target_ent[q] for q in target_ent):
                gs.move(
                    tuple(range(src_ys[0], src_ys[-1] + 1)),
                    tuple(range(src_xs[0], src_xs[-1] + 1)),
                    dst_ys,
                    dst_xs,
                )
                return
        # Fallback: move row by row
        by_y: dict[int, list[int]] = defaultdict(list)
        for q in target_ent:
            by_y[gs.positions[q][1]].append(q)
        for y in sorted(by_y, reverse=True):
            for q in by_y[y]:
                gs.move_site(q, target_ent[q])



    # ---------------------------------------------------------------------------
    # Stage 4b: Graph Matching, Single-Zone AOD Batching & Blocking-CZ Engine
    # ---------------------------------------------------------------------------


    def _ent_booths(geom: ArchGeometry) -> list[tuple[Site, Site]]:
        """Return all interaction pairs (slot0, slot1) in `entangling`, ordered closest to `storage` first."""
        ent = geom.entangling
        ex0, ex1 = ent.x
        ey0, ey1 = ent.y
        st = geom.storage
        st_cy = 0.5 * (st.y[0] + st.y[1])
        st_cx = 0.5 * (st.x[0] + st.x[1])
        booths: list[tuple[Site, Site]] = []
        if ent.pair_axis == "y":
            xs = list(range(ex0, ex1 + 1))
            xs.sort(key=lambda x: abs(x - st_cx))
            ys = list(range(ey0, ey1, 2))
            ys.sort(key=lambda y: abs(y - st_cy))
            for y in ys:
                for x in xs:
                    booths.append(((x, y), (x, y + 1)))
        else:
            ys = list(range(ey0, ey1 + 1))
            ys.sort(key=lambda y: abs(y - st_cy))
            for y in ys:
                for x in range(ex0, ex1, 2):
                    booths.append(((x, y), (x + 1, y)))
        return booths


    def _max_cut_bipartition(vertices: Sequence[int], edges: Sequence[tuple[int, int]]) -> tuple[set[int], set[int]]:
        """Fast greedy + 1-flip local search Max-Cut bipartition (A, B)."""
        adj: dict[int, list[int]] = defaultdict(list)
        for u, v in edges:
            if u != v:
                adj[u].append(v)
                adj[v].append(u)
        side: dict[int, int] = {}
        # Order by descending degree for strong initial cut
        order = sorted(vertices, key=lambda q: -len(adj[q]))
        for q in order:
            c0 = sum(1 for nb in adj[q] if side.get(nb) == 0)
            c1 = sum(1 for nb in adj[q] if side.get(nb) == 1)
            side[q] = 1 if c0 >= c1 else 0
        for _ in range(10):
            improved = False
            for q in order:
                s = side[q]
                same = sum(1 for nb in adj[q] if side.get(nb) == s)
                diff = len(adj[q]) - same
                if same > diff:
                    side[q] = 1 - s
                    improved = True
            if not improved:
                break
        part_a = {q for q, s in side.items() if s == 0}
        part_b = {q for q, s in side.items() if s == 1}
        return part_a, part_b


    def _find_large_matching(
        edges: list[tuple[int, int, Any]],
        priority: dict[tuple[int, int], float] | None = None,
    ) -> list[int]:
        """Return indices into `edges` forming a large (exact max if bipartite) matching."""
        if not edges:
            return []
        m = len(edges)
        adj_v: dict[int, list[tuple[int, int]]] = defaultdict(list)
        deg: Counter[int] = Counter()
        for idx, (u, v, _) in enumerate(edges):
            adj_v[u].append((v, idx))
            adj_v[v].append((u, idx))
            deg[u] += 1
            deg[v] += 1

        # Test bipartiteness via BFS
        color: dict[int, int] = {}
        is_bip = True
        for start in deg:
            if start in color:
                continue
            color[start] = 0
            stack = [start]
            while stack and is_bip:
                u = stack.pop()
                cu = color[u]
                for v, _ in adj_v[u]:
                    if v not in color:
                        color[v] = 1 - cu
                        stack.append(v)
                    elif color[v] == cu:
                        is_bip = False
                        break
            if not is_bip:
                break

        if is_bip:
            # Kuhn's exact maximum bipartite matching, prioritizing max-degree + high-priority vertices
            left = [v for v, c in color.items() if c == 0]
            # Order each vertex's incident edges by priority/degree
            for v in left:
                adj_v[v].sort(
                    key=lambda item: (
                        -(priority.get((min(v, item[0]), max(v, item[0])), 0.0) if priority else 0.0),
                        -deg[item[0]],
                    )
                )
            left.sort(
                key=lambda v: (
                    -deg[v],
                    -(max((priority.get((min(v, nb), max(v, nb)), 0.0) for nb, _ in adj_v[v]), default=0.0) if priority else 0.0),
                )
            )
            match_r: dict[int, tuple[int, int]] = {}  # right_v -> (left_v, edge_idx)

            def dfs(u: int, seen: set[int]) -> bool:
                for v, eidx in adj_v[u]:
                    if v in seen:
                        continue
                    seen.add(v)
                    if v not in match_r or dfs(match_r[v][0], seen):
                        match_r[v] = (u, eidx)
                        return True
                return False

            for u in left:
                dfs(u, set())
            return [eidx for _, eidx in match_r.values()]

        # Non-bipartite: greedy max-degree/priority + length-3 augmenting-path polish
        order = sorted(
            range(m),
            key=lambda i: (
                -(priority.get((min(edges[i][0], edges[i][1]), max(edges[i][0], edges[i][1])), 0.0) if priority else 0.0),
                -(deg[edges[i][0]] + deg[edges[i][1]]),
                -max(deg[edges[i][0]], deg[edges[i][1]]),
            ),
        )
        matched_v: dict[int, tuple[int, int]] = {}  # v -> (partner, edge_idx)
        for i in order:
            u, v, _ = edges[i]
            if u not in matched_v and v not in matched_v:
                matched_v[u] = (v, i)
                matched_v[v] = (u, i)

        # 1-step augmenting path improvement (u - v == w - z)
        improved = True
        while improved:
            improved = False
            free_vs = [v for v in deg if v not in matched_v]
            for u in free_vs:
                if u in matched_v:
                    continue
                for v, e_uv in adj_v[u]:
                    if v not in matched_v:
                        matched_v[u] = (v, e_uv)
                        matched_v[v] = (u, e_uv)
                        improved = True
                        break
                    w, _ = matched_v[v]
                    if w == u:
                        continue
                    aug_z = None
                    for z, e_wz in adj_v[w]:
                        if z not in matched_v and z != u:
                            aug_z = (z, e_wz)
                            break
                    if aug_z is not None:
                        z, e_wz = aug_z
                        del matched_v[v]
                        del matched_v[w]
                        matched_v[u] = (v, e_uv)
                        matched_v[v] = (u, e_uv)
                        matched_v[w] = (z, e_wz)
                        matched_v[z] = (w, e_wz)
                        improved = True
                        break

        chosen: set[int] = set()
        for _, eidx in matched_v.values():
            chosen.add(eidx)
        return sorted(chosen)


    def _edge_color_matchings(
        edges: list[tuple[int, int, Any]],
        max_matching_size: int | None = None,
        priority: dict[tuple[int, int], float] | None = None,
    ) -> list[list[tuple[int, int, Any]]]:
        """Partition `edges` into a minimal number of large matchings, ordered by max vertex overlap."""
        rem = list(edges)
        raw_matchings: list[list[tuple[int, int, Any]]] = []
        while rem:
            idxs = _find_large_matching(rem, priority=priority)
            idx_set = set(idxs)
            if max_matching_size is not None and len(idxs) > max_matching_size:
                idxs = idxs[:max_matching_size]
                idx_set = set(idxs)
            raw_matchings.append([rem[i] for i in idxs])
            rem = [e for i, e in enumerate(rem) if i not in idx_set]
        if len(raw_matchings) <= 2:
            return raw_matchings

        # Order matchings greedily: start with the largest matching, then pick highest vertex overlap
        verts = [{u for u, _, _ in ms} | {v for _, v, _ in ms} for ms in raw_matchings]
        used = [False] * len(raw_matchings)
        first = max(range(len(raw_matchings)), key=lambda i: len(raw_matchings[i]))
        order = [first]
        used[first] = True
        while len(order) < len(raw_matchings):
            prev_v = verts[order[-1]]
            nxt = max(
                (i for i in range(len(raw_matchings)) if not used[i]),
                key=lambda i: (len(prev_v & verts[i]), len(raw_matchings[i])),
            )
            order.append(nxt)
            used[nxt] = True
        return [raw_matchings[i] for i in order]


    def _execute_aod_batches(
        gs: GridState,
        moves_map: dict[int, Site],
        src_zone_name: str,
        dst_zone_name: str,
    ) -> None:
        """Move every `q -> dst` in `moves_map` (all sources in `src_zone_name`, all targets in `dst_zone_name`)
        using greedy maximal ghost-free monotone rectangular AOD batches. Handles permutation cycles safely."""
        geom = gs.geom
        todo = {q: tgt for q, tgt in moves_map.items() if gs.positions[q] != tgt}
        if not todo:
            return

        # Fast check: can ALL of `todo` move in ONE single rectangular move?
        def _try_single_rect(sub: dict[int, Site]) -> bool:
            if not sub:
                return True
            xmap: dict[int, int] = {}
            ymap: dict[int, int] = {}
            sub_srcs: set[Site] = set()
            for q, dst in sub.items():
                s = gs.positions[q]
                sub_srcs.add(s)
                if s[0] in xmap and xmap[s[0]] != dst[0]:
                    return False
                if s[1] in ymap and ymap[s[1]] != dst[1]:
                    return False
                xmap[s[0]] = dst[0]
                ymap[s[1]] = dst[1]
            xs = sorted(xmap)
            ys = sorted(ymap)
            txs = [xmap[x] for x in xs]
            tys = [ymap[y] for y in ys]
            if any(txs[i] >= txs[i + 1] for i in range(len(txs) - 1)):
                return False
            if any(tys[i] >= tys[i + 1] for i in range(len(tys) - 1)):
                return False
            # Ghost check in bounding box
            for x in range(xs[0], xs[-1] + 1):
                for y in range(ys[0], ys[-1] + 1):
                    if (x, y) in gs.occupant and (x, y) not in sub_srcs:
                        return False
            # All mapped intersections must be in valid zones; all atom targets in dst_zone & collision-free
            for x in xs:
                tx = xmap[x]
                for y in ys:
                    ty = ymap[y]
                    if (tx, ty) not in geom.zone_at:
                        return False
            for q, dst in sub.items():
                if geom.zone_at[dst].name != dst_zone_name:
                    return False
                occ = gs.occupant.get(dst)
                if occ is not None and gs.positions[occ] not in sub_srcs:
                    return False
            gs.move(tuple(range(ys[0], ys[-1] + 1)), tuple(range(xs[0], xs[-1] + 1)), tuple(tys), tuple(txs))
            return True

        if _try_single_rect(todo):
            return

        # Greedy maximal ghost-free monotone rectangular batching
        dst_zone = geom.zones[dst_zone_name]
        while todo:
            # 1. Try rigid same-displacement (dx, dy) batches across ALL `todo` (supports simultaneous shifts!)
            by_disp: dict[tuple[int, int], list[int]] = defaultdict(list)
            for q, t in todo.items():
                s = gs.positions[q]
                by_disp[(t[0] - s[0], t[1] - s[1])].append(q)
            best_disp_grp: dict[int, Site] | None = None
            best_disp_gain = 0.0
            for _, qlist in by_disp.items():
                if len(qlist) < 2:
                    continue
                qlist.sort(key=lambda qq: (gs.positions[qq][1], gs.positions[qq][0]))
                grp: dict[int, Site] = {}
                for q in qlist:
                    cand = dict(grp)
                    cand[q] = todo[q]
                    xs = sorted({gs.positions[qq][0] for qq in cand})
                    ys = sorted({gs.positions[qq][1] for qq in cand})
                    srcs = {gs.positions[qq] for qq in cand}
                    ok = True
                    for x in range(xs[0], xs[-1] + 1):
                        for y in range(ys[0], ys[-1] + 1):
                            if (x, y) in gs.occupant and (x, y) not in srcs:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        # Also check no target collides with an atom outside `srcs`
                        for qq, dst in cand.items():
                            occ = gs.occupant.get(dst)
                            if occ is not None and gs.positions[occ] not in srcs:
                                ok = False
                                break
                    if ok:
                        grp = cand
                if len(grp) >= 2:
                    d_sum = sum(geom.dist(gs.positions[q], t) for q, t in grp.items())
                    d_max = max(geom.dist(gs.positions[q], t) for q, t in grp.items())
                    gain = d_sum - d_max
                    if gain > best_disp_gain:
                        best_disp_gain = gain
                        best_disp_grp = grp

            if best_disp_grp is not None and _try_single_rect(best_disp_grp):
                for q in best_disp_grp:
                    del todo[q]
                continue

            # 1b. 1D Row-Slice & Col-Slice Monotone Run Batcher (groups atoms with different dx/dy on same row/col!)
            best_1d_grp: dict[int, Site] | None = None
            best_1d_gain = 0.0

            by_row_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
            by_col_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
            for q, t in todo.items():
                s = gs.positions[q]
                by_row_pair[(s[1], t[1])].append(q)
                by_col_pair[(s[0], t[0])].append(q)

            for (sy, _ty), qlist in by_row_pair.items():
                if len(qlist) < 2:
                    continue
                qlist.sort(key=lambda qq: gs.positions[qq][0])
                n_q = len(qlist)
                for a in range(n_q):
                    run: dict[int, Site] = {qlist[a]: todo[qlist[a]]}
                    prev_x = gs.positions[qlist[a]][0]
                    prev_tx = todo[qlist[a]][0]
                    occ0 = gs.occupant.get(todo[qlist[a]])
                    if occ0 is not None and occ0 != qlist[a]:
                        # Wait: allow if occ0 is also in the same run (checked at validation)
                        pass
                    for b in range(a + 1, n_q):
                        qb = qlist[b]
                        xb = gs.positions[qb][0]
                        txb = todo[qb][0]
                        # Check no non-run atom sits on row sy between prev_x and xb
                        if any((cx, sy) in gs.occupant for cx in range(prev_x + 1, xb)):
                            break
                        if txb <= prev_tx:
                            break
                        run[qb] = todo[qb]
                        prev_x = xb
                        prev_tx = txb
                        if len(run) >= 2:
                            run_set = set(run)
                            if all(gs.occupant.get(dst) is None or gs.occupant.get(dst) in run_set for dst in run.values()):
                                d_sum = sum(geom.dist(gs.positions[qq], dst) for qq, dst in run.items())
                                d_max = max(geom.dist(gs.positions[qq], dst) for qq, dst in run.items())
                                if d_sum - d_max > best_1d_gain:
                                    best_1d_gain = d_sum - d_max
                                    best_1d_grp = dict(run)

            for (sx, _tx), qlist in by_col_pair.items():
                if len(qlist) < 2:
                    continue
                qlist.sort(key=lambda qq: gs.positions[qq][1])
                n_q = len(qlist)
                for a in range(n_q):
                    run = {qlist[a]: todo[qlist[a]]}
                    prev_y = gs.positions[qlist[a]][1]
                    prev_ty = todo[qlist[a]][1]
                    for b in range(a + 1, n_q):
                        qb = qlist[b]
                        yb = gs.positions[qb][1]
                        tyb = todo[qb][1]
                        if any((sx, cy) in gs.occupant for cy in range(prev_y + 1, yb)):
                            break
                        if tyb <= prev_ty:
                            break
                        run[qb] = todo[qb]
                        prev_y = yb
                        prev_ty = tyb
                        if len(run) >= 2:
                            run_set = set(run)
                            if all(gs.occupant.get(dst) is None or gs.occupant.get(dst) in run_set for dst in run.values()):
                                d_sum = sum(geom.dist(gs.positions[qq], dst) for qq, dst in run.items())
                                d_max = max(geom.dist(gs.positions[qq], dst) for qq, dst in run.items())
                                if d_sum - d_max > best_1d_gain:
                                    best_1d_gain = d_sum - d_max
                                    best_1d_grp = dict(run)

            if best_1d_grp is not None:
                # Try extending `best_1d_grp` into a 2D multi-row/multi-col monotone rectangle!
                xmap_1d = {gs.positions[q][0]: t[0] for q, t in best_1d_grp.items()}
                ymap_1d = {gs.positions[q][1]: t[1] for q, t in best_1d_grp.items()}
                srcs_1d = {gs.positions[q] for q in best_1d_grp}
                for q_ext, t_ext in sorted(todo.items(), key=lambda item: -geom.dist(gs.positions[item[0]], item[1])):
                    if q_ext in best_1d_grp:
                        continue
                    s_ext = gs.positions[q_ext]
                    if s_ext[0] in xmap_1d and xmap_1d[s_ext[0]] != t_ext[0]:
                        continue
                    if s_ext[1] in ymap_1d and ymap_1d[s_ext[1]] != t_ext[1]:
                        continue
                    nxm = dict(xmap_1d)
                    nym = dict(ymap_1d)
                    nxm[s_ext[0]] = t_ext[0]
                    nym[s_ext[1]] = t_ext[1]
                    xs = sorted(nxm)
                    ys = sorted(nym)
                    txs = [nxm[x] for x in xs]
                    tys = [nym[y] for y in ys]
                    if any(txs[i] >= txs[i + 1] for i in range(len(txs) - 1)):
                        continue
                    if any(tys[i] >= tys[i + 1] for i in range(len(tys) - 1)):
                        continue
                    nsrcs = srcs_1d | {s_ext}
                    if any(
                        (x, y) in gs.occupant and (x, y) not in nsrcs
                        for x in range(xs[0], xs[-1] + 1)
                        for y in range(ys[0], ys[-1] + 1)
                    ):
                        continue
                    occ_t = gs.occupant.get(t_ext)
                    if occ_t is not None and gs.positions[occ_t] not in nsrcs:
                        continue
                    if any((nxm[x], nym[y]) not in geom.zone_at for x in xs for y in ys):
                        continue
                    best_1d_grp[q_ext] = t_ext
                    xmap_1d, ymap_1d, srcs_1d = nxm, nym, nsrcs
                if _try_single_rect(best_1d_grp):
                    for q in best_1d_grp:
                        del todo[q]
                    continue

            # 2. Seed general monotone rectangle from atoms whose target is currently free
            free_ready = [q for q, dst in todo.items() if dst not in gs.occupant]
            if free_ready:
                # Try growing a general monotone rectangle from the farthest free_ready atom
                free_ready.sort(key=lambda qq: -geom.dist(gs.positions[qq], todo[qq]))
                seed_q = free_ready[0]
                grp = {seed_q: todo[seed_q]}
                xmap = {gs.positions[seed_q][0]: todo[seed_q][0]}
                ymap = {gs.positions[seed_q][1]: todo[seed_q][1]}
                grp_srcs = {gs.positions[seed_q]}
                if len(free_ready) >= 2:
                    for q in free_ready[1:]:
                        s, t = gs.positions[q], todo[q]
                        if s[0] in xmap and xmap[s[0]] != t[0]:
                            continue
                        if s[1] in ymap and ymap[s[1]] != t[1]:
                            continue
                        nxmap = dict(xmap)
                        nymap = dict(ymap)
                        nxmap[s[0]] = t[0]
                        nymap[s[1]] = t[1]
                        xs = sorted(nxmap)
                        ys = sorted(nymap)
                        txs = [nxmap[x] for x in xs]
                        tys = [nymap[y] for y in ys]
                        if any(txs[i] >= txs[i + 1] for i in range(len(txs) - 1)):
                            continue
                        if any(tys[i] >= tys[i + 1] for i in range(len(tys) - 1)):
                            continue
                        nsrcs = grp_srcs | {s}
                        ghost = False
                        for x in range(xs[0], xs[-1] + 1):
                            for y in range(ys[0], ys[-1] + 1):
                                if (x, y) in gs.occupant and (x, y) not in nsrcs:
                                    ghost = True
                                    break
                            if ghost:
                                break
                        if ghost:
                            continue
                        if any((nxmap[x], nymap[y]) not in geom.zone_at for x in xs for y in ys):
                            continue
                        grp[q] = t
                        xmap, ymap, grp_srcs = nxmap, nymap, nsrcs
                if _try_single_rect(grp):
                    for q in grp:
                        del todo[q]
                    continue
                # Fallback: move seed_q alone
                gs.move_site(seed_q, todo.pop(seed_q))
                continue

            # 2. All remaining atoms in `todo` have occupied targets (permutation cycle inside `src_zone_name` == `dst_zone_name`!)
            # First check if a cyclic 2-atom or k-atom simultaneous rectangular shift works (e.g. a row/col shift)
            # Otherwise park one atom temporarily at the nearest empty site inside `dst_zone` to break the cycle!
            q0 = next(iter(todo))
            s0 = gs.positions[q0]
            empty_in_zone = [
                (x, y)
                for y in range(dst_zone.y[0], dst_zone.y[1] + 1)
                for x in range(dst_zone.x[0], dst_zone.x[1] + 1)
                if (x, y) not in gs.occupant
            ]
            if not empty_in_zone:
                raise RuntimeError(f"No empty site in zone {dst_zone_name!r} to break permutation cycle.")
            park = min(empty_in_zone, key=lambda s: geom.dist(s0, s))
            gs.move_site(q0, park)


    def _evict_to_storage(
        gs: GridState,
        qubits_to_evict: Sequence[int],
        avoid_rows: set[int] | None = None,
        preferred: dict[int, Site] | None = None,
    ) -> None:
        """Move `qubits_to_evict` from `entangling` to `storage` (strictly single-zone `ent -> st`)."""
        geom = gs.geom
        st = geom.storage
        ent = geom.entangling
        to_move = [q for q in qubits_to_evict if geom.zone_at[gs.positions[q]].name == ent.name]
        if not to_move:
            return
        avoid = avoid_rows or set()
        sx0, sx1 = st.x
        sy0, sy1 = st.y

        # Fast path: if ALL `to_move` form a ghost-free rectangle in `entangling` and fit in free storage rows!
        to_set = set(to_move)
        occ_ys = sorted({gs.positions[q][1] for q in to_move})
        occ_xs = sorted({gs.positions[q][0] for q in to_move})
        no_ghosts = not any(
            (x, y) in gs.occupant and gs.occupant[(x, y)] not in to_set
            for x in range(occ_xs[0], occ_xs[-1] + 1)
            for y in range(occ_ys[0], occ_ys[-1] + 1)
        )
        if no_ghosts and len(occ_ys) <= (sy1 - sy0 + 1) and len(occ_xs) <= (sx1 - sx0 + 1):
            tgt_xs = tuple(occ_xs) if all(sx0 <= x <= sx1 for x in occ_xs) else tuple(range(sx0, sx0 + len(occ_xs)))
            # Search for highest contiguous block of `len(occ_ys)` empty rows in storage (first avoiding `avoid`, then anywhere)
            for allow_avoid in (False, True):
                best_ys = None
                for top_y in range(sy1, sy0 + len(occ_ys) - 2, -1):
                    cand_ys = tuple(range(top_y - len(occ_ys) + 1, top_y + 1))
                    if not allow_avoid and any(y in avoid for y in cand_ys):
                        continue
                    if all((tx, ty) not in gs.occupant for tx in range(tgt_xs[0], tgt_xs[-1] + 1) for ty in cand_ys):
                        best_ys = cand_ys
                        break
                if best_ys is not None:
                    gs.move(
                        tuple(range(occ_ys[0], occ_ys[-1] + 1)),
                        tuple(range(occ_xs[0], occ_xs[-1] + 1)),
                        best_ys,
                        tgt_xs,
                    )
                    return

        # Assign free storage sites: keep adjacent horizontal pairs in `to_move` adjacent in storage so they drop in ONE move!
        free_pref = [
            (x, y)
            for y in range(sy1, sy0 - 1, -1)
            if y not in avoid
            for x in range(sx0, sx1 + 1)
            if (x, y) not in gs.occupant
        ]
        free_fall = [
            (x, y)
            for y in range(sy1, sy0 - 1, -1)
            if y in avoid
            for x in range(sx0, sx1 + 1)
            if (x, y) not in gs.occupant
        ]
        avail_pref = set(free_pref)
        avail_fall = set(free_fall)
        moves_map: dict[int, Site] = {}
        remaining = set(to_move)

        # First pair up horizontal adjacent atoms (x, y) & (x+1, y) in `remaining` -> adjacent storage slots (tx, ty), (tx+1, ty)!
        for q in sorted(to_move, key=lambda qq: (gs.positions[qq][1], gs.positions[qq][0])):
            if q not in remaining:
                continue
            s = gs.positions[q]
            s_right = (s[0] + 1, s[1])
            q_right = gs.occupant.get(s_right)
            if q_right is not None and q_right in remaining:
                adj_slots = [(x, y) for (x, y) in avail_pref if (x + 1, y) in avail_pref]
                if adj_slots:
                    best_l = min(adj_slots, key=lambda t: max(geom.dist(s, t), geom.dist(s_right, (t[0] + 1, t[1]))))
                    best_r = (best_l[0] + 1, best_l[1])
                    avail_pref.remove(best_l)
                    avail_pref.remove(best_r)
                    moves_map[q] = best_l
                    moves_map[q_right] = best_r
                    remaining.remove(q)
                    remaining.remove(q_right)

        for q in sorted(remaining, key=lambda qq: (gs.positions[qq][1], gs.positions[qq][0])):
            s = gs.positions[q]
            if preferred and q in preferred and preferred[q] in avail_pref and geom.dist(s, preferred[q]) <= 3.0:
                tgt = preferred[q]
                avail_pref.remove(tgt)
            elif avail_pref:
                tgt = min(avail_pref, key=lambda t: geom.dist(s, t))
                avail_pref.remove(tgt)
            else:
                tgt = min(avail_fall, key=lambda t: geom.dist(s, t))
                avail_fall.remove(tgt)
            moves_map[q] = tgt
        _execute_aod_batches(gs, moves_map, ent.name, st.name)


    def _assign_matching_booths(
        gs: GridState,
        pairs: list[tuple[int, int]],
        target_hint: set[int] | None = None,
        booths_all: list[tuple[Site, Site]] | None = None,
        extra_rows: int = 0,
    ) -> dict[int, Site]:
        """Assign each pair `(u, v)` to an interaction booth `(bL, bR)` in the lowest needed rows of `entangling`,
        minimizing total movement distance via greedy + 2-opt booth swap polish."""
        geom = gs.geom
        ent = geom.entangling
        booths = booths_all if booths_all is not None else _ent_booths(geom)
        m = len(pairs)
        if m == 0:
            return {}

        # Limit candidate booths to the minimum number of rows needed (+ optional breathing row!)
        if ent.pair_axis == "x":
            k_per_row = (ent.x[1] - ent.x[0] + 1) // 2
            r_needed = max(1, math.ceil(m / k_per_row)) + extra_rows
            cand_booths = booths[: min(len(booths), r_needed * k_per_row)]
        else:
            cand_booths = booths[: min(len(booths), max(m, 5) + extra_rows)]

        n_b = len(cand_booths)
        th = target_hint or set()
        # Slot-0 bias for small matchings (target on slot 0 dips distance 1.000 instead of 1.221!)
        slot0_bias = 0.35 if (m <= 2 and th) else 0.0

        # Precompute pair-to-booth cost and best orientation (with vertical-alignment, monotone X-order,
        # and convex dy^2 tie-break so storage rows map in ASCENDING Y-order (6->10, 7->11, 8->12, 9->13)
        # for 1-move 2D bulk lifts instead of inverted (9->10, 6->13)!)
        # cost[i][b] = (dist, (tgt_u, tgt_v))
        cost_mat: list[list[tuple[float, Site, Site]]] = []
        for u, v in pairs:
            su, sv = gs.positions[u], gs.positions[v]
            # Strongly prefer preserving left-to-right X order when u and v sit on the same row (enables 1-move 2-atom lift!)
            same_row_ord = 0.55 if su[1] == sv[1] else 0.15
            ord_1 = -same_row_ord if su[0] < sv[0] else (same_row_ord if su[0] > sv[0] else 0.0)
            ord_2 = -ord_1
            row_c: list[tuple[float, Site, Site]] = []
            for bL, bR in cand_booths:
                a1 = (-0.25 if su[0] == bL[0] else 0.0) + (-0.25 if sv[0] == bR[0] else 0.0)
                a2 = (-0.25 if su[0] == bR[0] else 0.0) + (-0.25 if sv[0] == bL[0] else 0.0)
                dy_pen = 0.06 * ((bL[1] - su[1]) ** 2 + (bR[1] - sv[1]) ** 2)
                # Never invert a pair that is ALREADY sitting on this exact booth with zero distance!
                d1_raw = geom.dist(su, bL) + geom.dist(sv, bR)
                d2_raw = geom.dist(su, bR) + geom.dist(sv, bL)
                c1 = d1_raw + (0.0 if d1_raw < 1e-6 else (a1 + ord_1 + dy_pen - (slot0_bias if u in th else 0.0)))
                c2 = d2_raw + (0.0 if d2_raw < 1e-6 else (a2 + ord_2 + dy_pen - (slot0_bias if v in th else 0.0)))
                if c1 <= c2:
                    row_c.append((c1, bL, bR))
                else:
                    row_c.append((c2, bR, bL))
            cost_mat.append(row_c)

        # Greedy assignment: lock zero-cost (already paired!) edges first, then smallest best cost
        assigned: list[int] = [-1] * m
        used_b = [False] * n_b
        pair_order = sorted(range(m), key=lambda i: min(cost_mat[i][b][0] for b in range(n_b)))
        for i in pair_order:
            best_b = min((b for b in range(n_b) if not used_b[b]), key=lambda b: cost_mat[i][b][0])
            assigned[i] = best_b
            used_b[best_b] = True

        # 2-opt booth swap + relocate to unused booth polish (super fast for m <= 25!)
        for _ in range(4):
            improved = False
            # Relocate to unused booth
            free_bs = [b for b in range(n_b) if not used_b[b]]
            if free_bs:
                for i in range(m):
                    bi = assigned[i]
                    ci = cost_mat[i][bi][0]
                    best_fb = min(free_bs, key=lambda b: cost_mat[i][b][0])
                    if cost_mat[i][best_fb][0] + 1e-6 < ci:
                        used_b[bi] = False
                        used_b[best_fb] = True
                        assigned[i] = best_fb
                        free_bs.remove(best_fb)
                        free_bs.append(bi)
                        improved = True
            # Pairwise swaps
            for i in range(m):
                bi = assigned[i]
                ci = cost_mat[i][bi][0]
                for j in range(i + 1, m):
                    bj = assigned[j]
                    if cost_mat[i][bj][0] + cost_mat[j][bi][0] + 1e-6 < ci + cost_mat[j][bj][0]:
                        assigned[i], assigned[j] = bj, bi
                        bi = bj
                        ci = cost_mat[i][bi][0]
                        improved = True
            if not improved:
                break

        target_pos: dict[int, Site] = {}
        for i, (u, v) in enumerate(pairs):
            _, tu, tv = cost_mat[i][assigned[i]]
            target_pos[u] = tu
            target_pos[v] = tv
        return target_pos


    def _assemble_matching_in_entangling(
        gs: GridState,
        pairs: list[tuple[int, int]],
        target_hint: set[int] | None = None,
        reserve_top_st_rows: int = 0,
        preferred_st: dict[int, Site] | None = None,
        extra_rows: int = 0,
    ) -> None:
        """Arrange `pairs` in `entangling` with ZERO idle atoms and strictly single-zone moves."""
        geom = gs.geom
        ent = geom.entangling
        st = geom.storage
        active = {q for u, v in pairs for q in (u, v)}

        # 1. Evict all atoms in `entangling` NOT in `active` to `storage`
        avoid_rows: set[int] = set()
        if reserve_top_st_rows > 0:
            sy0, sy1 = st.y
            avoid_rows = set(range(max(sy0, sy1 - reserve_top_st_rows + 1), sy1 + 1))
        idle_ent = [q for q, s in gs.positions.items() if geom.zone_at[s].name == ent.name and q not in active]
        if idle_ent:
            _evict_to_storage(gs, idle_ent, avoid_rows=avoid_rows, preferred=preferred_st)

        # 2. Assign pair booths
        target_pos = _assign_matching_booths(gs, pairs, target_hint=target_hint, extra_rows=extra_rows)

        # 3a. Rearrange atoms ALREADY in `entangling` (strictly `ent -> ent`)
        in_ent_moves = {
            q: tgt for q, tgt in target_pos.items() if geom.zone_at[gs.positions[q]].name == ent.name and gs.positions[q] != tgt
        }
        if in_ent_moves:
            _execute_aod_batches(gs, in_ent_moves, ent.name, ent.name)

        # 3b. Lift entering atoms from `storage` to `entangling` (strictly `st -> ent`)
        from_st_moves = {
            q: tgt for q, tgt in target_pos.items() if geom.zone_at[gs.positions[q]].name == st.name
        }
        if from_st_moves:
            _execute_aod_batches(gs, from_st_moves, st.name, ent.name)


    def _dip_entangling_for_1q(gs: GridState, q1_ops: dict[int, list[tuple[str, float]]]) -> None:
        """Execute 1Q gates `q1_ops` on qubits (returning dipped entangling qubits to their exact sites)."""
        geom = gs.geom
        st = geom.storage
        ent = geom.entangling
        sx0, sx1 = st.x
        sy0, sy1 = st.y

        # 1. Execute any qubits already in storage for free
        need_ent: list[int] = []
        for q, ops_q in q1_ops.items():
            if not ops_q:
                continue
            if geom.zone_at[gs.positions[q]].name == st.name:
                for gn, ang in ops_q:
                    gs.gate(gn, (q,), (ang,))
            else:
                need_ent.append(q)
        if not need_ent:
            return

        # 2. Evaluate Option A1: direct 1-move bulk dip into free storage rows (no push)
        #    Option A2: 1-move push storage rows to bottom + 1-move bulk dip to top storage rows + pull back up
        #    Option A3: chunked row-band bulk dip into available top free storage rows (0 storage push!)
        all_ent_qs = [q for q, s in gs.positions.items() if geom.zone_at[s].name == ent.name]
        bulk_direct = None
        bulk_push = None
        bulk_chunked = None
        if all_ent_qs:
            occ_ys = sorted({gs.positions[q][1] for q in all_ent_qs})
            occ_xs = sorted({gs.positions[q][0] for q in all_ent_qs})
            if len(occ_ys) <= (sy1 - sy0 + 1) and len(occ_xs) <= (sx1 - sx0 + 1):
                tgt_xs = tuple(occ_xs) if all(sx0 <= x <= sx1 for x in occ_xs) else tuple(range(sx0, sx0 + len(occ_xs)))
                best_ys = None
                for top_y in range(sy1, sy0 + len(occ_ys) - 2, -1):
                    cand_ys = tuple(range(top_y - len(occ_ys) + 1, top_y + 1))
                    if all((tx, ty) not in gs.occupant for tx in range(tgt_xs[0], tgt_xs[-1] + 1) for ty in cand_ys):
                        best_ys = cand_ys
                        break
                if best_ys is not None:
                    xmap = dict(zip(occ_xs, tgt_xs))
                    ymap = dict(zip(occ_ys, best_ys))
                    d_one_way = max(
                        geom.dist(gs.positions[q], (xmap[gs.positions[q][0]], ymap[gs.positions[q][1]]))
                        for q in all_ent_qs
                    )
                    bulk_direct = (2.0 * d_one_way, occ_ys, occ_xs, best_ys, tgt_xs)

                # Option A2: check pushing all occupied storage rows down to sy0..sy0+k_st-1 so top `len(occ_ys)` rows are free!
                top_dip_ys = tuple(range(sy1 - len(occ_ys) + 1, sy1 + 1))
                if best_ys != top_dip_ys:
                    st_qs = [q for q, s in gs.positions.items() if geom.zone_at[s].name == st.name]
                    if st_qs:
                        st_ys = sorted({gs.positions[q][1] for q in st_qs})
                        st_xs = sorted({gs.positions[q][0] for q in st_qs})
                        if len(st_ys) + len(occ_ys) <= (sy1 - sy0 + 1):
                            push_ys = tuple(range(sy0, sy0 + len(st_ys)))
                            d_push = max(abs(y1 - y2) for y1, y2 in zip(st_ys, push_ys))
                            xmap = dict(zip(occ_xs, tgt_xs))
                            ymap = dict(zip(occ_ys, top_dip_ys))
                            d_dip = max(
                                geom.dist(gs.positions[q], (xmap[gs.positions[q][0]], ymap[gs.positions[q][1]]))
                                for q in all_ent_qs
                            )
                            bulk_push = (2.0 * d_push + 2.0 * d_dip, st_ys, st_xs, push_ys, occ_ys, occ_xs, top_dip_ys, tgt_xs)

                    # Option A3: count how many top rows `sy1 - f + 1 .. sy1` are completely free across `tgt_xs[0]..tgt_xs[-1]`
                    f_top = 0
                    for ry in range(sy1, sy0 - 1, -1):
                        if all((tx, ry) not in gs.occupant for tx in range(tgt_xs[0], tgt_xs[-1] + 1)):
                            f_top += 1
                        else:
                            break
                    if 1 <= f_top < len(occ_ys):
                        # Partition only the entangling rows that actually contain a qubit in `need_ent` into bands of `<= f_top`!
                        need_rows = sorted({gs.positions[q][1] for q in need_ent})
                        chunks: list[tuple[list[int], list[int], tuple[int, ...], tuple[int, ...]]] = []
                        tot_chunk_cost = 0.0
                        idx_r = 0
                        while idx_r < len(need_rows):
                            # Take a contiguous slice of `occ_ys` starting at `need_rows[idx_r]` of length up to `f_top`
                            y_start = need_rows[idx_r]
                            pos_in_occ = occ_ys.index(y_start)
                            band_ys = occ_ys[pos_in_occ : min(len(occ_ys), pos_in_occ + f_top)]
                            band_set = set(band_ys)
                            band_qs = [q for q in all_ent_qs if gs.positions[q][1] in band_set]
                            band_xs = sorted({gs.positions[q][0] for q in band_qs})
                            b_tgt_xs = tuple(band_xs) if all(sx0 <= x <= sx1 for x in band_xs) else tuple(range(sx0, sx0 + len(band_xs)))
                            b_tgt_ys = tuple(range(sy1 - len(band_ys) + 1, sy1 + 1))
                            bxmap = dict(zip(band_xs, b_tgt_xs))
                            bymap = dict(zip(band_ys, b_tgt_ys))
                            d_b = max(
                                geom.dist(gs.positions[q], (bxmap[gs.positions[q][0]], bymap[gs.positions[q][1]]))
                                for q in band_qs
                            )
                            tot_chunk_cost += 2.0 * d_b
                            chunks.append((band_ys, band_xs, b_tgt_ys, b_tgt_xs))
                            while idx_r < len(need_rows) and need_rows[idx_r] <= band_ys[-1]:
                                idx_r += 1
                        bulk_chunked = (tot_chunk_cost, chunks)

        # 3. Evaluate Option B: selective dip of ONLY `need_ent` atoms (individually / batched)
        free_st = [(x, y) for y in range(sy1, sy0 - 1, -1) for x in range(sx0, sx1 + 1) if (x, y) not in gs.occupant]
        sel_cost = float("inf")
        sel_assign: dict[int, Site] = {}
        if len(free_st) >= len(need_ent):
            avail = set(free_st)
            tot_sel = 0.0
            for q in sorted(need_ent, key=lambda qq: gs.positions[qq][1]):
                s = gs.positions[q]
                bt = min(avail, key=lambda t: geom.dist(s, t))
                avail.remove(bt)
                sel_assign[q] = bt
                tot_sel += 2.0 * geom.dist(s, bt)
            sel_cost = tot_sel

        c_dir = bulk_direct[0] if bulk_direct is not None else float("inf")
        c_psh = bulk_push[0] if bulk_push is not None else float("inf")
        c_chk = bulk_chunked[0] if bulk_chunked is not None else float("inf")
        best_c = min(c_dir, c_psh, c_chk, sel_cost)

        if c_dir == best_c and bulk_direct is not None:
            _, occ_ys, occ_xs, best_ys, tgt_xs = bulk_direct
            gs.move(tuple(range(occ_ys[0], occ_ys[-1] + 1)), tuple(range(occ_xs[0], occ_xs[-1] + 1)), best_ys, tgt_xs)
            for q in need_ent:
                for gn, ang in q1_ops[q]:
                    gs.gate(gn, (q,), (ang,))
            gs.move(best_ys, tuple(range(tgt_xs[0], tgt_xs[-1] + 1)), tuple(occ_ys), tuple(occ_xs))
            return

        if c_chk == best_c and bulk_chunked is not None:
            _, chunks = bulk_chunked
            need_set = set(need_ent)
            for band_ys, band_xs, b_tgt_ys, b_tgt_xs in chunks:
                band_y_set = set(band_ys)
                band_need = [q for q in need_ent if q in need_set and gs.positions[q][1] in band_y_set]
                gs.move(tuple(range(band_ys[0], band_ys[-1] + 1)), tuple(range(band_xs[0], band_xs[-1] + 1)), b_tgt_ys, b_tgt_xs)
                for q in band_need:
                    for gn, ang in q1_ops[q]:
                        gs.gate(gn, (q,), (ang,))
                    need_set.remove(q)
                gs.move(b_tgt_ys, tuple(range(b_tgt_xs[0], b_tgt_xs[-1] + 1)), tuple(band_ys), tuple(band_xs))
            return

        if c_psh == best_c and bulk_push is not None:
            _, st_ys, st_xs, push_ys, occ_ys, occ_xs, top_dip_ys, tgt_xs = bulk_push
            # 1. Push storage idle rows down to bottom in 1 move
            gs.move(tuple(range(st_ys[0], st_ys[-1] + 1)), tuple(range(st_xs[0], st_xs[-1] + 1)), push_ys, tuple(st_xs))
            # 2. Bulk dip entangling atoms to topmost storage rows in 1 move
            gs.move(tuple(range(occ_ys[0], occ_ys[-1] + 1)), tuple(range(occ_xs[0], occ_xs[-1] + 1)), top_dip_ys, tgt_xs)
            for q in need_ent:
                for gn, ang in q1_ops[q]:
                    gs.gate(gn, (q,), (ang,))
            # 3. Bulk lift entangling atoms back in 1 move
            gs.move(top_dip_ys, tuple(range(tgt_xs[0], tgt_xs[-1] + 1)), tuple(occ_ys), tuple(occ_xs))
            # 4. Pull storage idle rows right back up to top in 1 move
            pull_ys = tuple(range(sy1 - len(st_ys) + 1, sy1 + 1))
            gs.move(push_ys, tuple(range(st_xs[0], st_xs[-1] + 1)), pull_ys, tuple(st_xs))
            return

        if sel_assign:
            orig = {q: gs.positions[q] for q in sel_assign}
            _execute_aod_batches(gs, sel_assign, ent.name, st.name)
            for q in need_ent:
                for gn, ang in q1_ops[q]:
                    gs.gate(gn, (q,), (ang,))
            _execute_aod_batches(gs, orig, st.name, ent.name)
            return

        # Fallback chunked dip (if storage is nearly full, e.g. 95+ qubits)
        for q in need_ent:
            orig_s = gs.positions[q]
            free_now = [(x, y) for y in range(sy1, sy0 - 1, -1) for x in range(sx0, sx1 + 1) if (x, y) not in gs.occupant]
            bt = min(free_now, key=lambda t: geom.dist(orig_s, t))
            gs.move_site(q, bt)
            for gn, ang in q1_ops[q]:
                gs.gate(gn, (q,), (ang,))
            gs.move_site(q, orig_s)


    def _apply_1q_in_storage(
        gs: GridState,
        q1_ops: dict[int, list[tuple[str, float]]],
        has_future_2q: set[int],
        keep_in_storage: bool = False,
    ) -> None:
        """Execute 1Q gates in storage; one-way drops qubits with no future 2Q (or when `keep_in_storage=True`)."""
        if not any(q1_ops.values()):
            return
        geom = gs.geom
        st = geom.storage
        ent = geom.entangling

        # Execute qubits already in storage
        need_ent: list[int] = []
        for q, ops_q in q1_ops.items():
            if not ops_q:
                continue
            if geom.zone_at[gs.positions[q]].name == st.name:
                for gn, ang in ops_q:
                    gs.gate(gn, (q,), (ang,))
            else:
                need_ent.append(q)
        if not need_ent:
            return

        one_way = [q for q in need_ent if keep_in_storage or q not in has_future_2q]
        round_trip = [q for q in need_ent if q not in set(one_way)]

        if one_way and not round_trip:
            # If all entangling atoms have no future 2Q (or keep_in_storage), drop all entangling atoms cleanly if cheaper!
            all_ent = [q for q, s in gs.positions.items() if geom.zone_at[s].name == ent.name]
            if len(one_way) >= max(3, len(all_ent) // 2) and all(q not in has_future_2q or keep_in_storage for q in all_ent):
                _evict_to_storage(gs, all_ent)
            else:
                _evict_to_storage(gs, one_way)
            for q in one_way:
                for gn, ang in q1_ops[q]:
                    gs.gate(gn, (q,), (ang,))
            return

        if one_way:
            _evict_to_storage(gs, one_way)
            for q in one_way:
                for gn, ang in q1_ops[q]:
                    gs.gate(gn, (q,), (ang,))

        if round_trip:
            _dip_entangling_for_1q(gs, {q: q1_ops[q] for q in round_trip})


    def _execute_cz_matching_batch(
        gs: GridState,
        cz_pairs: list[tuple[int, int]],
        fast_mode: bool = False,
    ) -> None:
        """Execute a commuting set of pure `cz` gates via blocking-CZ matchings."""
        if not cz_pairs:
            return
        booths = _ent_booths(gs.geom)
        max_m = len(booths)
        edges = [(u, v, None) for u, v in cz_pairs]
        # Prioritize pairs already sitting in entangling!
        ent_name = gs.geom.entangling.name
        prio: dict[tuple[int, int], float] = {}
        for u, v, _ in edges:
            k = (min(u, v), max(u, v))
            in_e = (1 if gs.geom.zone_at[gs.positions[u]].name == ent_name else 0) + (
                1 if gs.geom.zone_at[gs.positions[v]].name == ent_name else 0
            )
            prio[k] = float(in_e) + (1.0 / (1.0 + gs.geom.dist(gs.positions[u], gs.positions[v])))
        matchings = _edge_color_matchings(edges, max_matching_size=max_m, priority=prio)
        for m_edges in matchings:
            pairs = [(u, v) for u, v, _ in m_edges]
            _assemble_matching_in_entangling(gs, pairs)
            for u, v in pairs:
                gs.gate("cz", (u, v))


    def _execute_rzx_matching(
        gs: GridState,
        rzx_edges: list[tuple[int, int, float]],
        preferred_st: dict[int, Site] | None = None,
    ) -> None:
        """Execute a vertex-disjoint matching of `R_ZX(t)_{ctrl, tgt} = CZ(ctrl, tgt) · RX(t)_tgt · CZ(ctrl, tgt)`,
        micro-branching over eviction reservation & clean-slate bulk drop to guarantee minimum move distance."""
        if not rzx_edges:
            return
        geom = gs.geom
        ent = geom.entangling
        k_per_row = max(1, (ent.x[1] - ent.x[0] + 1) // 2) if ent.pair_axis == "x" else 1
        r_m = math.ceil(len(rzx_edges) / k_per_row)
        pairs = [(u, v) for u, v, _ in rzx_edges]
        tgts = {v for _, v, _ in rzx_edges}
        q1_rx = {v: [("rx", ang)] for _, v, ang in rzx_edges}

        best_gs: GridState | None = None
        best_sc = float("inf")
        res_opts = sorted({0, min(2, r_m), r_m})
        ext_opts = (0, 1) if len(rzx_edges) >= 6 else (0,)
        for ext_r in ext_opts:
            for res_r in res_opts:
                if ext_r == 1 and res_r not in (0, min(2, r_m)):
                    continue
                g_cand = gs.clone()
                _assemble_matching_in_entangling(
                    g_cand, pairs, target_hint=tgts, reserve_top_st_rows=res_r, preferred_st=preferred_st, extra_rows=ext_r
                )
                for u, v, _ in rzx_edges:
                    g_cand.gate("cz", (u, v))
                _dip_entangling_for_1q(g_cand, q1_rx)
                for u, v, _ in rzx_edges:
                    g_cand.gate("cz", (u, v))
                sc = g_cand.total_distance + 2.9221 * g_cand.depth
                if sc < best_sc:
                    best_sc = sc
                    best_gs = g_cand

        # Also test Clean-Slate 1-move bulk drop of current entangling atoms before assembling `pairs`!
        cur_ent = [q for q, s in gs.positions.items() if geom.zone_at[s].name == ent.name]
        active = {q for u, v in pairs for q in (u, v)}
        if len(cur_ent) >= 4 and any(q not in active for q in cur_ent):
            g_cs = gs.clone()
            _evict_to_storage(g_cs, cur_ent, preferred=preferred_st)
            _assemble_matching_in_entangling(g_cs, pairs, target_hint=tgts, reserve_top_st_rows=0, preferred_st=preferred_st)
            for u, v, _ in rzx_edges:
                g_cs.gate("cz", (u, v))
            _dip_entangling_for_1q(g_cs, q1_rx)
            for u, v, _ in rzx_edges:
                g_cs.gate("cz", (u, v))
            sc = g_cs.total_distance + 2.9221 * g_cs.depth
            if sc < best_sc:
                best_sc = sc
                best_gs = g_cs

        assert best_gs is not None
        gs.positions = best_gs.positions
        gs.occupant = best_gs.occupant
        gs.total_distance = best_gs.total_distance
        gs.q_depth = best_gs.q_depth
        gs._layer_2q_qubits = best_gs._layer_2q_qubits
        gs._layer_2q_zones = best_gs._layer_2q_zones
        gs.ops = best_gs.ops


    def _execute_hub_star_rzx(
        gs: GridState,
        hub: int,
        star_leaves: list[tuple[int, float]],
        h_is_target: bool = False,
    ) -> None:
        """Execute a star of `R_ZX(t_i)_{hub, v_i}` (or `R_ZX(t_i)_{v_i, hub}` if `h_is_target`)
        using the 2-Leaf Pipelined Row-Rolling Hub Sweep (depth 2 per leaf, 0 idle atoms, strictly single-zone)."""
        if not star_leaves:
            return
        geom = gs.geom
        st = geom.storage
        ent = geom.entangling
        sx0, sx1 = st.x
        sy0, sy1 = st.y
        ex0, ex1 = ent.x
        ey0, ey1 = ent.y

        # Evict all non-hub atoms from entangling to storage
        non_hub_ent = [q for q, s in gs.positions.items() if geom.zone_at[s].name == ent.name and q != hub]
        if non_hub_ent:
            _evict_to_storage(gs, non_hub_ent)

        h_ops = _matrix_to_zxz(_MAT_1Q["h"]())

        # If `h_is_target` and >= 7 leaves, conjugate `hub` and every leaf by `H` in storage to flip target to leaves
        # (cuts hub depth from 3k to 2k + 6!).
        use_h_flip = h_is_target and len(star_leaves) >= 7
        if use_h_flip:
            if geom.zone_at[gs.positions[hub]].name == ent.name:
                _evict_to_storage(gs, [hub])
            for gn, ang in h_ops:
                gs.gate(gn, (hub,), (ang,))
            for v, _ in star_leaves:
                for gn, ang in h_ops:
                    gs.gate(gn, (v,), (ang,))

        # If `h_is_target` and < 4 leaves (no H-flip), run directly with `hub` dipping for `rx`
        if h_is_target and not use_h_flip:
            for v, ang in star_leaves:
                sv = gs.positions[v]
                # Pick closest booth on ey0 above sv
                booths_0 = _ent_booths(geom)[: max(1, (ex1 - ex0 + 1) // 2)]
                bL, bR = min(booths_0, key=lambda b: min(geom.dist(sv, b[0]), geom.dist(sv, b[1])))
                tv, th = (bL, bR) if geom.dist(sv, bL) <= geom.dist(sv, bR) else (bR, bL)
                # Move hub to th, lift v to tv
                if geom.zone_at[gs.positions[hub]].name == ent.name:
                    gs.move_site(hub, th)
                else:
                    gs.move_site(hub, th)
                gs.move_site(v, tv)
                gs.gate("cz", (v, hub))
                # Dip hub to vacated sv (guaranteed empty in storage!)
                gs.move_site(hub, sv)
                gs.gate("rx", (hub,), (ang,))
                gs.move_site(hub, th)
                gs.gate("cz", (v, hub))
                gs.move_site(v, sv)
            return

        # Standard mode: `hub` is control (Z), every leaf `v` is target (X)!
        # Shift all storage atoms up so the topmost occupied row sits at `sy1` (if row `sy1` is currently empty)
        st_atoms = [q for q, s in gs.positions.items() if geom.zone_at[s].name == st.name and q != hub]
        if st_atoms:
            st_ys = sorted({gs.positions[q][1] for q in st_atoms})
            if st_ys[-1] < sy1:
                # Make sure hub isn't sitting in storage blocking the shift!
                if geom.zone_at[gs.positions[hub]].name == st.name:
                    gs.move_site(hub, (ex0 + 1, ey0))
                st_xs = sorted({gs.positions[q][0] for q in st_atoms})
                dy_up = sy1 - st_ys[-1]
                gs.move(
                    tuple(range(st_ys[0], st_ys[-1] + 1)),
                    tuple(range(st_xs[0], st_xs[-1] + 1)),
                    tuple(y + dy_up for y in st_ys),
                    tuple(st_xs),
                )

        # Ensure `hub` is in `entangling` on row `ey0` (and is the ONLY atom in `entangling`)
        if geom.zone_at[gs.positions[hub]].name != ent.name:
            hx = min(max(gs.positions[hub][0], ex0), ex1)
            gs.move_site(hub, (hx, ey0))
        elif gs.positions[hub][1] != ey0:
            gs.move_site(hub, (gs.positions[hub][0], ey0))

        rem: dict[int, float] = {v: ang for v, ang in star_leaves}
        can_pipe_x = (ent.pair_axis == "x") and (sx0 >= ex0) and (sx1 <= ex1)
        k_booths = (ex1 - ex0 + 1) // 2

        def _leaf_slot_options(sv: Site) -> list[tuple[Site, Site]]:
            if can_pipe_x and ex0 <= sv[0] <= ex1:
                t_str = (sv[0], ey0)
                t_par = geom.partner_site(t_str)
                assert t_par is not None
                return [(t_str, t_par), (t_par, t_str)]
            bL, bR = min(_ent_booths(geom)[: max(1, k_booths)], key=lambda b: min(geom.dist(sv, b[0]), geom.dist(sv, b[1])))
            return [(bL, bR), (bR, bL)]

        def _do_single_leaf(v: int, ang: float) -> None:
            sv = gs.positions[v]
            h_cur = gs.positions[hub]
            tv, th = min(
                _leaf_slot_options(sv),
                key=lambda opt: geom.dist(h_cur, opt[1]) + 4.0 * geom.dist(sv, opt[0]),
            )
            if gs.positions[hub] != th:
                gs.move_site(hub, th)
            gs.move_site(v, tv)
            gs.gate("cz", (hub, v))
            gs.move_site(v, sv)
            gs.gate("rx", (v,), (ang,))
            gs.move_site(v, tv)
            gs.gate("cz", (hub, v))
            gs.move_site(v, sv)

        def _do_booth_pair(vL: int, angL: float, vR: int, angR: float, left_to_right: bool) -> None:
            """2-leaf pipelined booth: depth 3 for BOTH edges, booth exit aligns with sweep direction (step 1.70!)."""
            sL = gs.positions[vL]
            sR = gs.positions[vR]
            tL = (sL[0], ey0)
            tR = (sR[0], ey0)
            if left_to_right:
                # Start with vR (hub at tL), end with vL (hub at tR -> 1.70 to next booth's tL!)
                seq = ((vR, angR, sR, tR, tL), (vL, angL, sL, tL, tR))
            else:
                # Start with vL (hub at tR), end with vR (hub at tL -> 1.70 to prev booth's tR!)
                seq = ((vL, angL, sL, tL, tR), (vR, angR, sR, tR, tL))
            # Pass 1: CZ1 + RX for both leaves
            for v, ang, sv, tv, th in seq:
                if gs.positions[hub] != th:
                    gs.move_site(hub, th)
                gs.move_site(v, tv)
                gs.gate("cz", (hub, v))
                gs.move_site(v, sv)
                gs.gate("rx", (v,), (ang,))
            # Pass 2: CZ2 for both leaves
            for v, _, sv, tv, th in seq:
                if gs.positions[hub] != th:
                    gs.move_site(hub, th)
                gs.move_site(v, tv)
                gs.gate("cz", (hub, v))
                gs.move_site(v, sv)

        def _do_two_leaves(v1: int, ang1: float, v2: int, ang2: float) -> None:
            """2-leaf pipelined execution for ANY two leaves (optimizing intra-booth slot orientation to minimize 3x hub shuttling!)."""
            s1 = gs.positions[v1]
            s2 = gs.positions[v2]
            h_cur = gs.positions[hub]
            opts1 = _leaf_slot_options(s1)
            opts2 = _leaf_slot_options(s2)
            (tv1, th1), (tv2, th2) = min(
                ((o1, o2) for o1 in opts1 for o2 in opts2),
                key=lambda pair: (
                    geom.dist(h_cur, pair[0][1])
                    + 3.0 * geom.dist(pair[0][1], pair[1][1])
                    + 4.0 * geom.dist(s1, pair[0][0])
                    + 4.0 * geom.dist(s2, pair[1][0])
                ),
            )
            seq = ((v1, ang1, s1, tv1, th1), (v2, ang2, s2, tv2, th2))
            for v, ang, sv, tv, th in seq:
                if gs.positions[hub] != th:
                    gs.move_site(hub, th)
                gs.move_site(v, tv)
                gs.gate("cz", (hub, v))
                gs.move_site(v, sv)
                gs.gate("rx", (v,), (ang,))
            for v, _, sv, tv, th in seq:
                if gs.positions[hub] != th:
                    gs.move_site(hub, th)
                gs.move_site(v, tv)
                gs.gate("cz", (hub, v))
                gs.move_site(v, sv)

        def _sweep_row_leaves(row_leaves: list[int], ltr: bool) -> None:
            by_booth: dict[int, list[int]] = defaultdict(list)
            for v in row_leaves:
                b = (gs.positions[v][0] - ex0) // 2
                by_booth[b].append(v)
            b_order = sorted(by_booth.keys(), reverse=(not ltr))
            pending_single: int | None = None
            for idx_b, b in enumerate(b_order):
                vs = sorted(by_booth[b], key=lambda qq: gs.positions[qq][0])
                if len(vs) == 2:
                    if pending_single is not None:
                        # Only cascade through a 2-leaf booth if the IMMEDIATELY NEXT booth is a 1-leaf booth within 1 booth step!
                        next_is_adj_single = (
                            idx_b + 1 < len(b_order)
                            and len(by_booth[b_order[idx_b + 1]]) == 1
                            and abs(b_order[idx_b + 1] - b) == 1
                        )
                        v_first = vs[1] if ltr else vs[0]
                        v_second = vs[0] if ltr else vs[1]
                        if next_is_adj_single and abs(gs.positions[pending_single][0] - gs.positions[v_first][0]) <= 2:
                            _do_two_leaves(pending_single, rem.pop(pending_single), v_first, rem.pop(v_first))
                            pending_single = v_second
                            continue
                        # Check if there is a 1-leaf booth later on this row within 2 booths of pending_single:
                        # if so, keep pending_single waiting and run this 2-leaf booth intact!
                        future_close_single = any(
                            len(by_booth[b_order[k]]) == 1
                            and abs(gs.positions[by_booth[b_order[k]][0]][0] - gs.positions[pending_single][0]) <= 4
                            for k in range(idx_b + 1, len(b_order))
                        )
                        if not future_close_single:
                            _do_single_leaf(pending_single, rem.pop(pending_single))
                            pending_single = None
                    vL, vR = vs[0], vs[1]
                    _do_booth_pair(vL, rem.pop(vL), vR, rem.pop(vR), ltr)
                else:
                    v = vs[0]
                    if pending_single is not None:
                        if abs(gs.positions[pending_single][0] - gs.positions[v][0]) <= 4:
                            _do_two_leaves(pending_single, rem.pop(pending_single), v, rem.pop(v))
                            pending_single = None
                        else:
                            _do_single_leaf(pending_single, rem.pop(pending_single))
                            pending_single = v
                    else:
                        pending_single = v
            if pending_single is not None:
                _do_single_leaf(pending_single, rem.pop(pending_single))

        sweep_ltr = gs.positions[hub][0] <= (ex0 + ex1) // 2

        while rem:
            top_leaves = [v for v in rem if gs.positions[v][1] == sy1]
            if top_leaves and can_pipe_x:
                _sweep_row_leaves(top_leaves, sweep_ltr)
                sweep_ltr = not sweep_ltr
                continue

            if not rem:
                break

            # All remaining leaves are on rows < sy1: roll rows up to sy1 when cheaper than direct lifts!
            y_hi = max(gs.positions[v][1] for v in rem)
            y_lo = min(gs.positions[v][1] for v in rem)
            gap = sy1 - y_hi
            rolled = False
            if can_pipe_x and len(rem) >= max(3, 2 * gap + 1):
                occ_on_sy1 = [x for x in range(sx0, sx1 + 1) if (x, sy1) in gs.occupant]
                if not occ_on_sy1:
                    # Row sy1 is completely empty: shift all occupied storage rows in [y_lo..y_hi] up to sy1!
                    mid = [(x, y) for y in range(y_lo, y_hi + 1) for x in range(sx0, sx1 + 1) if (x, y) in gs.occupant]
                    if mid:
                        mys = sorted({y for _, y in mid})
                        mxs = sorted({x for x, _ in mid})
                        dy = sy1 - mys[-1]
                        if dy > 0 and all(
                            (x, y + dy) not in gs.occupant or (x, y + dy) in set(mid)
                            for x, y in mid
                        ):
                            gs.move(
                                tuple(range(mys[0], mys[-1] + 1)),
                                tuple(range(mxs[0], mxs[-1] + 1)),
                                tuple(y + dy for y in mys),
                                tuple(mxs),
                            )
                            rolled = True
                else:
                    # Find highest block of `gap` empty storage rows below `y_hi` (to move `gap` finished rows in 1 move!)
                    empty_below = [
                        yc for yc in range(y_hi - 1, sy0 - 1, -1)
                        if not any((x, yc) in gs.occupant for x in range(sx0, sx1 + 1))
                    ]
                    if len(empty_below) >= gap:
                        # Check if `empty_below[:gap]` is contiguous and below `y_hi`
                        park_ys = tuple(sorted(empty_below[:gap]))
                        fin_rows = tuple(range(sy1 - gap + 1, sy1 + 1))
                        fin_occ = [
                            (x, y)
                            for y in fin_rows
                            for x in range(sx0, sx1 + 1)
                            if (x, y) in gs.occupant
                        ]
                        if fin_occ:
                            fys = sorted({y for _, y in fin_occ})
                            fxs = sorted({x for x, _ in fin_occ})
                            # Move finished rows `fys` -> `park_ys[-len(fys):]` in ONE move!
                            tgt_park = park_ys[-len(fys):]
                            gs.move(
                                tuple(range(fys[0], fys[-1] + 1)),
                                tuple(range(fxs[0], fxs[-1] + 1)),
                                tgt_park,
                                tuple(fxs),
                            )
                            # Shift all occupied rows in [tgt_park[-1]+1 .. y_hi] UP by `+gap` in ONE move!
                            mid = [
                                (x, y)
                                for y in range(tgt_park[-1] + 1, y_hi + 1)
                                for x in range(sx0, sx1 + 1)
                                if (x, y) in gs.occupant
                            ]
                            if mid:
                                mys = sorted({y for _, y in mid})
                                mxs = sorted({x for x, _ in mid})
                                if all((x, y + gap) not in gs.occupant or (x, y + gap) in set(mid) for x, y in mid):
                                    gs.move(
                                        tuple(range(mys[0], mys[-1] + 1)),
                                        tuple(range(mxs[0], mxs[-1] + 1)),
                                        tuple(y + gap for y in mys),
                                        tuple(mxs),
                                    )
                            rolled = True

            if not rolled:
                if can_pipe_x:
                    hi_leaves = [v for v in rem if gs.positions[v][1] == y_hi]
                    _sweep_row_leaves(hi_leaves, sweep_ltr)
                    sweep_ltr = not sweep_ltr
                else:
                    rem_list = sorted(rem.keys(), key=lambda v: (-gs.positions[v][1], gs.positions[v][0]))
                    for v in rem_list:
                        _do_single_leaf(v, rem.pop(v))

        # Close H-flip if we flipped target from hub to leaves!
        if use_h_flip:
            for v, _ in star_leaves:
                for gn, ang in h_ops:
                    gs.gate(gn, (v,), (ang,))
            _evict_to_storage(gs, [hub])
            for gn, ang in h_ops:
                gs.gate(gn, (hub,), (ang,))

        # Compact any row gap left below `sy1` by row-rolling (e.g. rows 4..7 + row 9 -> shift 4..7 up to 5..8 in 1 move of dist 1.0!)
        st_after = [q for q, s in gs.positions.items() if geom.zone_at[s].name == st.name]
        if st_after:
            ys_a = sorted({gs.positions[q][1] for q in st_after})
            if len(ys_a) >= 2 and ys_a[-1] == sy1 and ys_a[-2] < sy1 - 1:
                gap_a = (sy1 - 1) - ys_a[-2]
                low_qs = [q for q in st_after if gs.positions[q][1] <= ys_a[-2]]
                lys = sorted({gs.positions[q][1] for q in low_qs})
                lxs = sorted({gs.positions[q][0] for q in low_qs})
                gs.move(
                    tuple(range(lys[0], lys[-1] + 1)),
                    tuple(range(lxs[0], lxs[-1] + 1)),
                    tuple(y + gap_a for y in lys),
                    tuple(lxs),
                )


    # ---------------------------------------------------------------------------
    # Stage 5: Specialized Pipelines (HEA, IHVA-CZ, QAOA-CZ, General-CZ)
    # ---------------------------------------------------------------------------


    def _slice_dag_fronts(
        ops: list[tuple[str, tuple[int, ...], tuple[float, ...]]], nq: int
    ) -> list[tuple[str, Any]]:
        """Topological DAG front-slicing into alternating ('1q', dict) and ('2q', list) fronts."""
        n_ops = len(ops)
        preds = [0] * n_ops
        succs: list[list[int]] = [[] for _ in range(n_ops)]
        last_on_q: list[int | None] = [None] * nq
        for i, (_, qs, _) in enumerate(ops):
            for q in qs:
                prev = last_on_q[q]
                if prev is not None:
                    preds[i] += 1
                    succs[prev].append(i)
                last_on_q[q] = i

        ready_1q = [i for i in range(n_ops) if preds[i] == 0 and len(ops[i][1]) == 1]
        ready_2q = [i for i in range(n_ops) if preds[i] == 0 and len(ops[i][1]) == 2]
        fronts: list[tuple[str, Any]] = []
        rem_ops = n_ops
        while rem_ops > 0:
            if ready_1q:
                q1: dict[int, list[tuple[str, float]]] = defaultdict(list)
                while ready_1q:
                    i = ready_1q.pop()
                    rem_ops -= 1
                    nm, qs, ps = ops[i]
                    q1[qs[0]].append((nm, ps[0]))
                    for s in succs[i]:
                        preds[s] -= 1
                        if preds[s] == 0:
                            if len(ops[s][1]) == 1:
                                ready_1q.append(s)
                            else:
                                ready_2q.append(s)
                fronts.append(("1q", q1))
            elif ready_2q:
                q2: list[tuple[str, tuple[int, int], tuple[float, ...]]] = []
                while ready_2q:
                    i = ready_2q.pop()
                    rem_ops -= 1
                    nm, qs, ps = ops[i]
                    q2.append((nm, qs, ps))
                    for s in succs[i]:
                        preds[s] -= 1
                        if preds[s] == 0:
                            if len(ops[s][1]) == 2:
                                ready_2q.append(s)
                            else:
                                ready_1q.append(s)
                fronts.append(("2q", q2))
        return fronts


    def _build_m1_m2_loading(
        nq: int,
        m1_pairs: Sequence[tuple[int, int]],
        m2_pairs: Sequence[tuple[int, int]],
        base_ent_pos: dict[int, Site],
        geom: ArchGeometry,
    ) -> dict[int, Site]:
        """Construct an initial storage loading aligned with the first two matchings `m1_pairs` and `m2_pairs`
        so `m1_pairs` lifts in 1-2 bulk moves with 0 ghosts and 0 storage push, and `m2_pairs \\ m1_pairs` sits
        contiguously in monotone X-order on the next storage row down."""
        st = geom.storage
        sx0, sx1 = st.x
        sy0, sy1 = st.y
        w_st = sx1 - sx0 + 1
        h_st = sy1 - sy0 + 1
        k_per_row = w_st // 2
        if k_per_row < 1 or not m1_pairs:
            return _storage_row_for_ent(base_ent_pos, geom)

        # Order m1_pairs by base_ent_pos (y, x) and orient each pair left-to-right
        ordered_m1: list[tuple[int, int]] = []
        for u, v in m1_pairs:
            if base_ent_pos[u][0] <= base_ent_pos[v][0]:
                ordered_m1.append((u, v))
            else:
                ordered_m1.append((v, u))
        ordered_m1.sort(
            key=lambda p: (
                base_ent_pos[p[0]][1] + base_ent_pos[p[1]][1],
                base_ent_pos[p[0]][0] + base_ent_pos[p[1]][0],
            )
        )

        m1_len = len(ordered_m1)
        r_1 = math.ceil(m1_len / k_per_row)
        rem_count = nq - 2 * m1_len
        r_rem_atoms = math.ceil(max(0, rem_count) / w_st)
        if r_1 + r_rem_atoms > h_st:
            return _storage_row_for_ent(base_ent_pos, geom)

        loading: dict[int, Site] = {}
        m1_set: set[int] = set()
        f_full = m1_len // k_per_row
        part_pairs = m1_len % k_per_row

        # Place `f_full` full rows at the very top `sy1 - f_full + 1 .. sy1` (ascending y -> ascending entangling y!)
        idx_p = 0
        for r_off in range(f_full):
            y_st = sy1 - f_full + 1 + r_off
            for k in range(k_per_row):
                u, v = ordered_m1[idx_p]
                idx_p += 1
                loading[u] = (sx0 + 2 * k, y_st)
                loading[v] = (sx0 + 2 * k + 1, y_st)
                m1_set.add(u)
                m1_set.add(v)

        # Place any partial row of `part_pairs` on `sy1 - r_1 + 1` (with rest of that row left empty -> 0 ghosts!)
        if part_pairs > 0:
            y_part = sy1 - r_1 + 1
            for k in range(part_pairs):
                u, v = ordered_m1[idx_p]
                idx_p += 1
                loading[u] = (sx0 + 2 * k, y_part)
                loading[v] = (sx0 + 2 * k + 1, y_part)
                m1_set.add(u)
                m1_set.add(v)

        # Order remaining `nq - 2*m1_len` atoms: first `M2 \ M1` sorted by M2 partner's x-coord, then the rest
        m2_partner_x: dict[int, float] = {}
        for u, v in m2_pairs:
            if u not in m1_set and v in m1_set:
                m2_partner_x[u] = float(loading[v][0])
            elif v not in m1_set and u in m1_set:
                m2_partner_x[v] = float(loading[u][0])
            elif u not in m1_set and v not in m1_set:
                avg_x = 0.5 * (base_ent_pos[u][0] + base_ent_pos[v][0])
                m2_partner_x[u] = avg_x - 0.1
                m2_partner_x[v] = avg_x + 0.1

        m2_entering = sorted(m2_partner_x.keys(), key=lambda q: m2_partner_x[q])
        m2_set = set(m2_entering)
        other_idle = sorted(
            [q for q in range(nq) if q not in m1_set and q not in m2_set],
            key=lambda q: (base_ent_pos[q][1], base_ent_pos[q][0]),
        )

        avail_lower = [
            (x, y)
            for y in range(sy1 - r_1, sy0 - 1, -1)
            for x in range(sx0, sx1 + 1)
        ]
        for q, s in zip(m2_entering + other_idle, avail_lower):
            loading[q] = s
        return loading


    def _try_compile_hea(
        circuit: QuantumCircuit,
        ops: list[tuple[str, tuple[int, ...], tuple[float, ...]]],
        trailing_meas: list[tuple[int, int]],
        geom: ArchGeometry,
        raw_spec: ArchitectureSpec | None,
    ) -> tuple[QuantumCircuit, Architecture, float, int] | None:
        """Blocking-CZ 2D Snake pipeline for 1D-chain / HEA circuits (ZERO idle atoms!)."""
        nq = circuit.num_qubits
        if nq < 4 or geom.entangling.pair_axis != "x":
            return None
        edges_2q = {(min(qs), max(qs)) for name, qs, _ in ops if len(qs) == 2}
        if not edges_2q:
            return None
        if not all(v == u + 1 for u, v in edges_2q):
            return None
        if any(name == "rzz" for name, _, _ in ops):
            return None

        fronts = _slice_dag_fronts(ops, nq)

        sx0, sx1 = geom.storage.x
        sy0, sy1 = geom.storage.y
        ex0, ex1 = geom.entangling.x
        ey0, ey1 = geom.entangling.y
        W = min(sx1 - sx0 + 1, ex1 - ex0 + 1)
        if nq <= 12 and W >= 6:
            W = 6
        elif W % 2 == 1:
            W -= 1
        if W < 4:
            return None
        R = math.ceil(nq / W)
        if R > (sy1 - sy0 + 1) or R > (ey1 - ey0 + 1):
            return None

        # Snake order: row 0 (ey0 / sy1) L->R, row 1 R->L, etc.
        ent_pos: dict[int, Site] = {}
        q = 0
        for r in range(R):
            cols = range(ex0, ex0 + W) if r % 2 == 0 else range(ex0 + W - 1, ex0 - 1, -1)
            for c in cols:
                if q < nq:
                    ent_pos[q] = (c, ey0 + r)
                    q += 1
        loading = _storage_row_for_ent(ent_pos, geom)
        gs = GridState(geom, loading)
        gs._raw_spec = raw_spec
        sy_lo = sy1 - R + 1

        for f_idx, (ftype, fdata) in enumerate(fronts):
            if ftype == "1q":
                # Drop any atoms in entangling back to exact loading sites
                in_ent = [qq for qq in range(nq) if geom.zone_at[gs.positions[qq]].name == geom.entangling.name]
                if in_ent:
                    if len(in_ent) == nq and all(gs.positions[qq] == ent_pos[qq] for qq in in_ent):
                        gs.move(
                            tuple(range(ey0, ey0 + R)),
                            tuple(range(ex0, ex0 + W)),
                            tuple(range(sy_lo, sy1 + 1)),
                            tuple(range(sx0, sx0 + W)),
                        )
                    else:
                        _execute_aod_batches(gs, {qq: loading[qq] for qq in in_ent}, geom.entangling.name, geom.storage.name)
                for qq, ops_q in fdata.items():
                    for gn, ang in ops_q:
                        gs.gate(gn, (qq,), (ang,))
            else:
                rem_pairs: set[tuple[int, int]] = {(min(gqs), max(gqs)) for _, gqs, _ in fdata}
                all_in_st = all(gs.positions[qq] == loading[qq] for qq in range(nq))
                all_even = all(u % 2 == 0 and v == u + 1 for u, v in rem_pairs)
                all_odd = all(u % 2 == 1 and v == u + 1 for u, v in rem_pairs)

                if all_in_st and all_even and nq % 2 == 0 and len(rem_pairs) == nq // 2:
                    # Perfect Even Layer: ALL nq atoms participate -> 1 bulk lift, 1 blocking CZ layer!
                    _lift_storage_to_ent(gs, ent_pos)
                    for u, v in sorted(rem_pairs):
                        gs.gate("cz", (u, v))
                    rem_pairs.clear()
                    # Check if next front is 2Q (odd) -> drop back to loading so odd layer starts cleanly in storage
                    if f_idx + 1 < len(fronts) and fronts[f_idx + 1][0] == "2q":
                        gs.move(
                            tuple(range(ey0, ey0 + R)),
                            tuple(range(ex0, ex0 + W)),
                            tuple(range(sy_lo, sy1 + 1)),
                            tuple(range(sx0, sx0 + W)),
                        )
                elif all_in_st and all_odd and nq % W == 0 and (ex0 + W - 1) <= ex1:
                    # 1. Interior odd pairs: lift ONLY cols sx0+1 .. sx0+W-2 (W-2 EVEN cols -> 0 idle atoms!)
                    interior_pairs = {
                        (u, v)
                        for u, v in rem_pairs
                        if (sx0 + 1 <= loading[u][0] <= sx0 + W - 2) and (sx0 + 1 <= loading[v][0] <= sx0 + W - 2)
                    }
                    if len(interior_pairs) == R * (W - 2) // 2:
                        gs.move(
                            tuple(range(sy_lo, sy1 + 1)),
                            tuple(range(sx0 + 1, sx0 + W - 1)),
                            tuple(range(ey0, ey0 + R)),
                            tuple(range(ex0, ex0 + W - 2)),
                        )
                        for u, v in sorted(interior_pairs):
                            gs.gate("cz", (u, v))
                            rem_pairs.remove((u, v))
                        gs.move(
                            tuple(range(ey0, ey0 + R)),
                            tuple(range(ex0, ex0 + W - 2)),
                            tuple(range(sy_lo, sy1 + 1)),
                            tuple(range(sx0 + 1, sx0 + W - 1)),
                        )

                    # 2. Right-side turnaround pairs in col sx0+W-1 (ONLY paired atoms -> 0 idle atoms!)
                    right_pairs = sorted(
                        [(u, v) for u, v in rem_pairs if loading[u][0] == sx0 + W - 1 and loading[v][0] == sx0 + W - 1],
                        key=lambda p: min(loading[p[0]][1], loading[p[1]][1]),
                    )
                    if right_pairs:
                        r_atoms = sorted({qq for p in right_pairs for qq in p}, key=lambda qq: gs.positions[qq][1])
                        r_st_ys = tuple(gs.positions[qq][1] for qq in r_atoms)
                        r_ent_ys = tuple(ey0 + i for i in range(len(r_atoms)))
                        # Lift ONLY the paired right-turnaround atoms to col ex0+W-2 (contiguous in storage!)
                        gs.move(tuple(range(r_st_ys[0], r_st_ys[-1] + 1)), (sx0 + W - 1,), r_ent_ys, (ex0 + W - 2,))
                        for i in range(0, len(r_atoms), 2):
                            q_hi = r_atoms[i + 1]
                            gs.move_site(q_hi, (ex0 + W - 1, ey0 + i))
                        # Now ALL atoms in entangling are paired -> 1 blocking CZ layer!
                        for u, v in right_pairs:
                            gs.gate("cz", (u, v))
                            rem_pairs.remove((u, v))
                        # Drop both columns back to storage col sx0+W-1 in 2 clean bulk moves!
                        lo_ys = tuple(ey0 + i for i in range(0, len(r_atoms), 2))
                        lo_dst = tuple(loading[r_atoms[i]][1] for i in range(0, len(r_atoms), 2))
                        hi_dst = tuple(loading[r_atoms[i + 1]][1] for i in range(0, len(r_atoms), 2))
                        gs.move(tuple(range(lo_ys[0], lo_ys[-1] + 1)), (ex0 + W - 2,), lo_dst, (sx0 + W - 1,))
                        gs.move(tuple(range(lo_ys[0], lo_ys[-1] + 1)), (ex0 + W - 1,), hi_dst, (sx0 + W - 1,))

                    # 3. Left-side turnaround pairs in col sx0 (ONLY paired atoms -> 0 idle atoms!)
                    left_pairs = sorted(
                        [(u, v) for u, v in rem_pairs if loading[u][0] == sx0 and loading[v][0] == sx0],
                        key=lambda p: min(loading[p[0]][1], loading[p[1]][1]),
                    )
                    if left_pairs:
                        l_atoms = sorted({qq for p in left_pairs for qq in p}, key=lambda qq: gs.positions[qq][1])
                        l_st_ys = tuple(gs.positions[qq][1] for qq in l_atoms)
                        l_ent_ys = tuple(ey0 + i for i in range(len(l_atoms)))
                        gs.move(tuple(range(l_st_ys[0], l_st_ys[-1] + 1)), (sx0,), l_ent_ys, (ex0,))
                        for i in range(0, len(l_atoms), 2):
                            q_hi = l_atoms[i + 1]
                            gs.move_site(q_hi, (ex0 + 1, ey0 + i))
                        for u, v in left_pairs:
                            gs.gate("cz", (u, v))
                            rem_pairs.remove((u, v))
                        lo_ys = tuple(ey0 + i for i in range(0, len(l_atoms), 2))
                        lo_dst = tuple(loading[l_atoms[i]][1] for i in range(0, len(l_atoms), 2))
                        hi_dst = tuple(loading[l_atoms[i + 1]][1] for i in range(0, len(l_atoms), 2))
                        gs.move(tuple(range(lo_ys[0], lo_ys[-1] + 1)), (ex0,), lo_dst, (sx0,))
                        gs.move(tuple(range(lo_ys[0], lo_ys[-1] + 1)), (ex0 + 1,), hi_dst, (sx0,))

                    if rem_pairs:
                        _execute_cz_matching_batch(gs, sorted(rem_pairs))
                else:
                    _execute_cz_matching_batch(gs, sorted(rem_pairs))

        compiled, arch = gs.build_circuit(circuit, trailing_meas)
        return compiled, arch, gs.total_distance, gs.depth


    def _compile_ihva_cz(
        circuit: QuantumCircuit,
        init_1q: list[tuple[str, tuple[int, ...], tuple[float, ...]]],
        blocks: list[tuple[int, int, float]],
        trailing_meas: list[tuple[int, int]],
        geom: ArchGeometry,
        raw_spec: ArchitectureSpec | None,
    ) -> tuple[QuantumCircuit, Architecture]:
        """CZ-only, Blocking-CZ IHVA compiler via Global S-Conjugation + Pauli-Commutation DAG Matching + 2-Leaf Hub Sweep."""
        nq = circuit.num_qubits
        ex0, ex1 = geom.entangling.x
        ey0, ey1 = geom.entangling.y
        ent_w = ex1 - ex0 + 1
        ent_h = ey1 - ey0 + 1
        st_h = geom.storage.y[1] - geom.storage.y[0] + 1
        hub = _detect_hub(nq, [(u, v) for u, v, _ in blocks])
        booths = _ent_booths(geom)
        max_m = len(booths)

        # Global S-conjugation: conjugate all target qubits by rz(-pi/2) at start and rz(+pi/2) at end!
        # Every R_ZY(t)_{u, v} becomes R_ZX(t)_{u, v} = CZ(u, v) · RX(t)_v · CZ(u, v) with ZERO basis switches!
        target_qs = {v for _, v, _ in blocks}
        init_mats: list[np.ndarray] = [np.eye(2, dtype=complex) for _ in range(nq)]
        for name, (q,), ps in init_1q:
            m = _gate_to_matrix(name, ps)
            if m is not None:
                init_mats[q] = m @ init_mats[q]
        for q in target_qs:
            init_mats[q] = _mat_rz(-math.pi / 2) @ init_mats[q]

        # Precompute Pauli-commutation DAG once
        n_b = len(blocks)
        preds_dag: list[set[int]] = [set() for _ in range(n_b)]
        succs_dag: list[list[int]] = [[] for _ in range(n_b)]
        for j in range(n_b):
            uj, vj, _ = blocks[j]
            for k in range(j):
                uk, vk, _ = blocks[k]
                if ((uj == vk) + (vj == uk)) % 2 == 1:
                    preds_dag[j].add(k)
                    succs_dag[k].append(j)
        crit = [1] * n_b
        for j in range(n_b - 1, -1, -1):
            if succs_dag[j]:
                crit[j] = 1 + max(crit[s] for s in succs_dag[j])

        block_epoch = [-1] * n_b
        epochs: list[tuple[str, list[int]]] = []
        for j, (u, v, _) in enumerate(blocks):
            if hub is not None and (u == hub or v == hub):
                ori = "C" if u == hub else "T"
                if not epochs or epochs[-1][0] != ori:
                    epochs.append((ori, [j]))
                else:
                    epochs[-1][1].append(j)
                block_epoch[j] = len(epochs) - 1
        sy1 = geom.storage.y[1]

        def _run_dag_matching(
            ent_pos: dict[int, Site],
            fold_hub: bool = True,
            hub_star_thresh: int = 5,
            custom_loading: dict[int, Site] | None = None,
            booth_pair_hub: bool = False,
        ) -> tuple[float, QuantumCircuit, Architecture, list[list[tuple[int, int]]]]:
            loading = custom_loading if custom_loading is not None else _storage_row_for_ent(ent_pos, geom)
            gs = GridState(geom, loading)
            gs._raw_spec = raw_spec
            for q in range(nq):
                for gn, ang in _matrix_to_zxz(init_mats[q]):
                    gs.gate(gn, (q,), (ang,))

            rem_preds = [len(preds_dag[j]) for j in range(n_b)]
            done = [False] * n_b
            rem_cnt = n_b
            ent_name = geom.entangling.name
            first_two_m: list[list[tuple[int, int]]] = []

            while rem_cnt > 0:
                ready = [j for j in range(n_b) if not done[j] and rem_preds[j] == 0]
                # Merge duplicate same-(u, v) ready blocks!
                by_uv: dict[tuple[int, int], list[int]] = defaultdict(list)
                for j in ready:
                    u, v, _ = blocks[j]
                    by_uv[(u, v)].append(j)

                # Drop any merged blocks whose summed angle is 0!
                zero_dropped = False
                for uv, idxs in list(by_uv.items()):
                    tot_ang = _norm_angle(sum(blocks[j][2] for j in idxs))
                    if _is_zero_angle(tot_ang):
                        for j in idxs:
                            done[j] = True
                            rem_cnt -= 1
                            for s in succs_dag[j]:
                                rem_preds[s] -= 1
                        del by_uv[uv]
                        zero_dropped = True
                if zero_dropped:
                    continue

                # Split ready merged blocks into non-hub vs hub-ctrl vs hub-tgt
                non_hub_items: list[tuple[int, int, tuple[float, list[int]]]] = []
                hub_ctrl_items: list[tuple[int, float, list[int]]] = []
                hub_tgt_items: list[tuple[int, float, list[int]]] = []
                for (u, v), idxs in by_uv.items():
                    ang = _norm_angle(sum(blocks[j][2] for j in idxs))
                    if hub is not None and (u == hub or v == hub):
                        if u == hub:
                            hub_ctrl_items.append((v, ang, idxs))
                        else:
                            hub_tgt_items.append((u, ang, idxs))
                    else:
                        non_hub_items.append((u, v, (ang, idxs)))

                # Trigger Hub Star when no non-hub blocks ready (or very few non-hub blocks ready while >= hub_star_thresh hub blocks ready!)
                if not non_hub_items or (
                    len(non_hub_items) <= 2 and max(len(hub_ctrl_items), len(hub_tgt_items)) >= hub_star_thresh
                ):
                    if hub is not None and (hub_ctrl_items or hub_tgt_items):
                        # Choose larger (or higher crit-sum) orientation of the hub star!
                        sc_c = sum(max(crit[j] for j in idxs) for _, _, idxs in hub_ctrl_items)
                        sc_t = sum(max(crit[j] for j in idxs) for _, _, idxs in hub_tgt_items)
                        if sc_c >= sc_t and hub_ctrl_items:
                            chosen = hub_ctrl_items
                            is_t = False
                        else:
                            chosen = hub_tgt_items
                            is_t = True
                        _execute_hub_star_rzx(gs, hub, [(v, ang) for v, ang, _ in chosen], h_is_target=is_t)
                        for _, _, idxs in chosen:
                            for j in idxs:
                                done[j] = True
                                rem_cnt -= 1
                                for s in succs_dag[j]:
                                    rem_preds[s] -= 1
                        continue

                # Build a maximum-cardinality matching from non-hub ready blocks (+ 1 folded hub block if fold_hub)
                prio: dict[tuple[int, int], float] = {}
                all_cand = list(non_hub_items)
                if fold_hub and hub is not None:
                    for v, ang, idxs in hub_ctrl_items:
                        all_cand.append((hub, v, (ang, idxs)))
                    for u, ang, idxs in hub_tgt_items:
                        all_cand.append((u, hub, (ang, idxs)))

                # Deduplicate unordered pair (u, v) in `all_cand` (keep highest-crit orientation if both (u,v) & (v,u) ready)
                dedup: dict[tuple[int, int], tuple[int, int, tuple[float, list[int]]]] = {}
                for u, v, payload in all_cand:
                    k = (min(u, v), max(u, v))
                    c_val = max(crit[j] for j in payload[1])
                    if k not in dedup or c_val > max(crit[j] for j in dedup[k][2][1]):
                        dedup[k] = (u, v, payload)
                cand_list = list(dedup.values())

                for u, v, (_, idxs) in cand_list:
                    k = (min(u, v), max(u, v))
                    in_e = (1 if geom.zone_at[gs.positions[u]].name == ent_name else 0) + (
                        1 if geom.zone_at[gs.positions[v]].name == ent_name else 0
                    )
                    max_c = max(crit[j] for j in idxs)
                    prio[k] = 3.0 * max_c + 2.0 * in_e + 1.0 / (1.0 + geom.dist(gs.positions[u], gs.positions[v]))

                m_idxs = _find_large_matching(cand_list, priority=prio)
                if len(m_idxs) > max_m:
                    m_idxs = m_idxs[:max_m]
                rzx_batch: list[tuple[int, int, float]] = []
                used_in_m = {q for mi in m_idxs for q in (cand_list[mi][0], cand_list[mi][1])}
                for mi in m_idxs:
                    u, v, (ang, idxs) = cand_list[mi]
                    if (
                        booth_pair_hub
                        and hub is not None
                        and (u == hub or v == hub)
                        and max(len(hub_ctrl_items), len(hub_tgt_items)) >= 3
                    ):
                        cur_c = max(crit[j] for j in idxs)
                        ep_id = block_epoch[idxs[0]]
                        ep_leaves: set[int] = set()
                        if ep_id >= 0:
                            for j_ep in epochs[ep_id][1]:
                                if not done[j_ep]:
                                    uj, vj, _ = blocks[j_ep]
                                    ep_leaves.add(vj if uj == hub else uj)
                        b_cnt: Counter[tuple[int, int]] = Counter()
                        r_cnt: Counter[int] = Counter()
                        for lv in ep_leaves:
                            lx, ly = gs.positions[lv]
                            py = ly if geom.zone_at[(lx, ly)].name != ent_name else (sy1 - (ly - ey0))
                            b_cnt[(py, (lx - ex0) // 2)] += 1
                            r_cnt[py] += 1
                        other_used = used_in_m - {u, v}
                        pool = [(hub, lv, lang, lidxs) for lv, lang, lidxs in hub_ctrl_items] + [
                            (lu, hub, lang, lidxs) for lu, lang, lidxs in hub_tgt_items
                        ]
                        valid_swaps = [
                            (hu, hv, lang, lidxs)
                            for hu, hv, lang, lidxs in pool
                            if (hv if hu == hub else hu) not in other_used and max(crit[j] for j in lidxs) >= cur_c
                        ]
                        if valid_swaps:
                            def _sw_key(item: tuple[int, int, float, list[int]]) -> tuple[float, ...]:
                                hu, hv, _, lidxs = item
                                lv = hv if hu == hub else hu
                                lx, ly = gs.positions[lv]
                                py = ly if geom.zone_at[(lx, ly)].name != ent_name else (sy1 - (ly - ey0))
                                return (
                                    float(-max(crit[j] for j in lidxs)),
                                    float(max(0, gs.q_depth[lv] - gs.q_depth[hub])),
                                    0.0 if b_cnt[(py, (lx - ex0) // 2)] == 1 else 1.0,
                                    0.0 if r_cnt[py] % 2 == 1 else 1.0,
                                    geom.dist(gs.positions[hub], gs.positions[lv]),
                                )

                            valid_swaps.sort(key=_sw_key)
                            u, v, ang, idxs = valid_swaps[0]
                    rzx_batch.append((u, v, ang))
                    for j in idxs:
                        done[j] = True
                        rem_cnt -= 1
                        for s in succs_dag[j]:
                            rem_preds[s] -= 1
                if len(first_two_m) < 2:
                    first_two_m.append([(u, v) for u, v, _ in rzx_batch])
                _execute_rzx_matching(gs, rzx_batch, preferred_st=loading)

            # Final rz(+pi/2) on all target_qs (one-way drop to storage!)
            final_ops = {q: [("rz", math.pi / 2)] for q in target_qs}
            _apply_1q_in_storage(gs, final_ops, set(), keep_in_storage=True)
            compiled, arch = gs.build_circuit(circuit, trailing_meas)
            return gs.total_distance + 2.9221 * gs.depth, compiled, arch, first_two_m

        def _run_wave_matching(waves: list[list[tuple[int, int, float]]], ent_pos: dict[int, Site]) -> tuple[float, QuantumCircuit, Architecture]:
            loading = _storage_row_for_ent(ent_pos, geom)
            gs = GridState(geom, loading)
            gs._raw_spec = raw_spec
            for q in range(nq):
                for gn, ang in _matrix_to_zxz(init_mats[q]):
                    gs.gate(gn, (q,), (ang,))

            for w_blocks in waves:
                # Merge duplicate (u, v) within wave
                agg: dict[tuple[int, int], float] = defaultdict(float)
                for u, v, ang in w_blocks:
                    agg[(u, v)] = _norm_angle(agg[(u, v)] + ang)
                non_hub: list[tuple[int, int, float]] = []
                hub_ctrl: list[tuple[int, float]] = []
                hub_tgt: list[tuple[int, float]] = []
                for (u, v), ang in agg.items():
                    if _is_zero_angle(ang):
                        continue
                    if hub is not None and u == hub:
                        hub_ctrl.append((v, ang))
                    elif hub is not None and v == hub:
                        hub_tgt.append((u, ang))
                    else:
                        non_hub.append((u, v, ang))

                # Color non-hub edges into matchings, folding 1 hub edge into each matching when possible
                matchings = _edge_color_matchings(non_hub, max_matching_size=max_m)
                rem_hc = list(hub_ctrl)
                rem_ht = list(hub_tgt)
                for ms in matchings:
                    if hub is not None and len(ms) < max_m:
                        used_v = {q for u, v, _ in ms for q in (u, v)}
                        if hub not in used_v:
                            folded = False
                            for idx, (v, ang) in enumerate(rem_hc):
                                if v not in used_v:
                                    ms.append((hub, v, ang))
                                    rem_hc.pop(idx)
                                    folded = True
                                    break
                            if not folded:
                                for idx, (u, ang) in enumerate(rem_ht):
                                    if u not in used_v:
                                        ms.append((u, hub, ang))
                                        rem_ht.pop(idx)
                                        break
                    _execute_rzx_matching(gs, ms, preferred_st=loading)

                if hub is not None and rem_hc:
                    _execute_hub_star_rzx(gs, hub, rem_hc, h_is_target=False)
                if hub is not None and rem_ht:
                    _execute_hub_star_rzx(gs, hub, rem_ht, h_is_target=True)

            final_ops = {q: [("rz", math.pi / 2)] for q in target_qs}
            _apply_1q_in_storage(gs, final_ops, set(), keep_in_storage=True)
            compiled, arch = gs.build_circuit(circuit, trailing_meas)
            return gs.total_distance + 2.9221 * gs.depth, compiled, arch

        # Build SA edge weights (excluding hub so non-hub graph clusters tightly!)
        ew_cnt: Counter[tuple[int, int]] = Counter()
        for u, v, _ in blocks:
            if u != hub and v != hub:
                ew_cnt[(min(u, v), max(u, v))] += 1
        w_edges = [(u, v, float(w)) for (u, v), w in ew_cnt.items()]
        C = min(10, ent_w) if nq > 20 else (min(6, ent_w) if nq <= 12 else min(8, ent_w))
        R = math.ceil(nq / C)
        if R > ent_h or R > st_h:
            C = ent_w
            R = math.ceil(nq / C)
        sites = [(x, ey0 + y) for y in range(R) for x in range(ex0, ex0 + C)]
        fixed = {hub: (ex0, ey0)} if hub is not None else None

        candidates: list[tuple[float, QuantumCircuit, Architecture]] = []
        sa_steps = 30000 if nq <= 20 else 45000
        for sd in (42, 7, 13):
            ep = _sa_layout(nq, sites, w_edges, geom, cost_mode="compact", fixed_sites=fixed, steps=sa_steps, seed=sd)
            for hst in (5, 9):
                for bph in (False, True):
                    sc0, c0, a0, ftm = _run_dag_matching(ep, fold_hub=True, hub_star_thresh=hst, booth_pair_hub=bph)
                    candidates.append((sc0, c0, a0))
                    if ftm:
                        m1_p = ftm[0]
                        m2_p = ftm[1] if len(ftm) > 1 else []
                        aligned_ld = _build_m1_m2_loading(nq, m1_p, m2_p, ep, geom)
                        sc_a, c_a, a_a, _ = _run_dag_matching(
                            ep, fold_hub=True, hub_star_thresh=hst, custom_loading=aligned_ld, booth_pair_hub=bph
                        )
                        candidates.append((sc_a, c_a, a_a))
            ep_m2 = _sa_layout(nq, sites, w_edges, geom, cost_mode="mesh2", fixed_sites=fixed, steps=sa_steps, seed=sd)
            for hst in (5, 9):
                for bph in (False, True):
                    sc3, c3, a3, ftm3 = _run_dag_matching(ep_m2, fold_hub=True, hub_star_thresh=hst, booth_pair_hub=bph)
                    candidates.append((sc3, c3, a3))
                    if ftm3:
                        ld3 = _build_m1_m2_loading(nq, ftm3[0], ftm3[1] if len(ftm3) > 1 else [], ep_m2, geom)
                        sc3a, c3a, a3a, _ = _run_dag_matching(
                            ep_m2, fold_hub=True, hub_star_thresh=hst, custom_loading=ld3, booth_pair_hub=bph
                        )
                        candidates.append((sc3a, c3a, a3a))
            if sd == 42:
                sc2, c2, a2, _ = _run_dag_matching(ep, fold_hub=False, hub_star_thresh=4)
                candidates.append((sc2, c2, a2))
                for mode in ("alap", "hub_alap"):
                    waves = _schedule_ihva_waves(nq, blocks, mode=mode)
                    if waves:
                        candidates.append(_run_wave_matching(waves, ep))

        candidates.sort(key=lambda x: x[0])
        return candidates[0][1], candidates[0][2]


    def _compile_qaoa_cz(
        circuit: QuantumCircuit,
        fronts: list[tuple[str, Any]],
        trailing_meas: list[tuple[int, int]],
        geom: ArchGeometry,
        raw_spec: ArchitectureSpec | None,
    ) -> tuple[QuantumCircuit, Architecture]:
        """CZ-only, Blocking-CZ QAOA / Commuting-RZZ compiler with Persistent X-Basis + Max-Cut Matchings + 2-Leaf Hub Sweep."""
        nq = circuit.num_qubits
        ex0, ex1 = geom.entangling.x
        ey0, ey1 = geom.entangling.y
        ent_w = ex1 - ex0 + 1
        ent_h = ey1 - ey0 + 1
        st_h = geom.storage.y[1] - geom.storage.y[0] + 1
        booths = _ent_booths(geom)
        max_m = len(booths)
        h_mat = _MAT_1Q["h"]()
        h_ops = _matrix_to_zxz(h_mat)

        # Compute future 2Q participation per front
        future_2q: list[set[int]] = [set() for _ in range(len(fronts))]
        acc: set[int] = set()
        for t in range(len(fronts) - 1, -1, -1):
            if fronts[t][0] == "2q":
                for _, gqs, _ in fronts[t][1]:
                    acc.update(gqs)
            future_2q[t] = set(acc)

        all_rzz_edges = [(gqs[0], gqs[1]) for ft, fd in fronts if ft == "2q" for gn, gqs, _ in fd if gn == "rzz"]
        hub = _detect_hub(nq, all_rzz_edges)
        non_hub_vs = [q for q in range(nq) if q != hub]
        non_hub_all = [(u, v) for u, v in all_rzz_edges if u != hub and v != hub]
        part_a, part_b = _max_cut_bipartition(non_hub_vs, non_hub_all)

        # Check if circuit starts with H on all qubits -> B in X-basis cancels initial H!
        # Desired starting X-basis for first 2Q front: `part_b` = True, `part_a` = False, `hub` = False!
        desired_b = {q: (q in part_b) for q in range(nq)}

        def _run_qaoa_candidate(
            ent_pos: dict[int, Site],
            fold_hub: bool = True,
            unified_coloring: bool = False,
            mesh_guided: bool = False,
            palindrome: bool = False,
            custom_loading: dict[int, Site] | None = None,
        ) -> tuple[float, QuantumCircuit, Architecture, list[list[tuple[int, int]]]]:
            loading = custom_loading if custom_loading is not None else _storage_row_for_ent(ent_pos, geom)
            gs = GridState(geom, loading)
            gs._raw_spec = raw_spec
            in_x: list[bool] = [False] * nq
            sy1 = geom.storage.y[1]
            first_two_m: list[list[tuple[int, int]]] = []

            def _set_basis(target_basis: dict[int, bool], f_idx_cur: int) -> None:
                flips = {q: h_ops for q, want in target_basis.items() if in_x[q] != want}
                if flips:
                    _apply_1q_in_storage(gs, flips, future_2q[f_idx_cur])
                    for q in flips:
                        in_x[q] = target_basis[q]

            def _cur_proj_ent() -> dict[int, Site]:
                """Project current atom positions vertically into entangling coordinates for dynamic round-aware matching!"""
                proj: dict[int, Site] = {}
                for q, (x, y) in gs.positions.items():
                    if geom.zone_at[(x, y)].name == geom.entangling.name:
                        proj[q] = (x, y)
                    else:
                        proj[q] = (min(max(x, ex0), ex1), min(ey1, ey0 + max(0, sy1 - y)))
                return proj

            def _build_dynamic_prio(proj: dict[int, Site]) -> dict[tuple[int, int], float]:
                dp: dict[tuple[int, int], float] = {}
                for u, v in non_hub_all:
                    k = (min(u, v), max(u, v))
                    if k not in dp:
                        d_uv = geom.dist(proj[u], proj[v])
                        bon = 8.0 if are_paired(geom.entangling, proj[u], proj[v]) else (
                            4.0 if (proj[u][1] == proj[v][1] and abs(proj[u][0] - proj[v][0]) == 1) else 0.0
                        )
                        dp[k] = bon + 3.0 / (0.3 + d_uv)
                return dp

            r2q_idx = 0
            prev_round_matchings: list[list[tuple[int, int, float]]] = []
            for f_idx, (ftype, fdata) in enumerate(fronts):
                if ftype == "1q":
                    has_next_2q = bool(future_2q[f_idx])
                    eff_ops: dict[int, list[tuple[str, float]]] = {}
                    for q in range(nq):
                        u_q = np.eye(2, dtype=complex)
                        if in_x[q]:
                            u_q = h_mat @ u_q
                        for gn, ang in fdata.get(q, ()):
                            m = _mat_rx(ang) if gn == "rx" else _mat_rz(ang)
                            u_q = m @ u_q
                        if not (has_next_2q and q in future_2q[f_idx]):
                            want_after = False
                        elif palindrome and r2q_idx >= 1:
                            # Keep current X-basis across the mixer (after Hub Sweep all leaves are in X-basis -> mixer is 1 RZ gate!)
                            want_after = in_x[q] if q != hub else False
                        else:
                            want_after = desired_b[q]
                        if want_after:
                            u_q = h_mat @ u_q
                        zxz = _matrix_to_zxz(u_q)
                        if zxz:
                            eff_ops[q] = zxz
                        in_x[q] = want_after
                    _apply_1q_in_storage(gs, eff_ops, future_2q[f_idx], keep_in_storage=(not has_next_2q))
                else:
                    is_rev_round = palindrome and (r2q_idx % 2 == 1)
                    r2q_idx += 1

                    # Separate any pure `cz` gates vs `rzz` gates
                    cz_list: list[tuple[int, int]] = []
                    rzz_agg: dict[tuple[int, int], float] = defaultdict(float)
                    for gn, gqs, gps in fdata:
                        u, v = min(gqs), max(gqs)
                        if gn == "cz":
                            cz_list.append((u, v))
                        elif gn == "rzz":
                            rzz_agg[(u, v)] = _norm_angle(rzz_agg[(u, v)] + gps[0])

                    # Handle any pure `cz` gates first (closing X-basis on their endpoints)
                    if cz_list:
                        cz_qs = {q for u, v in cz_list for q in (u, v)}
                        _set_basis({q: False for q in cz_qs}, f_idx)
                        _execute_cz_matching_batch(gs, cz_list)

                    if unified_coloring:
                        # Color ALL non-hub edges together into minimal matchings (6-7 instead of 10-12!)
                        non_hub_e: list[tuple[int, int, float]] = []
                        hub_edges: list[tuple[int, float]] = []
                        for (u, v), ang in rzz_agg.items():
                            if _is_zero_angle(ang):
                                continue
                            if hub is not None and (u == hub or v == hub):
                                other = v if u == hub else u
                                hub_edges.append((other, ang))
                            else:
                                non_hub_e.append((u, v, ang))

                        if non_hub_e:
                            proj_now = _cur_proj_ent()
                            dist_prio = _build_dynamic_prio(proj_now)
                            # Check if we can reuse the previous round's matchings in REVERSE order (M_L -> ... -> M_1)!
                            cur_nh_map = {(min(u, v), max(u, v)): ang for u, v, ang in non_hub_e}
                            prev_nh_keys = {
                                (min(u, v), max(u, v))
                                for ms in prev_round_matchings
                                for u, v, _ in ms
                                if u != hub and v != hub
                            }
                            if is_rev_round and prev_round_matchings and prev_nh_keys == set(cur_nh_map.keys()):
                                # Reverse matching sequence! Rotate storage rows by 1 if row sy1 was the non-M_L row!
                                st_qs_now = [q for q, s in gs.positions.items() if geom.zone_at[s].name == geom.storage.name]
                                if st_qs_now:
                                    mL_set = {q for u, v, _ in prev_round_matchings[-1] if u != hub and v != hub for q in (u, v)}
                                    top_row_qs = [q for q in st_qs_now if gs.positions[q][1] == sy1]
                                    # If row sy1 contains mostly non-M_L atoms (left at sy1 after Hub Sweep!), roll sy1 to bottom so M_L sits at top!
                                    if top_row_qs and sum(1 for q in top_row_qs if q in mL_set) < len(top_row_qs) // 2:
                                        ys_st = sorted({gs.positions[q][1] for q in st_qs_now})
                                        if ys_st[0] > geom.storage.y[0]:
                                            y_bot = ys_st[0] - 1
                                            txs_top = sorted(gs.positions[q][0] for q in top_row_qs)
                                            gs.move((sy1,), tuple(range(txs_top[0], txs_top[-1] + 1)), (y_bot,), tuple(txs_top))
                                            rem_st = [q for q in st_qs_now if gs.positions[q][1] < sy1]
                                            rys = sorted({gs.positions[q][1] for q in rem_st})
                                            rxs = sorted({gs.positions[q][0] for q in rem_st})
                                            gs.move(
                                                tuple(range(rys[0], rys[-1] + 1)),
                                                tuple(range(rxs[0], rxs[-1] + 1)),
                                                tuple(y + 1 for y in rys),
                                                tuple(rxs),
                                            )
                                matchings = [
                                    [(u, v, cur_nh_map[(min(u, v), max(u, v))]) for u, v, _ in ms if u != hub and v != hub]
                                    for ms in reversed(prev_round_matchings)
                                ]
                            elif mesh_guided and geom.entangling.pair_axis == "x":
                                # Extract Phase 0 (same-booth) and Phase 1 (cross-booth dx=1, dy=0) from CURRENT positions!
                                phase0 = [e for e in non_hub_e if are_paired(geom.entangling, proj_now[e[0]], proj_now[e[1]])]
                                p0_set = {(min(u, v), max(u, v)) for u, v, _ in phase0}
                                rem_e = [e for e in non_hub_e if (min(e[0], e[1]), max(e[0], e[1])) not in p0_set]
                                phase1 = [
                                    e for e in rem_e
                                    if proj_now[e[0]][1] == proj_now[e[1]][1] and abs(proj_now[e[0]][0] - proj_now[e[1]][0]) == 1
                                ]
                                p1_set = {(min(u, v), max(u, v)) for u, v, _ in phase1}
                                rem_e2 = [e for e in rem_e if (min(e[0], e[1]), max(e[0], e[1])) not in p1_set]
                                matchings = []
                                if phase0:
                                    matchings.extend(_edge_color_matchings(phase0, max_matching_size=max_m, priority=dist_prio))
                                if phase1:
                                    matchings.extend(_edge_color_matchings(phase1, max_matching_size=max_m, priority=dist_prio))
                                if rem_e2:
                                    matchings.extend(_edge_color_matchings(rem_e2, max_matching_size=max_m, priority=dist_prio))
                            else:
                                matchings = _edge_color_matchings(non_hub_e, max_matching_size=max_m, priority=dist_prio)
                            prev_round_matchings = [list(ms) for ms in matchings]
                            rem_hub = list(hub_edges)
                            for ms in matchings:
                                if fold_hub and hub is not None and len(ms) < max_m:
                                    used_v = {q for u, v, _ in ms for q in (u, v)}
                                    if hub not in used_v:
                                        # Pick a hub leaf that doesn't stall hub depth, empties 1-leaf booths/odd rows, and is close to hub!
                                        h_dep = gs.q_depth[hub]
                                        avail_h = [
                                            (idx, leaf, ang)
                                            for idx, (leaf, ang) in enumerate(rem_hub)
                                            if leaf not in used_v and gs.q_depth[leaf] <= h_dep + 1
                                        ]
                                        if avail_h:
                                            p_cur = _cur_proj_ent()
                                            b_cnt = Counter((p_cur[lv][1], (p_cur[lv][0] - ex0) // 2) for lv, _ in rem_hub)
                                            r_cnt = Counter(p_cur[lv][1] for lv, _ in rem_hub)
                                            hz = geom.zone_at[gs.positions[hub]].name
                                            avail_h.sort(
                                                key=lambda item: (
                                                    0 if in_x[item[1]] != in_x[hub] else 1,
                                                    max(0, gs.q_depth[item[1]] - h_dep),
                                                    0 if b_cnt[(p_cur[item[1]][1], (p_cur[item[1]][0] - ex0) // 2)] == 1 else 1,
                                                    0 if r_cnt[p_cur[item[1]][1]] % 2 == 1 else 1,
                                                    0 if geom.zone_at[gs.positions[item[1]]].name == hz else 1,
                                                    geom.dist(p_cur[hub], p_cur[item[1]]),
                                                )
                                            )
                                            picked_idx, leaf, ang = avail_h[0]
                                            rem_hub.pop(picked_idx)
                                            ms.append((hub, leaf, ang))
                                nb_m: dict[int, bool] = {}
                                rzx_m: list[tuple[int, int, float]] = []
                                st_name = geom.storage.name
                                for u, v, ang in ms:
                                    if in_x[u] and not in_x[v]:
                                        rzx_m.append((v, u, ang))
                                    elif in_x[v] and not in_x[u]:
                                        rzx_m.append((u, v, ang))
                                    else:
                                        # Flip whichever endpoint is ALREADY in storage (0 extra move!), else v
                                        flip_u = (geom.zone_at[gs.positions[u]].name == st_name) and (
                                            geom.zone_at[gs.positions[v]].name != st_name
                                        )
                                        if flip_u:
                                            nb_m[u] = not in_x[v]
                                            rzx_m.append((v, u, ang) if not in_x[v] else (u, v, ang))
                                        else:
                                            nb_m[v] = not in_x[u]
                                            rzx_m.append((u, v, ang) if not in_x[u] else (v, u, ang))
                                if nb_m:
                                    _set_basis(nb_m, f_idx)
                                if len(first_two_m) < 2:
                                    first_two_m.append([(u, v) for u, v, _ in rzx_m])
                                _execute_rzx_matching(gs, rzx_m, preferred_st=loading)
                            hub_edges = rem_hub

                        if hub is not None and hub_edges:
                            _evict_to_storage(
                                gs, [q for q in range(nq) if geom.zone_at[gs.positions[q]].name == geom.entangling.name]
                            )
                            nb_h = {hub: False}
                            for leaf, _ in hub_edges:
                                nb_h[leaf] = True
                            _set_basis(nb_h, f_idx)
                            _execute_hub_star_rzx(gs, hub, hub_edges, h_is_target=False)
                        continue

                    # Separate rzz into:
                    # (1) crossing edges between A (in_x=False) and B (in_x=True)
                    # (2) same-side edges inside A or inside B
                    # (3) hub star edges
                    cross_edges: list[tuple[int, int, float]] = []
                    same_a_edges: list[tuple[int, int, float]] = []
                    same_b_edges: list[tuple[int, int, float]] = []
                    hub_edges = []
                    for (u, v), ang in rzz_agg.items():
                        if _is_zero_angle(ang):
                            continue
                        if hub is not None and (u == hub or v == hub):
                            other = v if u == hub else u
                            hub_edges.append((other, ang))
                        elif (u in part_a and v in part_b):
                            cross_edges.append((u, v, ang))
                        elif (v in part_a and u in part_b):
                            cross_edges.append((v, u, ang))
                        elif u in part_a and v in part_a:
                            same_a_edges.append((u, v, ang))
                        else:
                            same_b_edges.append((u, v, ang))

                    # 1. Execute crossing edges (all A in Z-basis, all B in X-basis -> ZERO basis flips!)
                    if cross_edges:
                        need_b = {u: False for u, _, _ in cross_edges}
                        for _, v, _ in cross_edges:
                            need_b[v] = True
                        if fold_hub and hub is not None and hub_edges:
                            need_b[hub] = False
                        _set_basis(need_b, f_idx)

                        matchings = _edge_color_matchings(cross_edges, max_matching_size=max_m)
                        rem_hub = list(hub_edges)
                        for ms in matchings:
                            if fold_hub and hub is not None and len(ms) < max_m:
                                used_v = {q for u, v, _ in ms for q in (u, v)}
                                if hub not in used_v:
                                    # Pick a hub leaf in `part_b` (ALREADY in X-basis -> 0 extra H gates!)
                                    for idx, (leaf, ang) in enumerate(rem_hub):
                                        if leaf not in used_v and in_x[leaf]:
                                            ms.append((hub, leaf, ang))
                                            rem_hub.pop(idx)
                                            break
                            _execute_rzx_matching(gs, ms, preferred_st=loading)
                        hub_edges = rem_hub

                    # 2. Execute same-side edges inside A and inside B via matchings (flip 1 endpoint per edge!)
                    for same_edges, orig_x in ((same_a_edges, False), (same_b_edges, True)):
                        if not same_edges:
                            continue
                        # Re-bipartition this small residual graph to minimize basis flips!
                        vs_sub = sorted({q for u, v, _ in same_edges for q in (u, v)})
                        sub_a, sub_b = _max_cut_bipartition(vs_sub, [(u, v) for u, v, _ in same_edges])
                        s_cross = []
                        s_rem = []
                        for u, v, ang in same_edges:
                            if u in sub_a and v in sub_b:
                                s_cross.append((u, v, ang))
                            elif v in sub_a and u in sub_b:
                                s_cross.append((v, u, ang))
                            else:
                                s_rem.append((u, v, ang))
                        if s_cross:
                            nb_sub = {u: False for u, _, _ in s_cross}
                            for _, v, _ in s_cross:
                                nb_sub[v] = True
                            _set_basis(nb_sub, f_idx)
                            for ms in _edge_color_matchings(s_cross, max_matching_size=max_m):
                                _execute_rzx_matching(gs, ms, preferred_st=loading)
                        if s_rem:
                            for ms in _edge_color_matchings(s_rem, max_matching_size=max_m):
                                nb_m = {}
                                rzx_m = []
                                for u, v, ang in ms:
                                    if in_x[u] and not in_x[v]:
                                        rzx_m.append((v, u, ang))
                                    elif in_x[v] and not in_x[u]:
                                        rzx_m.append((u, v, ang))
                                    else:
                                        # Flip v to opposite of u
                                        nb_m[v] = not in_x[u]
                                        rzx_m.append((u, v, ang) if not in_x[u] else (v, u, ang))
                                if nb_m:
                                    _set_basis(nb_m, f_idx)
                                _execute_rzx_matching(gs, rzx_m, preferred_st=loading)

                    # 3. Execute remaining hub star edges via 2-Leaf Pipelined Row-Rolling Hub Sweep!
                    if hub is not None and hub_edges:
                        # Put hub in Z-basis, all remaining star leaves in X-basis (all in storage -> ZERO extra moves!)
                        _evict_to_storage(gs, [q for q in range(nq) if geom.zone_at[gs.positions[q]].name == geom.entangling.name])
                        nb_h = {hub: False}
                        for leaf, _ in hub_edges:
                            nb_h[leaf] = True
                        _set_basis(nb_h, f_idx)
                        _execute_hub_star_rzx(gs, hub, hub_edges, h_is_target=False)

            # Close any still-open X-basis at end of circuit
            close_flips = {q: h_ops for q in range(nq) if in_x[q]}
            if close_flips:
                _apply_1q_in_storage(gs, close_flips, set(), keep_in_storage=True)

            compiled, arch = gs.build_circuit(circuit, trailing_meas)
            return gs.total_distance + 2.9221 * gs.depth, compiled, arch, first_two_m

        # Build SA layout on non-hub edges
        ew_cnt: Counter[tuple[int, int]] = Counter()
        for u, v in non_hub_all:
            ew_cnt[(min(u, v), max(u, v))] += 1
        w_edges = [(u, v, float(w)) for (u, v), w in ew_cnt.items()]
        C = min(10, ent_w) if nq > 20 else (min(6, ent_w) if nq <= 12 else min(8, ent_w))
        R = math.ceil(nq / C)
        if R > ent_h or R > st_h:
            C = ent_w
            R = math.ceil(nq / C)
        sites = [(x, ey0 + y) for y in range(R) for x in range(ex0, ex0 + C)]
        fixed = {hub: (ex0, ey0)} if hub is not None else None

        n_2q_fronts = sum(1 for ft, _ in fronts if ft == "2q")
        candidates: list[tuple[float, QuantumCircuit, Architecture]] = []
        sa_steps = 30000 if nq <= 20 else 45000

        def _eval_qaoa(
            ep_cur: dict[int, Site],
            fold_hub: bool = True,
            unified_coloring: bool = True,
            mesh_guided: bool = False,
            palindrome: bool = False,
            try_aligned: bool = True,
        ) -> None:
            sc, c, a, ftm = _run_qaoa_candidate(
                ep_cur, fold_hub=fold_hub, unified_coloring=unified_coloring, mesh_guided=mesh_guided, palindrome=palindrome
            )
            candidates.append((sc, c, a))
            if try_aligned and ftm:
                ld_a = _build_m1_m2_loading(nq, ftm[0], ftm[1] if len(ftm) > 1 else [], ep_cur, geom)
                sc2, c2, a2, _ = _run_qaoa_candidate(
                    ep_cur,
                    fold_hub=fold_hub,
                    unified_coloring=unified_coloring,
                    mesh_guided=mesh_guided,
                    palindrome=palindrome,
                    custom_loading=ld_a,
                )
                candidates.append((sc2, c2, a2))

        for sd in (42, 7, 13):
            ep = _sa_layout(nq, sites, w_edges, geom, cost_mode="compact", fixed_sites=fixed, steps=sa_steps, seed=sd)
            _eval_qaoa(ep, fold_hub=True, unified_coloring=True, mesh_guided=False, palindrome=False)
            _eval_qaoa(ep, fold_hub=True, unified_coloring=True, mesh_guided=True, palindrome=False)
            if n_2q_fronts >= 2:
                _eval_qaoa(ep, fold_hub=True, unified_coloring=True, mesh_guided=True, palindrome=True)
                _eval_qaoa(ep, fold_hub=True, unified_coloring=True, mesh_guided=False, palindrome=True)
            if sd == 42:
                _eval_qaoa(ep, fold_hub=True, unified_coloring=False, mesh_guided=False, palindrome=False, try_aligned=False)
                _eval_qaoa(ep, fold_hub=False, unified_coloring=True, mesh_guided=True, palindrome=(n_2q_fronts >= 2))
            ep_m2 = _sa_layout(nq, sites, w_edges, geom, cost_mode="mesh2", fixed_sites=fixed, steps=sa_steps, seed=sd)
            _eval_qaoa(ep_m2, fold_hub=True, unified_coloring=True, mesh_guided=True, palindrome=False)
            if n_2q_fronts >= 2:
                _eval_qaoa(ep_m2, fold_hub=True, unified_coloring=True, mesh_guided=True, palindrome=True)
                _eval_qaoa(ep_m2, fold_hub=True, unified_coloring=True, mesh_guided=False, palindrome=True)
            else:
                _eval_qaoa(ep_m2, fold_hub=True, unified_coloring=True, mesh_guided=False, palindrome=False)

        candidates.sort(key=lambda x: x[0])
        return candidates[0][1], candidates[0][2]


    def _compile_general_cz(
        circuit: QuantumCircuit,
        ops: list[tuple[str, tuple[int, ...], tuple[float, ...]]],
        trailing_meas: list[tuple[int, int]],
        geom: ArchGeometry,
        raw_spec: ArchitectureSpec | None,
    ) -> tuple[QuantumCircuit, Architecture]:
        """CZ-only, Blocking-CZ compiler for arbitrary circuits (up to 70q+ and arbitrary specs)."""
        nq = circuit.num_qubits
        ex0, ex1 = geom.entangling.x
        ey0, ey1 = geom.entangling.y
        ent_w = ex1 - ex0 + 1
        st_w = geom.storage.x[1] - geom.storage.x[0] + 1
        st_h = geom.storage.y[1] - geom.storage.y[0] + 1

        # Expand any virtual rzz -> native {rx, rz, cz} and re-compress!
        cz_ops = _expand_virtual_rzz_to_cz(ops, allow_pi_simplify=(nq <= 19))
        cz_ops = compress_native_ops(cz_ops, nq)
        fronts = _slice_dag_fronts(cz_ops, nq)

        future_2q: list[set[int]] = [set() for _ in range(len(fronts))]
        acc: set[int] = set()
        for t in range(len(fronts) - 1, -1, -1):
            if fronts[t][0] == "2q":
                for _, gqs, _ in fronts[t][1]:
                    acc.update(gqs)
            future_2q[t] = set(acc)

        all_2q_edges = [(gqs[0], gqs[1], 1.0) for ft, fd in fronts if ft == "2q" for _, gqs, _ in fd]
        if not all_2q_edges:
            loading = {
                q: (geom.storage.x[0] + q % st_w, geom.storage.y[1] - q // st_w)
                for q in range(nq)
            }
            gs = GridState(geom, loading)
            gs._raw_spec = raw_spec
            for _, fd in fronts:
                _apply_1q_in_storage(gs, fd, set(), keep_in_storage=True)
            return gs.build_circuit(circuit, trailing_meas)

        is_large = nq > 35 or len(cz_ops) > 2000
        C = min(ent_w, st_w, 10)
        R = math.ceil(nq / C)
        if R > st_h:
            C = st_w
            R = math.ceil(nq / C)
        sites = [(geom.storage.x[0] + x, geom.storage.y[1] - y) for y in range(R) for x in range(C)]
        if len(sites) < nq:
            sites = [
                (x, y)
                for y in range(geom.storage.y[1], geom.storage.y[0] - 1, -1)
                for x in range(geom.storage.x[0], geom.storage.x[1] + 1)
            ]

        # Weight earlier 2Q fronts higher for initial storage layout
        w_edges: list[tuple[int, int, float]] = []
        r_cnt = 0
        for ft, fd in fronts:
            if ft == "2q":
                wt = 1.0 + 1.0 / (1.0 + r_cnt)
                for _, gqs, _ in fd:
                    w_edges.append((gqs[0], gqs[1], wt))
                r_cnt += 1

        sa_steps = 4000 if is_large else 25000
        loading = _sa_layout(nq, sites, w_edges, geom, cost_mode="compact", steps=sa_steps, seed=42)
        gs = GridState(geom, loading)
        gs._raw_spec = raw_spec

        for f_idx, (ftype, fdata) in enumerate(fronts):
            if ftype == "1q":
                _apply_1q_in_storage(gs, fdata, future_2q[f_idx])
            else:
                cz_pairs = [(min(gqs), max(gqs)) for _, gqs, _ in fdata]
                _execute_cz_matching_batch(gs, cz_pairs, fast_mode=is_large)

        return gs.build_circuit(circuit, trailing_meas)


    # ---------------------------------------------------------------------------
    # Public Entry Point
    # ---------------------------------------------------------------------------



    """Compile `circuit` for the zoned neutral-atom `architecture_spec` (CZ-only, blocking CZ, single-zone moves)."""
    geom = ArchGeometry.from_spec(architecture_spec)

    # 1. Check IHVA Rzy fast-path (compiled via Global S-Conjugation to CZ-only!)
    ihva_parsed = _try_parse_ihva(circuit)
    if ihva_parsed is not None:
        init_1q, blocks, trailing_meas = ihva_parsed
        return _compile_ihva_cz(circuit, init_1q, blocks, trailing_meas, geom, architecture_spec)

    # 2. General Stage 1 algebraic front-end (uses virtual `rzz` internally ONLY; never emitted!)
    raw_ops, trailing_meas = _extract_circuit_gates(circuit, recognize_virtual_rzz=True)
    ops = compress_native_ops(raw_ops, circuit.num_qubits)

    # 3. Check HEA 1D-chain specialization (pure CZ)
    hea_res = _try_compile_hea(circuit, ops, trailing_meas, geom, architecture_spec)
    if hea_res is not None:
        return hea_res[0], hea_res[1]

    # 4. Check QAOA / Commuting-RZZ specialization (compiled via Persistent-Basis CZ-only!)
    has_virtual_rzz = any(name == "rzz" for name, _, _ in ops)
    if has_virtual_rzz and circuit.num_qubits <= 60 and len(ops) <= 2000:
        fronts = _slice_dag_fronts(ops, circuit.num_qubits)
        # Use QAOA-CZ pipeline whenever 2Q fronts are dominated by virtual rzz gates
        n_rzz = sum(1 for name, _, _ in ops if name == "rzz")
        n_2q = sum(1 for _, qs, _ in ops if len(qs) == 2)
        if n_rzz >= 0.7 * max(1, n_2q):
            return _compile_qaoa_cz(circuit, fronts, trailing_meas, geom, architecture_spec)

    # 5. General CZ-only Blocking-CZ compiler (handles arbitrary circuits up to 70q+)
    return _compile_general_cz(circuit, ops, trailing_meas, geom, architecture_spec)
