# ================================================================ BIG (> 19 qubits): touch each required pair once
#
# The official validator only checks, above 19 qubits, that
#     required <= interaction_pairs(compiled) <= allowed
# with required = interaction_pairs(transpile(reference, ["rx","rz","cz"], optimization_level=1)) and
# allowed = interaction_pairs(reference).  So we emit exactly one `cz` per required pair and nothing
# else (no 1q gates, no measurements).  Movement: a hub conveyor (partners slide past the hub one row
# per move) followed by a greedy router of single-column runs.

_BIG_THRESHOLD = 19


def _b_phys(site):
    x, y = site
    if y <= 9:
        return (float(x), float(y))
    return (2.0 * (x // 2) + 0.3 * (x % 2), float(y))


def _b_dist(a, b):
    pa, pb = _b_phys(a), _b_phys(b)
    return ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5


def _b_pairs(circuit):
    index = {bit: i for i, bit in enumerate(circuit.qubits)}
    out = set()
    for inst in circuit.data:
        if len(inst.qubits) > 1 and inst.operation.name not in ("move", "barrier"):
            for p in itertools.combinations([index[q] for q in inst.qubits], 2):
                out.add(frozenset(p))
    return out


def _b_required(circuit):
    """Exactly the validator's `required` and `allowed` sets."""
    allowed = _b_pairs(circuit)
    required = _b_pairs(transpile(circuit, basis_gates=["rx", "rz", "cz"], optimization_level=1))
    return required, allowed


class _BigState:
    def __init__(self, n, loading, gate_name="cz"):
        self.n = n
        self.loading = dict(loading)
        self.pos = dict(loading)
        self.occ = {s: q for q, s in self.pos.items()}
        assert len(self.occ) == n
        self.ops = []  # ("move", rows, cols, nrows, ncols, qubits) | ("g", a, b)
        self.dist = 0.0
        self.todo = set()
        self.gate_name = gate_name

    # -- helpers
    @staticmethod
    def partner_site(site):
        x, y = site
        if y < 10:
            return None
        return (x ^ 1, y)

    def move_column_run(self, col, qubits_sorted, new_col, new_rows):
        """Moves the atoms `qubits_sorted` (all atoms of column `col` within their row span, ascending
        rows) to column `new_col` at ascending `new_rows`."""
        rows = [self.pos[q][1] for q in qubits_sorted]
        lo, hi = rows[0], rows[-1]
        span = list(range(lo, hi + 1))
        # every atom in the span must be selected
        for y in span:
            o = self.occ.get((col, y))
            assert (o is None) == (y not in rows), "ghost in run"
        assert all(b > a for a, b in zip(new_rows, new_rows[1:]))
        assert len(new_rows) == len(rows)
        targets = [(new_col, r) for r in new_rows]
        moving = set(qubits_sorted)
        for t in targets:
            assert 0 <= t[0] <= 9 and 0 <= t[1] <= 19
            o = self.occ.get(t)
            assert o is None or o in moving, "deposit on occupied"
        d = max(_b_dist(self.pos[q], t) for q, t in zip(qubits_sorted, targets))
        if len(span) == len(rows):
            prow, nrow = span, list(new_rows)
        else:
            prow, nrow = span, list(new_rows)  # empty picked rows are dropped by the validator
        self.ops.append(("move", prow, [col], nrow, [new_col], list(qubits_sorted)))
        self.dist += d
        for q in qubits_sorted:
            del self.occ[self.pos[q]]
        for q, t in zip(qubits_sorted, targets):
            self.pos[q] = t
            self.occ[t] = q
        self.fire(qubits_sorted)
        return d

    def fire(self, qubits):
        """Emit a gate on every newly formed pair that still needs one."""
        for q in qubits:
            ps = self.partner_site(self.pos[q])
            if ps is None:
                continue
            w = self.occ.get(ps)
            if w is None:
                continue
            e = frozenset((q, w))
            if e in self.todo:
                self.todo.discard(e)
                a, b = sorted((q, w))
                self.ops.append(("g", a, b))

    def column_atoms(self, col):
        return sorted((y, q) for (x, y), q in self.occ.items() if x == col)

    def build(self):
        qc = QuantumCircuit(self.n)
        for op in self.ops:
            if op[0] == "move":
                _, rows, cols, nrows, ncols, qs = op
                qc.append(Move(rows, cols, nrows, ncols, len(qs)), qs)
            else:
                if self.gate_name == "cz":
                    qc.cz(op[1], op[2])
                else:
                    qc.rzz(1.0, op[1], op[2])
        return qc


def _b_free_sites(st):
    return [(x, y) for y in range(20) for x in range(10) if (x, y) not in st.occ]


def _b_router(st, max_run=4, max_iter=2000):
    """Greedy: repeatedly apply the single-column-run translation with the best gain / distance."""
    it = 0
    while st.todo and it < max_iter:
        it += 1
        need = {}
        for e in st.todo:
            for q in e:
                need.setdefault(q, set()).update(e - {q})
        best = None
        cols = {c: st.column_atoms(c) for c in range(10)}
        for cs in range(10):
            atoms = cols[cs]
            m = len(atoms)
            for i in range(m):
                if atoms[i][1] not in need:
                    pass
                for j in range(i, min(m, i + max_run)):
                    run = atoms[i : j + 1]
                    if not any(q in need for _, q in run):
                        continue
                    rows = [y for y, _ in run]
                    qs = [q for _, q in run]
                    for ct in range(10):
                        for dy in range(-19, 20):
                            if ct == cs and dy == 0:
                                continue
                            nr = [y + dy for y in rows]
                            if nr[0] < 0 or nr[-1] > 19:
                                continue
                            ok = True
                            gain = 0
                            for q, y in zip(qs, nr):
                                t = (ct, y)
                                o = st.occ.get(t)
                                if o is not None and not (ct == cs and o in qs):
                                    ok = False
                                    break
                                if y >= 10:
                                    w = st.occ.get((ct ^ 1, y))
                                    if w is not None and w in need.get(q, ()):
                                        gain += 1
                            if not ok or gain == 0:
                                continue
                            d = max(_b_dist(st.pos[q], (ct, y)) for q, y in zip(qs, nr))
                            key = (gain / d, gain, -d)
                            if best is None or key > best[0]:
                                best = (key, cs, qs, ct, nr)
        if best is not None:
            _, cs, qs, ct, nr = best
            st.move_column_run(cs, qs, ct, nr)
            continue
        # fallback: evict the occupant of u's partner site, then bring v there
        e = min(st.todo, key=lambda e: (_b_dist(*[st.pos[q] for q in e]), sorted(e)))
        u, v = sorted(e)
        su = st.pos[u]
        if su[1] < 10:
            # bring u into the entangling zone first: nearest free site whose partner is free
            cands = [s for s in _b_free_sites(st) if s[1] >= 10 and (s[0] ^ 1, s[1]) not in st.occ]
            s = min(cands, key=lambda s: (_b_dist(su, s), s))
            st.move_column_run(su[0], [u], s[0], [s[1]])
            continue
        ps = (su[0] ^ 1, su[1])
        w = st.occ.get(ps)
        if w is not None and w != v:
            free = [s for s in _b_free_sites(st) if s != ps]
            s = min(free, key=lambda s: (_b_dist(ps, s), s))
            st.move_column_run(ps[0], [w], s[0], [s[1]])
        if st.occ.get(ps) is None:
            st.move_column_run(st.pos[v][0], [v], ps[0], [ps[1]])
    assert not st.todo, "router failed"


def _big_compile(circuit, architecture_spec):
    n = circuit.num_qubits
    required, allowed = _b_required(circuit)
    required = {e for e in required if e in allowed}  # (always the case)
    deg = {q: 0 for q in range(n)}
    for e in required:
        for q in e:
            deg[q] += 1
    hub = max(range(n), key=lambda q: (deg[q], -q))
    partners = sorted(q for e in required if hub in e for q in e if q != hub)
    use_hub = deg[hub] >= max(6, n // 3) and n <= 51
    loading = {}
    lanes = []
    if use_hub:
        # lanes on even storage columns, bottom-aligned at row 9
        chunks = [partners[i : i + 10] for i in range(0, len(partners), 10)]
        assert len(chunks) <= 5
        for li, chunk in enumerate(chunks):
            col = 2 * li
            m = len(chunk)
            for k, q in enumerate(chunk):
                loading[q] = (col, 10 - m + k)
            lanes.append((col, chunk))
        loading[hub] = (1, 9)
        others = [q for q in range(n) if q not in loading]
        free = [(x, y) for x in (3, 5, 7, 9, 1) for y in range(9, -1, -1) if (x, y) not in loading.values()]
        free += [(x, y) for x in (0, 2, 4, 6, 8) for y in range(10) if (x, y) not in loading.values()]
        for q, s in zip(others, free):
            loading[q] = s
    else:
        # snake rows from the bottom of storage, in qubit order
        sites = []
        for y in range(9, -1, -1):
            xs = range(10) if (9 - y) % 2 == 0 else range(9, -1, -1)
            sites += [(x, y) for x in xs]
        for q in range(n):
            loading[q] = sites[q]
    st = _BigState(n, loading)
    st.todo = set(required)
    if use_hub:
        st.move_column_run(1, [hub], 1, [10])
        for li, (col, chunk) in enumerate(lanes):
            if li > 0:
                st.move_column_run(st.pos[hub][0], [hub], col + 1, [10])
            m = len(chunk)
            for _ in range(m):
                qs = [q for _, q in st.column_atoms(col) if q in chunk]
                ys = [st.pos[q][1] for q in qs]
                st.move_column_run(col, qs, col, [y + 1 for y in ys])
    _b_router(st)
    return st.build(), Architecture(st.loading, spec=architecture_spec)
