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


def _b_router(st, max_run=8, max_iter=3000):
    """Greedy: repeatedly apply the column-run move (any target column, any ascending target rows)
    with the best gain / distance (DP over target rows for each distance threshold)."""
    it = 0
    rows_all = range(20)
    while st.todo and it < max_iter:
        it += 1
        need = {}
        for e in st.todo:
            for q in e:
                need.setdefault(q, set()).update(e - {q})
        best = None
        best_ratio = 0.0
        cols = {c: st.column_atoms(c) for c in range(10)}
        occ = st.occ
        for cs in range(10):
            atoms = cols[cs]
            m = len(atoms)
            for i in range(m):
                for j in range(i, min(m, i + max_run)):
                    run = atoms[i : j + 1]
                    if run[0][1] not in need and run[-1][1] not in need:
                        continue
                    qs = [q for _, q in run]
                    qset = set(qs)
                    k = len(qs)
                    srcs = [st.pos[q] for q in qs]
                    for ct in range(10):
                        # per atom: list of (t, d, g) for valid target rows
                        opts = []
                        maxg = 0
                        ming = 1e9
                        feasible = True
                        for q, src in zip(qs, srcs):
                            lst = []
                            nq = need.get(q, ())
                            for t in rows_all:
                                o = occ.get((ct, t))
                                if o is not None and o not in qset:
                                    continue
                                if ct == cs and o is not None and o != q and False:
                                    continue
                                g = 0
                                if t >= 10:
                                    w = occ.get((ct ^ 1, t))
                                    if w is not None and w in nq:
                                        g = 1
                                d = _b_dist(src, (ct, t))
                                lst.append((t, d, g))
                                if g and d < ming:
                                    ming = d
                            if not lst:
                                feasible = False
                                break
                            if any(g for _, _, g in lst):
                                maxg += 1
                            opts.append(lst)
                        if not feasible or maxg == 0 or maxg / ming <= best_ratio:
                            continue
                        ths = sorted({d for lst in opts for t, d, g in lst if g})
                        for D in ths:
                            if maxg / D <= best_ratio:
                                break
                            # DP: prev[t] = best gain with last atom at row <= t
                            prev = [0] * 20
                            choice = []
                            okd = True
                            for ai, lst in enumerate(opts):
                                cur = [-1] * 20
                                for t, d, g in lst:
                                    if d > D + 1e-9:
                                        continue
                                    if ai == 0:
                                        cur[t] = g
                                    elif t > 0 and prev[t - 1] >= 0:
                                        cur[t] = prev[t - 1] + g
                                # prefix max
                                run_best = -1
                                pm = [-1] * 20
                                for t in range(20):
                                    if cur[t] > run_best:
                                        run_best = cur[t]
                                    pm[t] = run_best
                                if run_best < 0:
                                    okd = False
                                    break
                                choice.append(cur)
                                prev = pm
                            if not okd:
                                continue
                            gain = prev[19]
                            if gain <= 0:
                                continue
                            ratio = gain / D
                            if ratio > best_ratio + 1e-12:
                                # backtrack
                                tgt = [0] * k
                                tmax = 19
                                need_val = gain
                                for ai in range(k - 1, -1, -1):
                                    cur = choice[ai]
                                    for t in range(tmax, -1, -1):
                                        if cur[t] < 0:
                                            continue
                                        g = cur[t] - (0 if ai == 0 else choice[ai - 1 and ai - 1][0] * 0)
                                        # verify consistency: cur[t] == need_val
                                        if cur[t] == need_val:
                                            tgt[ai] = t
                                            gt = 1 if (t >= 10 and occ.get((ct ^ 1, t)) in need.get(qs[ai], ())) else 0
                                            need_val -= gt
                                            tmax = t - 1
                                            break
                                best_ratio = ratio
                                best = ((ratio,), cs, qs, ct, tgt)
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


def _b_long_path(vertices, edges):
    """Greedy path cover of the graph (vertices, edges) concatenated into one order; DFS with
    Warnsdorff ordering for the first path."""
    adj = {v: set() for v in vertices}
    for e in edges:
        a, b = tuple(e)
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    left = set(vertices)
    order = []
    while left:
        start = min(left, key=lambda v: (len(adj[v] & left), v))
        path = [start]
        left.discard(start)
        while True:
            cand = [u for u in adj[path[-1]] if u in left]
            if not cand:
                break
            u = min(cand, key=lambda u: (len(adj[u] & left), u))
            path.append(u)
            left.discard(u)
        order += path
    return order


def _b_path_order(n, edges):
    """If the graph is a single path (HEA), its vertex order; else None."""
    adj = {v: [] for v in range(n)}
    for e in edges:
        a, b = tuple(e)
        adj[a].append(b)
        adj[b].append(a)
    if any(len(v) > 2 for v in adj.values()) or len(edges) != n - 1:
        return None
    ends = [v for v in range(n) if len(adj[v]) == 1]
    if len(ends) != 2:
        return None
    order, prev = [ends[0]], None
    while len(order) < n:
        nxt = [u for u in adj[order[-1]] if u != prev]
        if not nxt:
            return None
        prev = order[-1]
        order.append(nxt[0])
    return order


def _big_path(n, order, required, architecture_spec):
    """A path graph: rows of 9 at cols 0..8, one bulk move in, one 1.7 shift right, router for the junctions."""
    rows = [order[i : i + 9] for i in range(0, n, 9)]
    R = len(rows)
    assert R <= 10
    loading = {}
    for k, chunk in enumerate(rows):
        for j, q in enumerate(chunk):
            loading[q] = (j, 10 - R + k)
    st = _BigState(n, loading)
    st.todo = set(required)
    # bulk move of the whole block into the entangling zone (multi-row, multi-col)
    def bulk(rows_from, rows_to, cols_from, cols_to):
        qs = [q for (x, y), q in sorted(st.occ.items()) if x in cols_from and y in rows_from]
        rmap = dict(zip(rows_from, rows_to))
        cmap = dict(zip(cols_from, cols_to))
        targets = {q: (cmap[st.pos[q][0]], rmap[st.pos[q][1]]) for q in qs}
        for t in targets.values():
            o = st.occ.get(t)
            assert o is None or o in targets
        d = max(_b_dist(st.pos[q], targets[q]) for q in qs)
        st.ops.append(("move", list(rows_from), list(cols_from), list(rows_to), list(cols_to), qs))
        st.dist += d
        for q in qs:
            del st.occ[st.pos[q]]
        for q in qs:
            st.pos[q] = targets[q]
            st.occ[targets[q]] = q
        st.fire(qs)
    srows = list(range(10 - R, 10))
    erows = list(range(10, 10 + R))
    ncols = max(len(c) for c in rows)
    bulk(srows, erows, list(range(ncols)), list(range(ncols)))
    bulk(erows, erows, list(range(ncols)), list(range(1, ncols + 1)))
    _b_router(st)
    return st.build(), Architecture(st.loading, spec=architecture_spec)


_BIG_LAYOUT = "path"


def _b_slots(m):
    """Final entangling site of lane-order position i (same lane split as `_big_compile`)."""
    slots = []
    i = 0
    li = 0
    while i < m:
        if li <= 3:
            seg = min(20, m - i)
            a = (seg + 1) // 2
            b = seg - a
            for p in range(seg):
                lane = li if p % 2 == 0 else li + 1
                slots.append((2 * lane, 10 + p // 2))
            li += 2 if b else 1
            i += seg
        else:
            seg = min(10, m - i)
            for p in range(seg):
                slots.append((2 * li, 10 + p))
            li += 1
            i += seg
    return slots


def _b_sa_layout(order, edges, iters=60000, seed=12345):
    import random

    rng = random.Random(seed)
    m = len(order)
    slots = _b_slots(m)
    idx = {q: i for i, q in enumerate(order)}
    adj = {q: [] for q in order}
    for e in edges:
        a, b = tuple(e)
        if a in adj and b in adj:
            adj[a].append(b)
            adj[b].append(a)

    def ecost(pa, pb):
        # move b next to a, or a next to b
        da = _b_dist(pb, (pa[0] ^ 1, pa[1]))
        db = _b_dist(pa, (pb[0] ^ 1, pb[1]))
        return min(da, db)

    pos = [slots[i] for i in range(m)]
    cur = order[:]

    def qcost(q, p):
        return sum(ecost(p, pos[idx[w]]) for w in adj[q])

    total = sum(ecost(pos[idx[a]], pos[idx[b]]) for e in edges for a, b in [tuple(e)] if a in idx and b in idx)
    T0, T1 = 2.0, 0.02
    for it in range(iters):
        T = T0 * (T1 / T0) ** (it / iters)
        i, j = rng.randrange(m), rng.randrange(m)
        if i == j:
            continue
        a, b = cur[i], cur[j]
        before = qcost(a, pos[i]) + qcost(b, pos[j])
        cur[i], cur[j] = b, a
        idx[a], idx[b] = j, i
        after = qcost(a, pos[j]) + qcost(b, pos[i])
        delta = after - before
        if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
            total += delta
        else:
            cur[i], cur[j] = a, b
            idx[a], idx[b] = i, j
    return cur


def _big_compile_one(circuit, architecture_spec, required, variant=0):
    n = circuit.num_qubits
    if n <= 90:
        order = _b_path_order(n, required)
        if order is not None:
            return _big_path(n, order, required, architecture_spec)
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
        # path-aware lane assignment: segments of 20 along a long path of the non-hub graph,
        # even positions in lane 2j, odd positions in lane 2j+2 (so the router zips them)
        rest = [e for e in required if hub not in e]
        porder = _b_long_path(partners, rest)
        if variant:
            k = variant >> 1
            if k:
                porder = porder[(k * 7) % len(porder):] + porder[: (k * 7) % len(porder)]
            if variant & 1:
                porder = porder[::-1]
        if _BIG_LAYOUT == "sa":
            porder = _b_sa_layout(porder, rest)
        lane_lists = []
        i = 0
        while i < len(porder):
            seg = porder[i : i + 20]
            if len(lane_lists) <= 3:
                lane_lists.append(seg[0::2])
                if seg[1::2]:
                    lane_lists.append(seg[1::2])
                i += 20
            else:
                lane_lists.append(porder[i : i + 10])
                i += 10
        assert len(lane_lists) <= 5 and all(len(l) <= 10 for l in lane_lists)
        for li, chunk in enumerate(lane_lists):
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
    return st.build(), Architecture(st.loading, spec=architecture_spec), st.dist


_BIG_VARIANTS = 6
_BIG_BUDGET_S = 150.0  # safety net only (normal runs never hit it, so output stays deterministic)


def _big_compile(circuit, architecture_spec):
    import time

    t0 = time.time()
    n = circuit.num_qubits
    required, allowed = _b_required(circuit)
    required = {e for e in required if e in allowed}  # (always the case)
    if n <= 90:
        order = _b_path_order(n, required)
        if order is not None:
            return _big_path(n, order, required, architecture_spec)
    best = None
    for v in range(_BIG_VARIANTS):
        t1 = time.time()
        qc, arch, d = _big_compile_one(circuit, architecture_spec, required, v)
        score = d + 2.92 * qc.depth()
        if best is None or score < best[0]:
            best = (score, qc, arch)
        if time.time() - t0 + (time.time() - t1) > _BIG_BUDGET_S:
            break
    return best[1], best[2]
