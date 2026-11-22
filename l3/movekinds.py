import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as M
tot = defaultdict(lambda: [0, 0.0])
for path in sys.argv[1:]:
    circ, loading = M.load_cached(path)
    st = M.State(loading); qi = {b: i for i, b in enumerate(circ.qubits)}
    agg = defaultdict(lambda: [0, 0.0]); rows_hist = defaultdict(lambda: [0, 0.0])
    for ins in circ.data:
        if ins.operation.name != 'move': continue
        qs = [qi[q] for q in ins.qubits]
        rows, cols, nr, nc = ins.operation.params
        occ_r = [y for y in rows if any(st.pos[q][1] == y for q in qs)]
        occ_c = [x for x in cols if any(st.pos[q][0] == x for q in qs)]
        if len(nr) == len(rows): occ_r = list(rows)
        if len(nc) == len(cols): occ_c = list(cols)
        rm, cm = dict(zip(occ_r, nr)), dict(zip(occ_c, nc))
        grp = {q: (cm[st.pos[q][0]], rm[st.pos[q][1]]) for q in qs}
        d = max(M.dist(st.pos[q], grp[q]) for q in qs)
        k = M.zone_of(st.pos[qs[0]]) + '>' + M.zone_of(grp[qs[0]])
        agg[k][0] += 1; agg[k][1] += d
        for q in qs: del st.occ[st.pos[q]]
        for q, t in grp.items(): st.pos[q] = t; st.occ[t] = q
    for k, v in agg.items(): tot[k][0] += v[0]; tot[k][1] += v[1]
    print(Path(path).stem[9:], ' '.join(f"{k}: {v[0]} mv {v[1]:.0f} ({v[1]/v[0]:.2f})" for k, v in sorted(agg.items())))
print('TOTAL', ' '.join(f"{k}: {v[0]} mv {v[1]:.0f} ({v[1]/v[0]:.2f})" for k, v in sorted(tot.items())))
