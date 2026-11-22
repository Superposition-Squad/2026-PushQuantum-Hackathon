"""Move-kind breakdown of any circuit (zone->zone etc.), with #atoms histogram."""
import sys
from collections import defaultdict
sys.path.insert(0, __import__('os').path.dirname(__file__))
import model as M

def kinds(circ, loading):
    st = M.State(loading); qi = {b: i for i, b in enumerate(circ.qubits)}
    agg = defaultdict(lambda: [0, 0.0, 0])
    for ins in circ.data:
        if ins.operation.name != 'move': continue
        qs = [qi[q] for q in ins.qubits]; rows, cols, nr, nc = ins.operation.params
        occ_r = [y for y in rows if any(st.pos[q][1] == y for q in qs)]; occ_c = [x for x in cols if any(st.pos[q][0] == x for q in qs)]
        if len(nr) == len(rows): occ_r = list(rows)
        if len(nc) == len(cols): occ_c = list(cols)
        rm, cm = dict(zip(occ_r, nr)), dict(zip(occ_c, nc)); g = {q: (cm[st.pos[q][0]], rm[st.pos[q][1]]) for q in qs}
        d = max(M.dist(st.pos[q], g[q]) for q in qs)
        k = M.zone_of(st.pos[qs[0]]) + '>' + M.zone_of(g[qs[0]])
        agg[k][0] += 1; agg[k][1] += d; agg[k][2] += len(qs)
        for q in qs: del st.occ[st.pos[q]]
        for q, t in g.items(): st.pos[q] = t; st.occ[t] = q
    return ' '.join(f"{k}: {v[0]}mv {v[1]:.0f} ({v[1]/v[0]:.2f}, {v[2]/v[0]:.1f}at)" for k, v in sorted(agg.items()))
