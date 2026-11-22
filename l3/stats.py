"""Schedule statistics and where the source compiler spends its distance, per gap type."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as M  # noqa: E402


def gap_costs(circ, loading):
    """distance and #moves of the source output between consecutive CZ layers (gap k = before layer k)."""
    st = M.State(loading)
    qi = {b: i for i, b in enumerate(circ.qubits)}
    gaps = [[0.0, 0]]
    in_layer = False
    for ins in circ.data:
        name = ins.operation.name
        if name == "cz":
            in_layer = True
        elif name == "move":
            if in_layer:
                gaps.append([0.0, 0])
                in_layer = False
            qs = [qi[q] for q in ins.qubits]
            rows, cols, nr, nc = ins.operation.params
            occ_r = [y for y in rows if any(st.pos[q][1] == y for q in qs)]
            occ_c = [x for x in cols if any(st.pos[q][0] == x for q in qs)]
            if len(nr) == len(rows):
                occ_r = list(rows)
            if len(nc) == len(cols):
                occ_c = list(cols)
            rm, cm = dict(zip(occ_r, nr)), dict(zip(occ_c, nc))
            grp = {q: (cm[st.pos[q][0]], rm[st.pos[q][1]]) for q in qs}
            d = max(M.dist(st.pos[q], grp[q]) for q in qs)
            for q in qs:
                del st.occ[st.pos[q]]
            for q, t in grp.items():
                st.pos[q] = t
                st.occ[t] = q
            gaps[-1][0] += d
            gaps[-1][1] += 1
    return gaps


if __name__ == "__main__":
    for path in sys.argv[1:]:
        circ, loading = M.load_cached(path)
        sch = M.extract(circ)
        assert M.schedule_signature(sch) == M.signature(circ), "extraction round trip failed"
        layers = gap_costs(circ, loading)
        K0 = sch.K
        rem = M.cancel_dummies(sch)
        mem = sch.members()
        owe = sch.owe()
        sizes = Counter(len(P) for P in sch.pulses)
        bind = sum(len(mem[k] & mem[k - 1] & owe[k]) for k in range(1, sch.K))
        D = sum(g[0] for g in layers)
        nm = sum(g[1] for g in layers)
        # crude lower bound: per gap, >=1.0 if someone must leave the zone, >=1.0 if someone must enter
        lb = 0.0
        for k in range(sch.K):
            prev = mem[k - 1] if k else set()
            leave = (prev - mem[k]) | (prev & mem[k] & owe[k])
            enter = (mem[k] - prev) | (prev & mem[k] & owe[k])
            lb += (1.0 if leave else 0) + (1.0 if enter else 0)
        lb += 1.0 if mem[-1] & sch.final_owe() else 0
        print(f"{Path(path).stem[9:]:24s} n={sch.n:2d} layers={len(layers)-1:4d} pulses={K0:4d}->{sch.K:4d} "
              f"dummyCZ={rem:3d} D={D:7.1f} moves={nm:5d} D/pulse={D/sch.K:4.2f} LB={lb:6.1f} "
              f"binding={bind:4d} sizes={dict(sorted(sizes.items())[:6])}")
