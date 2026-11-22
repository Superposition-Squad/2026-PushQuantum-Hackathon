"""Exact (MILP) re-solve of single gaps of an incumbent plan with fixed endpoints."""
import sys, time
sys.path.insert(0, __import__('os').path.dirname(__file__))
import model as M, chain as C, milp as MI
from check import check

f = sys.argv[1]; ks = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else None
tl = float(sys.argv[3]) if len(sys.argv) > 3 else 60
c, l = M.load_cached(f); s = M.extract(c)
X, gaps = C.incumbent(c, l, s); B = C.binding(s)
newg = list(gaps); tot_i = tot_m = 0
for k in (ks or range(s.K + 1)):
    ci = C.gap_cost(X[k], gaps[k]); T = len(gaps[k])
    if T == 0: continue
    moved = {q for g in gaps[k] for q in g}
    dip = [(q, 1, T - 1) for q in B[k]] if k < len(B) and B[k] else []
    t = time.time()
    r = MI.solve_window(X[k], [], [], [], dip, end=X[k + 1], T=T, active=moved, time_limit=tl)
    if r is None:
        print(k, 'infeasible/none', T, len(moved)); continue
    mv = [g for g in r['moves'] if g]
    try:
        cm = C.gap_cost(X[k], mv)
    except Exception as e:
        print(k, 'decode error', e); continue
    tot_i += ci; tot_m += min(ci, cm)
    if cm < ci - 1e-9: newg[k] = mv
    print(f'gap {k:3d} T={T:2d} atoms={len(moved):2d} incumbent {ci:6.2f} milp {cm:6.2f} (obj {r["obj"]:.2f}, {r["msg"][:30]}) build {r["build_s"]:.1f}s solve {r["solve_s"]:.1f}s', flush=True)
print('window total incumbent', round(tot_i, 2), 'best', round(tot_m, 2))
out, load, d = C.assemble(s, l, newg); print('validated', check(out, load))
