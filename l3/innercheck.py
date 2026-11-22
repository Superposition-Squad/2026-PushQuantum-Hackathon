"""Inner solver quality: re-solve every gap of the incumbent (same endpoints) and compare."""
import sys, glob, time
sys.path.insert(0, __import__('os').path.dirname(__file__))
import model as M, chain as C
from check import check
T = [0, 0, 0]
for f in sorted(glob.glob(sys.argv[1] + '/*.pkl')):
    if len(sys.argv) > 2 and sys.argv[2] not in f: continue
    c, l = M.load_cached(f); s = M.extract(c)
    X, gaps = C.incumbent(c, l, s)
    B = C.binding(s)
    t = time.time(); inc = mine = best = 0.0; worse = better = 0; newgaps = []
    for k in range(s.K + 1):
        ci = C.gap_cost(X[k], gaps[k])
        mv, cm = C.transition(X[k], X[k + 1], B[k] if k < len(B) else set())
        inc += ci; mine += cm
        if cm < ci - 1e-9: better += 1; newgaps.append(mv); best += cm
        else: newgaps.append(gaps[k]); best += ci
        worse += cm > ci + 1e-9
    out, load, d = C.assemble(s, l, newgaps)
    r = check(out, load)
    T[0] += inc; T[1] += mine; T[2] += r['distance']
    print(f.split('/')[-1][9:-4], f'incumbent {inc:7.1f} inner {mine:7.1f} (better {better}, worse {worse}) best-of {best:7.1f} validated {r["distance"]:7.1f} {time.time()-t:.1f}s', flush=True)
print('TOTAL', [round(x, 1) for x in T])
