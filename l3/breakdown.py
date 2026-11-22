import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as M
from stats import gap_costs
for path in sys.argv[1:]:
    circ, loading = M.load_cached(path)
    sch = M.extract(circ)
    gaps = gap_costs(circ, loading)   # gaps[k] before layer k; one layer may hold 2 pulses
    # map layers->pulses: count pulses per layer again
    qi = {b: i for i, b in enumerate(circ.qubits)}
    layer_sizes = []; cur = None
    for ins in circ.data:
        nm = ins.operation.name
        if nm == 'cz':
            cur = (cur or 0) + 1
        elif nm == 'move' and cur is not None:
            layer_sizes.append(cur); cur = None
    if cur: layer_sizes.append(cur)
    agg = defaultdict(lambda: [0, 0.0, 0])
    tail = gaps[len(layer_sizes)][0] if len(gaps) > len(layer_sizes) else 0
    for k, sz in enumerate(layer_sizes):
        c = 'single' if sz == 1 else ('2-3' if sz <= 3 else '4+')
        agg[c][0] += 1; agg[c][1] += gaps[k][0]; agg[c][2] += gaps[k][1]
    tot = sum(g[0] for g in gaps)
    print(Path(path).stem[9:], f"D={tot:.0f}", ' '.join(f"{c}: {v[0]} layers, D={v[1]:.0f} ({v[1]/v[0]:.2f}/layer, {v[2]/v[0]:.1f} moves)" for c, v in sorted(agg.items())), f"tail={tail:.1f}")
