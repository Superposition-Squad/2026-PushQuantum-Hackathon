#!/usr/bin/env python3
"""Add PLAN_RANK + guarded rollout finalists to a relay-family compiler (relay_plus*):
    python l3/exp/add_rollout.py <src compiler.py> <dst compiler.py>
The rollout machinery is taken verbatim from compilers/relay_plus_roll/compiler.py."""
import re, sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
ref = open('compilers/relay_plus_roll/compiler.py').read()
roll = ref[ref.index('\n# ---------------------------------------------------------------- rollout lookahead'):ref.index('\ndef compile_ihva(')]
# PLAN_RANK selection (same edit as l3/exp/relay_roll.py)
a = ref.index('        if self.RECT_SIM > 1 and len(plans) > 1:\n            sims = []')
b = ref.index('            rc = None', a) + len('            rc = None')
new_sel = ref[a:b]
a2 = s.index('        if self.RECT_SIM > 1 and len(plans) > 1:\n            best = None')
b2 = s.index('            rc = None', a2) + len('            rc = None')
s = s[:a2] + new_sel + s[b2:]
s = s.replace('    GATHER = True\n    GATHER_MAX = 5\n', '    GATHER = True\n    GATHER_MAX = 5\n    PLAN_RANK = 0  # rollout: take the r-th best simulated band plan\n', 1)
i = s.index('\ndef compile_ihva(')
s = s[:i] + roll + s[i:]
a = ref.index('    best = None\n    for _, _, cand, lay in finalists:')
b = ref.index('    _, qc, home, ex = best', a) + len('    _, qc, home, ex = best')
a2 = s.index('    best = None\n    for _, _, cand, lay in finalists:')
b2 = s.index('    _, qc, home = best', a2) + len('    _, qc, home = best')
s = s[:a2] + ref[a:b] + s[b2:]
open(dst, 'w').write(s)
print('wrote', dst)
