"""Exact Clifford check: angles -> random multiples of pi/2, compile, compare stabilizer tableaux.
1q runs of the compiled circuit are multiplied out and matched against the 24 single-qubit Cliffords."""
import sys, importlib.util, math, random, itertools
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, Operator
def load(p):
    spec=importlib.util.spec_from_file_location("c",p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
# 24 single-qubit Cliffords
words=[""]
cl=[]
H=np.array([[1,1],[1,-1]])/np.sqrt(2); S=np.diag([1,1j])
seen=[]
def key(u):
    i=np.argmax(np.abs(u.flatten())>1e-6); ph=u.flatten()[i]/abs(u.flatten()[i]); return u/ph
for L in range(0,7):
    for w in itertools.product("HS",repeat=L):
        u=np.eye(2,dtype=complex)
        for g in w: u=(H if g=="H" else S)@u
        k=key(u)
        if not any(np.allclose(k,s) for s,_ in seen): seen.append((k,w))
assert len(seen)==24, len(seen)
def as_cliff(u):
    k=key(u)
    for s,w in seen:
        if np.allclose(k,s,atol=1e-6): return w
    return None
def strip(q):
    n=q.num_qubits; r=QuantumCircuit(n); acc={}
    def flush(i):
        u=acc.pop(i,None)
        if u is None: return
        w=as_cliff(u)
        if w is None: raise SystemExit("non-Clifford 1q run")
        for g in w: (r.h if g=="H" else r.s)(i)
    for inst in q.data:
        nm=inst.operation.name
        if nm in("move","barrier","measure"): continue
        qs=[q.find_bit(b).index for b in inst.qubits]
        if len(qs)==1:
            acc[qs[0]]=Operator(inst.operation).data@acc.get(qs[0],np.eye(2))
        else:
            for i in qs: flush(i)
            r.append(inst.operation,qs)
    for i in list(acc): flush(i)
    return r
comp=load(sys.argv[1]); ok=True
for f in sys.argv[2:]:
    for seed in (1,2):
        rnd=random.Random(seed)
        src=QuantumCircuit.from_qasm_file(f)
        c=QuantumCircuit(*src.qregs,*src.cregs)
        for inst in src.data:
            op=inst.operation.copy()
            if op.params and op.name not in ("barrier","measure"):
                op.params=[rnd.choice([1,2,3])*math.pi/2 for _ in op.params]
            c.append(op,inst.qubits,inst.clbits)
        out,arch=comp.compile_circuit(c.copy(),None)
        eq = Clifford(strip(out))==Clifford(strip(c))
        ok&=eq
        print(f.split('/')[-1],seed,"EQUAL" if eq else "DIFFERENT",flush=True)
print("ALL OK" if ok else "FAIL")
