from itertools import product
import numpy as np

def separated(edges,A,B,given):
    graph={}
    for a,b in edges:graph.setdefault(a,set()).add(b);graph.setdefault(b,set()).add(a)
    seen=set(given);stack=list(A)
    while stack:
        n=stack.pop()
        if n in seen:continue
        if n in B:return False
        seen.add(n);stack.extend(graph.get(n,set())-seen)
    return True

def enumerate_factors(factors):
    nodes=list(dict.fromkeys(v for f in factors for v in f.variables));joint={}
    for vals in product(range(2),repeat=len(nodes)):
        a=dict(zip(nodes,vals));joint[vals]=float(np.prod([f.values[tuple(a[v] for v in f.variables)] for f in factors]))
    Z=sum(joint.values())
    if Z<=0:raise ValueError('Zero partition function')
    return nodes,{k:v/Z for k,v in joint.items()},Z
