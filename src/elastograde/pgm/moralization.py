from itertools import combinations

def moralize(network):
    edges=set()
    for child,parents in network.parents.items():
        for parent in parents:edges.add(tuple(sorted((child,parent))))
        for a,b in combinations(parents,2):edges.add(tuple(sorted((a,b))))
    return sorted(edges),network.factors()

def verify(network):
    from .markov_network import enumerate_factors
    edges,factors=moralize(network);nodes,joint,Z=enumerate_factors(factors)
    error=max(abs(p-network.joint(dict(zip(nodes,values)))) for values,p in joint.items())
    return {'edges':edges,'partition_function':Z,'max_joint_error':error,'triangulation_performed':False}
