from itertools import product
import numpy as np
from scipy.ndimage import convolve

def energy(labels,unary,beta=.5):
    z,x=np.indices(labels.shape);data=unary[z,x,labels].sum()
    pair=(labels[1:]!=labels[:-1]).sum()+(labels[:,1:]!=labels[:,:-1]).sum()
    return float(data+beta*pair)

def icm(unary,beta=.5,iterations=10,quality=None):
    """Conditional Potts model when unaries depend on observed mechanics."""
    unary=np.asarray(unary,float)
    if quality is not None:unary=unary*np.asarray(quality)[...,None]
    labels=unary.argmin(-1);h,w=labels.shape;history=[energy(labels,unary,beta)]
    for _ in range(iterations):
        changes=0
        for z in range(h):
            for x in range(w):
                neighbors=[labels[zz,xx] for zz,xx in [(z-1,x),(z+1,x),(z,x-1),(z,x+1)] if 0<=zz<h and 0<=xx<w]
                costs=unary[z,x]+beta*np.array([sum(v!=k for v in neighbors) for k in range(unary.shape[-1])]);new=costs.argmin();changes+=int(new!=labels[z,x]);labels[z,x]=new
        history.append(energy(labels,unary,beta))
        if not changes:break
    return labels,history

def exact_map(unary,beta=.5):
    h,w,k=unary.shape
    if k**(h*w)>100000:raise ValueError('Exact graph too large')
    states=(np.array(s).reshape(h,w) for s in product(range(k),repeat=h*w))
    best=min(states,key=lambda s:energy(s,unary,beta));return best,energy(best,unary,beta)
