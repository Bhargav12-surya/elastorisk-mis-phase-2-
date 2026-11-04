import numpy as np
from copy import deepcopy

def corrupt(sample,kind,severity=.1,seed=42):
    s=deepcopy(sample);rng=np.random.default_rng(seed)
    if kind=='noise':s.post=s.post+severity*np.std(s.post)*rng.normal(size=s.post.shape)
    elif kind=='gain':s.post=s.post*(1+severity)
    elif kind=='dropout':
        n=max(1,int(s.post.shape[0]*severity));s.post[s.post.shape[0]//2:s.post.shape[0]//2+n]=0
    elif kind=='missing_metadata':s.metadata['spacing_mm']=None
    else:raise ValueError('Unknown corruption')
    s.metadata['corruption']={'kind':kind,'severity':severity,'seed':seed};return s
