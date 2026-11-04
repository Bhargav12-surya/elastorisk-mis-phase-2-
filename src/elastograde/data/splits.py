import numpy as np
from .audit import audit

def grouped_split(records,group_key='geometry_id',seed=42):
    if any(not r.get(group_key) for r in records):raise ValueError('Verified independent group IDs are required')
    groups=sorted(set(r[group_key] for r in records))
    if len(groups)<4:raise ValueError('Need >=4 independent groups for train/validation/calibration/test')
    rng=np.random.default_rng(seed);rng.shuffle(groups)
    n=len(groups);nt=max(1,int(n*.6));nv=max(1,int(n*.15));nc=max(1,int(n*.1))
    while nt+nv+nc>=n:nt-=1
    roles=['train']*nt+['validation']*nv+['calibration']*nc+['test']*(n-nt-nv-nc)
    assignments=dict(zip(groups,roles))
    out=[dict(r,split=assignments[r[group_key]]) for r in records]
    audit(out,group_key);return out
