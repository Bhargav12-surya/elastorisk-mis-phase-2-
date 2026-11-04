import numpy as np
from elastograde.mechanics.warping import warp

def quality(pre,post,flow,reverse,valid):
    moved,inside=warp(post,flow);back,inside_back=warp(reverse,flow);mask=valid&inside&inside_back
    if not mask.any():raise ValueError('No valid quality support')
    residual=np.abs(pre-moved);cycle=np.linalg.norm(flow+back,axis=0)
    return {'residual':residual,'cycle':cycle,'valid':mask,'residual_mean':float(residual[mask].mean()),'cycle_mean':float(cycle[mask].mean()),'confidence_status':'uncalibrated_quality_proxy'}

def ensemble_disagreement(flows):
    if len(flows)<2:raise ValueError('At least two independently fitted models required')
    f=np.stack(flows);return np.sqrt(np.mean(np.var(f,axis=0,ddof=1),axis=0))
