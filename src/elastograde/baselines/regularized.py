import numpy as np
from scipy.ndimage import gaussian_filter

def refine(flow,confidence,lam=.4,iterations=15):
    """Confidence-weighted quadratic smoothing, not a published named solver."""
    if not 0<=lam<=1:raise ValueError('lam outside [0,1]')
    u=flow.copy();c=np.clip(confidence,0,1)[None]
    for _ in range(iterations):
        smooth=np.stack([gaussian_filter(v,1) for v in u])
        u=(c*flow+lam*smooth)/(c+lam+1e-8)
    return u
