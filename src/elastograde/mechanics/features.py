import numpy as np
from scipy.ndimage import binary_erosion

def mechanical_features(axial,valid,roi=None,spacing=None):
    mask=np.asarray(valid,bool)&np.isfinite(axial)
    if not mask.any():raise ValueError('No valid strain support')
    v=axial[mask];med=float(np.median(v))
    out={'median_strain':med,'mad_strain':float(np.median(abs(v-med))),'q10':float(np.quantile(v,.1)),'q90':float(np.quantile(v,.9)),'invalid_fraction':float(1-mask.mean()),'strain_ratio_background_over_roi':None,'cnr':None,'contrast':None}
    if roi is not None:
        r=mask&roi;b=mask&~roi
        if r.any() and b.any():
            a,c=axial[r],axial[b];ma,mc=float(a.mean()),float(c.mean())
            out['strain_ratio_background_over_roi']=float(np.median(abs(c))/np.median(abs(a))) if np.median(abs(a))>1e-6 else None
            den=np.sqrt(a.var()+c.var());out['cnr']=float(abs(ma-mc)/den) if den>1e-10 else None
            out['contrast']=float(abs(ma-mc));out['roi_pixels']=int(roi.sum())
            boundary=roi&~binary_erosion(roi);g=np.hypot(*np.gradient(axial))
            out['boundary_gradient_pixel']=float(g[boundary&mask].mean()) if (boundary&mask).any() else None
            out['roi_area_mm2']=float(roi.sum()*np.prod(spacing)) if spacing is not None else None
    return out

def feature_vector(features):
    return np.array([features[k] for k in ['median_strain','mad_strain','q10','q90','invalid_fraction']],float)
