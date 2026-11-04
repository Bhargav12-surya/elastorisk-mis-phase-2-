import numpy as np
from scipy.signal import savgol_filter
from .coordinates import to_mm,require_spacing

def strain(flow,spacing=None,method='finite',window=7):
    """Normal strains are dimensionless. Physical shear needs spacing."""
    u=np.asarray(flow,float)
    if spacing is None:
        dz=dx=1.0
    else:
        dz,dx=require_spacing(spacing);u=to_mm(u,spacing)
    def derivative(a,axis,step):
        if method=='finite':return np.gradient(a,step,axis=axis,edge_order=2)
        if method=='least_squares':
            if window%2!=1 or window>min(a.shape) or window<3:raise ValueError('Odd derivative window >=3 required')
            return savgol_filter(a,window,1,deriv=1,delta=step,axis=axis,mode='interp')
        raise ValueError('Unknown derivative estimator')
    ux_x=derivative(u[0],1,dx);uz_z=derivative(u[1],0,dz)
    shear=0.5*(derivative(u[0],0,dz)+derivative(u[1],1,dx))
    return {'axial':uz_z,'lateral':ux_x,'shear':shear if spacing is not None else None,'pixel_shear':shear if spacing is None else None}

def jacobian(flow):
    ux_z,ux_x=np.gradient(flow[0]);uz_z,uz_x=np.gradient(flow[1])
    return (1+ux_x)*(1+uz_z)-ux_z*uz_x
