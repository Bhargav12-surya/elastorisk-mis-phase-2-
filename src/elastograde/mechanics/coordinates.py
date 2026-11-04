import numpy as np

def require_spacing(spacing):
    if spacing is None or len(spacing)!=2 or not np.all(np.isfinite(spacing)) or min(spacing)<=0:
        raise ValueError('Physical operation needs verified positive [dz,dx] spacing')
    return tuple(spacing)

def to_mm(flow,spacing):
    dz,dx=require_spacing(spacing)
    return flow*np.array([dx,dz])[:,None,None]

def axial_spacing(sampling_hz,sound_speed_m_s=1540,decimation=1):
    if sampling_hz<=0 or decimation<1:raise ValueError('Invalid acquisition sampling metadata')
    return 1000*sound_speed_m_s*decimation/(2*sampling_hz)

def flip_pair(pre,post,flow,axis):
    if axis not in (0,1):raise ValueError('axis must be axial 0 or lateral 1')
    u=np.flip(flow,axis=axis+1).copy();u[1 if axis==0 else 0]*=-1
    return np.flip(pre,axis).copy(),np.flip(post,axis).copy(),u
