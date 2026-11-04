import numpy as np
from scipy.signal import hilbert,butter,sosfiltfilt,resample_poly

def envelope(rf):return np.abs(hilbert(rf,axis=0))
def bmode(rf,dynamic_range=60):
    e=envelope(rf);return np.clip(20*np.log10(np.maximum(e,1e-12)/max(e.max(),1e-12)),-dynamic_range,0)
def normalize_pair(a,b):
    scale=np.sqrt((np.mean(a*a)+np.mean(b*b))/2)
    if not np.isfinite(scale) or scale<1e-10:raise ValueError('Invalid pair normalization')
    mean=(a.mean()+b.mean())/2
    return ((a-mean)/scale).astype('float32'),((b-mean)/scale).astype('float32')
def bandpass(rf,sampling_hz,low_hz,high_hz):
    if not 0<low_hz<high_hz<sampling_hz/2:raise ValueError('Bandpass violates Nyquist bounds')
    return sosfiltfilt(butter(4,[low_hz,high_hz],btype='bandpass',fs=sampling_hz,output='sos'),rf,axis=0)
def resample_axial(rf,flow,spacing,up,down):
    from elastograde.mechanics.coordinates import require_spacing
    dz,dx=require_spacing(spacing)
    out=resample_poly(rf,up,down,axis=0)
    u=resample_poly(flow,up,down,axis=1);u[1]*=up/down
    return out,u,(dz*down/up,dx)
def patches(sample,shape=(64,48),stride=(32,24)):
    h,w=sample.pre.shape
    if shape[0]>h or shape[1]>w:raise ValueError('Patch exceeds source')
    for z in range(0,h-shape[0]+1,stride[0]):
        for x in range(0,w-shape[1]+1,stride[1]):
            yield z,x,sample.pre[z:z+shape[0],x:x+shape[1]],sample.post[z:z+shape[0],x:x+shape[1]]
def signal_quality(rf):
    return {'finite_fraction':float(np.isfinite(rf).mean()),'zero_fraction':float((rf==0).mean()),'rms':float(np.sqrt(np.mean(rf**2))),'max_abs':float(np.abs(rf).max())}
