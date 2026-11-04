import numpy as np
from scipy.ndimage import map_coordinates

def warp(image,flow,order=1):
    """Sample moving image at reference grid + [ux,uz], both in pixels."""
    if flow.shape!=(2,*image.shape[-2:]):raise ValueError('Flow shape mismatch')
    if not np.isfinite(flow).all():raise ValueError('Nonfinite flow')
    z,x=np.indices(image.shape[-2:]);xx=x+flow[0];zz=z+flow[1]
    mask=(xx>=0)&(xx<=x.max())&(zz>=0)&(zz<=z.max())
    if image.ndim==3:out=np.stack([map_coordinates(c,[zz,xx],order=order,mode='nearest') for c in image])
    else:out=map_coordinates(image,[zz,xx],order=order,mode='nearest')
    return out,mask

def compose(a,b):
    """Compose phi_b(phi_a(x)); returns a(x)+b(x+a(x)), with support."""
    bw,mask=warp(b,a);return a+bw,mask

def torch_warp(image,flow):
    import torch
    import torch.nn.functional as F
    b,c,h,w=image.shape
    z,x=torch.meshgrid(torch.arange(h,device=image.device,dtype=flow.dtype),torch.arange(w,device=image.device,dtype=flow.dtype),indexing='ij')
    xx=x[None]+flow[:,0];zz=z[None]+flow[:,1]
    grid=torch.stack((2*(xx+0.5)/w-1,2*(zz+0.5)/h-1),-1)
    valid=((xx>=0)&(xx<=w-1)&(zz>=0)&(zz<=h-1))[:,None]
    return F.grid_sample(image,grid,align_corners=False,padding_mode='border'),valid
