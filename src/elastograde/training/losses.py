import torch
import torch.nn.functional as F
from elastograde.mechanics.warping import torch_warp

def masked_mean(x,mask):
    mask=mask.expand_as(x)
    if not mask.any():raise ValueError('Empty loss support')
    return x[mask].mean()

def local_ncc(a,b,mask,window=7):
    # Masked window sums; includes no padded samples in the local moments.
    m=mask.float().expand_as(a);pool=lambda x:F.avg_pool2d(x,window,1,window//2)
    support=pool(m).clamp_min(1e-7)
    ma=pool(a*m)/support;mb=pool(b*m)/support
    va=(pool(a*a*m)/support-ma*ma).clamp_min(0);vb=(pool(b*b*m)/support-mb*mb).clamp_min(0)
    cov=pool(a*b*m)/support-ma*mb
    corr=(cov/torch.sqrt(va*vb+1e-6)).clamp(-1,1)
    return 1-corr

def objective(a,b,forward,reverse,valid,weights,target=None,env_a=None,env_b=None):
    u=forward['flow'].float();v=reverse['flow'].float();a=a.float();b=b.float()
    bw,inside=torch_warp(b,u);source_valid,_=torch_warp(valid.float(),u)
    mask=valid.bool()&inside&(source_valid>.999)
    rfmap=local_ncc(a,bw,mask)
    if weights.get('confidence',False) and forward['log_variance'] is not None:
        # Normalized heteroscedastic Gaussian residual objective, not an error probability.
        lv=forward['log_variance'].float()
        rf=.5*masked_mean(torch.exp(-lv)*(a-bw).square()+lv,mask)
    else:rf=masked_mean(rfmap,mask)
    vv,bi=torch_warp(v,u);fb=masked_mean((u+vv).abs(),mask&bi)
    dz=u[:,:,1:]-u[:,:,:-1];dx=u[:,:,:,1:]-u[:,:,:,:-1]
    smooth=.75*dz.abs().mean()+.25*dx.abs().mean()
    uxz,uxx=torch.gradient(u[:,0],dim=(1,2));uzz,uzx=torch.gradient(u[:,1],dim=(1,2))
    jac=(1+uxx)*(1+uzz)-uxz*uzx
    folding=F.relu(.05-jac).mean()
    env=u.sum()*0
    if env_a is not None and weights.get('env',0):
        ew,_=torch_warp(env_b.float(),u);env=masked_mean(local_ncc(env_a.float(),ew,mask),mask)
    sup=u.sum()*0
    if weights.get('sup',0):
        if target is None:raise ValueError('Supervised mode needs verified target displacement')
        sup=masked_mean(F.smooth_l1_loss(u,target,reduction='none'),valid.bool())
    terms={'rf':rf,'env':env,'fb':fb,'smooth':smooth,'jacobian':folding,'sup':sup}
    total=sum(weights.get(k,0)*v for k,v in terms.items())
    if not torch.isfinite(total):raise FloatingPointError('Nonfinite training objective')
    return total,{k:float(v.detach()) for k,v in terms.items()}
