import numpy as np
from scipy.ndimage import uniform_filter
from elastograde.mechanics.warping import warp

def estimate(pre,post,radius=(3,2),window=(9,7)):
    """Dense windowed signed NCC search; parabolic subpixel refinement."""
    scores=[];shifts=[];h,w=pre.shape
    ma=uniform_filter(pre,size=window);va=np.maximum(uniform_filter(pre*pre,size=window)-ma*ma,0)
    for dz in range(-radius[0],radius[0]+1):
        for dx in range(-radius[1],radius[1]+1):
            f=np.zeros((2,h,w));f[0]=dx;f[1]=dz
            p,m=warp(post,f);mb=uniform_filter(p,size=window)
            vb=np.maximum(uniform_filter(p*p,size=window)-mb*mb,0)
            cov=uniform_filter(pre*p,size=window)-ma*mb
            score=cov/np.sqrt(va*vb+1e-10);score[~m]=-1
            scores.append(score);shifts.append((dx,dz))
    scores=np.stack(scores);best=scores.argmax(0);grid=np.indices((h,w));u=np.asarray(shifts)[best].transpose(2,0,1).astype(float)
    width=2*radius[1]+1
    for component,step,bound in [(0,1,radius[1]),(1,width,radius[0])]:
        interior=np.abs(u[component])<bound
        lo=np.maximum(best-step,0);hi=np.minimum(best+step,len(scores)-1)
        a=scores[lo,*grid];b=scores[best,*grid];c=scores[hi,*grid]
        den=a-2*b+c
        delta=np.divide(.5*(a-c),den,out=np.zeros_like(a),where=abs(den)>1e-8)
        u[component]+=np.where(interior,np.clip(delta,-.5,.5),0)
    conf=scores[best,*grid];valid=np.ones((h,w),bool)
    bz=window[0]//2+radius[0];bx=window[1]//2+radius[1]
    valid[:bz]=False;valid[-bz:]=False;valid[:,:bx]=False;valid[:,-bx:]=False
    return u.astype('float32'),conf,valid
