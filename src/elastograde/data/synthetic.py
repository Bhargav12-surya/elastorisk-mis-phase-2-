import numpy as np
from scipy.ndimage import gaussian_filter,map_coordinates
from .schemas import Sample

def make_pair(seed=42,shape=(64,48),kind='translation',shift=(0.6,1.1),compression=0.025):
    """Toy band-limited RF-like fixture; no acoustic/FEM validity claimed."""
    rng=np.random.default_rng(seed);z,x=np.indices(shape,dtype=float)
    pre=gaussian_filter(rng.normal(size=shape),(0.65,1.1))
    pre=pre/np.std(pre)
    u=np.zeros((2,*shape));valid=np.ones(shape,bool)
    if kind=='zero':post=pre.copy()
    elif kind=='translation':
        u[0]=shift[0];u[1]=shift[1]
        post=map_coordinates(pre,[z-shift[1],x-shift[0]],order=3,mode='reflect')
    elif kind in ('compression','heterogeneous','dropout'):
        u[1]=-compression*(z-shape[0]/2)
        if kind=='heterogeneous':u[1]+=0.25*np.sin(2*np.pi*x/shape[1])*np.sin(np.pi*z/shape[0])
        # Invert smooth reference -> moving coordinates by fixed point.
        invz=z.copy();invx=x.copy()
        for _ in range(30):
            invz=z-map_coordinates(u[1],[invz,invx],order=1,mode='nearest')
        post=map_coordinates(pre,[invz,invx],order=3,mode='reflect')
        if kind=='dropout':post[shape[0]//3:shape[0]//2]=0;valid[shape[0]//3:shape[0]//2]=False
    else:raise ValueError('Unknown fixture kind')
    valid[:4]=False;valid[-4:]=False;valid[:,:4]=False;valid[:,-4:]=False
    roi=((z-shape[0]/2)**2/(shape[0]/5)**2+(x-shape[1]/2)**2/(shape[1]/5)**2)<1
    meta={'sample_id':f'fixture-{seed}-{kind}','dataset_id':'toy-fixtures','data_origin':'synthetic_fixture','domain':'toy','geometry_id':f'geometry-{seed}','frame_ids':[f'{seed}-pre',f'{seed}-post'],'spacing_mm':[0.04,0.2],'axes':'zx','reference_convention':'reference_to_moving','roi_provenance':'analytic illustrative region; not a physical inclusion'}
    return Sample(pre.astype('float32'),post.astype('float32'),meta,u.astype('float32'),valid,roi).validate()
