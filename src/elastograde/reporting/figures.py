from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def panel(sample,result,path):
    from elastograde.data.preprocessing import bmode
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    items=[('B-mode (dB)',bmode(sample.pre),'gray'),('Axial displacement (pixels)',result['flow'][1],'coolwarm'),('Axial strain',result['axial_strain'],'coolwarm'),('Residual RF amplitude',result['residual'],'magma'),('Cycle mismatch (pixels)',result['cycle'],'viridis'),('Valid support',result['valid'],'gray')]
    for ax,(title,array,cmap) in zip(axes.flat,items):
        arr=array if title.startswith('B-mode') or title=='Valid support' else np.ma.masked_where(~result['valid'],array)
        im=ax.imshow(arr,cmap=cmap,aspect='auto');ax.set_title(title);ax.set_xlabel('Lateral sample');ax.set_ylabel('Axial sample');fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle(f"ElastoGrade Research | {result['data_origin']} | {result['method']}")
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);fig.savefig(p,dpi=140);plt.close(fig)
    return p
