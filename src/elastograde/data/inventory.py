from pathlib import Path
import numpy as np
from scipy.io import whosmat
from elastograde.reporting.provenance import sha256

def inspect_file(path):
    p=Path(path);out={'name':p.name,'bytes':p.stat().st_size,'sha256':sha256(p),'arrays':[]}
    try:
        if p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as d:
                out['arrays']=[{'key':k,'shape':list(d[k].shape),'dtype':str(d[k].dtype)} for k in d.files]
        elif p.suffix in ('.mat','.h5','.hdf5'):
            import h5py
            if h5py.is_hdf5(p):
                with h5py.File(p,'r') as f:
                    def visit(k,v):
                        if isinstance(v,h5py.Dataset):out['arrays'].append({'key':k,'shape':list(v.shape),'dtype':str(v.dtype)})
                    f.visititems(visit)
            else:out['arrays']=[{'key':k,'shape':list(s),'dtype':t} for k,s,t in whosmat(p)]
        else:out['format_status']='metadata_only'
    except Exception as e:out['error']=str(e)
    return out

def inventory(root):
    root=Path(root)
    if not root.is_dir():raise FileNotFoundError(root)
    return {'root':str(root),'files':[dict(inspect_file(p),relative_path=str(p.relative_to(root))) for p in sorted(root.rglob('*')) if p.is_file()], 'axes':'not_verified','pathology_grades':'not_verified'}
