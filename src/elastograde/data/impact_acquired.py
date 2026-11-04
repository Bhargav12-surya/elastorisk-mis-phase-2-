from pathlib import Path
import numpy as np
from scipy.io import loadmat
from .schemas import Sample,save_sample
from elastograde.reporting.provenance import sha256,digest

def read_array(path,key):
    p=Path(path)
    if p.suffix=='.npz':
        with np.load(p,allow_pickle=False) as f:return np.array(f[key])
    import h5py
    if h5py.is_hdf5(p):
        with h5py.File(p,'r') as f:return np.asarray(f[key])
    obj=loadmat(p,struct_as_record=False,squeeze_me=False)
    # Explicit dotted struct traversal; no hidden squeeze of imaging dimensions.
    parts=key.split('.')
    value=obj[parts[0]]
    for part in parts[1:]:
        if isinstance(value,np.ndarray) and value.size==1:value=value.item()
        value=getattr(value,part)
    return np.asarray(value)

def prepare(config):
    required=['root','output','mapping','records','axes','dataset_id']
    for k in required:
        if k not in config:raise ValueError(f'Missing explicit adapter field: {k}')
    if config['axes'] not in ('zx','xz'):raise ValueError('Verify axes zx or xz explicitly')
    if not config['records']:raise ValueError('No records: inventory archives and fill explicit records first')
    root=Path(config['root']);out=Path(config['output']);manifest=[]
    for record in config['records']:
        source=root/record['path'];arrays={}
        for role,key in config['mapping'].items():
            if key is None:continue
            a=read_array(source,key)
            if a.ndim!=2:raise ValueError(f'{key}: use explicit 2D pair files; discovered shape {a.shape}')
            arrays[role]=a.T if config['axes']=='xz' else a
        if not {'pre','post'}<=arrays.keys():raise ValueError('Mapping needs pre and post')
        meta=dict(record);meta.update(dataset_id=config['dataset_id'],source_sha256=sha256(source),source_relative_path=record['path'],axes='zx',data_origin='acquired_phantom',spacing_mm=config.get('spacing_mm'),adapter_config_hash=digest(config))
        disp=None
        if 'ux' in arrays and 'uz' in arrays:
            if config.get('displacement_convention')!='reference_to_moving' or config.get('displacement_units')!='pixels':
                raise ValueError('Convert reference convention and units explicitly before importing displacement')
            disp=np.stack([arrays['ux'],arrays['uz']])
        s=Sample(arrays['pre'],arrays['post'],meta,disp,arrays.get('valid'),arrays.get('roi'))
        dest=save_sample(out/(record['sample_id']+'.npz'),s)
        manifest.append(dict(meta,path=str(dest.resolve())))
    return manifest
