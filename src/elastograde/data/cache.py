"""Optional chunked HDF5 cache with content/config/version identity."""
from pathlib import Path
import json
from elastograde.reporting.provenance import digest

def cache_key(source_checksum,config,code_version):
    return digest({'source':source_checksum,'config':config,'code':code_version})

def write_cache(path,sample,source_checksum,config,code_version):
    import h5py
    sample.validate();p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    key=cache_key(source_checksum,config,code_version)
    with h5py.File(p,'w') as f:
        f.attrs['cache_key']=key;f.attrs['metadata']=json.dumps(sample.metadata)
        for name in ['pre','post','displacement','valid','roi']:
            a=getattr(sample,name)
            if a is not None:f.create_dataset(name,data=a,chunks=True,compression='gzip')
    return key

def read_patch(path,key,z,x,height,width):
    import h5py
    with h5py.File(path,'r') as f:
        if f.attrs['cache_key']!=key:raise ValueError('Stale cache key')
        return {name:d[...,z:z+height,x:x+width] for name,d in f.items()}
