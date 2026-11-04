from .impact_acquired import prepare as _prepare
from .schemas import load_sample,save_sample

def prepare(config):
    records=_prepare(config)
    for r in records:
        r['data_origin']='simulated_phantom'
        s=load_sample(r['path']);s.metadata['data_origin']='simulated_phantom';save_sample(r['path'],s)
    return records
