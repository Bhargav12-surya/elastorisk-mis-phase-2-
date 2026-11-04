from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np, json
from elastograde.reporting.provenance import sha256

@dataclass
class Sample:
    pre: np.ndarray
    post: np.ndarray
    metadata: dict = field(default_factory=dict)
    displacement: np.ndarray | None = None
    valid: np.ndarray | None = None
    roi: np.ndarray | None = None
    def validate(self):
        if self.pre.ndim!=2 or self.pre.shape!=self.post.shape or min(self.pre.shape)<8:
            raise ValueError('Expected matching 2D [z,x] arrays, each dimension >=8')
        for a in [self.pre,self.post]:
            if not np.isfinite(a).all() or np.std(a)<1e-10:raise ValueError('Nonfinite or constant RF input')
        if self.displacement is not None and (self.displacement.shape!=(2,*self.pre.shape) or not np.isfinite(self.displacement).all()):
            raise ValueError('Displacement must be finite [ux,uz] in pixels with shape [2,z,x]')
        for a in [self.valid,self.roi]:
            if a is not None and a.shape!=self.pre.shape:raise ValueError('Mask shape mismatch')
        if self.valid is not None and not self.valid.any():raise ValueError('No valid support')
        spacing=self.metadata.get('spacing_mm')
        if spacing is not None and (len(spacing)!=2 or not all(np.isfinite(x) and x>0 for x in spacing)):
            raise ValueError('spacing_mm is positive [dz,dx]')
        if self.metadata.get('axes','zx')!='zx':raise ValueError('Canonical axes must be zx')
        return self

METADATA_FIELDS=['sample_id','dataset_id','domain','acquisition_id','sequence_id','frame_ids','phantom_id','region_id','geometry_id','scatterer_id','patient_id','lesion_id','study_id','site_id','scanner_id','center_frequency_hz','sampling_frequency_hz','spacing_mm','timestamps','compression','loading','pathology_grade','grading_system','label_provenance']

def save_sample(path,sample):
    sample.validate();p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    arrays={'pre':sample.pre.astype('float32'),'post':sample.post.astype('float32')}
    for key in ['displacement','valid','roi']:
        val=getattr(sample,key)
        if val is not None:arrays[key]=val
    np.savez_compressed(p,**arrays)
    meta={k:None for k in METADATA_FIELDS};meta.update(sample.metadata)
    meta.update(schema_version=1,axes='zx',displacement_units='pixels',displacement_channels=['ux','uz'],canonical_sha256=sha256(p))
    p.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    return p

def load_sample(path):
    p=Path(path);meta=json.loads(p.with_suffix('.json').read_text())
    if meta.get('canonical_sha256')!=sha256(p):raise ValueError('Canonical data checksum mismatch')
    with np.load(p,allow_pickle=False) as d:
        s=Sample(d['pre'],d['post'],meta,**{k:d[k] if k in d else None for k in ['displacement','valid','roi']})
    return s.validate()
