import time
import numpy as np
from elastograde.baselines.ncc import estimate
from elastograde.baselines.regularized import refine
from elastograde.mechanics.strain import strain,jacobian
from elastograde.mechanics.features import mechanical_features
from elastograde.uncertainty.quality import quality
from elastograde.uncertainty.selective import threshold_action
from elastograde.pgm.bayesian_quality import illustrative_network
from elastograde.pgm.decisions import choose
from elastograde.reporting.provenance import code_hash,sha256

def infer(sample,method='ncc',checkpoint=None,device='cpu'):
    sample.validate();a,b=sample.pre,sample.post;t0=time.perf_counter();ckhash=None
    if checkpoint:
        import torch
        if device=='cpu':torch.set_num_threads(2)
        from elastograde.training.engine import build_model,tensors
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=ck['config'];model=build_model(cfg).to(device);model.load_state_dict(ck['model']);model.eval()
        aa,bb,m,_,_,_,meta=tensors(sample,cfg.get('model',{}).get('inputs','rf'),device)
        with torch.no_grad():u=model(aa,bb,m,meta)['flow'][0].cpu().numpy();rev=model(bb,aa,m,meta)['flow'][0].cpu().numpy()
        mask=np.ones(a.shape,bool);method=cfg.get('model',{}).get('name','pyramid');ckhash=sha256(checkpoint)
    elif method=='zero':u=np.zeros((2,*a.shape));rev=-u;mask=np.ones(a.shape,bool)
    elif method in ['ncc','regularized']:
        u,c,mask=estimate(a,b);rev,rc,rm=estimate(b,a);mask&=rm
        if method=='regularized':u=refine(u,c);rev=refine(rev,rc)
    else:raise ValueError('Neural inference requires checkpoint')
    if sample.valid is not None:mask&=sample.valid
    q=quality(a,b,u,rev,mask);mask=q['valid'];spacing=sample.metadata.get('spacing_mm');s=strain(u,spacing,'least_squares')
    features=mechanical_features(s['axial'],mask,sample.roi,spacing)
    decision=threshold_action(q['residual_mean'],q['cycle_mean'],features['invalid_fraction'])
    evidence={'residual':int(q['residual_mean']>.5),'cycle':int(q['cycle_mean']>1),'uncertainty':int(features['invalid_fraction']>.4)}
    bn=illustrative_network();p=float(bn.query('reliable',evidence)[1])
    output={'flow':u,'axial_strain':s['axial'],'lateral_strain':s['lateral'],'valid':mask,'residual':q['residual'],'cycle':q['cycle'],'jacobian':jacobian(u),'sample_id':sample.metadata.get('sample_id'),'method':method,'data_origin':sample.metadata.get('data_origin','unknown'),'displacement_units':'pixels','spacing_mm':spacing,'strain_units':'dimensionless','features':features,'quality':{'residual_mean':q['residual_mean'],'cycle_mean':q['cycle_mean'],'status':q['confidence_status']},'decision':decision,'educational_bn':{'p_reliable':p,'evidence':evidence,'cpd_provenance':'assumed, not fitted','decision':choose(p)},'grade_status':'not_available_no_validated_clinical_mapping','grade_probabilities':None,'absolute_modulus_kpa':None,'seconds':time.perf_counter()-t0,'code_hash':code_hash(),'checkpoint_sha256':ckhash,'source_sha256':sample.metadata.get('canonical_sha256'),'roi_provenance':sample.metadata.get('roi_provenance','not_verified')}
    if s['shear'] is not None:
        from elastograde.mechanics.coordinates import to_mm
        output['shear_strain']=s['shear'];output['flow_mm']=to_mm(u,spacing)
    return output
