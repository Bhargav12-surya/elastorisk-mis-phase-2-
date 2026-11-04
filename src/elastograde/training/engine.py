from pathlib import Path
import json,random,time
import numpy as np,torch
from elastograde.config import save_json
from elastograde.data.synthetic import make_pair
from elastograde.data.schemas import load_sample,save_sample
from elastograde.data.splits import grouped_split
from elastograde.data.audit import audit
from elastograde.data.preprocessing import normalize_pair,envelope
from elastograde.models.compact_unet import CompactUNet
from elastograde.models.pyramid_flow import PyramidFlow
from elastograde.training.losses import objective
from elastograde.training import checkpoints
from elastograde.reporting.provenance import environment,code_hash,digest

def build_model(config):
    m=config.get('model',{});channels=2 if m.get('inputs','rf')=='fusion' else 1
    cls=CompactUNet if m.get('name')=='unet' else PyramidFlow
    return cls(channels=channels,width=m.get('width',12),**({k:m[k] for k in ['radius','attention','confidence'] if k in m} if cls==PyramidFlow else {}))

def tensors(sample,inputs,device):
    a,b=normalize_pair(sample.pre,sample.post);ea,eb=normalize_pair(envelope(sample.pre),envelope(sample.post))
    if inputs=='rf':aa,bb=a[None],b[None]
    elif inputs=='envelope':aa,bb=ea[None],eb[None]
    elif inputs=='fusion':aa,bb=np.stack([a,ea]),np.stack([b,eb])
    else:raise ValueError('inputs must be rf/envelope/fusion')
    def t(x):return torch.as_tensor(x,dtype=torch.float32,device=device)[None]
    valid=sample.valid if sample.valid is not None else np.ones_like(a,bool)
    meta=sample.metadata;freq=meta.get('center_frequency_hz');rate=meta.get('sampling_frequency_hz')
    metadata=torch.tensor([[0 if freq is None else freq/1e7,0 if rate is None else rate/4e7,float(freq is not None),float(rate is not None)]],device=device)
    return t(aa),t(bb),t(valid[None]),t(sample.displacement) if sample.displacement is not None else None,t(ea[None]),t(eb[None]),metadata

def get_records(config,out):
    if config.get('data_mode')=='demo':
        records=[]
        for i in range(config.get('samples',12)):
            s=make_pair(config.get('seed',42)+i,shape=tuple(config.get('shape',[48,32])),kind='compression',compression=.01+.003*(i%5))
            p=save_sample(out/'fixtures'/f'{i:03}.npz',s);records.append(dict(s.metadata,path=str(p.resolve())))
        return grouped_split(records,'geometry_id',config.get('seed',42))
    path=config.get('manifest')
    if not path:raise ValueError('Real training requires a verified split manifest')
    records=json.loads(Path(path).read_text());audit(records,config.get('group_key','geometry_id'));return records


def crop_sample(sample,shape,random_crop=False):
    from copy import deepcopy
    if shape is None:return sample
    h,w=sample.pre.shape;ph,pw=min(shape[0],h),min(shape[1],w)
    z=int(np.random.randint(h-ph+1)) if random_crop else (h-ph)//2
    x=int(np.random.randint(w-pw+1)) if random_crop else (w-pw)//2
    s=deepcopy(sample)
    for key in ['pre','post','displacement','valid','roi']:
        a=getattr(s,key)
        if a is not None:setattr(s,key,a[...,z:z+ph,x:x+pw])
    s.metadata['patch_origin_zx']=[z,x]
    return s.validate()

def train(config,resume=None,callback=None,stop_event=None):
    out=Path(config.get('output','reports/training'));out.mkdir(parents=True,exist_ok=True)
    seed=config.get('seed',42);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(config.get('threads',2))
    device=config.get('device','cpu')
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA not available; choose CPU config')
    model=build_model(config).to(device);opt=torch.optim.AdamW(model.parameters(),lr=config.get('lr',.001))
    records=get_records(config,out);save_json(out/'manifest.json',records);save_json(out/'config.json',config)
    splits={role:[r for r in records if r['split']==role] for role in ['train','validation','calibration','test']}
    if not splits['train'] or not splits['validation']:raise ValueError('Missing train/validation groups')
    regime=config.get('regime','self_supervised');weights=dict(config.get('loss',{'rf':1,'env':.2,'fb':.1,'smooth':.02,'jacobian':.01,'sup':0}))
    if regime=='self_supervised' and weights.get('sup',0):raise ValueError('Self-supervised arm cannot use target labels')
    if regime not in ['self_supervised','simulation_supervised','hybrid']:raise ValueError('Unknown regime')
    if regime=='simulation_supervised' and not weights.get('sup',0):raise ValueError('Supervised arm must enable sup loss')
    if regime=='hybrid' and not config.get('pretrained') and resume is None:raise ValueError('Hybrid adaptation requires simulation-pretrained checkpoint')
    start=0;best=float('inf')
    if config.get('pretrained') and not resume:checkpoints.load(config['pretrained'],model)
    if resume:
        ck=checkpoints.load(resume,model,opt,True)
        for key in ['model','regime','loss','seed']:
            if ck['config'].get(key)!=config.get(key):raise ValueError('Resume configuration mismatch: '+key)
        start=ck['epoch']+1;best=ck['best']
    train_inputs=config.get('model',{}).get('inputs','rf');history=[];t0=time.perf_counter();status='running'
    save_json(out/'status.json',{'status':status})
    amp=bool(config.get('amp',False) and device.startswith('cuda'));scaler=torch.amp.GradScaler('cuda',enabled=amp)
    accumulation=max(1,config.get('accumulation',1))
    try:
        for epoch in range(start,config.get('epochs',2)):
            if stop_event is not None and stop_event.is_set():status='cancelled';break
            model.train();opt.zero_grad();tot=0;terms={}
            for index,r in enumerate(splits['train']):
                s=crop_sample(load_sample(r['path']),config.get('shape'),True);a,b,m,target,ea,eb,meta=tensors(s,train_inputs,device)
                if weights.get('sup',0) and s.metadata.get('data_origin') not in ['simulated_phantom','synthetic_fixture']:raise ValueError('Verified simulation target required for sup loss')
                with torch.autocast(device_type='cuda' if device.startswith('cuda') else 'cpu',enabled=amp):
                    f=model(a,b,m,meta);v=model(b,a,m,meta)
                loss,terms=objective(a[:,:1],b[:,:1],f,v,m,weights,target,ea,eb)
                group_start=(index//accumulation)*accumulation
                group_size=min(accumulation,len(splits['train'])-group_start)
                scaler.scale(loss/group_size).backward();tot+=float(loss.detach())
                if (index+1)%accumulation==0 or index==len(splits['train'])-1:
                    scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(opt);scaler.update();opt.zero_grad()
            model.eval();val=[]
            with torch.no_grad():
                for r in splits['validation']:
                    a,b,m,target,ea,eb,meta=tensors(crop_sample(load_sample(r['path']),config.get('shape')),train_inputs,device)
                    l,_=objective(a[:,:1],b[:,:1],model(a,b,m,meta),model(b,a,m,meta),m,weights,target,ea,eb);val.append(float(l))
            row={'epoch':epoch,'train_loss':tot/len(splits['train']),'validation_loss':float(np.mean(val)),**terms,'seconds':time.perf_counter()-t0}
            history.append(row)
            with (out/'history.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
            improved=row['validation_loss']<best
            if improved:best=row['validation_loss']
            extra={'manifest_hash':digest(records),'data_origin':'synthetic_fixture' if config.get('data_mode')=='demo' else 'verified_manifest','regime':regime,'code_hash':code_hash()}
            checkpoints.save(out/'last.pt',model,opt,epoch,config,best,extra)
            if improved:checkpoints.save(out/'best.pt',model,opt,epoch,config,best,extra)
            if callback:callback(row)
        if status=='running':status='completed'
    except KeyboardInterrupt:status='cancelled'
    except Exception:
        save_json(out/'status.json',{'status':'failed'});raise
    summary={'status':status,'run':str(out),'epochs_executed':len(history),'data_origin':'synthetic_fixture' if config.get('data_mode')=='demo' else 'verified_manifest','regime':regime,'best_validation_loss':best if np.isfinite(best) else None,'parameter_count':sum(p.numel() for p in model.parameters()),'seconds':time.perf_counter()-t0,'device':device,'environment':environment(),'code_hash':code_hash(),'manifest_hash':digest(records),'clinical_validation':False,'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device.startswith('cuda') else None}
    save_json(out/'status.json',summary);return summary
