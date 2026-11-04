from pathlib import Path
import torch, random, numpy as np

def save(path,model,optimizer,epoch,config,best,extra=None):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'config':config,'best':best,'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'extra':extra or {}},tmp);tmp.replace(p)

def load(path,model,optimizer=None,restore_rng=False):
    # Own local checkpoints only: optimizer/RNG tuples require non-weights loading.
    ck=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
    if optimizer:optimizer.load_state_dict(ck['optimizer'])
    if restore_rng:
        torch.set_rng_state(ck['torch_rng']);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
    return ck
