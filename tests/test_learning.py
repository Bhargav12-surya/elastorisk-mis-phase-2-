import numpy as np,torch,pytest
from elastograde.models.pyramid_flow import PyramidFlow
from elastograde.models.compact_unet import CompactUNet
from elastograde.training.losses import masked_mean,objective
from elastograde.models.ordinal import OrdinalHead,OrdinalLogistic
from elastograde.training.engine import train
from elastograde.config import load
@pytest.mark.parametrize('cls',[PyramidFlow,CompactUNet])
def test_model_gradient(cls):
 torch.set_num_threads(2);m=cls(width=8);a=torch.randn(1,1,24,16);o=m(a,a);assert o['flow'].shape==(1,2,24,16);o['flow'].sum().backward();assert any(p.grad is not None for p in m.parameters())
def test_masked_loss_empty():
 with pytest.raises(ValueError):masked_mean(torch.ones(1,1,3,3),torch.zeros(1,1,3,3,dtype=torch.bool))
def test_warp_gradcheck():
 from elastograde.mechanics.warping import torch_warp
 a=torch.randn(1,1,5,5,dtype=torch.double);u=torch.full((1,2,5,5),.2,dtype=torch.double,requires_grad=True)
 assert torch.autograd.gradcheck(lambda v:torch_warp(a,v)[0],(u,),eps=1e-6,atol=1e-4)
def test_ordinal_monotonic():
 m=OrdinalHead(3,4);p=m.probabilities(torch.randn(5,3));assert torch.all(p>=0);torch.testing.assert_close(p.sum(1),torch.ones(5))
def test_ordinal_fit():
 rng=np.random.default_rng(1);X=rng.normal(size=(90,2));y=np.digitize(X[:,0],[-.4,.4]);m=OrdinalLogistic().fit(X,y,['train']*90);p=m.predict_proba(X);assert (p.argmax(1)==y).mean()>.8
def test_checkpoint_resume(tmp_path):
 c=load('configs/demo.yaml');c.update(output=str(tmp_path),samples=8,epochs=1);train(c);c['epochs']=2;r=train(c,tmp_path/'last.pt');assert r['epochs_executed']==1
