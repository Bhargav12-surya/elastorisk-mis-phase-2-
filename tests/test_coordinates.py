import numpy as np,pytest
from elastograde.data.synthetic import make_pair
from elastograde.mechanics.warping import warp,compose,torch_warp
from elastograde.mechanics.strain import strain
from elastograde.mechanics.coordinates import flip_pair,to_mm
from elastograde.baselines.ncc import estimate

def test_identity():
 s=make_pair(kind='zero');out,m=warp(s.post,np.zeros_like(s.displacement));np.testing.assert_allclose(out,s.pre);assert m.all()
def test_translation():
 s=make_pair(shape=(64,48));u,c,m=estimate(s.pre,s.post);assert abs(np.median(u[1,m])-1.1)<.15;assert abs(np.median(u[0,m])-.6)<.15
@pytest.mark.parametrize('method',['finite','least_squares'])
def test_affine_strain(method):
 s=make_pair(kind='compression');v=strain(s.displacement,s.metadata['spacing_mm'],method);np.testing.assert_allclose(v['axial'][8:-8,8:-8],-.025,atol=1e-6)
def test_flip_vectors():
 s=make_pair();_,_,u=flip_pair(s.pre,s.post,s.displacement,1);np.testing.assert_allclose(u[0],-.6)
def test_composition():
 a=np.zeros((2,10,10));a[0]=1;b=a.copy();c,m=compose(a,b);np.testing.assert_allclose(c[0,m],2)
def test_units_require_spacing():
 with pytest.raises(ValueError):to_mm(np.zeros((2,10,10)),None)
def test_torch_warp_matches_numpy():
 import torch
 s=make_pair();out,m=warp(s.post,s.displacement);t,tm=torch_warp(torch.tensor(s.post)[None,None],torch.tensor(s.displacement)[None]);np.testing.assert_allclose(t.numpy()[0,0][m],out[m],atol=1e-5)
def test_tiling_constant_predictor():
 from elastograde.inference.tiling import tiled_predict
 s=make_pair(shape=(100,80));u=tiled_predict(s.pre,s.post,lambda a,b:np.ones((2,*a.shape)),(48,32),12);np.testing.assert_allclose(u,1)
