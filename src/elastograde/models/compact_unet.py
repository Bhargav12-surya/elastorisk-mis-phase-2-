import torch
from torch import nn
import torch.nn.functional as F

def block(a,b):return nn.Sequential(nn.Conv2d(a,b,(5,3),padding=(2,1)),nn.GroupNorm(1,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.SiLU())

class CompactUNet(nn.Module):
    def __init__(self,channels=1,width=12,**kwargs):
        super().__init__();self.enc=block(2*channels+1,width);self.low=block(width,width*2);self.dec=block(width*3,width);self.out=nn.Conv2d(width,2,3,padding=1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,a,b,valid=None,metadata=None):
        if valid is None:valid=torch.ones_like(a[:,:1])
        e=self.enc(torch.cat([a,b,valid],1));low=self.low(F.avg_pool2d(e,2));d=self.dec(torch.cat([e,F.interpolate(low,size=e.shape[-2:],mode='bilinear',align_corners=False)],1))
        return {'flow':self.out(d),'log_variance':None,'embedding':low.mean((2,3))}
