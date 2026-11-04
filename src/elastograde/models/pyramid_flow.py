import torch
from torch import nn
import torch.nn.functional as F
from .compact_unet import block
from elastograde.mechanics.warping import torch_warp

def resize_flow(u,size):
    h,w=u.shape[-2:];v=F.interpolate(u,size=size,mode='bilinear',align_corners=False)
    return v*torch.tensor([size[1]/w,size[0]/h],device=u.device,dtype=u.dtype)[None,:,None,None]

def correlation(a,b,radius=2):
    a=F.normalize(a,dim=1);b=F.normalize(b,dim=1);h,w=a.shape[-2:]
    padded=F.pad(b,(radius,radius,radius,radius));vol=[]
    for z in range(2*radius+1):
        for x in range(2*radius+1):vol.append((a*padded[:,:,z:z+h,x:x+w]).mean(1,keepdim=True))
    return torch.cat(vol,1)

class PyramidFlow(nn.Module):
    def __init__(self,channels=1,width=12,radius=2,attention=False,confidence=False):
        super().__init__();self.radius=radius;self.confidence=confidence
        self.enc1=block(channels,width);self.enc2=block(width,width);self.enc3=block(width,width)
        self.refiners=nn.ModuleList([nn.Sequential(block(width+(2*radius+1)**2+3+4,width),nn.Conv2d(width,2,3,padding=1)) for _ in range(3)])
        for r in self.refiners:nn.init.zeros_(r[-1].weight);nn.init.zeros_(r[-1].bias)
        self.attention=nn.MultiheadAttention(width,2,batch_first=True) if attention else None
        self.var=nn.Conv2d(width,1,3,padding=1) if confidence else None
    def encode(self,a):
        e1=self.enc1(a);e2=self.enc2(F.avg_pool2d(e1,2));e3=self.enc3(F.avg_pool2d(e2,2));return [e1,e2,e3]
    def forward(self,a,b,valid=None,metadata=None):
        ea,eb=self.encode(a),self.encode(b)
        if self.attention:
            aa=ea[-1].flatten(2).transpose(1,2);bb=eb[-1].flatten(2).transpose(1,2)
            if aa.shape[1]>4096:raise ValueError('Attention token budget exceeded; use patches')
            q,_=self.attention(aa,bb,bb,need_weights=False);ea[-1]=ea[-1]+q.transpose(1,2).reshape_as(ea[-1])
        u=a.new_zeros((a.shape[0],2,*ea[-1].shape[-2:]))
        if valid is None:valid=torch.ones_like(a[:,:1])
        if metadata is None:metadata=a.new_zeros((a.shape[0],4))
        for index in (2,1,0):
            size=ea[index].shape[-2:];u=resize_flow(u,size)
            bw,_=torch_warp(eb[index],u);cv=correlation(ea[index],bw,self.radius)
            m=F.interpolate(valid.float(),size=size,mode='nearest');meta=metadata[:,:,None,None].expand(-1,-1,*size)
            u=u+self.refiners[index](torch.cat([ea[index],cv,u,m,meta],1))
        return {'flow':u,'log_variance':self.var(ea[0]).clamp(-6,3) if self.var else None,'embedding':ea[-1].mean((2,3))}
