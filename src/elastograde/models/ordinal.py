import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

class OrdinalLogistic:
    """Proportional odds with ordered cutpoints optimized by likelihood."""
    def fit(self,X,y,roles,classes=None):
        if set(roles)!={'train'}:raise ValueError('Fit on training data only')
        X=np.asarray(X,float);y=np.asarray(y,int);self.classes=int(classes or y.max()+1)
        if self.classes<2 or set(y)!=set(range(self.classes)):raise ValueError('Every ordered class must occur in training')
        self.mean=X.mean(0);self.scale=np.maximum(X.std(0),1e-6);Z=(X-self.mean)/self.scale;d=Z.shape[1]
        def predict(theta):
            cuts=np.r_[theta[d],theta[d]+np.cumsum(np.logaddexp(0,theta[d+1:]))]
            q=expit(Z@theta[:d,None]-cuts)
            return np.concatenate([1-q[:,:1],q[:,:-1]-q[:,1:],q[:,-1:]],1)
        def loss(theta):return -np.log(np.clip(predict(theta)[np.arange(len(y)),y],1e-12,1)).mean()+1e-3*np.sum(theta[:d]**2)
        result=minimize(loss,np.zeros(d+self.classes-1),method='L-BFGS-B')
        if not result.success:raise RuntimeError(result.message)
        self.theta=result.x;return self
    def predict_proba(self,X):
        Z=(np.asarray(X)-self.mean)/self.scale;d=Z.shape[1];t=self.theta
        cuts=np.r_[t[d],t[d]+np.cumsum(np.logaddexp(0,t[d+1:]))];q=expit(Z@t[:d,None]-cuts)
        return np.concatenate([1-q[:,:1],q[:,:-1]-q[:,1:],q[:,-1:]],1)

import torch
from torch import nn
import torch.nn.functional as F
class OrdinalHead(nn.Module):
    def __init__(self,features,classes):
        super().__init__();self.score=nn.Linear(features,1);self.base=nn.Parameter(torch.zeros(1));self.increments=nn.Parameter(torch.zeros(classes-2))
    def forward(self,x):
        cuts=torch.cat([self.base,self.base+torch.cumsum(F.softplus(self.increments),0)])
        return self.score(x)-cuts
    def probabilities(self,x):
        q=torch.sigmoid(self(x));return torch.cat([1-q[:,:1],q[:,:-1]-q[:,1:],q[:,-1:]],1)
    def loss(self,x,y):
        logits=self(x);target=(y[:,None]>torch.arange(logits.shape[1],device=y.device)).float()
        return F.binary_cross_entropy_with_logits(logits,target)
