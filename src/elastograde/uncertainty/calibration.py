import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize_scalar

class TemperatureScaler:
    def fit(self,logits,labels,roles):
        if set(roles)!={'calibration'}:raise ValueError('Independent calibration role required')
        logits=np.asarray(logits);labels=np.asarray(labels,int)
        result=minimize_scalar(lambda t:-np.log(np.clip(softmax(logits/np.exp(t),axis=1)[np.arange(len(labels)),labels],1e-12,1)).mean(),bounds=(-4,4),method='bounded')
        self.temperature=float(np.exp(result.x));return self
    def predict(self,logits):return softmax(np.asarray(logits)/self.temperature,axis=1)

def conformal_radius(group_scores,alpha=.1):
    scores=np.asarray(group_scores,float);n=len(scores)
    if n==0 or not 0<alpha<1:raise ValueError('Need calibration groups and alpha in (0,1)')
    k=int(np.ceil((n+1)*(1-alpha)))
    return float(np.sort(scores)[k-1]) if k<=n else float('inf')

def reliability(probabilities,labels,bins=10):
    p=np.asarray(probabilities);y=np.asarray(labels);c=p.max(1);correct=p.argmax(1)==y;out=[]
    for i in range(bins):
        m=(c>=i/bins)&(c< (i+1)/bins if i<bins-1 else c<=1)
        if m.any():out.append({'confidence':float(c[m].mean()),'accuracy':float(correct[m].mean()),'count':int(m.sum())})
    return out
