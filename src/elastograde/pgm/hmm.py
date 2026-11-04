import numpy as np
from scipy.special import logsumexp

class CategoricalHMM:
    def __init__(self,initial,transition,emission):
        self.initial=np.asarray(initial,float);self.transition=np.asarray(transition,float);self.emission=np.asarray(emission,float)
        for a in [self.initial,self.transition,self.emission]:
            if np.any(a<=0) or not np.allclose(a.sum(axis=-1),1):raise ValueError('Strictly positive normalized HMM probabilities required')
    def infer(self,observations,ordered=False):
        if not ordered:raise ValueError('Verified ordering or explicit synthetic sequence required')
        obs=np.asarray(observations,int)
        if len(obs)==0:raise ValueError('Empty sequence')
        n=len(obs);k=len(self.initial);logA=np.log(self.transition);e=np.log(self.emission[:,obs].T)
        f=np.empty((n,k));f[0]=np.log(self.initial)+e[0]
        for t in range(1,n):f[t]=e[t]+logsumexp(f[t-1,:,None]+logA,axis=0)
        b=np.zeros((n,k))
        for t in range(n-2,-1,-1):b[t]=logsumexp(logA+e[t+1,None,:]+b[t+1,None,:],axis=1)
        filt=np.exp(f-logsumexp(f,axis=1,keepdims=True));smooth=np.exp(f+b-logsumexp(f+b,axis=1,keepdims=True))
        d=f.copy();back=np.zeros((n,k),int)
        for t in range(1,n):
            scores=d[t-1,:,None]+logA;back[t]=scores.argmax(0);d[t]=scores.max(0)+e[t]
        path=np.zeros(n,int);path[-1]=d[-1].argmax()
        for t in range(n-2,-1,-1):path[t]=back[t+1,path[t+1]]
        return {'filtered':filt,'smoothed':smooth,'viterbi':path,'log_likelihood':float(logsumexp(f[-1]))}
    @classmethod
    def fit_supervised(cls,sequences,state_sequences,roles,states=3,symbols=3,alpha=1):
        if set(roles)!={'train'}:raise ValueError('HMM fitting requires training sequences')
        pi=np.full(states,alpha,float);a=np.full((states,states),alpha,float);b=np.full((states,symbols),alpha,float)
        for obs,hidden in zip(sequences,state_sequences):
            if len(obs)!=len(hidden) or len(obs)==0:raise ValueError('Invalid training sequence')
            pi[hidden[0]]+=1
            for i,(o,s) in enumerate(zip(obs,hidden)):
                b[s,o]+=1
                if i:a[hidden[i-1],s]+=1
        return cls(pi/pi.sum(),a/a.sum(1,keepdims=True),b/b.sum(1,keepdims=True))

def illustrative_hmm():return CategoricalHMM([.8,.1,.1],[[.9,.08,.02],[.05,.8,.15],[.3,.1,.6]],[[.85,.1,.05],[.1,.75,.15],[.3,.2,.5]])
