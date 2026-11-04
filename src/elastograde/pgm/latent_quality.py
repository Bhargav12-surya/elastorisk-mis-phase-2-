import numpy as np
from scipy.special import logsumexp

class BernoulliMixture:
    """EM latent quality representation; component labels are not identifiable."""
    def fit(self,X,roles,states=2,iterations=50,seed=42):
        if set(roles)!={'train'}:raise ValueError('EM fit uses training observations')
        X=np.asarray(X,float)
        if not np.isin(X,[0,1]).all():raise ValueError('Binary observations required')
        rng=np.random.default_rng(seed);self.p=rng.uniform(.2,.8,(states,X.shape[1]));self.weights=np.ones(states)/states;self.history=[]
        for _ in range(iterations):
            logp=self._logp(X);norm=logsumexp(logp,axis=1);r=np.exp(logp-norm[:,None]);nk=r.sum(0)
            self.weights=(nk+1)/(len(X)+states);self.p=np.clip((r.T@X+1)/(nk[:,None]+2),1e-5,1-1e-5);self.history.append(float(norm.mean()))
        return self
    def _logp(self,X):return X@np.log(self.p).T+(1-X)@np.log(1-self.p).T+np.log(self.weights)
    def posterior(self,X):
        a=self._logp(np.asarray(X));return np.exp(a-logsumexp(a,axis=1,keepdims=True))
