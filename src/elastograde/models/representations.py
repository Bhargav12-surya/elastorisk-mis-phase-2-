import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.linear_model import Ridge

class MechanicalPhenotypes:
    def fit(self,X,roles,components=(2,3),seed=42,pca_dim=None):
        if set(roles)!={'train'}:raise ValueError('Phenotype fitting accepts training partition only')
        X=np.asarray(X,float)
        if not np.isfinite(X).all():raise ValueError('Nonfinite features')
        self.scaler=StandardScaler().fit(X);Z=self.scaler.transform(X);self.pca=None
        if pca_dim:self.pca=PCA(min(pca_dim,*Z.shape),random_state=seed).fit(Z);Z=self.pca.transform(Z)
        candidates=[GaussianMixture(k,random_state=seed,reg_covar=1e-5,n_init=3).fit(Z) for k in components if k<len(Z)]
        if not candidates:raise ValueError('Insufficient training samples')
        self.model=min(candidates,key=lambda m:m.bic(Z));self.train_nll=-self.model.score_samples(Z)
        self.threshold=float(np.quantile(self.train_nll,.99));return self
    def transform(self,X):
        z=self.scaler.transform(X);return self.pca.transform(z) if self.pca else z
    def predict(self,X):
        z=self.transform(X);p=self.model.predict_proba(z)
        return {'probabilities':p,'names':[f'mechanical_cluster_{i}' for i in p.argmax(1)],'outlier':-self.model.score_samples(z)>self.threshold}
    def stability(self,X,repeats=10,seed=42):
        z=self.transform(X);base=self.model.predict(z);rng=np.random.default_rng(seed);scores=[]
        for i in range(repeats):
            ix=rng.integers(len(z),size=len(z));m=GaussianMixture(self.model.n_components,reg_covar=1e-5,random_state=seed+i).fit(z[ix]);scores.append(adjusted_rand_score(base,m.predict(z)))
        return {'ari_mean':float(np.mean(scores)),'ari_values':scores,'unit':'feature rows; use one row per independent group'}

def nuisance_probe(train_features,train_nuisance,test_features,test_nuisance):
    model=Ridge(alpha=1).fit(train_features,train_nuisance)
    return {'held_out_r2':float(model.score(test_features,test_nuisance))}

def variance_preserving_loss(a,b):
    import torch
    invariance=(a-b).square().mean()
    var=torch.relu(1-torch.sqrt(a.var(0,unbiased=False)+1e-4)).mean()+torch.relu(1-torch.sqrt(b.var(0,unbiased=False)+1e-4)).mean()
    return invariance+var
