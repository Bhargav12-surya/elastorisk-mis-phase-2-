import numpy as np
from sklearn.covariance import LedoitWolf
class FeatureOOD:
    def fit(self,X,roles):
        if set(roles)!={'train'}:raise ValueError('OOD reference must use train only')
        self.model=LedoitWolf().fit(X);self.threshold=float(np.quantile(self.model.mahalanobis(X),.99));return self
    def score(self,X):return self.model.mahalanobis(X)
    def reject(self,X):return self.score(X)>self.threshold
