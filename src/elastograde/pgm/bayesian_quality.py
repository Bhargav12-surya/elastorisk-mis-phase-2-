from itertools import product
import numpy as np

class Factor:
    def __init__(self,variables,values):self.variables=list(variables);self.values=np.asarray(values,float)
    def condition(self,evidence):
        values=self.values;variables=self.variables.copy()
        for key,value in evidence.items():
            if key in variables:
                axis=variables.index(key);values=np.take(values,int(value),axis=axis);variables.pop(axis)
        return Factor(variables,values)
    def multiply(self,other):
        variables=self.variables+[v for v in other.variables if v not in self.variables]
        def aligned(f):
            order=[v for v in variables if v in f.variables]
            a=np.transpose(f.values,[f.variables.index(v) for v in order]) if order else f.values
            return a.reshape([f.values.shape[f.variables.index(v)] if v in f.variables else 1 for v in variables])
        return Factor(variables,aligned(self)*aligned(other))
    def sum_out(self,key):
        axis=self.variables.index(key);return Factor([v for v in self.variables if v!=key],self.values.sum(axis))

class BayesianNetwork:
    def __init__(self,parents,cpds):
        self.parents=parents;self.cpds={k:np.asarray(v,float) for k,v in cpds.items()};self.nodes=list(parents)
        seen=set()
        for k in self.nodes:
            if not set(parents[k])<=seen:raise ValueError('Nodes must be topologically ordered DAG')
            c=self.cpds[k]
            if c.shape!=(2,)*(len(parents[k])+1) or not np.allclose(c.sum(-1),1) or np.any(c<0):raise ValueError('Invalid binary CPD '+k)
            seen.add(k)
    def factors(self):return [Factor(self.parents[k]+[k],self.cpds[k]) for k in self.nodes]
    def joint(self,assignment):
        return float(np.prod([self.cpds[k][tuple(assignment[p] for p in self.parents[k])+ (assignment[k],)] for k in self.nodes]))
    def enumerate_query(self,key,evidence=None):
        evidence=evidence or {};out=np.zeros(2)
        for vals in product(range(2),repeat=len(self.nodes)):
            a=dict(zip(self.nodes,vals))
            if all(a[k]==v for k,v in evidence.items()):out[a[key]]+=self.joint(a)
        if out.sum()==0:raise ValueError('Impossible evidence')
        return out/out.sum()
    def query(self,key,evidence=None):
        evidence=evidence or {}
        if key in evidence:return np.eye(2)[evidence[key]]
        factors=[f.condition(evidence) for f in self.factors()]
        for v in self.nodes:
            if v==key or v in evidence:continue
            selected=[f for f in factors if v in f.variables];factors=[f for f in factors if v not in f.variables]
            if selected:
                f=selected[0]
                for g in selected[1:]:f=f.multiply(g)
                factors.append(f.sum_out(v))
        f=factors[0]
        for g in factors[1:]:f=f.multiply(g)
        if f.values.sum()<=0:raise ValueError('Impossible evidence')
        return f.values/f.values.sum()
    def fit(self,rows,roles,alpha=1):
        if set(roles)!={'train'}:raise ValueError('CPD learning requires training records')
        for k in self.nodes:
            counts=np.full_like(self.cpds[k],alpha)
            for row in rows:counts[tuple(row[p] for p in self.parents[k])+(row[k],)]+=1
            self.cpds[k]=counts/counts.sum(-1,keepdims=True)
        return self
    def log_likelihood(self,rows):return float(np.mean([np.log(max(self.joint(r),1e-15)) for r in rows]))

def illustrative_network():
    # All CPDs are educational assumptions; not clinical or phantom calibration.
    parents={'domain':[],'quality':['domain'],'residual':['quality'],'cycle':['quality'],'uncertainty':['quality'],'reliable':['domain','quality']}
    cpds={'domain':[.8,.2],'quality':[[.15,.85],[.55,.45]],'residual':[[.15,.85],[.9,.1]],'cycle':[[.2,.8],[.95,.05]],'uncertainty':[[.25,.75],[.9,.1]],'reliable':[[[.95,.05],[.05,.95]],[[.99,.01],[.3,.7]]]}
    return BayesianNetwork(parents,cpds)

class TrainingDiscretizer:
    def fit(self,X,roles):
        if set(roles)!={'train'}:raise ValueError('Train-only thresholds')
        self.threshold=np.median(X,axis=0);return self
    def transform(self,X):return (np.asarray(X)>self.threshold).astype(int)
