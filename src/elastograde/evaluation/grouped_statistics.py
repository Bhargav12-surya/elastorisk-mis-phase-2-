import numpy as np

def grouped_bootstrap(values,groups,repeats=1000,seed=42):
    values=np.asarray(values,float);groups=np.asarray(groups);unique=np.unique(groups)
    if len(unique)<2:raise ValueError('At least two independent groups needed')
    means=np.array([values[groups==g].mean() for g in unique]);rng=np.random.default_rng(seed)
    samples=means[rng.integers(len(means),size=(repeats,len(means)))].mean(1)
    return {'mean':float(means.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),'independent_groups':len(unique),'weighting':'equal group means'}

def paired_comparison(a,b,groups,seed=42):return grouped_bootstrap(np.asarray(a)-np.asarray(b),groups,seed=seed)

def holm_adjust(pvalues):
    p=np.asarray(pvalues);order=np.argsort(p);adjust=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]);out=np.zeros_like(p);out[order]=np.minimum(adjust,1);return out
