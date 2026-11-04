import numpy as np

def risk_coverage(errors,uncertainties):
    e=np.asarray(errors);u=np.asarray(uncertainties)
    if len(e)!=len(u) or len(e)==0 or not np.isfinite(e).all() or not np.isfinite(u).all():raise ValueError('Finite matched group arrays required')
    ix=np.argsort(u);n=np.arange(1,len(e)+1)
    return {'coverage':n/len(e),'risk':np.cumsum(e[ix])/n,'order':ix}

def threshold_action(residual,cycle,invalid_fraction,limits=(.5,1.,.4)):
    reasons=[name for name,value,limit in zip(['large_residual','cycle_mismatch','insufficient_support'],[residual,cycle,invalid_fraction],limits) if value>limit]
    return {'action':'reacquire' if reasons else 'accept_mechanics','reasons':reasons,'policy':'illustrative thresholds; requires validation'}
