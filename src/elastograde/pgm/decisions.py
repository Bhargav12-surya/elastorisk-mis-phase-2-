import numpy as np
ACTIONS=['accept_mechanics','reacquire','expert_review']
DEFAULT_LOSS=np.array([[10.,0.],[1.,1.],[2.,.5]]) # columns unreliable,reliable

def choose(p_reliable,loss=None):
    if not 0<=p_reliable<=1:raise ValueError('Probability outside [0,1]')
    loss=DEFAULT_LOSS if loss is None else np.asarray(loss,float)
    if loss.shape!=(3,2) or not np.isfinite(loss).all():raise ValueError('Need finite 3x2 action-state losses')
    expected=loss@np.array([1-p_reliable,p_reliable]);i=expected.argmin()
    return {'action':ACTIONS[i],'expected_loss':dict(zip(ACTIONS,expected.tolist())),'utility_provenance':'illustrative educational costs; not a clinical policy'}

def sensitivity(probabilities,costs):
    rows=[]
    for cost in costs:
        loss=DEFAULT_LOSS.copy();loss[0,0]=cost
        for p in probabilities:rows.append({'p_reliable':float(p),'false_accept_cost':float(cost),**choose(p,loss)})
    return rows
