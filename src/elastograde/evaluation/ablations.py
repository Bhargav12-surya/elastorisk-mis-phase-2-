from copy import deepcopy

def configurations(base):
    variants=[('unet',{'model.name':'unet'}),('candidate',{}),('envelope',{'model.inputs':'envelope'}),('fusion',{'model.inputs':'fusion'}),('no_cycle',{'loss.fb':0}),('no_smoothness',{'loss.smooth':0}),('confidence',{'model.confidence':True,'loss.confidence':True}),('simulation_supervised',{'regime':'simulation_supervised','loss.sup':1.,'loss.rf':0.,'loss.env':0.})]
    out={}
    for name,updates in variants:
        c=deepcopy(base)
        for key,value in updates.items():
            if '.' in key:a,b=key.split('.');c.setdefault(a,{})[b]=value
            else:c[key]=value
        c['output']=str(base.get('output','reports/ablations'))+'/'+name;out[name]=c
    return out
