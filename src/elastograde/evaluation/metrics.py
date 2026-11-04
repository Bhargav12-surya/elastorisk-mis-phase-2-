import numpy as np
from elastograde.mechanics.coordinates import to_mm

def displacement_metrics(pred,true,mask,spacing=None,units='pixels'):
    if units=='mm':pred=to_mm(pred,spacing);true=to_mm(true,spacing)
    elif units!='pixels':raise ValueError('units pixels or mm')
    m=mask&np.isfinite(pred).all(0)&np.isfinite(true).all(0)
    if not m.any():raise ValueError('Empty metric support')
    d=pred[:,m]-true[:,m]
    return {'axial_mae':float(abs(d[1]).mean()),'lateral_mae':float(abs(d[0]).mean()),'axial_rmse':float(np.sqrt((d[1]**2).mean())),'lateral_rmse':float(np.sqrt((d[0]**2).mean())),'endpoint_error':float(np.linalg.norm(d,axis=0).mean()),'valid_pixels':int(m.sum()),'valid_fraction':float(m.mean()),'units':units}

def strain_metrics(pred,true,mask):
    d=pred[mask]-true[mask]
    if len(d)==0:raise ValueError('Empty support')
    return {'strain_mae':float(abs(d).mean()),'strain_rmse':float(np.sqrt((d*d).mean())),'units':'dimensionless'}

def segmentation_metrics(pred,true):
    p=np.asarray(pred,bool);t=np.asarray(true,bool);inter=(p&t).sum();union=(p|t).sum()
    return {'dice':float(2*inter/(p.sum()+t.sum())) if p.sum()+t.sum() else 1.,'iou':float(inter/union) if union else 1.}

def clinical_metrics(probabilities,labels):
    from sklearn.metrics import f1_score,balanced_accuracy_score,confusion_matrix,cohen_kappa_score,recall_score,log_loss
    p=np.asarray(probabilities);y=np.asarray(labels);pred=p.argmax(1);classes=list(range(p.shape[1]))
    return {'macro_f1':float(f1_score(y,pred,average='macro',labels=classes,zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'ordinal_mae':float(abs(y-pred).mean()),'quadratic_kappa':float(cohen_kappa_score(y,pred,weights='quadratic')),'confusion_matrix':confusion_matrix(y,pred,labels=classes).tolist(),'per_class_recall':recall_score(y,pred,labels=classes,average=None,zero_division=0).tolist(),'nll':float(log_loss(y,p,labels=classes)),'brier':float(((p-np.eye(p.shape[1])[y])**2).sum(1).mean())}
