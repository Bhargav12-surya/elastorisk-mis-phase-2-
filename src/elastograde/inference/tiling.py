import numpy as np

def tiled_predict(pre,post,predict,tile=(96,64),overlap=24):
    h,w=pre.shape;th,tw=tile
    if not 0<overlap<min(tile):raise ValueError('Overlap must be between zero and tile size')
    if h<=th and w<=tw:return predict(pre,post)
    def starts(n,t):return sorted(set(list(range(0,max(1,n-t+1),t-overlap))+[max(0,n-t)]))
    out=np.zeros((2,h,w));weight=np.zeros((h,w))
    for z in starts(h,th):
        for x in starts(w,tw):
            a=pre[z:z+th,x:x+tw];b=post[z:z+th,x:x+tw];u=predict(a,b)
            win=np.maximum(np.outer(np.hanning(a.shape[0]),np.hanning(a.shape[1])),.001)
            out[:,z:z+a.shape[0],x:x+a.shape[1]]+=u*win;weight[z:z+a.shape[0],x:x+a.shape[1]]+=win
    return out/weight
