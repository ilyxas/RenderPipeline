import copy
import numpy as np
from .bundle import AnimationBundle


def continuous(q):
    q=np.array(q,copy=True)
    for i in range(1,len(q)):
        q[i]=np.where((np.sum(q[i-1]*q[i],axis=-1)<0)[...,None],-q[i],q[i])
    return q


def slerp(a,b,t):
    dot=np.sum(a*b,axis=-1,keepdims=True)
    b=np.where(dot<0,-b,b); dot=np.abs(dot).clip(0,1)
    angle=np.arccos(dot); sine=np.sin(angle)
    safe=np.where(sine<1e-8,1,sine)
    q=np.where(dot>0.9995,(1-t)*a+t*b,(np.sin((1-t)*angle)*a+np.sin(t*angle)*b)/safe)
    return q/np.linalg.norm(q,axis=-1,keepdims=True)


def sample(bundle,times):
    source=bundle.arrays; old=source['sample_times_s']; times=np.asarray(times,dtype=float)
    if np.any(times<old[0]) or np.any(times>old[-1]+1e-8): raise ValueError('Sampling outside bundle support')
    hi=np.searchsorted(old,times,side='right').clip(1,len(old)-1) if len(old)>1 else np.zeros(len(times),int)
    lo=np.maximum(0,hi-1); span=old[hi]-old[lo]
    alpha=np.divide(times-old[lo],span,out=np.zeros_like(times),where=span!=0)
    a={k:v.copy() for k,v in source.items()}
    a['sample_times_s']=times
    for k in ('root_translation','face_weights','source_time_map'):
        t=alpha.reshape((-1,)+(1,)*(source[k].ndim-1)); a[k]=source[k][lo]*(1-t)+source[k][hi]*t
    a['local_rotation_delta']=slerp(source['local_rotation_delta'][lo],source['local_rotation_delta'][hi],alpha[:,None,None])
    for g in ('rotation','root','face'):
        for suffix in ('validity','confidence','provenance'):
            key=g+'_'+suffix
            # Both endpoints must support interpolation; exact keys retain masks.
            t=alpha.reshape((-1,)+(1,)*(source[key].ndim-1))
            if suffix=='validity': a[key]=np.where(t==0,source[key][lo],np.where(t==1,source[key][hi],source[key][lo]&source[key][hi]))
            elif suffix=='confidence': a[key]=source[key][lo]*(1-t)+source[key][hi]*t
            else: a[key]=np.where(t<0.5,source[key][lo],source[key][hi])
        invalid=~a[g+'_validity']; a[g+'_confidence'][invalid]=0; a[g+'_provenance'][invalid]=0
        a[{'rotation':'local_rotation_delta','root':'root_translation','face':'face_weights'}[g]][invalid]=[0,0,0,1] if g=='rotation' else 0
    return AnimationBundle(copy.deepcopy(bundle.metadata),a)
