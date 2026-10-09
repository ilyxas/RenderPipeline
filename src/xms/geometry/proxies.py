"""Analytical capsule/plane proxies; these are not deformed-surface distances."""
import numpy as np


def segment_distance(a,b,c,d):
    a,b,c,d=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float),np.asarray(c,float),np.asarray(d,float))
    u=b-a;v=d-c;w=a-c;aa=np.sum(u*u,axis=-1);bb=np.sum(u*v,axis=-1);cc=np.sum(v*v,axis=-1);dd=np.sum(u*w,axis=-1);ee=np.sum(v*w,axis=-1);den=aa*cc-bb*bb
    s=np.clip(np.divide(bb*ee-cc*dd,den,out=np.zeros_like(den),where=den>1e-12),0,1)
    t=np.divide(bb*s+ee,cc,out=np.zeros_like(cc),where=cc>1e-12)
    s=np.where(t<0,np.clip(np.divide(-dd,aa,out=np.zeros_like(aa),where=aa>1e-12),0,1),s)
    s=np.where(t>1,np.clip(np.divide(bb-dd,aa,out=np.zeros_like(aa),where=aa>1e-12),0,1),s);t=np.clip(t,0,1)
    return np.linalg.norm(w+s[...,None]*u-t[...,None]*v,axis=-1)


def capsule_depth(a,b,radius,c,d,other_radius,margin=0):
    return np.maximum(0,radius+other_radius+margin-segment_distance(a,b,c,d))


def plane_depth(points,normal,offset,radius=0):
    return np.maximum(0,radius-(np.asarray(points)@np.asarray(normal)-offset))


def terms(positions,lookup,definitions):
    depths=[]
    for pair in definitions.get('pairs',[]):
        left,right=(definitions['capsules'][k] for k in pair)
        depths.append(capsule_depth(positions[lookup[left['start']]],positions[lookup[left['end']]],left['radius_m'],positions[lookup[right['start']]],positions[lookup[right['end']]],right['radius_m'],definitions['margin_m']))
    return np.stack(depths,axis=-1) if depths else np.empty((len(next(iter(positions.values()))),0))
