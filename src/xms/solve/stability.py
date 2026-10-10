"""Time-based trajectory repair. Estimates are marked separately from observations."""
import numpy as np
from xms.animation.sample import slerp,continuous
from .contacts import spans


def angle(a,b):
    return 2*np.arccos(np.clip(abs(np.dot(a,b)),0,1))


def rotations(values,valid,confidence,times,bridge=.3,fade=.4,tau=.035,max_speed=None):
    q=values.copy();v=valid.copy();c=confidence.copy();estimated=np.zeros(len(v),bool)
    identity=np.array([0.,0,0,1.])
    # A single inconsistent observation between agreeing neighbours is an outlier,
    # not a fast coherent gesture. Work on SO(3), never Euler/rotvec branches.
    for i in range(1,len(q)-1):
        if v[i-1:i+2].all():
            midpoint=slerp(q[i-1],q[i+1],.5)
            if angle(q[i],midpoint)>np.deg2rad(25) and angle(q[i-1],q[i+1])<np.deg2rad(20):
                q[i]=midpoint;c[i]*=.5;estimated[i]=True
    for start,end in spans(~v,1):
        left=start-1;right=end
        if left>=0 and right<len(q) and times[right]-times[left]<=bridge:
            for i in range(start,end):
                q[i]=slerp(q[left],q[right],(times[i]-times[left])/(times[right]-times[left]))
                v[i]=True;c[i]=min(c[left],c[right])*.4;estimated[i]=True
        else:
            for i in range(start,end):
                choices=[]
                if left>=0 and times[i]-times[left]<fade:choices.append((1-(times[i]-times[left])/fade,q[left],c[left]))
                if right<len(q) and times[right]-times[i]<fade:choices.append((1-(times[right]-times[i])/fade,q[right],c[right]))
                if choices:
                    weight,source,conf=max(choices,key=lambda x:x[0]);q[i]=slerp(identity,source,weight);v[i]=True;c[i]=conf*.25;estimated[i]=True
    # Symmetric filtering avoids a frame delay. Coherent fast motion raises the
    # cutoff; angular outliers and noisy slow poses receive more stabilization.
    original=continuous(q)
    for direction in (range(1,len(q)),range(len(q)-2,-1,-1)):
        for i in direction:
            previous=i-1 if direction.step==1 else i+1
            if not (v[i] and v[previous]):continue
            dt=abs(times[i]-times[previous]);speed=angle(original[i],original[previous])/max(dt,1e-8)
            cutoff=tau/(1+speed/5)
            alpha=1-np.exp(-dt/max(cutoff,1e-5));q[i]=slerp(q[previous],q[i],alpha)
    if max_speed is not None:
        # Unsupported rotational estimates use a declared bounded fallback.
        # Ordinary coherent gestures below this physical rate are untouched.
        for i in range(1,len(q)):
            if not (v[i] and v[i-1]):continue
            step=angle(q[i-1],q[i]);budget=np.deg2rad(max_speed)*(times[i]-times[i-1])
            if step>budget:
                q[i]=slerp(q[i-1],q[i],budget/step);c[i]*=.25;estimated[i]=True
    q[~v]=identity;c[~v]=0
    return continuous(q),v,c,estimated


def stabilize_bundle(bundle,joints=None,max_speed=None):
    a=bundle.arrays;count=0
    for j in range(len(bundle.metadata['joint_names'])) if joints is None else joints:
        q,v,c,estimated=rotations(a['local_rotation_delta'][:,j],a['rotation_validity'][:,j],a['rotation_confidence'][:,j],a['sample_times_s'],max_speed=max_speed)
        a['local_rotation_delta'][:,j]=q;a['rotation_validity'][:,j]=v;a['rotation_confidence'][:,j]=c
        a['rotation_provenance'][v & estimated,j]=2;a['rotation_provenance'][~v,j]=0;count+=int(estimated.sum())
    return count
