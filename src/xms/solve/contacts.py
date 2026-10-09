"""Conservative skeletal foot contacts, never a per-frame floor clamp."""
import numpy as np
from xms.animation.fk import evaluate


def spans(mask,min_samples=3):
    edges=np.diff(np.r_[False,mask,False].astype(int));return [(int(a),int(b)) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if b-a>=min_samples]


def detect(positions,times,confidence,floor_height,enter_height=.04,exit_height=.07,speed_limit=.15):
    positions=np.asarray(positions);n=len(times)
    speed=np.zeros(n)
    if n>1:speed[1:]=np.linalg.norm(np.diff(positions,axis=0),axis=1)/np.diff(times);speed[0]=speed[1]
    mask=np.zeros(n,bool);active=False
    for i in range(n):
        active=bool(confidence[i]>=.5 and speed[i]<=speed_limit*(1.5 if active else 1) and abs(positions[i,1]-floor_height)<=(exit_height if active else enter_height))
        mask[i]=active
    return spans(mask)


def constraints(bundle):
    a=bundle.arrays;names=bundle.metadata['joint_names'];times=a['sample_times_s'];world=np.array([evaluate(bundle,i) for i in range(len(times))])
    # Rest bone height is a skeletal proxy, not verified skin sole geometry.
    from xms.animation.bundle import AnimationBundle
    import copy
    rest=copy.deepcopy(bundle);rest.arrays['local_rotation_delta'][:]=[0,0,0,1];rest.arrays['root_translation'][:]=0
    rest_world=evaluate(rest,0);records=[];terms=[];masks=[]
    for side in ('l','r'):
        j=names.index('foot_'+side);pos=world[:,j,:3,3];mask=np.zeros(len(times),bool)
        for start,end in detect(pos,times,a['rotation_confidence'][:,names.index('calf_'+side)],rest_world[j,1,3]):
            anchor=np.median(pos[start:end],axis=0);mask[start:end]=True
            valid=np.zeros(len(times),bool);valid[start:end]=True;conf=a['rotation_confidence'][:,names.index('calf_'+side)]*valid
            terms.append((j,anchor,valid,conf));records.append({'start_s':float(times[start]),'end_s':float(times[end-1]),'joint':names[j],'kind':'foot','confidence':float(conf[start:end].mean())})
        masks.append(mask)
    bundle.metadata['contacts']=records
    bundle.metadata['contact_anchors']=[{'joint':names[j],'anchor_world_m':anchor.tolist(),'sample_indices':np.flatnonzero(valid).tolist()} for j,anchor,valid,conf in terms]
    bundle.metadata['contact_policy']='Skeletal height/velocity/visibility hysteresis; unidentifiable floor/depth yields unavailable, no floor clamp.'
    # Flight requires both feet reliably above their rest-level proxy.
    flight=np.ones(len(times),bool)
    for side in ('l','r'):
        j=names.index('foot_'+side);flight&=(world[:,j,1,3]-rest_world[j,1,3]>.08)&(a['rotation_confidence'][:,names.index('calf_'+side)]>=.5)
    bundle.metadata['events']=[{'time_s':float(times[start]),'kind':'jump','channels':['foot_l','foot_r']} for start,end in spans(flight)]
    return terms
