"""Calibrated hand residuals on the solved arm; no second arm IK or palm solver."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
from xms.animation.fk import evaluate,quat_matrix,matrix_quat
from .head import rotation,local_delta_for_world
from .baseline_hands import hand_frame
from .stability import rotations
from .image_body import unit


def continuous_twist(rotations_xyzw,axes):
    """Signed swing/twist angle, unwrapped before applying anatomical limits."""
    q=np.asarray(rotations_xyzw);projected=np.sum(q[:,:3]*axes,axis=1)
    angle=np.unwrap(2*np.arctan2(projected,q[:,3]))
    # The first pose supplies the branch nearest the calibrated neutral.
    if len(angle):angle-=2*np.pi*round(angle[0]/(2*np.pi))
    return angle


def solve(body,observations,timeline):
    _,obs=observations;idx=np.array(timeline['output_observation_indices']);t=body.arrays['sample_times_s'];n=len(t);fps=1/np.median(np.diff(t)) if n>1 else 24;names=body.metadata['joint_names'];ids={s:i for i,s in enumerate(names)}
    work=copy.deepcopy(body);a=work.arrays;ref=copy.deepcopy(body);ref.arrays['local_rotation_delta'][:]=[0,0,0,1];ref.arrays['root_translation'][:]=0;rest=evaluate(ref,0);selected=np.array(body.metadata.get('image_retarget_diagnostics',{}).get('neutral_sample_indices',[]),int)
    valid=np.zeros_like(a['rotation_validity']);confidence=np.zeros_like(a['rotation_confidence']);claimed=[];repaired=0
    for si,side in enumerate(('l','r')):
        wrist=ids['hand_'+side];elbow=ids['lowerarm_'+side];joints=[wrist]+[ids[f'{f}_{k:02d}_{side}'] for f in ('thumb','index','middle','ring','pinky') for k in (1,2,3)];claimed+=joints
        lm=obs['world_landmarks_raw'][idx,si]*[1,-1,-1];v=obs['validity'][idx,si]&(obs['confidence'][idx,si]>.4);c=obs['confidence'][idx,si].copy();q=np.tile([0.,0,0,1.],(n,1))
        for i in np.flatnonzero(v):
            try:q[i]=matrix_quat(hand_frame(lm[i,0],lm[i,9],lm[i,5],lm[i,17]))
            except ValueError:v[i]=False
        reference=hand_frame(rest[wrist,:3,3],rest[ids['middle_01_'+side],:3,3],rest[ids['index_01_'+side],:3,3],rest[ids['pinky_01_'+side],:3,3])
        neutral_samples=selected[v[selected]] if len(selected) else np.array([],int)
        origin=Rotation.from_quat(q[neutral_samples]).mean().as_matrix() if len(neutral_samples) else reference
        q,v,c,estimated=rotations(q,v,c,t,bridge=.2,fade=0,tau=.06);repaired+=int(estimated.sum())
        frames=np.flatnonzero(v)
        desired_all=Rotation.from_quat(q).as_matrix()@origin.T@rotation(rest[wrist])
        fk_cache={i:evaluate(work,i) for i in frames}
        relative=[];axes=[]
        for i in frames:
            fk=fk_cache[i];base_world=rotation(fk[elbow])
            hand_neutral=base_world@rotation(rest[elbow]).T@rotation(rest[wrist])
            forearm_world=unit(fk[wrist,:3,3]-fk[elbow,:3,3])
            relative.append(Rotation.from_matrix(hand_neutral.T@desired_all[i]).as_quat())
            axes.append(hand_neutral.T@forearm_world)
        rolls=np.zeros(n)
        if len(frames):
            values=continuous_twist(np.array(relative),np.array(axes))
            rolls=np.interp(t,t[frames],values)
            rolls=gaussian_filter1d(np.clip(rolls,-np.pi*.75,np.pi*.75),max(.05*fps,.01),mode='nearest')
        for i in frames:
            fk=fk_cache[i];base_world=rotation(fk[elbow]);parent=a['parent_indices'][elbow]
            forearm_world=unit(fk[wrist,:3,3]-fk[elbow,:3,3])
            twist=Rotation.from_rotvec(forearm_world*rolls[i]).as_matrix()
            a['local_rotation_delta'][i,elbow]=local_delta_for_world(twist@base_world,fk[parent],a['rest_rotation'][elbow])
            body.arrays['local_rotation_delta'][i,elbow]=a['local_rotation_delta'][i,elbow]
            parent_world=fk[elbow].copy();parent_world[:3,:3]=twist@parent_world[:3,:3]
            delta=local_delta_for_world(desired_all[i],parent_world,a['rest_rotation'][wrist]);rv=Rotation.from_quat(delta).as_rotvec();angle=np.linalg.norm(rv)
            if angle>np.deg2rad(75):delta=Rotation.from_rotvec(rv*(np.deg2rad(75)/angle)).as_quat()
            a['local_rotation_delta'][i,wrist]=delta;valid[i,wrist]=True;confidence[i,wrist]=c[i]
        a['local_rotation_delta'][:,wrist],valid[:,wrist],confidence[:,wrist],_=rotations(a['local_rotation_delta'][:,wrist],valid[:,wrist],confidence[:,wrist],t,bridge=.2,fade=.12,tau=.045,max_speed=720)
        for finger,offset in [('thumb',1),('index',5),('middle',9),('ring',13),('pinky',17)]:
            for k in (1,2,3):
                j=ids[f'{finger}_{k:02d}_{side}'];u=lm[:,offset+k-1]-lm[:,offset+k-2] if k>1 else lm[:,offset]-lm[:,0];z=lm[:,offset+k]-lm[:,offset+k-1]
                curl=np.arccos(np.clip(np.sum(unit(u)*unit(z),axis=1),-1,1));good=np.flatnonzero(v)
                if not len(good):continue
                curl=np.interp(t,t[good],curl[good]);curl=gaussian_filter1d(curl,max(.055*fps,.01),mode='nearest');bias=float(np.median(curl[neutral_samples])) if len(neutral_samples) else 0.;curl=np.clip(curl-bias,-.15,np.deg2rad(95))
                child=ids[f'{finger}_{k+1:02d}_{side}'] if k<3 else j;direction=rest[child,:3,3]-rest[j,:3,3] if k<3 else rest[j,:3,3]-rest[ids[f'{finger}_02_{side}'],:3,3]
                inward=reference[:,2]*(1 if side=='l' else -1);axis=unit(rotation(rest[j]).T@np.cross(direction,inward))
                a['local_rotation_delta'][:,j]=Rotation.from_rotvec(curl[:,None]*axis).as_quat();valid[:,j]=v;confidence[:,j]=c
    return {'indices':claimed,'rotations':a['local_rotation_delta'],'validity':valid,'confidence':confidence,'provenance':valid.astype(np.uint8)*2,'diagnostics':{'policy':'neutral-calibrated palm rotation; forearm-axis pronation; relative finger hinge angles','repaired_samples':repaired,'quality_accepted':False}}
