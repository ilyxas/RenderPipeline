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


def axial_roll(relative_quaternions,axes):
    """Least wrist swing after axial rotation; unwrap without clipping a basis angle."""
    q=np.asarray(relative_quaternions)
    projected=np.sum(q[:,:3]*axes,axis=-1)
    return np.unwrap(2*np.arctan2(projected,q[:,3]))


def finger_rotation(current,target,hinge,mcp=False,thumb=False):
    """Primary finger anatomy: hinge distally, bounded splay at the knuckle."""
    current=unit(current);target=unit(target);hinge=unit(hinge)
    projected=unit(target-hinge*np.sum(target*hinge,axis=-1,keepdims=True))
    angle=np.arctan2(np.sum(hinge*np.cross(current,projected),axis=-1),np.sum(current*projected,axis=-1))
    bounded=np.clip(angle,np.deg2rad(-35),np.deg2rad(105))
    change=Rotation.from_rotvec(hinge*bounded[:,None]).as_matrix()
    limited=np.abs(angle-bounded)>1e-6
    if not mcp:limited|=np.abs(np.sum(target*hinge,axis=-1))>np.sin(np.deg2rad(3))
    if mcp:
        splay=-np.arcsin(np.clip(np.sum(target*hinge,axis=-1),-1,1));limit=np.deg2rad(40 if thumb else 15)
        capped=np.clip(splay,-limit,limit);normal=unit(np.cross(hinge,current))
        change=Rotation.from_rotvec(normal*capped[:,None]).as_matrix()@change
        limited|=np.abs(splay-capped)>1e-6
    return change,limited


def solve(body,observations,timeline):
    _,obs=observations;idx=np.array(timeline['output_observation_indices']);t=body.arrays['sample_times_s'];n=len(t);fps=1/np.median(np.diff(t)) if n>1 else 24;names=body.metadata['joint_names'];ids={s:i for i,s in enumerate(names)}
    work=copy.deepcopy(body);a=work.arrays;ref=copy.deepcopy(body);ref.arrays['local_rotation_delta'][:]=[0,0,0,1];ref.arrays['root_translation'][:]=0;rest=evaluate(ref,0);selected=np.array(body.metadata.get('image_retarget_diagnostics',{}).get('neutral_sample_indices',[]),int)
    valid=np.zeros_like(a['rotation_validity']);confidence=np.zeros_like(a['rotation_confidence']);claimed=[];repaired=0;bounded_palms=0;bounded_fingers=0;anatomical_fingers=0
    for si,side in enumerate(('l','r')):
        wrist=ids['hand_'+side];elbow=ids['lowerarm_'+side];shoulder=ids['upperarm_'+side];joints=[wrist]+[ids[f'{f}_{k:02d}_{side}'] for f in ('thumb','index','middle','ring','pinky') for k in (1,2,3)];claimed+=joints
        lm=obs['world_landmarks_raw'][idx,si]*[1,-1,-1];v=obs['validity'][idx,si]&(obs['confidence'][idx,si]>.4);c=obs['confidence'][idx,si].copy();q=np.tile([0.,0,0,1.],(n,1))
        for i in np.flatnonzero(v):
            try:q[i]=matrix_quat(hand_frame(lm[i,0],lm[i,9],lm[i,5],lm[i,17]))
            except ValueError:v[i]=False
        reference=hand_frame(rest[wrist,:3,3],rest[ids['middle_01_'+side],:3,3],rest[ids['index_01_'+side],:3,3],rest[ids['pinky_01_'+side],:3,3])
        neutral_samples=selected[v[selected]] if len(selected) else np.array([],int)
        origin=Rotation.from_quat(q[neutral_samples]).mean().as_matrix() if len(neutral_samples) else reference
        detected=v.copy();observed_frames=Rotation.from_quat(q).as_matrix()
        q,v,c,estimated=rotations(q,v,c,t,bridge=.2,fade=0,tau=.06,max_speed=900);repaired+=int(estimated.sum());bounded_palms+=int(estimated.sum())
        frames=np.flatnonzero(v)
        desired_all=Rotation.from_quat(q).as_matrix()@origin.T@rotation(rest[wrist])
        fk_cache={i:evaluate(work,i) for i in range(n)}
        # Palm lateral direction fixes forearm axial orientation. A twist
        # angle relative to a moving swing frame is not an anatomical joint
        # angle: clipping its unwrapped branch rotates the visible palm away.
        arm_rolls={}
        for joint,child in ((shoulder,elbow),(elbow,wrist)):
            relative=[];axes=[];upper_angles=[]
            for i in frames:
                fk=fk_cache[i];axis=unit(fk[child,:3,3]-fk[joint,:3,3])
                lateral=desired_all[i]@rotation(rest[wrist]).T@reference[:,0]
                if joint==shoulder:
                    # The elbow hinge follows the bend plane, not the fingers.
                    # Its normal has two equivalent signs; choose the branch
                    # continuous with neutral, and hold near a straight elbow.
                    forearm=unit(fk[wrist,:3,3]-fk[elbow,:3,3]);bend=-np.cross(axis,forearm)
                    amount=np.linalg.norm(bend);base=rotation(fk[shoulder])[:,1]
                    previous=upper_angles[-1] if upper_angles else 0.
                    if amount>.10:
                        lateral=unit(bend)
                        theta=np.arctan2(axis@np.cross(base,lateral),base@lateral)
                        theta+=np.pi*round((previous-theta)/np.pi)
                    else:theta=previous
                    upper_angles.append(theta)
                    continue
                hand_neutral=rotation(fk[elbow])@rotation(rest[elbow]).T@rotation(rest[wrist])
                relative.append(Rotation.from_matrix(hand_neutral.T@desired_all[i]).as_quat());axes.append(hand_neutral.T@axis)
            rolls=np.zeros(n)
            if len(frames):
                values=np.array(upper_angles) if joint==shoulder else axial_roll(np.array(relative),np.array(axes))
                rolls=np.interp(t,t[frames],values)
                rolls=gaussian_filter1d(rolls,max(.065*fps,.01),mode='nearest')
                if joint==elbow:
                    distance=np.min(np.abs(t[:,None]-t[frames][None,:]),axis=1)
                    rolls*=np.clip((.65-distance)/.4,0,1)
            arm_rolls[joint]=rolls
        for i in range(n):
            fk=fk_cache[i]
            upper_axis=unit(fk[elbow,:3,3]-fk[shoulder,:3,3]);lower_axis=unit(fk[wrist,:3,3]-fk[elbow,:3,3])
            upper=Rotation.from_rotvec(upper_axis*arm_rolls[shoulder][i]).as_matrix()@rotation(fk[shoulder])
            lower=Rotation.from_rotvec(lower_axis*arm_rolls[elbow][i]).as_matrix()@rotation(fk[elbow])
            # Both axial rotations preserve joint centres. Recompute the lower
            # local channel against the changed upper world transform once.
            parent=int(a['parent_indices'][shoulder]);upper_parent=fk[shoulder].copy();upper_parent[:3,:3]=upper
            a['local_rotation_delta'][i,shoulder]=local_delta_for_world(upper,fk[parent],a['rest_rotation'][shoulder])
            a['local_rotation_delta'][i,elbow]=local_delta_for_world(lower,upper_parent,a['rest_rotation'][elbow])
            for j in (shoulder,elbow):body.arrays['local_rotation_delta'][i,j]=a['local_rotation_delta'][i,j]
            if not v[i]:continue
            parent_world=fk[elbow].copy();parent_world[:3,:3]=lower
            delta=local_delta_for_world(desired_all[i],parent_world,a['rest_rotation'][wrist]);rv=Rotation.from_quat(delta).as_rotvec();angle=np.linalg.norm(rv)
            if angle>np.deg2rad(75):delta=Rotation.from_rotvec(rv*(np.deg2rad(75)/angle)).as_quat()
            a['local_rotation_delta'][i,wrist]=delta;valid[i,wrist]=True;confidence[i,wrist]=c[i]
        a['local_rotation_delta'][:,wrist],valid[:,wrist],confidence[:,wrist],_=rotations(a['local_rotation_delta'][:,wrist],valid[:,wrist],confidence[:,wrist],t,bridge=.2,fade=.12,tau=.045,max_speed=720)
        # Finger directions live in the observed palm frame. Transfer them
        # to the solved rig palm, then convert each primary bone locally once.
        # This can straighten or splay a finger from its curved rig rest pose.
        wrist_world=[]
        for i in range(n):
            fk=evaluate(work,i);wrist_world.append(rotation(fk[wrist]))
        wrist_world=np.array(wrist_world);palm_world=wrist_world@rotation(rest[wrist]).T@reference
        for finger,offset in [('thumb',1),('index',5),('middle',9),('ring',13),('pinky',17)]:
            parent_world=wrist_world.copy()
            for k in (1,2,3):
                j=ids[f'{finger}_{k:02d}_{side}'];direction=lm[:,offset+k]-lm[:,offset+k-1]
                relative=np.einsum('nji,nj->ni',observed_frames,unit(direction));good=np.flatnonzero(detected)
                if not len(good):continue
                for d in range(3):relative[:,d]=np.interp(t,t[good],relative[good,d])
                relative=unit(gaussian_filter1d(relative,max(.055*fps,.01),axis=0,mode='nearest'))
                target=np.einsum('nij,nj->ni',palm_world,relative)
                base=parent_world@quat_matrix(a['rest_rotation'][j])
                if k<3:axis=a['rest_translation'][ids[f'{finger}_{k+1:02d}_{side}']]
                else:axis=rotation(rest[j]).T@(rest[j,:3,3]-rest[ids[f'{finger}_02_{side}'],:3,3])
                current=np.einsum('nij,j->ni',base,axis)
                rest_direction=rotation(rest[j])@axis
                inward=reference[:,2]*(1 if side=='r' else -1)
                hinge_local=rotation(rest[j]).T@unit(np.cross(rest_direction,inward))
                hinge=np.einsum('nij,j->ni',base,hinge_local)
                change,limited=finger_rotation(current,target,hinge,mcp=k==1,thumb=finger=='thumb');anatomical_fingers+=int(limited.sum())
                delta=np.swapaxes(base,-1,-2)@change@base
                values=Rotation.from_matrix(delta).as_quat()
                values,fv,fc,estimated=rotations(values,v,c,t,bridge=.2,fade=.12,tau=.04,max_speed=1000)
                bounded_fingers+=int(estimated.sum());a['local_rotation_delta'][:,j]=values
                parent_world=base@Rotation.from_quat(values).as_matrix()
                valid[:,j]=fv;confidence[:,j]=fc
    return {'indices':claimed,'rotations':a['local_rotation_delta'],'validity':valid,'confidence':confidence,'provenance':valid.astype(np.uint8)*2,'diagnostics':{'policy':'calibrated palm-plane forearm orientation; local observed phalanx directions','repaired_samples':repaired,'bounded_palm_samples':bounded_palms,'bounded_finger_samples':bounded_fingers,'finger_rate_fallback_deg_s':1000,'anatomical_finger_samples':anatomical_fingers,'finger_dofs':'MCP curl and bounded splay; PIP/DIP hinge','palm_rate_fallback_deg_s':900,'arm_policy':'elbow bend plane and palm lateral axis; joint centres preserved','quality_accepted':False}}
