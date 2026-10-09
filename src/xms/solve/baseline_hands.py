import copy
import numpy as np
from xms.animation.fk import evaluate,trs,quat_matrix,matrix_quat
from xms.solve.head import rotation,local_delta_for_world
from xms.solve.baseline_body import align
from xms.animation.sample import slerp


def hand_frame(wrist,middle,index,pinky):
    y=middle-wrist;y=y/max(np.linalg.norm(y),1e-10);x=index-pinky;x-=y*(x@y);x=x/max(np.linalg.norm(x),1e-10);z=np.cross(x,y)
    if np.linalg.norm(z)<.9:raise ValueError('Degenerate hand frame')
    return np.stack([x,y,z],axis=1)


def solve_hands(body,observations,timeline):
    meta,obs=observations;indices=np.array(timeline['output_observation_indices']);names=body.metadata['joint_names'];lookup={n:i for i,n in enumerate(names)}
    neutral=copy.deepcopy(body);neutral.arrays['local_rotation_delta'][:]=[0,0,0,1];neutral.arrays['root_translation'][:]=0;rest=evaluate(neutral,0);work=copy.deepcopy(body)
    claimed=[]
    for side in ('l','r'):claimed+=[lookup['hand_'+side]]+[lookup[f'{f}_{k:02d}_{side}'] for f in ['thumb','index','middle','ring','pinky'] for k in [1,2,3]]
    valid=np.zeros((len(indices),len(names)),bool);confidence=np.zeros_like(valid,dtype=float)
    for i,source_i in enumerate(indices):
        for side_id,side in enumerate(('l','r')):
            if not obs['validity'][source_i,side_id] or obs['confidence'][source_i,side_id]<.2:continue
            landmarks=obs['world_landmarks_raw'][source_i,side_id]*[1,-1,-1]
            j=lookup['hand_'+side]
            try:
                reference=hand_frame(rest[j,:3,3],rest[lookup['middle_01_'+side],:3,3],rest[lookup['index_01_'+side],:3,3],rest[lookup['pinky_01_'+side],:3,3]);observed=hand_frame(landmarks[0],landmarks[9],landmarks[5],landmarks[17])
            except ValueError:continue
            world=evaluate(work,i);parent=work.arrays['parent_indices'][j];target=observed@reference.T@rotation(rest[j]);q=local_delta_for_world(target,world[parent],work.arrays['rest_rotation'][j]);angle=2*np.arccos(np.clip(q[3],-1,1))
            if angle>np.deg2rad(95):q=slerp(np.array([0.,0,0,1]),q,np.deg2rad(95)/angle)
            work.arrays['local_rotation_delta'][i,j]=q;valid[i,j]=True;confidence[i,j]=obs['confidence'][source_i,side_id]
            world=evaluate(work,i);actual_frame=rotation(world[j])@rotation(rest[j]).T@reference
            # Map observed finger directions into the solved wrist frame. This
            # keeps wrist/forearm ownership local instead of applying world motion twice.
            for finger,offset in [('thumb',1),('index',5),('middle',9),('ring',13),('pinky',17)]:
                for k in [1,2,3]:
                    bone=lookup[f'{finger}_{k:02d}_{side}'];parent=int(work.arrays['parent_indices'][bone]);world=evaluate(work,i)
                    local=trs(work.arrays['rest_translation'][bone],work.arrays['rest_rotation'][bone],work.arrays['rest_scale'][bone]);base=world[parent]@local
                    if k<3:axis=work.arrays['rest_translation'][lookup[f'{finger}_{k+1:02d}_{side}']]
                    else:axis=rotation(rest[bone]).T@(rest[bone,:3,3]-rest[lookup[f'{finger}_02_{side}'],:3,3])
                    target=actual_frame@observed.T@(landmarks[offset+k]-landmarks[offset+k-1]);current=base[:3,:3]@axis
                    if np.linalg.norm(target)<1e-7 or np.linalg.norm(current)<1e-7:continue
                    r=rotation(base);delta=r.T@align(current,target,100)@r;work.arrays['local_rotation_delta'][i,bone]=matrix_quat(delta);valid[i,bone]=True;confidence[i,bone]=obs['confidence'][source_i,side_id]
    return {'indices':claimed,'rotations':work.arrays['local_rotation_delta'],'validity':valid,'confidence':confidence}
