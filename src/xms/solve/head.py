import numpy as np
from xms.animation.fk import evaluate,quat_matrix,matrix_quat
from xms.animation.sample import slerp


def rotation(matrix):
    u,_,vt=np.linalg.svd(matrix[:3,:3]);r=u@vt
    if np.linalg.det(r)<0:u[:,-1]*=-1;r=u@vt
    return r


def local_delta_for_world(target,parent_world,rest_rotation):
    return matrix_quat((rotation(parent_world)@quat_matrix(rest_rotation)).T@target)


def solve_head(body,face,timeline):
    m,a=face;indices=np.array(timeline['output_observation_indices']);j=body.metadata['joint_names'].index('head');parent=body.arrays['parent_indices'][j]
    # Face canonical neutral faces +Z; matrix is camera x-right/y-up/z-toward-viewer.
    from xms.animation.bundle import AnimationBundle
    import copy
    rest=copy.deepcopy(body);rest.arrays['local_rotation_delta'][:]=[0,0,0,1];rest.arrays['root_translation'][:]=0
    neutral_rotation=rotation(evaluate(rest,0)[j]);values=body.arrays['local_rotation_delta'][:,j].copy();valid=a['validity'][indices]&(a['confidence'][indices]>=.2)
    for i,source_i in enumerate(indices):
        if not valid[i]:continue
        parent_world=evaluate(body,i)[parent];target=rotation(a['face_matrix_raw'][source_i])@neutral_rotation
        q=local_delta_for_world(target,parent_world,body.arrays['rest_rotation'][j]);angle=2*np.arccos(np.clip(q[3],-1,1))
        if angle>np.deg2rad(80):q=slerp(np.array([0.,0,0,1]),q,np.deg2rad(80)/angle)
        values[i]=q
    return {'index':j,'rotations':values,'face_validity':valid,'confidence':a['confidence'][indices]}
