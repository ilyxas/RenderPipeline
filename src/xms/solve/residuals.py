"""Coupled Stage 8 window objective; no contacts or collision terms."""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from xms.animation.fk import quat_matrix
from xms.qa.pose import project


def temporal_weights(initial, confidence):
    speed=np.linalg.norm(relative_steps(initial),axis=-1)
    return .2+.8/(1+(speed/np.deg2rad(12))**2)+.35*(1-np.minimum(confidence[:-1],confidence[1:]))


def relative_steps(rotvec):
    shape=rotvec.shape
    rotations=Rotation.from_rotvec(rotvec.reshape(-1,3)).as_matrix().reshape(shape[:-1]+(3,3))
    relative=np.swapaxes(rotations[:-1],-1,-2)@rotations[1:]
    return Rotation.from_matrix(relative.reshape(-1,3,3)).as_rotvec().reshape((shape[0]-1,)+shape[1:])


def forward(x,c):
    """Vectorized canonical rest @ delta FK over only the relevant ancestors."""
    state=x.reshape(c['n'],c['width']);v=c['initial']+state[:,2:].reshape(c['n'],c['k'],3)
    rotations=Rotation.from_rotvec(v.reshape(-1,3)).as_matrix().reshape(c['n'],c['k'],3,3)
    transforms={}
    for j in c['closure']:
        local=np.repeat(c['rest'][j][None],c['n'],axis=0)
        if j in c['variable']:local[:,:3,:3]=c['rest_rotation'][j]@rotations[:,c['variable'][j]]@np.diag(c['scale'][j])
        else:local[:,:3,:3]=c['fixed_rotation'][:,j]
        parent=int(c['parents'][j]);transforms[j]=(c['armature'] if parent<0 else transforms[parent])@local
    positions={j:m[:,:3,3]+c['root_initial']+np.pad(state[:,:2],((0,0),(0,1))) for j,m in transforms.items()}
    return positions,v,c['root_initial'][:,:2]+state[:,:2]


def residual(x,c,terms=False):
    positions,v,root=forward(x,c);parts={};data=[];directions=[]
    for j,target,valid,confidence in c['points']:
        pred=project(positions[j],c['camera'])
        data.append(((pred-target)*[c['camera']['display_aspect'],1]*np.sqrt(confidence)[:,None])[valid].ravel())
    for j,child,target,valid,confidence in c['directions']:
        d=positions[child]-positions[j];d/=np.maximum(np.linalg.norm(d,axis=-1,keepdims=True),1e-10)
        directions.append(((d-target)*np.sqrt(confidence)[:,None]*.12)[valid].ravel())
    parts['reprojection']=np.concatenate(data);parts['soft_3d_prior']=np.concatenate(directions)
    delta=x.reshape(c['n'],c['width'])[:,2:].reshape(c['n'],c['k'],3)
    parts['initializer_anchor']=(delta*(.018+.055*(1-c['confidence']))[...,None]).ravel()
    steps=relative_steps(v)/(np.diff(c['times'])[:,None,None]*24)
    parts['velocity']=(steps*np.sqrt(c['weights'])[...,None]*.055).ravel()
    acceleration=np.diff(steps,axis=0)/(((np.diff(c['times'])[1:]+np.diff(c['times'])[:-1])/2)[:,None,None]*24)
    parts['acceleration']=(acceleration*np.sqrt(np.minimum(c['weights'][:-1],c['weights'][1:]))[...,None]*.24).ravel()
    parts['joint_limit']=(np.maximum(np.linalg.norm(v,axis=-1)-c['caps'],0)*8).ravel()
    parts['root_anchor']=((root-c['root_initial'][:,:2])*.04).ravel()
    root_steps=np.diff(root,axis=0)/(np.diff(c['times'])[:,None]*24)
    parts['root_velocity']=(root_steps*.08).ravel()
    parts['root_acceleration']=(np.diff(root_steps,axis=0)*.55).ravel()
    return {key:float(value@value) for key,value in parts.items()} if terms else np.concatenate(list(parts.values()))


def sparsity(c):
    rows=[];w=c['width'];k=c['k'];n=c['n']
    def point_columns(j,t):
        columns=[t*w,t*w+1];parent=int(c['parents'][j])
        while parent>=0:
            if parent in c['variable']:columns.extend(t*w+2+c['variable'][parent]*3+np.arange(3))
            parent=int(c['parents'][parent])
        return columns
    for j,target,valid,confidence in c['points']:
        for t in np.flatnonzero(valid):rows.extend([point_columns(j,t)]*2)
    for j,child,target,valid,confidence in c['directions']:
        for t in np.flatnonzero(valid):rows.extend([sorted(set(point_columns(j,t)+point_columns(child,t)))]*3)
    for t in range(n):
        for j in range(k):rows.extend([list(t*w+2+j*3+np.arange(3))]*3)
    for order in (1,2):
        for t in range(n-order):
            for j in range(k):rows.extend([[int((t+o)*w+2+j*3+d) for o in range(order+1) for d in range(3)]]*3)
    for t in range(n):
        for j in range(k):rows.append(list(t*w+2+j*3+np.arange(3)))
    for order in (0,1,2):
        for t in range(n-order):
            for d in range(2):rows.append([(t+o)*w+d for o in range(order+1)])
    matrix=lil_matrix((len(rows),n*w),dtype=np.int8)
    for i,columns in enumerate(rows):matrix[i,columns]=1
    return matrix.tocsr()
