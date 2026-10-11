"""Image-conditioned retargeting: one calibrated trajectory, one rotation solve.

Image-plane landmarks determine visible motion. Tracker depth supplies only a
smoothed sign/depth prior. World-delta rotations are converted to local channels
once, after the hierarchy has been solved; there is no downstream arm swivel.
"""
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d,median_filter
from scipy.spatial.transform import Rotation
from xms.animation.bundle import neutral
from xms.animation.fk import evaluate,quat_matrix
from xms.animation.validate import validate
from .calibration import calibrate


def unit(v):return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-9)
def mv(r,v):return np.einsum('...ij,...j->...i',r,v)
def basis(x,y):
    y=unit(y);x=unit(x-np.sum(x*y,axis=-1,keepdims=True)*y);return np.stack((x,y,np.cross(x,y)),axis=-1)
def swing(a,b):
    a,b=np.broadcast_arrays(unit(a),unit(b));axis=np.cross(a,b);angle=np.arctan2(np.linalg.norm(axis,axis=-1),np.sum(a*b,axis=-1));return Rotation.from_rotvec(unit(axis)*angle[...,None]).as_matrix()
def blend(a,b,w):
    relative=Rotation.from_matrix(np.swapaxes(a,-1,-2)@b).as_rotvec();return a@Rotation.from_rotvec(relative*np.asarray(w)[...,None]).as_matrix()
def smooth(x,tau,fps):return gaussian_filter1d(x,max(tau*fps,.01),axis=0,mode='nearest')
def rotation_smooth(r,fps):
    q=Rotation.from_matrix(r).as_quat()
    for i in range(1,len(q)):
        if q[i]@q[i-1]<0:q[i]*=-1
    return Rotation.from_quat(unit(smooth(q,.035,fps))).as_matrix()


def solve(observations,timeline,profile,rig):
    meta,raw=observations;idx=np.array(timeline['output_observation_indices']);times=np.array(timeline['output_times_s']);n=len(times);fps=1/np.median(np.diff(times)) if n>1 else 24
    b=neutral(profile,rig,times,np.array(timeline['source_time_map']));a=b.arrays;names=profile['joint_names'];lookup={s:i for i,s in enumerate(names)};rest=evaluate(b,0);H={s:rest[j,:3,3] for s,j in lookup.items()};R={s:Rotation.from_matrix(rest[j,:3,:3]/np.linalg.norm(rest[j,:3,:3],axis=0)).as_matrix() for s,j in lookup.items()}
    valid=raw['landmark_validity'][idx].copy();image=raw['image_landmarks_raw'][idx].copy();world=raw['world_landmarks_raw'][idx].copy()*[1,-1,-1];image_conf=np.minimum(raw['visibility'][idx],raw['presence'][idx]);aspect=meta['width']/meta['height']
    for j in range(33):
        good=np.flatnonzero(valid[:,j])
        if not len(good):continue
        for values in (image,world):
            for d in range(3):values[:,j,d]=np.interp(times,times[good],values[good,j,d])
    image=smooth(median_filter(image,size=(3,1,1)),.025,fps);world=smooth(median_filter(world,size=(3,1,1)),.07,fps)
    xy=image[:,:,:2]*[aspect,-1];hipxy=xy[:,[23,24]].mean(axis=1)
    # Select a neutral calibration interval only when the image actually has
    # straight legs and the rest arm directions; never zero an arbitrary pose.
    errors=[]
    for side,s,e,w,h,k,f in [('l',11,13,15,23,25,27),('r',12,14,16,24,26,28)]:
        for p,q,bone,child in [(s,e,'upperarm','lowerarm'),(e,w,'lowerarm','hand'),(h,k,'thigh','calf'),(k,f,'calf','foot')]:
            target=unit((H[child+'_'+side]-H[bone+'_'+side])[:2]);errors.append(np.degrees(np.arccos(np.clip(unit(xy[:,q]-xy[:,p])@target,-1,1))))
    eligible=(np.max(errors,axis=0)<22)&valid[:,[11,12,13,14,15,16,23,24,25,26,27,28]].all(axis=1)
    from .contacts import spans
    windows=[(s,e) for s,e in spans(eligible,2) if times[e-1]-times[s]>=.10]
    chosen=np.arange(windows[0][0],min(windows[0][1],windows[0][0]+max(3,round(.25*fps)))) if windows else np.array([],int)
    # Experiment switch: hands and head read the same (then empty) neutral samples.
    neutral_enabled=os.environ.get('XMS_NEUTRAL_CALIBRATION','1')!='0'
    if not neutral_enabled:chosen=np.array([],int)
    calibration=calibrate(observations,timeline,profile,rig);calibrated=len(chosen)>0
    torso_height=np.linalg.norm(xy[:,[11,12]].mean(axis=1)-hipxy,axis=1);target_height=np.linalg.norm((H['upperarm_l']+H['upperarm_r']-H['thigh_l']-H['thigh_r'])/2)
    scale=target_height/max(float(np.median(torso_height[chosen] if calibrated else torso_height)),.05)
    P=np.zeros_like(world);P[:,:,:2]=(xy-hipxy[:,None])*scale;P[:,:,2]=world[:,:,2]-world[:,[23,24],2].mean(axis=1)[:,None]
    # A neutral window calibrates per-landmark depth bias instead of treating
    # the tracker's asymmetric neutral hips/shoulders as a real body rotation.
    mapping={11:'upperarm_l',12:'upperarm_r',13:'lowerarm_l',14:'lowerarm_r',15:'hand_l',16:'hand_r',23:'thigh_l',24:'thigh_r',25:'calf_l',26:'calf_r',27:'foot_l',28:'foot_r',29:'foot_l',30:'foot_r',31:'ball_l',32:'ball_r'}
    midrest=(H['thigh_l']+H['thigh_r'])/2
    if calibrated:
        for j,bone in mapping.items():P[:,j,2]+=H[bone][2]-midrest[2]-np.median(P[chosen,j,2])
    P[:,:,2]=smooth(P[:,:,2],.09,fps)
    # Limb depth is constrained by its calibrated image length. Keep image
    # positions intact and avoid per-frame IK choices that change visible arms.
    for s,e,w,extras in [(11,13,15,[17,19,21]),(12,14,16,[18,20,22]),(23,25,27,[29,31]),(24,26,28,[30,32])]:
        for p,q,tail in [(s,e,[w]+extras),(e,w,extras)]:
            v=P[:,q]-P[:,p];length=np.linalg.norm(v[:,:2],axis=1);ref=float(np.median(np.linalg.norm(v[chosen],axis=1))) if calibrated else float(np.percentile(length,90));ref=max(ref,float(np.percentile(length,95)))
            needed=np.sqrt(np.maximum((ref*.97)**2-length**2,0));prior=1 if q in (13,14,15,16,25,26) else -1
            sign=np.sign(smooth(v[:,2],.10,fps)+prior*.025);sign[sign==0]=prior;depth=smooth(sign*np.maximum(np.abs(v[:,2]),needed),.05,fps);delta=depth-v[:,2]
            for j in [q]+tail:P[:,j,2]+=delta
    ident=np.broadcast_to(np.eye(3),(n,3,3)).copy();D={}
    def calibrated_frame(f,reference):
        origin=Rotation.from_matrix(f[chosen]).mean().as_matrix() if calibrated else reference
        return rotation_smooth(f@origin.T,fps)
    hip=P[:,23]-P[:,24];shoulder=P[:,11]-P[:,12];up=P[:,[11,12]].mean(axis=1)-P[:,[23,24]].mean(axis=1)
    restup=(H['upperarm_l']+H['upperarm_r'])/2-midrest
    D['pelvis']=calibrated_frame(basis(hip,unit(up)+[0,1,0]),basis(H['thigh_l']-H['thigh_r'],unit(restup)+[0,1,0]))
    chest=calibrated_frame(basis(shoulder,up),basis(H['upperarm_l']-H['upperarm_r'],restup))
    for k,name in enumerate(['spine_01','spine_02','spine_03','spine_04','spine_05']):D[name]=blend(D['pelvis'],chest,(k+1)/5)
    # Head from pose is a fallback. Full pipeline uses calibrated face rotation.
    D['neck_01']=chest;D['neck_02']=chest;D['head']=chest
    for side,s,e,w,h,k,f in [('l',11,13,15,23,25,27),('r',12,14,16,24,26,28)]:
        for prefix,p,q,parent in [('clavicle',s,e,'spine_05'),('upperarm',s,e,'clavicle_'+side),('lowerarm',e,w,'upperarm_'+side),('thigh',h,k,'pelvis'),('calf',k,f,'thigh_'+side)]:
            name=prefix+'_'+side
            if prefix=='clavicle':
                restarm=unit(H['lowerarm_'+side]-H['upperarm_'+side]);target=unit(P[:,e]-P[:,s])
                if calibrated:target=mv(swing(unit(np.median(target[chosen],axis=0)),restarm),target)
                elevation=swing(mv(chest,restarm),target)
                D[name]=blend(ident,elevation,.22)@chest
                continue
            child={'upperarm':'lowerarm','lowerarm':'hand','thigh':'calf','calf':'foot'}[prefix]+'_'+side;restvec=unit(H[child]-H[name]);target=unit(P[:,q]-P[:,p])
            if calibrated:target=mv(swing(unit(np.median(target[chosen],axis=0)),restvec),target)
            # Calf axial orientation is unobserved. Keep the pelvis facing
            # prior: inheriting a folded thigh's swing can flip the knee frame
            # at an antiparallel leg configuration while the foot barely moves.
            orientation_parent='pelvis' if prefix=='calf' else parent
            D[name]=swing(mv(D[orientation_parent],restvec),target)@D[orientation_parent]
            D[name]=rotation_smooth(D[name],fps)
        D['hand_'+side]=D['lowerarm_'+side]
        # Yaw from heel/toe, with neutral bias removed. Sole pitch is governed by
        # support and free-leg motion, not noisy independent ankle/ball aims.
        toe=31 if side=='l' else 32;heel=29 if side=='l' else 30;fv=P[:,toe]-P[:,heel];yaw=np.unwrap(np.arctan2(fv[:,0],fv[:,2]));yaw-=np.median(yaw[chosen]) if calibrated else np.arctan2((H['ball_'+side]-H['foot_'+side])[0],(H['ball_'+side]-H['foot_'+side])[2]);yaw=smooth(yaw,.08,fps)
        # A free foot follows visible heel-to-toe pitch. Only support owns a
        # horizontal sole; keeping every foot level erases ankle articulation.
        restfoot=H['ball_'+side]-H['foot_'+side]
        vertical=(xy[:,toe,1]-xy[:,heel,1])*scale
        pitch=np.arcsin(np.clip(-vertical/np.linalg.norm(restfoot),-.98,.98))
        pitch-=np.arctan2(-restfoot[1],np.linalg.norm(restfoot[[0,2]]))
        other=28 if side=='l' else 27
        clearance=(image[:,other,1]-image[:,f,1])*scale
        free_weight=np.clip((clearance-.025)/.10,0,1)
        pitch=smooth(np.clip(pitch,-.7,1.3)*free_weight,.055,fps)
        D['foot_'+side]=Rotation.from_rotvec(np.array([0,1,0])*yaw[:,None]).as_matrix()@Rotation.from_rotvec(np.array([1,0,0])*pitch[:,None]).as_matrix()
        D['ball_'+side]=D['foot_'+side]
    # Single hierarchical conversion. Unowned bones inherit their parent's
    # world delta, so no parent rotation is accidentally applied twice.
    actual=[]
    for j,name in enumerate(names):
        parent=int(a['parent_indices'][j]);parent_delta=ident if parent<0 else D[names[parent]]
        D.setdefault(name,parent_delta);desired=D[name]@R[name];parent_world=Rotation.from_matrix(a['armature_transform'][:3,:3]).as_matrix() if parent<0 else actual[parent]
        local=quat_matrix(a['rest_rotation'][j]).T@np.swapaxes(parent_world,-1,-2)@desired
        a['local_rotation_delta'][:,j]=Rotation.from_matrix(local).as_quat();actual.append(desired)
        if name in ['pelvis','head','neck_01','neck_02'] or name.startswith(('spine_','clavicle_','upperarm_','lowerarm_','thigh_','calf_','foot_','ball_','hand_')):
            a['rotation_validity'][:,j]=True;a['rotation_confidence'][:,j]=.6;a['rotation_provenance'][:,j]=2
    origin=np.median(hipxy[chosen],axis=0) if calibrated else np.median(hipxy,axis=0);a['root_translation'][:,:2]=(hipxy-origin)*scale;a['root_validity'][:]=True;a['root_confidence'][:]=.5;a['root_provenance'][:]=2
    calibration.update(policy='image_neutral_world_delta_v1',neutral_sample_indices=chosen.tolist(),image_plane_metres_per_normalized_height=float(scale),hip_image_origin=(origin/[aspect,-1]).tolist())
    calibration['reference_camera'].update(metres_per_normalized_image_height=float(scale),image_anchor=(origin/[aspect,-1]).tolist())
    b.metadata.update(observations_source_sha256=meta['source_sha256'],calibration=calibration,solver={'backend':'image_body','version':'1','channels':'body'},image_retarget_diagnostics={'neutral_calibrated':calibrated,'neutral_calibration_enabled':neutral_enabled,'neutral_sample_indices':chosen.tolist(),'image_scale':scale,'depth':'calibrated, length-constrained prior','rotation_policy':'single world-delta hierarchy conversion','quality_accepted':False},limitations=['Fixed-camera, upright single-person retarget; depth remains an estimate.','No verified camera motion: flight is not inferred from shared image translation.','Unobserved finger/pose details remain bounded estimates.'])
    from .grounding import ground
    ground(b,image,valid,scale)
    return validate(b)
