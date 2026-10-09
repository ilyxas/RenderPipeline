"""Unfiltered direction baseline. No bpy, old actions, IK, contact or face fusion."""
import numpy as np
from xms.animation.bundle import neutral
from xms.animation.fk import evaluate,trs,quat_matrix,matrix_quat
from xms.animation.sample import continuous
from xms.animation.validate import validate

LIMITATIONS=['Body-only unfiltered baseline; monocular depth and global translation are estimates, not metric ground truth.', 'No temporal optimization, foot anchoring, contact solve or collision correction.', 'Face, gaze, jaw, wrists and fingers are neutral/unobserved; head is a limited body-landmark prior.', 'Occluded/low-confidence segments fall back to local rest; this can cause discontinuities.', 'Root depth is unobserved and fixed; image-plane translation uses shoulder-scale and a clip-median origin.', 'Single-person input only; no temporal identity recovery or shot-cut handling.', 'Preview materials use natural canonical teeth; hair/shadow noise and clipping remain possible.']


def align(a,b,limit):
    a=a/np.linalg.norm(a);b=b/np.linalg.norm(b)
    axis=np.cross(a,b);s=np.linalg.norm(axis);c=float(np.clip(a@b,-1,1))
    angle=min(np.arctan2(s,c),np.deg2rad(limit))
    if s<1e-9:
        if c>0:return np.eye(3)
        axis=np.cross(a,[1,0,0] if abs(a[0])<.9 else [0,1,0]);s=np.linalg.norm(axis)
    axis/=s
    return quat_matrix(np.r_[axis*np.sin(angle/2),np.cos(angle/2)])


def solve(observations,timeline,profile,rig):
    meta,obs=observations;indices=np.array(timeline['output_observation_indices']);times=np.array(timeline['output_times_s'])
    b=neutral(profile,rig,times,timeline['source_time_map']);a=b.arrays;names=profile['joint_names'];lookup={n:i for i,n in enumerate(names)}
    rest=evaluate(b,0);world=obs['world_landmarks_raw'][indices]*[1,-1,-1];image=obs['image_landmarks_raw'][indices];valid=obs['landmark_validity'][indices];confidence=np.minimum(obs['visibility'][indices],obs['presence'][indices])
    aspect=meta['width']/meta['height'];hip=(image[:,23,:2]+image[:,24,:2])/2
    reliable=valid[:,23]&valid[:,24]&valid[:,11]&valid[:,12]
    if not reliable.any():raise ValueError('No reliable torso observations for baseline root estimate')
    shoulder=np.linalg.norm((image[:,11,:2]-image[:,12,:2])*[aspect,1],axis=1)
    target_width=np.linalg.norm(rest[lookup['upperarm_l'],:3,3]-rest[lookup['upperarm_r'],:3,3])
    scale=float(target_width/max(float(np.median(shoulder[reliable])),.05));origin=np.median(hip[reliable],axis=0)
    calibration={'schema_version':'xms.baseline_calibration.v1','image_plane_metres_per_normalized_height':scale,'hip_image_origin':origin.tolist(),'estimation':'clip-median reliable hips and shoulders; no neutral-pose assumption','root_depth':'unobserved_fixed_zero','camera':'orthographic image-plane translation approximation; no metric calibration'}
    for i in range(len(times)):
        if reliable[i]:
            a['root_translation'][i,:2]=(hip[i]-origin)*[aspect,-1]*scale
            a['root_validity'][i]=True;a['root_confidence'][i]=.3*min(confidence[i,[23,24]]);a['root_provenance'][i]=2
    segments={}
    for side,shoulder_id,elbow,wrist,hip_id,knee,ankle,heel,toe in [('l',11,13,15,23,25,27,29,31),('r',12,14,16,24,26,28,30,32)]:
        for bone,child,s,e,cap in [('upperarm','lowerarm',shoulder_id,elbow,160),('lowerarm','hand',elbow,wrist,155),('thigh','calf',hip_id,knee,110),('calf','foot',knee,ankle,145),('foot','ball',heel,toe,70)]:segments[bone+'_'+side]=(child+'_'+side,[s,e],cap)
    for i in range(len(times)):
        matrices=[]
        for j,name in enumerate(names):
            parent=int(rig['parent_indices'][j]);parent_world=rig['armature_transform'] if parent<0 else matrices[parent]
            local=trs(rig['rest_translation'][j],rig['rest_rotation'][j],rig['rest_scale'][j]);base=parent_world@local
            target=None;points=[];cap=0;prior=False
            if name in segments:
                child,points,cap=segments[name];current=base[:3,:3]@rig['rest_translation'][lookup[child]];target=world[i,points[1]]-world[i,points[0]]
            elif name=='pelvis':
                points=[11,12,23,24];cap=35;current=base[:3,:3]@rig['rest_translation'][lookup['spine_01']];target=(world[i,11]+world[i,12]-world[i,23]-world[i,24])/2
            elif name=='head':
                points=[0,7,8];cap=40;prior=True;rest_rot=rest[j,:3,:3]/np.linalg.norm(rest[j,:3,:3],axis=0);current=base[:3,:3]@(rest_rot.T@np.array([0,0,1.]));target=world[i,0]-(world[i,7]+world[i,8])/2
            if target is not None and valid[i,points].all() and np.linalg.norm(target)>1e-6 and np.linalg.norm(current)>1e-6:
                rotation=base[:3,:3]/np.linalg.norm(base[:3,:3],axis=0)
                delta=rotation.T@align(current,target,cap)@rotation
                q=matrix_quat(delta);a['local_rotation_delta'][i,j]=q
                a['rotation_validity'][i,j]=True;a['rotation_confidence'][i,j]=float(min(confidence[i,points]))*(.3 if prior else 1);a['rotation_provenance'][i,j]=2 if prior else 1
                local[:3,:3]=quat_matrix(rig['rest_rotation'][j])@quat_matrix(q)@np.diag(rig['rest_scale'][j]);base=parent_world@local
            matrices.append(base)
    a['local_rotation_delta']=continuous(a['local_rotation_delta'])
    b.metadata.update(solver={'backend':'baseline_body','version':'1','direction_caps_degrees':{'pelvis':35,'head':40,'upperarm':160,'lowerarm':155,'thigh':110,'calf':145,'foot':70}},calibration=calibration,limitations=LIMITATIONS,observations_source_sha256=meta['source_sha256'])
    return validate(b)
