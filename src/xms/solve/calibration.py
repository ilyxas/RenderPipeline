"""Deterministic reliable-window calibration; no neutral first-frame assumption."""
import hashlib,json
import numpy as np
from xms.animation.bundle import neutral
from xms.animation.fk import evaluate,matrix_quat

PAIRS=[('upperarm_l','lowerarm_l',11,13),('lowerarm_l','hand_l',13,15),('upperarm_r','lowerarm_r',12,14),('lowerarm_r','hand_r',14,16),('thigh_l','calf_l',23,25),('calf_l','foot_l',25,27),('thigh_r','calf_r',24,26),('calf_r','foot_r',26,28)]


def array_digest(arrays):
    h=hashlib.sha256()
    for key in sorted(arrays):
        a=np.ascontiguousarray(arrays[key]);h.update(key.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def calibrate(observations,timeline,profile,rig):
    meta,obs=observations;indices=np.array(timeline['output_observation_indices']);image=obs['image_landmarks_raw'][indices];world=obs['world_landmarks_raw'][indices]*[1,-1,-1];valid=obs['landmark_validity'][indices];confidence=np.minimum(obs['visibility'][indices],obs['presence'][indices]);names=profile['joint_names'];lookup={n:i for i,n in enumerate(names)}
    rest=evaluate(neutral(profile,rig,[0.]),0);aspect=meta['width']/meta['height'];hips=(image[:,23,:2]+image[:,24,:2])/2
    reliable=valid[:,[11,12,23,24]].all(axis=1);score=np.min(confidence[:,[11,12,23,24]],axis=1)
    if not reliable.any():raise ValueError('Calibration unavailable: no reliable torso samples')
    selected=[];windows=[]
    for group in np.array_split(np.arange(len(indices)),min(4,len(indices))):
        candidates=group[reliable[group]]
        if not len(candidates):continue
        chosen=int(candidates[np.argmax(score[candidates])]);selected.append(chosen)
        windows.append({'start_s':float(timeline['output_times_s'][group[0]]),'end_s':float(timeline['output_times_s'][group[-1]]),'representative_sample':chosen,'torso_confidence':float(score[chosen])})
    ratios=[];neutral_candidates=[]
    for i in selected:
        angles=[]
        for start,end,s,e in PAIRS:
            if not valid[i,[s,e]].all():continue
            target=rest[lookup[end],:3,3]-rest[lookup[start],:3,3];projected=(image[i,e,:2]-image[i,s,:2])*[aspect,1];length=np.linalg.norm(projected)
            if length>.02:ratios.append(float(np.linalg.norm(target)/length))
            observed=world[i,e]-world[i,s]
            if np.linalg.norm(observed)>1e-7:angles.append(float(np.degrees(np.arccos(np.clip(np.dot(observed,target)/(np.linalg.norm(observed)*np.linalg.norm(target)),-1,1)))))
        if len(angles)>=6 and np.median(angles)<15:neutral_candidates.append(i)
    if not ratios:raise ValueError('Calibration unavailable: no reliable segment length ratios')
    # Fixed target skeletal height; missing feet cannot be mistaken for full body height.
    target_height=float(rest[lookup['head'],1,3]-min(rest[lookup['foot_l'],1,3],rest[lookup['foot_r'],1,3]))
    heights=[]
    for i in selected:
        if valid[i,[0,27,28]].all():heights.append(float(max(image[i,27,1],image[i,28,1])-image[i,0,1]))
    segment_scale=float(np.median(ratios));height_scale=target_height/float(np.median(heights)) if heights and min(heights)>.1 else None
    # Fixed equal regularization weights selected before baseline evaluation.
    scale=segment_scale if height_scale is None else .5*segment_scale+.5*height_scale
    origin=np.median(hips[selected],axis=0);lateral=np.median(world[selected,23]-world[selected,24],axis=0);yaw=float(np.arctan2(-lateral[2],lateral[0]));orientation=[0,float(np.sin(yaw/2)),0,float(np.cos(yaw/2))]
    qconfidence=float(np.mean(score[selected]))*(.8 if neutral_candidates else .45)*(.65 if height_scale is None else 1.)
    root=rest[profile['root_carrier_index'],:3,3]
    return {'schema_version':'xms.calibration.v1','policy':'reliable_windows_rest_regularized_v1','source_sha256':meta['source_sha256'],'observations_digest':array_digest(obs),'character_profile_hash':profile['profile_hash'],'output_times_s':timeline['output_times_s'],'selected_windows':windows,'neutral_sample_indices':neutral_candidates,'first_frame_required_neutral':False,'confidence':qconfidence,'confidence_limitations':['Monocular depth and physical camera are not identifiable','No neutral pose: rest geometry regularization, reduced confidence'] if not neutral_candidates else ['Neutral candidates are tracker-based proxies, not ground truth'],'fixed_target_skeleton_height_m':target_height,'image_plane_metres_per_normalized_height':scale,'segment_scale_estimate':segment_scale,'full_height_scale_estimate':height_scale,'hip_image_origin':origin.tolist(),'global_orientation_prior_xyzw':orientation,'orientation_application':'Saved camera-frame prior for temporal solver; baseline directions already camera-aligned, so do not apply twice','root_depth':'unobserved_fixed_zero','reference_camera':{'model':'orthographic_prior','metres_per_normalized_image_height':scale,'display_aspect':aspect,'image_anchor':origin.tolist(),'world_anchor_m':root.tolist(),'world_to_camera_rotation':np.eye(3).tolist(),'frozen':True,'limitations':'Camera is a rest-regularized prior, not an optimized projection or metric camera calibration'}}


def validate_calibration(c,observations,timeline,profile):
    if c['schema_version']!='xms.calibration.v1' or c['character_profile_hash']!=profile['profile_hash'] or c['source_sha256']!=observations[0]['source_sha256'] or c['observations_digest']!=array_digest(observations[1]) or c['output_times_s']!=timeline['output_times_s']:raise ValueError('Calibration input/profile/timeline mismatch')
    if not np.isfinite(c['image_plane_metres_per_normalized_height']) or c['image_plane_metres_per_normalized_height']<=0:raise ValueError('Invalid calibration scale')
    return c
