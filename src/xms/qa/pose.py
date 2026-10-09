import numpy as np
from xms.animation.fk import evaluate
from .contract import metric

JOINTS={'left_shoulder':('upperarm_l',11),'right_shoulder':('upperarm_r',12),'left_elbow':('lowerarm_l',13),'right_elbow':('lowerarm_r',14),'left_wrist':('hand_l',15),'right_wrist':('hand_r',16),'left_hip':('thigh_l',23),'right_hip':('thigh_r',24),'left_knee':('calf_l',25),'right_knee':('calf_r',26),'left_ankle':('foot_l',27),'right_ankle':('foot_r',28)}


def project(points,camera):
    delta=(points-np.array(camera['world_anchor_m']))@np.array(camera['world_to_camera_rotation']).T;s=camera['metres_per_normalized_image_height'];u,v=camera['image_anchor'];aspect=camera['display_aspect']
    return np.stack([u+delta[...,0]/(s*aspect),v-delta[...,1]/s],axis=-1)


def measure_pose(bundle,observations,timeline,calibration,annotations):
    meta,a=observations;names=bundle.metadata['joint_names'];indices=np.array(timeline['output_observation_indices']);world=np.array([evaluate(bundle,i) for i in range(len(indices))]);camera=calibration['reference_camera'];aspect=camera['display_aspect'];all_errors=[];per_joint={};heights=[]
    # Visible skeleton span is a source proxy. Full person height unavailable on cropped holdout.
    for source_i in indices:
        visible=a['landmark_validity'][source_i]
        if visible.any():heights.append(float(np.ptp(a['image_landmarks_raw'][source_i,visible,1])))
    normalizer=float(np.percentile(heights,95)) if heights else 0
    for role,(joint,landmark) in JOINTS.items():
        if role not in annotations['confident_joints']:continue
        predicted=project(world[:,names.index(joint),:3,3],camera);target=a['image_landmarks_raw'][indices,landmark,:2];valid=a['landmark_validity'][indices,landmark]
        values=np.linalg.norm((predicted-target)*[aspect,1],axis=1)[valid]/max(normalizer,1e-8)
        if len(values):per_joint[role]={'median':float(np.median(values)),'p95':float(np.percentile(values,95)),'samples':len(values)};all_errors.extend(values.tolist())
    metrics={'normalized_reprojection_median':metric(float(np.median(all_errors)) if all_errors else None,'visible skeletal height',reason='No reliable annotated joints'),'normalized_reprojection_p95':metric(float(np.percentile(all_errors,95)) if all_errors else None,'visible skeletal height',reason='No reliable annotated joints'),'metric_depth_error':metric(reason='No metric 3D ground truth',unit='m')}
    return {'metrics':metrics,'per_joint':per_joint,'normalizer_source_image_height_fraction':normalizer,'normalizer_policy':'p95 visible skeletal vertical span; proxy, not person mesh height','full_body_normalization_available':annotations.get('full_body_visible',False),'reference_camera':camera},world
