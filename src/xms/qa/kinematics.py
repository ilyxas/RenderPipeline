import numpy as np
from .contract import metric


def rotation_velocity(q,times):
    a=q[:-1];b=q[1:];xyz=a[...,3,None]*b[...,:3]-b[...,3,None]*a[...,:3]-np.cross(a[...,:3],b[...,:3]);w=np.sum(a*b,axis=-1);sign=np.where(w<0,-1.,1.);xyz*=sign[...,None];w*=sign
    sine=np.linalg.norm(xyz,axis=-1);angle=2*np.arctan2(sine,np.clip(w,0,1));axis=xyz/np.maximum(sine[...,None],1e-12)
    return axis*angle[...,None]/np.diff(times)[:,None,None]


def measure_kinematics(bundle,world,timeline,annotations):
    a=bundle.arrays;names=bundle.metadata['joint_names'];times=a['sample_times_s'];q=a['local_rotation_delta'];velocity=rotation_velocity(q,times);valid=a['rotation_validity'][1:]&a['rotation_validity'][:-1];body=[i for i,n in enumerate(names) if n.startswith(('pelvis','spine','neck','head','upperarm_','lowerarm_','thigh_','calf_','foot_')) and not any(x in n for x in ('twist','corrective','_in_','_out_','_fwd_','_bck_','bicep','tricep'))];masks=valid[:,body];speed=np.degrees(np.linalg.norm(velocity[:,body],axis=-1));v=speed[masks]
    acceleration=np.diff(velocity[:,body],axis=0)/((np.diff(times)[1:]+np.diff(times)[:-1])/2)[:,None,None];amask=masks[1:]&masks[:-1];acc=np.degrees(np.linalg.norm(acceleration,axis=-1))[amask]
    jumps=np.degrees(np.linalg.norm(velocity,axis=-1)*np.diff(times)[:,None]);observed=jumps[valid]
    angles=np.degrees(2*np.arccos(np.clip(abs(q[...,3]),0,1)));caps=bundle.metadata['solver']['direction_caps_degrees'];limits=[]
    for j,n in enumerate(names):
        group=next((k for k in caps if n==k or n in (k+'_l',k+'_r')),None)
        cap=80 if n=='head' and bundle.metadata.get('compositor') else caps[group] if group else 95 if n in ('hand_l','hand_r') else 100 if any(n.startswith(x+'_') for x in ['thumb','index','middle','ring','pinky']) and n.split('_')[1] in ('01','02','03') else None
        if cap is not None:limits.extend((angles[:,j][a['rotation_validity'][:,j]]>cap+1e-4).tolist())
    root=a['root_translation'];root_steps=np.linalg.norm(np.diff(root,axis=0),axis=1);absolute=timeline['absolute_start_s']+times;drifts=[];stance_details=[]
    for stance in annotations['stance_intervals']:
        if stance['confidence']<.8:continue
        mask=(absolute>=stance['start_source_s'])&(absolute<stance['end_source_s']);joint='foot_'+stance['side'];j=names.index(joint)
        if mask.sum()<2:continue
        points=world[mask,j,:3,3];drift=float(np.max(np.linalg.norm(points[:,[0,2]]-points[0,[0,2]],axis=1)));drifts.append(drift);stance_details.append({**stance,'bone_proxy':joint,'max_horizontal_drift_m':drift})
    def percentile(x,p):return float(np.percentile(x,p)) if len(x) else None
    metrics={'body_angular_velocity_p95':metric(percentile(v,95),'degrees/s'),'body_angular_acceleration_p95':metric(percentile(acc,95),'degrees/s^2'),'observed_adjacent_rotation_jump_max':metric(float(max(observed)) if len(observed) else None,'degrees'),'root_adjacent_step_max':metric(float(max(root_steps)) if len(root_steps) else None,'m'),'joint_limit_violation_fraction':metric(float(np.mean(limits)) if limits else None,'fraction of scalar-cap samples'),'anatomical_joint_limits':metric(reason='Scalar swing caps are baseline constraints, not validated anatomical joint limits'),'foot_bone_drift_proxy_median':metric(percentile(drifts,50),'m',reason='No high-confidence annotated stance'),'foot_bone_drift_proxy_p95':metric(percentile(drifts,95),'m',reason='No high-confidence annotated stance'),'foot_skating':metric(reason='No verified sole contact geometry; foot-bone drift is only a proxy',unit='m'),'surface_penetration':metric(reason='Surface geometry QA is scheduled later',unit='m'),'root_speed_p95':metric(percentile(root_steps/np.diff(times),95),'m/s')}
    gestures=[]
    for event in annotations['fast_gestures']:
        mask=(absolute>=event['start_source_s'])&(absolute<event['end_source_s']);ii=np.flatnonzero(mask)
        for name in event['joints']:
            j=names.index(name)
            if not len(ii):continue
            distance=np.degrees(2*np.arccos(np.clip(np.abs(q[ii,j]@q[ii[0],j]),0,1)));k=int(np.argmax(distance));gestures.append({'label':event['label'],'joint':name,'amplitude_degrees':float(distance[k]),'peak_source_s':float(absolute[ii[k]])})
    return {'metrics':metrics,'stance_intervals':stance_details,'fast_gesture_reference':gestures,'root_trajectory':{'source_times_s':absolute.tolist(),'translation_m':root.tolist()},'policy':'Derivative metrics exclude missing endpoints; no interpolation. Foot drift is a horizontal bone-head proxy. Scalar caps are not anatomical validation.'}
