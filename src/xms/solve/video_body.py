"""Stable video retarget: torso orientation, image-guided two-bone arms and clearance."""
import copy
import numpy as np
from scipy.ndimage import median_filter
from xms.animation.fk import evaluate,quat_matrix,matrix_quat,trs
from xms.animation.validate import validate
from .baseline_body import solve as baseline,align
from .head import rotation,local_delta_for_world
from .stability import stabilize_bundle,rotations


def frame(lateral,up):
    up=up/max(np.linalg.norm(up),1e-8);x=lateral-up*np.dot(lateral,up)
    if np.linalg.norm(x)<1e-6:raise ValueError('Degenerate torso frame')
    x/=np.linalg.norm(x);return np.stack((x,up,np.cross(x,up)),axis=1)


def set_direction(b,i,j,child,target):
    a=b.arrays;world=evaluate(b,i);parent=int(a['parent_indices'][j]);base=(a['armature_transform'] if parent<0 else world[parent])@trs(a['rest_translation'][j],a['rest_rotation'][j],a['rest_scale'][j])
    current=base[:3,:3]@a['rest_translation'][child];r=rotation(base)
    a['local_rotation_delta'][i,j]=matrix_quat(r.T@align(current,target,180)@r)


def two_bone(shoulder,target,hint,l1,l2):
    delta=target-shoulder;distance=np.linalg.norm(delta);axis=delta/max(distance,1e-8)
    distance=np.clip(distance,abs(l1-l2)+.005,l1+l2-.005);target=shoulder+axis*distance
    along=(l1*l1-l2*l2+distance*distance)/(2*distance)
    bend=hint-shoulder; bend-=axis*np.dot(bend,axis)
    if np.linalg.norm(bend)<.005:
        bend=np.array([0.,0,1.]);bend-=axis*np.dot(bend,axis)
    if np.linalg.norm(bend)<1e-6:bend=np.cross(axis,[0.,1,0])
    elbow=shoulder+axis*along+bend/max(np.linalg.norm(bend),1e-8)*np.sqrt(max(0,l1*l1-along*along))
    return elbow,target


def clearance(point,bottom,top,margin=.18):
    axis=top-bottom;t=np.clip(np.dot(point-bottom,axis)/max(axis@axis,1e-8),0,1);center=bottom+t*axis
    # A visible wrist projected over the torso has an ambiguous depth. Choose
    # the camera-facing surface, rather than inventing a path through the body.
    dx=point[0]-center[0];dy=point[1]-center[1]
    radial_sq=dx*dx+dy*dy
    if radial_sq<margin*margin:
        point=point.copy();point[2]=max(point[2],center[2]+np.sqrt(max(0,margin*margin-radial_sq)))
    return point


def solve(observations,timeline,profile,rig):
    meta,raw=observations;clean={k:v.copy() for k,v in raw.items()};times=raw['source_times_s'];idx=np.asarray(timeline['output_observation_indices']);estimated=0
    # Keep raw observations immutable. Bridge short landmark dropouts before FK.
    for j in range(33):
        valid=raw['landmark_validity'][:,j];known=np.flatnonzero(valid)
        if not len(known):continue
        for key in ('image_landmarks_raw','world_landmarks_raw'):
            values=clean[key][:,j]
            for d in range(3):values[:,d]=np.interp(times,times[known],values[known,d])
            clean[key][:,j]=median_filter(values,size=(3,1),mode='nearest')
        for i in np.flatnonzero(~valid):
            left=known[known<i];right=known[known>i]
            if len(left) and len(right) and times[right[0]]-times[left[-1]]<=.3:
                clean['landmark_validity'][i,j]=True;clean['visibility'][i,j]=clean['presence'][i,j]=.25;estimated+=1
    b=baseline((meta,clean),timeline,profile,rig);a=b.arrays;names=b.metadata['joint_names'];lookup={n:j for j,n in enumerate(names)}
    # Calibration must refer to the original observations, not the repaired copy.
    from .calibration import calibrate
    b.metadata['calibration']=calibrate(observations,timeline,profile,rig)
    repairs=stabilize_bundle(b)
    restb=copy.deepcopy(b);restb.arrays['local_rotation_delta'][:]=[0,0,0,1];restb.arrays['root_translation'][:]=0;rest=evaluate(restb,0)
    world=clean['world_landmarks_raw'][idx]*[1,-1,-1];image=clean['image_landmarks_raw'][idx];valid=clean['landmark_validity'][idx];t=a['sample_times_s'];aspect=meta['width']/meta['height']
    pelvis=lookup['pelvis'];spine=lookup['spine_03'];neck=lookup['neck_01']
    rest_up=rest[neck,:3,3]-rest[pelvis,:3,3];rest_lateral=rest[lookup['upperarm_l'],:3,3]-rest[lookup['upperarm_r'],:3,3];reference=frame(rest_lateral,rest_up)
    torso_q=np.tile([0.,0,0,1.],(len(t),1));torso_v=valid[:,[11,12,23,24]].all(axis=1)
    for i in np.flatnonzero(torso_v):
        up=(world[i,11]+world[i,12]-world[i,23]-world[i,24])/2
        try:torso_q[i]=matrix_quat(frame(world[i,23]-world[i,24],up)@reference.T)
        except ValueError:torso_v[i]=False
    torso_q,torso_v,torso_c,_=rotations(torso_q,torso_v,torso_v.astype(float)*.7,t,tau=.07)
    for i in np.flatnonzero(torso_v):
        parent=a['parent_indices'][pelvis];pw=a['armature_transform'] if parent<0 else evaluate(b,i)[parent]
        a['local_rotation_delta'][i,pelvis]=local_delta_for_world(quat_matrix(torso_q[i])@rotation(rest[pelvis]),pw,a['rest_rotation'][pelvis]);a['rotation_validity'][i,pelvis]=True;a['rotation_confidence'][i,pelvis]=torso_c[i];a['rotation_provenance'][i,pelvis]=2
        # Shoulder yaw relative to hips supplies upper torso twist.
        up=(world[i,11]+world[i,12]-world[i,23]-world[i,24])/2
        try:
            target=frame(world[i,11]-world[i,12],up)@reference.T@rotation(rest[spine]);pw=evaluate(b,i)[a['parent_indices'][spine]]
            a['local_rotation_delta'][i,spine]=local_delta_for_world(target,pw,a['rest_rotation'][spine]);a['rotation_validity'][i,spine]=True;a['rotation_confidence'][i,spine]=.6;a['rotation_provenance'][i,spine]=2
        except ValueError:pass
    stabilize_bundle(b,[pelvis,spine])
    # Stable clip scale from torso height avoids scaling by foreshortened arms.
    source_height=np.linalg.norm(((image[:,11,:2]+image[:,12,:2]-image[:,23,:2]-image[:,24,:2])/2)*[aspect,1],axis=1)
    target_height=np.linalg.norm((rest[lookup['upperarm_l'],:3,3]+rest[lookup['upperarm_r'],:3,3])/2-rest[pelvis,:3,3])
    scale=target_height/max(float(np.median(source_height[torso_v])),.05) if torso_v.any() else b.metadata['calibration']['image_plane_metres_per_normalized_height']
    corrections=0;unreachable=0
    for side,s,e,w in [('l',11,13,15),('r',12,14,16)]:
        uj,ej,wj=(lookup[x+'_'+side] for x in ('upperarm','lowerarm','hand'))
        l1=np.linalg.norm(rest[ej,:3,3]-rest[uj,:3,3]);l2=np.linalg.norm(rest[wj,:3,3]-rest[ej,:3,3])
        # Visibility is an occlusion score, not an on/off pose switch. Retain
        # usable image estimates at reduced confidence instead of snapping arms.
        observed_conf=np.minimum(raw['visibility'][idx],raw['presence'][idx])
        reliable=clean['frame_validity'][idx] & (observed_conf[:,[s,e,w]].min(axis=1)>.15)
        # Every arm frame uses the same retarget convention. Frames without a
        # usable image estimate enter the gap policy, never an aim-only baseline.
        for j in (uj,ej):
            a['rotation_validity'][:,j]=False;a['rotation_confidence'][:,j]=0;a['rotation_provenance'][:,j]=0;a['local_rotation_delta'][:,j]=[0,0,0,1]
        previous_elbow=None;previous_normal=None
        rest_normal=np.cross(rest[ej,:3,3]-rest[uj,:3,3],rest[wj,:3,3]-rest[ej,:3,3])
        lower_reference=frame(rest_normal,rest[wj,:3,3]-rest[ej,:3,3])
        for i in np.flatnonzero(reliable):
            fk=evaluate(b,i);shoulder=fk[uj,:3,3];bottom=fk[pelvis,:3,3];top=fk[neck,:3,3]
            target=shoulder+np.r_[(image[i,w,:2]-image[i,s,:2])*[aspect,-1]*scale,(world[i,w,2]-world[i,s,2])*.65]
            hint=shoulder+np.r_[(image[i,e,:2]-image[i,s,:2])*[aspect,-1]*scale,(world[i,e,2]-world[i,s,2])*.65]
            safe=clearance(target,bottom,top,.20);corrections+=int(np.linalg.norm(safe-target)>.001);target=safe
            hint=clearance(hint,bottom,top,.18)
            unreachable+=int(np.linalg.norm(target-shoulder)>l1+l2-.005)
            elbow,target=two_bone(shoulder,target,hint,l1,l2)
            # Reach has an elbow swivel degree of freedom. Choose it jointly
            # with torso clearance, rather than pushing the wrist alone.
            from xms.geometry.proxies import segment_distance
            axis=target-shoulder;axis/=max(np.linalg.norm(axis),1e-8)
            center=shoulder+axis*np.dot(elbow-shoulder,axis);bend=elbow-center
            candidates=[]
            for degrees in (0,15,-15,30,-30,45,-45,60,-60,90,-90,120,-120,150,-150,180):
                theta=np.deg2rad(degrees);candidate=center+bend*np.cos(theta)+np.cross(axis,bend)*np.sin(theta)
                depth=max(0,.18-float(segment_distance(candidate,target,bottom,top)))
                image_cost=np.sum((candidate[:2]-hint[:2])**2)
                continuity=0 if previous_elbow is None else .15*np.sum((candidate-previous_elbow)**2)
                candidates.append((image_cost+100*depth*depth+continuity,candidate))
            elbow=min(candidates,key=lambda item:item[0])[1];previous_elbow=elbow
            normal=np.cross(elbow-shoulder,target-elbow)
            if previous_normal is not None and np.dot(normal,previous_normal)<0:normal=-normal
            normal/=max(np.linalg.norm(normal),1e-8);previous_normal=normal
            # Keep shoulder swing close to rest skinning. The elbow plane resolves
            # forearm roll without twisting the upper arm through the garment.
            set_direction(b,i,uj,ej,elbow-shoulder)
            for j,direction,reference_frame in ((ej,target-elbow,lower_reference),):
                fk=evaluate(b,i);parent=a['parent_indices'][j];desired=frame(normal,direction)@reference_frame.T@rotation(rest[j])
                a['local_rotation_delta'][i,j]=local_delta_for_world(desired,fk[parent],a['rest_rotation'][j])
            for j in (uj,ej):
                a['rotation_validity'][i,j]=True;a['rotation_confidence'][i,j]=float(min(clean['visibility'][idx[i],[s,e,w]]));a['rotation_provenance'][i,j]=2
    arm_repairs=stabilize_bundle(b,[lookup[x+'_'+s] for s in ('l','r') for x in ('upperarm','lowerarm')],max_speed=900)
    # Refit leg directions after changing the pelvis; otherwise torso yaw would
    # rotate baseline legs a second time.
    for side,h,k,f in [('l',23,25,27),('r',24,26,28)]:
        for name,child,s,e in [('thigh','calf',h,k),('calf','foot',k,f)]:
            j=lookup[name+'_'+side];cj=lookup[child+'_'+side]
            for i in np.flatnonzero(valid[:,[s,e]].all(axis=1)):
                direction=world[i,e]-world[i,s]
                if np.linalg.norm(direction)>1e-6:set_direction(b,i,j,cj,direction)
    stabilize_bundle(b,[lookup[x+'_'+s] for s in ('l','r') for x in ('thigh','calf')])
    # Root jitter is tracked image jitter, not intentional depth translation.
    for d in range(2):a['root_translation'][:,d]=median_filter(a['root_translation'][:,d],size=3,mode='nearest')
    # A stationary visible foot in the source is an anchor, not a noisy 3D
    # direction prior. Moving or occluded feet do not qualify for this fallback.
    planted={};poses=np.array([evaluate(b,i) for i in range(len(t))])
    from scipy.spatial.transform import Rotation
    for side,landmarks in [('l',[27,29,31]),('r',[28,30,32])]:
        mask=raw['landmark_validity'][idx][:,landmarks].all(axis=1)
        if mask.mean()<.9:continue
        xy=image[:,landmarks,:2]*[aspect,1]*scale;origin=np.median(xy[mask],axis=0)
        if np.percentile(np.linalg.norm(xy[mask]-origin,axis=-1),95)>.025:continue
        foot=lookup['foot_'+side];anchor=np.median(poses[mask,foot,:3,3],axis=0)
        planted[side]=(anchor,Rotation.from_matrix(np.array([rotation(w) for w in poses[mask,foot]])).mean().as_matrix())
    if planted:
        for i in range(len(t)):
            for side,(anchor,foot_rotation) in planted.items():
                hip,knee,foot=(lookup[x+'_'+side] for x in ('thigh','calf','foot'));fk=evaluate(b,i)
                l1=np.linalg.norm(rest[knee,:3,3]-rest[hip,:3,3]);l2=np.linalg.norm(rest[foot,:3,3]-rest[knee,:3,3])
                if len(planted)==2:
                    # Tracker scale noise must not overextend both planted legs.
                    excess=np.linalg.norm(anchor-fk[hip,:3,3])-(l1+l2-.005)
                    if excess>0:
                        a['root_translation'][i,1]-=min(excess,.03)
                        if not a['root_validity'][i]:a['root_validity'][i]=True;a['root_confidence'][i]=.05;a['root_provenance'][i]=2
                        fk=evaluate(b,i)
                knee_target,foot_target=two_bone(fk[hip,:3,3],anchor,fk[knee,:3,3],l1,l2)
                set_direction(b,i,hip,knee,knee_target-fk[hip,:3,3]);set_direction(b,i,knee,foot,foot_target-knee_target)
                fk=evaluate(b,i);a['local_rotation_delta'][i,foot]=local_delta_for_world(foot_rotation,fk[a['parent_indices'][foot]],a['rest_rotation'][foot])
                for j in (hip,knee,foot):a['rotation_validity'][i,j]=True;a['rotation_confidence'][i,j]=.5;a['rotation_provenance'][i,j]=2
    b.metadata['stationary_foot_diagnostics']={'anchored_sides':list(planted),'policy':'At least 90% visible; 95th-percentile image displacement <2.5cm in calibrated image plane; moving feet excluded','quality_accepted':False}
    b.metadata['solver'].update(backend='video_body',version='2',arm_fit='image-guided analytic two-bone IK; fixed rest lengths; camera-facing torso clearance')
    b.metadata['body_stability_diagnostics']={'bridged_landmark_samples':estimated,'repaired_rotation_samples':repairs,'clearance_target_samples':corrections,'unreachable_target_samples':unreachable,'arm_image_scale':float(scale),'arm_fallback_samples':arm_repairs,'max_arm_rate_deg_s':900,'quality_accepted':False}
    b.metadata['limitations']=['Monocular arm depth is a prior; wrists projected onto the torso use camera-facing clearance.','Two-bone reach clamps unreachable image targets; arbitrary occlusions and complex posture remain unsupported.']
    return validate(b)
