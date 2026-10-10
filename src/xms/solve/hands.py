"""Rest-length hand solve with bounded short gaps; fast observed gestures untouched."""
import numpy as np


def solve_hands(body,observations,timeline):
    import copy
    from xms.animation.fk import evaluate,trs,quat_matrix,matrix_quat
    from .head import rotation,local_delta_for_world
    from .baseline_body import align
    from .baseline_hands import hand_frame
    from .stability import rotations,angle
    meta,obs=observations;idx=np.asarray(timeline['output_observation_indices']);times=body.arrays['sample_times_s'];lookup={n:j for j,n in enumerate(body.metadata['joint_names'])}
    neutral=copy.deepcopy(body);neutral.arrays['local_rotation_delta'][:]=[0,0,0,1];neutral.arrays['root_translation'][:]=0;rest=evaluate(neutral,0);work=copy.deepcopy(body)
    valid=np.zeros_like(body.arrays['rotation_validity']);confidence=np.zeros_like(body.arrays['rotation_confidence']);claimed=[];flips=0;repairs=0;roll_fallback_total=0
    for si,side in enumerate(('l','r')):
        wrist=lookup['hand_'+side];elbow=lookup['lowerarm_'+side]
        joints=[wrist]+[lookup[f'{f}_{k:02d}_{side}'] for f in ('thumb','index','middle','ring','pinky') for k in (1,2,3)];claimed+=joints
        camera_q=np.tile([0.,0,0,1.],(len(times),1));v=obs['validity'][idx,si].copy();c=obs['confidence'][idx,si].copy();previous=None
        for i,source in enumerate(idx):
            if not v[i] or c[i]<.35:v[i]=False;continue
            lm=obs['world_landmarks_raw'][source,si]*[1,-1,-1]
            try:r=hand_frame(lm[0],lm[9],lm[5],lm[17])
            except ValueError:v[i]=False;continue
            q=matrix_quat(r)
            # A palm normal can reverse when depth is ambiguous. Resolve isolated
            # branch flips against the previous frame, not against global neutral.
            if previous is not None and angle(previous,q)>np.deg2rad(100):
                alternative=matrix_quat(r@np.diag([-1.,1.,-1.]))
                if angle(previous,alternative)+np.deg2rad(30)<angle(previous,q):q=alternative;flips+=1;c[i]*=.5
            camera_q[i]=q;previous=q
        camera_q,v,c,inferred=rotations(camera_q,v,c,times,bridge=.3,fade=0,tau=.045);repairs+=int(inferred.sum())
        reference=hand_frame(rest[wrist,:3,3],rest[lookup['middle_01_'+side],:3,3],rest[lookup['index_01_'+side],:3,3],rest[lookup['pinky_01_'+side],:3,3])
        rest_forearm=rest[wrist,:3,3]-rest[elbow,:3,3];rest_forearm/=np.linalg.norm(rest_forearm)
        from .video_body import frame
        forearm_reference=frame(reference[:,0],rest_forearm);previous_forearm=None;previous_roll=None;roll_fallbacks=0;forearm_base=body.arrays['local_rotation_delta'][:,elbow].copy()
        for i,source in enumerate(idx):
            if not v[i]:continue
            observed=quat_matrix(camera_q[i]);fk=evaluate(work,i);forearm=fk[wrist,:3,3]-fk[elbow,:3,3];forearm/=np.linalg.norm(forearm)
            if work.arrays['rotation_validity'][i,elbow]:
                # Share pronation with the forearm. Rotating around its length axis
                # leaves the solved elbow and wrist positions unchanged.
                forearm_frame=frame(observed[:,0],forearm)
                if previous_forearm is not None:
                    fq=matrix_quat(forearm_frame);alternative=forearm_frame@np.diag([-1.,1.,-1.])
                    if angle(previous_forearm,fq)>np.deg2rad(90) and angle(previous_forearm,matrix_quat(alternative))<angle(previous_forearm,fq):
                        forearm_frame=alternative;observed=observed@np.diag([-1.,1.,-1.]);flips+=1
                previous_forearm=matrix_quat(forearm_frame)
                target=forearm_frame@forearm_reference.T@rotation(rest[elbow]);parent=work.arrays['parent_indices'][elbow]
                q=local_delta_for_world(target,fk[parent],work.arrays['rest_rotation'][elbow])
                from scipy.spatial.transform import Rotation
                base=Rotation.from_quat(body.arrays['local_rotation_delta'][i,elbow]);relative=(base.inv()*Rotation.from_quat(q)).as_quat()
                axis=work.arrays['rest_translation'][wrist].copy();axis/=np.linalg.norm(axis)
                roll=2*np.arctan2(relative[:3]@axis,relative[3])
                if previous_roll is not None:
                    roll=previous_roll+(roll-previous_roll+np.pi)%(2*np.pi)-np.pi
                    limit=np.deg2rad(720)*(times[i]-times[max(0,i-1)])
                    if abs(roll-previous_roll)>limit:roll=previous_roll+np.clip(roll-previous_roll,-limit,limit);roll_fallbacks+=1
                previous_roll=roll
                q=(base*Rotation.from_rotvec(axis*roll)).as_quat();work.arrays['local_rotation_delta'][i,elbow]=q
                body.arrays['local_rotation_delta'][i,elbow]=q
            fk=evaluate(work,i)
            # Limit actual wrist bend relative to the forearm, not the arbitrary
            # rig's neutral quaternion sphere (which introduced 180-degree jumps).
            direction=align(forearm,observed[:,1],75)@forearm
            target=frame(observed[:,0],direction)@reference.T@rotation(rest[wrist]);parent=work.arrays['parent_indices'][wrist]
            work.arrays['local_rotation_delta'][i,wrist]=local_delta_for_world(target,fk[parent],work.arrays['rest_rotation'][wrist]);valid[i,wrist]=True;confidence[i,wrist]=c[i]
            if not obs['validity'][source,si] or inferred[i]:continue
            lm=obs['world_landmarks_raw'][source,si]*[1,-1,-1];raw_frame=hand_frame(lm[0],lm[9],lm[5],lm[17])
            actual_frame=rotation(evaluate(work,i)[wrist])@rotation(rest[wrist]).T@reference
            for finger,offset in [('thumb',1),('index',5),('middle',9),('ring',13),('pinky',17)]:
                for k in (1,2,3):
                    j=lookup[f'{finger}_{k:02d}_{side}'];parent=int(work.arrays['parent_indices'][j]);fk=evaluate(work,i)
                    base=fk[parent]@trs(work.arrays['rest_translation'][j],work.arrays['rest_rotation'][j],work.arrays['rest_scale'][j])
                    axis=work.arrays['rest_translation'][lookup[f'{finger}_{k+1:02d}_{side}']] if k<3 else rotation(rest[j]).T@(rest[j,:3,3]-rest[lookup[f'{finger}_02_{side}'],:3,3])
                    segment=lm[offset+k]-lm[offset+k-1]
                    if np.linalg.norm(segment)<1e-7:continue
                    if finger=='thumb':
                        target=actual_frame@raw_frame.T@segment;current=base[:3,:3]@axis
                        r=rotation(base);delta=matrix_quat(r.T@align(current,target,90)@r)
                    else:
                        # Finger joints are hinges, not independent 3D aim bones.
                        # Curl magnitude survives palm-depth ambiguity without
                        # sending a fingertip around a 180-degree quaternion branch.
                        previous_segment=raw_frame[:,1] if k==1 else lm[offset+k-1]-lm[offset+k-2]
                        curl=np.arccos(np.clip(np.dot(segment,previous_segment)/(np.linalg.norm(segment)*np.linalg.norm(previous_segment)), -1,1))
                        curl=min(curl,np.deg2rad(100))
                        # Mirrored anatomical hands have opposite palm normals.
                        inward=reference[:,2]*(1 if side=='l' else -1)
                        rest_axis=rotation(rest[j])@axis;hinge=np.cross(rest_axis,inward);hinge/=max(np.linalg.norm(hinge),1e-8)
                        hinge=rotation(rest[j]).T@hinge
                        delta=np.r_[hinge*np.sin(curl/2),np.cos(curl/2)]
                    work.arrays['local_rotation_delta'][i,j]=delta;valid[i,j]=True;confidence[i,j]=c[i]

        # Repair only the length-axis twist through palm dropouts. Applying a
        # full quaternion filter here would displace the solved wrist target.
        from scipy.spatial.transform import Rotation
        twists=(Rotation.from_quat(forearm_base).inv()*Rotation.from_quat(body.arrays['local_rotation_delta'][:,elbow])).as_quat()
        twists,_,_,_=rotations(twists,v,c,times,bridge=.3,fade=.4,tau=.05,max_speed=720)
        repaired=(Rotation.from_quat(forearm_base)*Rotation.from_quat(twists)).as_quat()
        for i in range(len(times)):
            desired=rotation(evaluate(work,i)[wrist]) if valid[i,wrist] else None
            body.arrays['local_rotation_delta'][i,elbow]=repaired[i];work.arrays['local_rotation_delta'][i,elbow]=repaired[i]
            if desired is not None:
                fk=evaluate(work,i);work.arrays['local_rotation_delta'][i,wrist]=local_delta_for_world(desired,fk[elbow],work.arrays['rest_rotation'][wrist])
        roll_fallback_total+=roll_fallbacks
    provenance=valid.astype(np.uint8)
    for j in claimed:
        q,v,c,inferred=rotations(work.arrays['local_rotation_delta'][:,j],valid[:,j],confidence[:,j],times,tau=.04,max_speed=720 if j in (lookup['hand_l'],lookup['hand_r']) else 1200)
        work.arrays['local_rotation_delta'][:,j]=q;valid[:,j]=v;confidence[:,j]=c;provenance[:,j]=np.where(inferred,2,v.astype(np.uint8));repairs+=int(inferred.sum())
    # Pronation is bounded about the length axis; solved wrist positions remain exact.
    return {'indices':claimed,'rotations':work.arrays['local_rotation_delta'],'validity':valid,'confidence':confidence,'provenance':provenance,'diagnostics':{'inferred_joint_samples':repairs,'resolved_palm_branch_flips':flips,'policy':'SO(3) outlier repair; time-based gap transitions; forearm pronation; anatomical wrist bend cone; adaptive symmetric stabilization','metacarpals':'rest/unobserved; twist remains estimated','max_wrist_rate_deg_s':720,'max_finger_rate_deg_s':1200,'forearm_roll_fallbacks':roll_fallback_total,'quality_accepted':False}}


def palm_correction(bundle,max_degrees=5):
    """Joint bilateral bounded solve requires close 3D palms AND opposing normals."""
    from scipy.optimize import least_squares
    from xms.animation.fk import evaluate
    from .baseline_hands import hand_frame
    names=bundle.metadata['joint_names'];lookup={name:i for i,name in enumerate(names)};a=bundle.arrays;logs=[]
    selected=[lookup[x+'_'+s] for s in ('l','r') for x in ('upperarm','lowerarm','hand')];limit=np.deg2rad(max_degrees)/np.sqrt(3)
    def palms(world):
        centers=[];normals=[]
        for side in ('l','r'):
            w=world[lookup['hand_'+side],:3,3];middle=world[lookup['middle_01_'+side],:3,3];index=world[lookup['index_01_'+side],:3,3];pinky=world[lookup['pinky_01_'+side],:3,3]
            centers.append((w+middle+index+pinky)/4);normals.append(hand_frame(w,middle,index,pinky)[:,2])
        return np.array(centers),np.array(normals)
    for i in range(len(a['sample_times_s'])):
        if min(a['rotation_confidence'][i,[lookup['hand_l'],lookup['hand_r']]])<.5:continue
        world=evaluate(bundle,i);centers,normals=palms(world);distance=np.linalg.norm(centers[0]-centers[1])
        if distance>.09 or normals[0]@normals[1]>-.5:continue
        if not a['rotation_validity'][i,selected].all():continue
        initial=a['local_rotation_delta'][i,selected].copy();target=centers.mean(axis=0)
        def cost(x):
            from scipy.spatial.transform import Rotation
            a['local_rotation_delta'][i,selected]=(Rotation.from_quat(initial)*Rotation.from_rotvec(x.reshape(-1,3))).as_quat();c,n=palms(evaluate(bundle,i))
            return np.r_[(c-target).ravel()*8,(n[0]+n[1])*.03,x*.1]
        before=float(np.linalg.norm(cost(np.zeros(18))));r=least_squares(cost,np.zeros(18),bounds=(-limit,limit),max_nfev=16,ftol=.01)
        after=float(np.linalg.norm(cost(r.x)))
        if after>=before:a['local_rotation_delta'][i,selected]=initial
        logs.append({'sample':i,'before':before,'after':min(before,after),'max_correction_degrees':float(np.degrees(np.linalg.norm(r.x.reshape(-1,3),axis=1).max())),'accepted':after<before})
    bundle.metadata['palm_diagnostics']={'samples':logs,'angular_budget_degrees':max_degrees,'quality_accepted':False}
    return bundle
