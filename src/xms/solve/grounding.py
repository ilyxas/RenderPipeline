"""Flat-floor support reconstruction in canonical Y-up metres, before publication.

Uses source evidence to retain flight; sole proxies belong to the registered rest
stance. No renderer offsets or video-specific constants are involved.
"""
import copy
import numpy as np
from xms.animation.fk import evaluate
from .contacts import spans
from .head import rotation,local_delta_for_world


def source_support(image,valid,times,scale):
    n=len(times);feet=image[:,[31,32],1];visible=valid[:,[27,28]] & valid[:,[31,32]]
    reliable=visible.any(axis=1);flight=np.zeros(n,bool)
    # Fixed-camera flat floor prior. Both feet AND hips must rise; a bent knee
    # or tracker depth noise alone cannot turn support into a jump.
    if visible.all(axis=1).any():
        floor=float(np.percentile(np.max(feet[visible.all(axis=1)],axis=1),90))
        hips=image[:,[23,24],1].mean(axis=1);hip_ok=valid[:,[23,24]].all(axis=1)
        hip_base=float(np.percentile(hips[hip_ok],80)) if hip_ok.any() else 0
        lifted=(floor-feet.max(axis=1))*scale>.10
        flight=visible.all(axis=1)&hip_ok&lifted&((hip_base-hips)*scale>.08)
        for a,b in spans(flight,1):
            if times[b-1]-times[a]<.06:flight[a:b]=False
    # Missing feet are not evidence of support or flight.
    return reliable,flight


def sole_proxies(bundle,rest):
    """Heel and ball proxies at registered floor, expressed in bone space.

These are skeletal sole estimates, not measured skin geometry. Profiles may
register precise sole points later without changing the motion contract.
"""
    names=bundle.metadata['joint_names'];points={}
    for side in ('l','r'):
        points[side]=[]
        for bone in ('foot','ball'):
            j=names.index(bone+'_'+side);p=rest[j,:3,3].copy();p[1]=0
            points[side].append((j,np.linalg.solve(rest[j,:3,:3],p-rest[j,:3,3])))
    return points


def sole_positions(fk,points):
    return np.array([[fk[j,:3,3]+fk[j,:3,:3]@p for j,p in points[s]] for s in ('l','r')])


def ground(bundle,image,valid,scale):
    from .video_body import foot_orientation,set_direction,two_bone,support_root_drop
    a=bundle.arrays;names=bundle.metadata['joint_names'];ids={n:i for i,n in enumerate(names)};t=a['sample_times_s'];n=len(t)
    neutral=copy.deepcopy(bundle);neutral.arrays['local_rotation_delta'][:]=[0,0,0,1];neutral.arrays['root_translation'][:]=0;rest=evaluate(neutral,0)
    points=sole_proxies(bundle,rest);reliable,flight=source_support(image,valid,t,scale);supported=reliable&~flight
    # Bounded short dropout holds the last floor correction, then leaves motion
    # untouched and diagnoses the unsupported samples.
    corrections=np.zeros(n);known=np.flatnonzero(supported);heights=[];bounded=0
    for i in known:
        fk=evaluate(bundle,i);soles=sole_positions(fk,points);side=('l','r')[int(np.argmin(soles[:,:,1].min(axis=1)))];foot=ids['foot_'+side];ball=ids['ball_'+side]
        # Flatten only the lowest foot near support; the lifted foot keeps pitch.
        forward=rotation(fk[foot])@rotation(rest[foot]).T@(rest[ball,:3,3]-rest[foot,:3,3])
        try:
            desired=foot_orientation(forward,rest[ball,:3,3]-rest[foot,:3,3],rotation(rest[foot]),planted=True)
            a['local_rotation_delta'][i,foot]=local_delta_for_world(desired,fk[a['parent_indices'][foot]],a['rest_rotation'][foot]);a['rotation_validity'][i,foot]=True;a['rotation_confidence'][i,foot]=.5;a['rotation_provenance'][i,foot]=2
        except ValueError:pass
        height=float(sole_positions(evaluate(bundle,i),points)[:,:,1].min());heights.append(height);corrections[i]=-np.clip(height,-.45,.45);bounded+=int(abs(height)>.45)
    # Flight interpolates the floor offset, not the airborne foot height.
    # Thus a source jump retains its original root excursion.
    if len(known):
        for i in range(n):
            if supported[i]:continue
            left=known[known<i];right=known[known>i]
            if flight[i] and len(left) and len(right):corrections[i]=np.interp(t[i],t[known],corrections[known])
            elif not reliable[i] and len(left) and t[i]-t[left[-1]]<=.25:corrections[i]=corrections[left[-1]]
    a['root_translation'][:,1]+=corrections
    changed=np.abs(corrections)>1e-8;a['root_validity'][changed]=True;a['root_confidence'][changed]=np.maximum(a['root_confidence'][changed],.25);a['root_provenance'][changed]=2
    # Local support intervals, not a whole-clip stationary predicate. Sliding
    # feet remain floor-aligned but are not locked horizontally.
    poses=np.array([evaluate(bundle,i) for i in range(n)]);records=[];unreachable=0
    for sidx,side in enumerate(('l','r')):
        hip,knee,foot=(ids[x+'_'+side] for x in ('thigh','calf','foot'));soles=np.array([sole_positions(fk,points)[sidx] for fk in poses]);pos=poses[:,foot,:3,3]
        speed=np.zeros(n)
        if n>1:speed[1:]=np.linalg.norm(np.diff(pos[:,[0,2]],axis=0),axis=1)/np.diff(t);speed[0]=speed[1]
        contact=supported&(soles[:,:,1].min(axis=1)<.025)&(speed<.25)&valid[:,27+sidx]&valid[:,31+sidx]
        l1=np.linalg.norm(rest[knee,:3,3]-rest[hip,:3,3]);l2=np.linalg.norm(rest[foot,:3,3]-rest[knee,:3,3])
        for start,end in spans(contact,2):
            if t[end-1]-t[start]<.06:continue
            anchor=np.median(pos[start:end],axis=0);records.append({'side':side,'start':start,'end':end,'anchor_world_m':anchor.tolist()})
            for i in range(start,end):
                fk=evaluate(bundle,i);target=fk[foot,:3,3].copy();ramp=min(1,(t[i]-t[start]+.02)/.08,(t[end-1]-t[i]+.02)/.08);target[[0,2]]+=(anchor[[0,2]]-target[[0,2]])*ramp
                drop=support_root_drop(fk[hip,:3,3][None],target[None],[l1+l2-.005]);a['root_translation'][i,1]-=drop;fk=evaluate(bundle,i)
                desired=rotation(fk[foot]);elbow,reach=two_bone(fk[hip,:3,3],target,fk[knee,:3,3],l1,l2);unreachable+=int(np.linalg.norm(reach-target)>.003)
                set_direction(bundle,i,hip,knee,elbow-fk[hip,:3,3]);set_direction(bundle,i,knee,foot,reach-elbow);fk=evaluate(bundle,i)
                a['local_rotation_delta'][i,foot]=local_delta_for_world(desired,fk[a['parent_indices'][foot]],a['rest_rotation'][foot])
                for j in (hip,knee,foot):a['rotation_validity'][i,j]=True;a['rotation_confidence'][i,j]=.5;a['rotation_provenance'][i,j]=2
    # IK reach corrections can change shared root height. Restore floor support
    # once, after both legs; never globally smooth rotations after this stage.
    for i in known:
        height=float(sole_positions(evaluate(bundle,i),points)[:,:,1].min());a['root_translation'][i,1]-=np.clip(height,-.12,.12)
    actual=np.array([sole_positions(evaluate(bundle,i),points)[:,:,1].min() for i in known]);errors=np.abs(actual)
    bundle.metadata['grounding_diagnostics']={'policy':'flat floor; source flight evidence; per-frame sole/root alignment; local support IK','support_sample_indices':known.tolist(),'flight_sample_indices':np.flatnonzero(flight).tolist(),'unsupported_samples':int((~reliable).sum()),'correction_bound_samples':bounded,'max_root_correction_m':float(np.abs(corrections).max(initial=0)),'contact_segments':records,'unreachable_anchor_samples':unreachable,'max_support_floor_error_m':float(errors.max(initial=0)),'p95_support_floor_error_m':float(np.percentile(errors,95)) if len(errors) else None,'sole_geometry':'rest-derived heel/ball skeletal proxies; skin sole unverified','quality_accepted':False}
    bundle.metadata['stationary_foot_diagnostics']={'anchored_sides':sorted({r['side'] for r in records}),'policy':'local support intervals; replaced whole-clip stationary locking','quality_accepted':False}
    return bundle
