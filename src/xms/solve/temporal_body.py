"""Bounded single-window root/joint increment optimization, experimental."""
import time
import numpy as np
from scipy.optimize import least_squares
from xms.animation.fk import trs,quat_matrix
from xms.animation.validate import validate
from xms.qa.pose import JOINTS
from xms.qa.contract import self_peak_bytes
from .baseline_body import solve as baseline_solve
from .parameterization import quaternions_to_rotvec,rotvec_to_quaternions,cap_rotvec
from .residuals import residual,sparsity,temporal_weights

LIMITATIONS=['Experimental single-window body optimizer; no stitching, contact or collision solving.',
 'Monocular fixed camera and depth prior do not recover metric ground truth.',
 'Missing body samples are explicit low-confidence body priors, not recovered observations.',
 'Scalar swing caps are not anatomical joint limits; face and hands use baseline backends.']
SEGMENTS={
 'pelvis':('spine_01',[23,24],[11,12],35),
 'upperarm_l':('lowerarm_l',[11],[13],160),'lowerarm_l':('hand_l',[13],[15],155),
 'upperarm_r':('lowerarm_r',[12],[14],160),'lowerarm_r':('hand_r',[14],[16],155),
 'thigh_l':('calf_l',[23],[25],110),'calf_l':('foot_l',[25],[27],145),'foot_l':('ball_l',[29],[31],70),
 'thigh_r':('calf_r',[24],[26],110),'calf_r':('foot_r',[26],[28],145),'foot_r':('ball_r',[30],[32],70)}


def context(bundle,observations,timeline,calibration):
    a=bundle.arrays;obs=observations[1];indices=np.asarray(timeline['output_observation_indices']);n=len(indices)
    image=obs['image_landmarks_raw'][indices,:,:2];world=obs['world_landmarks_raw'][indices]*[1,-1,-1]
    valid=obs['landmark_validity'][indices];confidence=np.minimum(obs['visibility'][indices],obs['presence'][indices])*valid
    names=bundle.metadata['joint_names'];lookup={name:i for i,name in enumerate(names)}
    selected=[lookup[name] for name in SEGMENTS];variable={j:i for i,j in enumerate(selected)}
    initial=np.stack([quaternions_to_rotvec(a['local_rotation_delta'][:,j]) for j in selected],axis=1)
    points=[(lookup[name],image[:,landmark],valid[:,landmark],confidence[:,landmark]) for name,landmark in JOINTS.values()]
    directions=[];joint_conf=[]
    for name,(child,start,end,cap) in SEGMENTS.items():
        direction=world[:,end].mean(axis=1)-world[:,start].mean(axis=1);length=np.linalg.norm(direction,axis=-1)
        mask=valid[:,start+end].all(axis=1)&(length>1e-8);conf=np.min(confidence[:,start+end],axis=1)*mask
        directions.append((lookup[name],lookup[child],direction/np.maximum(length[:,None],1e-10),mask,conf));joint_conf.append(conf)
    joint_conf=np.stack(joint_conf,axis=1);closure=set()
    for j in [p[0] for p in points]+[j for d in directions for j in d[:2]]:
        while j>=0:closure.add(j);j=int(a['parent_indices'][j])
    rest=[trs(t,q,s) for t,q,s in zip(a['rest_translation'],a['rest_rotation'],a['rest_scale'])]
    fixed=np.array([[quat_matrix(a['rest_rotation'][j])@quat_matrix(q)@np.diag(a['rest_scale'][j]) for j,q in enumerate(frame)] for frame in a['local_rotation_delta']])
    return dict(n=n,k=len(selected),width=2+3*len(selected),initial=initial,selected=selected,variable=variable,points=points,directions=directions,closure=sorted(closure),rest=rest,rest_rotation=[quat_matrix(q) for q in a['rest_rotation']],scale=a['rest_scale'],parents=a['parent_indices'],armature=a['armature_transform'],fixed_rotation=fixed,root_initial=a['root_translation'].copy(),camera=calibration['reference_camera'],times=a['sample_times_s'],confidence=joint_conf,weights=temporal_weights(initial,joint_conf),caps=np.deg2rad([s[3] for s in SEGMENTS.values()]))


def solve(observations,timeline,profile,rig,calibration=None,max_nfev=80,initializer=None,boundary=None,contacts=False):
    started=time.monotonic();b=baseline_solve(observations,timeline,profile,rig,calibration) if initializer is None else initializer;calibration=b.metadata['calibration'];c=context(b,observations,timeline,calibration)
    if c['n']<3: return b
    if c['times'][-1]-c['times'][0]>4:raise ValueError('Temporal window exceeds four seconds')
    if boundary is not None:c['boundary']=boundary
    if contacts:
        from .contacts import constraints
        c['contacts']=constraints(b)
    x0=np.zeros(c['n']*c['width'])
    bounds=(-np.inf,np.inf)
    if profile.get('collision_proxies'):
        c['collision_proxies']=profile['collision_proxies'];c['joint_lookup']={name:i for i,name in enumerate(b.metadata['joint_names'])}
        budget=np.full((c['n'],c['width']),np.deg2rad(profile['collision_proxies']['angular_correction_budget_degrees'])/np.sqrt(3));budget[:,:2]=profile['collision_proxies']['translation_correction_budget_m'];bounds=(-budget.ravel(),budget.ravel())
    before=residual(x0,c,True)
    def bounded_residual(x):
        if time.monotonic()-started>115:raise TimeoutError('Stage 8 115-second internal solve deadline exceeded')
        return residual(x,c)
    result=least_squares(bounded_residual,x0,bounds=bounds,jac_sparsity=sparsity(c),loss='soft_l1',f_scale=.015,max_nfev=max_nfev,ftol=1e-3,xtol=1e-3,gtol=1e-4)
    v=c['initial']+result.x.reshape(c['n'],c['width'])[:,2:].reshape(c['n'],c['k'],3)
    # Publish bounded rotations; record the projected objective separately.
    for i,j in enumerate(c['selected']):
        capped=cap_rotvec(v[:,i],np.degrees(c['caps'][i]));b.arrays['local_rotation_delta'][:,j]=rotvec_to_quaternions(capped);v[:,i]=capped
        missing=~b.arrays['rotation_validity'][:,j]
        if np.any(c['confidence'][:,i]>0):
            b.arrays['rotation_validity'][missing,j]=True;b.arrays['rotation_confidence'][missing,j]=.05;b.arrays['rotation_provenance'][missing,j]=2
        else:b.arrays['local_rotation_delta'][:,j]=[0,0,0,1]
    b.arrays['root_translation'][:,:2]+=result.x.reshape(c['n'],c['width'])[:,:2]
    missing=~b.arrays['root_validity'];b.arrays['root_validity'][missing]=True;b.arrays['root_confidence'][missing]=.05;b.arrays['root_provenance'][missing]=2
    published=result.x.reshape(c['n'],c['width']).copy();published[:,2:]=(v-c['initial']).reshape(c['n'],-1)
    diagnostics={'schema_version':'xms.temporal_diagnostics.v1','termination':'converged' if result.success else 'exhausted_or_failed','success':bool(result.success),'status':int(result.status),'message':result.message,'nfev':result.nfev,'njev':result.njev,'optimality':float(result.optimality),'robust_cost_after':float(result.cost),'objective_terms_before':before,'objective_terms_after':residual(published.ravel(),c,True),'seconds':time.monotonic()-started,'python_peak_rss_bytes':self_peak_bytes(),'max_nfev':max_nfev,'camera_refinement':False,'contact_weight':0,'collision_weight':0,'observed_body_mask_policy':'Original confidence preserved; temporal estimates in missing spans are body_prior with confidence .05'}
    caps=b.metadata['solver']['direction_caps_degrees'];b.metadata['solver']={'backend':'temporal_body','version':'1','direction_caps_degrees':caps,'window_samples':c['n'],'optimization':'simultaneous root XY and local rotvec increments; canonical FK reprojection, soft 3D directions, geodesic velocity/acceleration, scalar caps','robust_loss':'soft_l1','fast_motion_policy':'12 degrees/sample adaptive regularization, frozen before real evaluation','contacts_weight':0,'collision_weight':0}
    b.metadata.update(temporal_diagnostics=diagnostics,limitations=LIMITATIONS,quality='development_candidate')
    return validate(b)
