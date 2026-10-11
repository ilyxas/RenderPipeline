"""Anatomical arm limits: elbow hinge, bounded forearm roll, two-axis wrist.

A tracker palm flip can ask for a forearm roll no arm can make; the roll used
to be unwrapped and accepted. The roll is now read on the branch nearest the
rest forearm, and samples outside the anatomical range are treated as
unobserved and bridged from in-range neighbours. The elbow keeps only its
hinge flexion. The hand keeps its world orientation where the two wrist
hinges can reach it; the wrist itself has no axial roll.
"""
import copy,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from xms.animation.fk import evaluate

LIMITS=Path(__file__).resolve().parents[3]/'profiles/characters/xandra/arm_limits.v1.json'


def unit(v):return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12)
def wrap(x):return (x+np.pi)%(2*np.pi)-np.pi
def signed(a,b,axis):return np.arctan2(np.cross(a,b)@axis,np.sum(a*b,axis=-1))


def twist_split(q,axis):
    """q = swing * twist, twist about a rest-local axis on its nearest-rest branch."""
    v=q.as_quat();theta=wrap(2*np.arctan2(v[:,:3]@axis,v[:,3]));twist=Rotation.from_rotvec(axis*theta[:,None])
    return q*twist.inv(),theta


def apply(bundle,profile):
    spec=json.loads(LIMITS.read_text())
    if spec['asset_sha256']!=profile['asset_sha256']:
        bundle.metadata['arm_limit_diagnostics']={'status':'unavailable','reason':'Arm limits are registered for a different character asset'};return bundle
    a=bundle.arrays;ids={s:i for i,s in enumerate(bundle.metadata['joint_names'])};t=a['sample_times_s']
    ref=copy.deepcopy(bundle);ref.arrays['local_rotation_delta'][:]=[0,0,0,1];ref.arrays['root_translation'][:]=0;rest=evaluate(ref,0)
    def world_rotation(j):return rest[j,:3,:3]/np.linalg.norm(rest[j,:3,:3],axis=0)
    diagnostics={'status':'applied','source':spec['source']}
    for side in ('l','r'):
        u,l,h=(ids[x+'_'+side] for x in ('upperarm','lowerarm','hand'))
        q=Rotation.from_quat(a['local_rotation_delta'][:,l]);axis=unit(a['rest_translation'][h])
        swing,theta=twist_split(q,axis)
        # Elbow hinge in the lowerarm rest frame; flexion is counted from a straight arm.
        hinge=unit(world_rotation(l).T@np.array(spec['elbow']['axis'][side]))
        upper=world_rotation(l).T@unit(rest[l,:3,3]-rest[u,:3,3]);rest_bend=float(signed(upper,axis,hinge))
        if max(abs(upper@hinge),abs(axis@hinge))>.05:raise ValueError(f'Elbow hinge axis ({side}) is not perpendicular to the rest arm; check arm limits file')
        direction=swing.apply(axis);flexion=signed(np.broadcast_to(axis,direction.shape),direction-np.outer(direction@hinge,hinge),hinge)
        low,high=np.radians(spec['elbow']['flexion_from_straight_deg'])-rest_bend;bounded=np.clip(flexion,low,high)
        hinged=Rotation.from_rotvec(hinge*bounded[:,None]);off_plane=np.degrees(np.arccos(np.clip(np.sum(hinged.apply(axis)*direction,axis=1),-1,1)))
        low,high=np.radians(spec['forearm_roll']['range_deg']);inside=(theta>=low)&(theta<=high)
        roll=np.interp(t,t[inside],theta[inside]) if inside.any() else np.zeros_like(theta)
        limited=hinged*Rotation.from_rotvec(axis*roll[:,None])
        # Same hand world orientation under the new forearm, then wrist hinges only.
        rest_hand=Rotation.from_quat(a['rest_rotation'][h]);hand=Rotation.from_quat(a['local_rotation_delta'][:,h])
        desired=rest_hand.inv()*limited.inv()*q*rest_hand*hand
        hand_axis=unit(a['rest_translation'][ids['middle_01_'+side]]);wrist_swing,wrist_roll=twist_split(desired,hand_axis)
        flex_axis=unit(world_rotation(h).T@np.array(spec['wrist']['flexion_axis'][side]));dev_axis=unit(world_rotation(h).T@np.array(spec['wrist']['deviation_axis'][side]))
        vector=wrist_swing.as_rotvec();flex=vector@flex_axis;dev=vector@dev_axis
        flex_bounded=np.clip(flex,*np.radians(spec['wrist']['flexion_deg']));dev_bounded=np.clip(dev,*np.radians(spec['wrist']['deviation_deg']))
        a['local_rotation_delta'][:,l]=limited.as_quat();a['local_rotation_delta'][:,h]=Rotation.from_rotvec(np.outer(flex_bounded,flex_axis)+np.outer(dev_bounded,dev_axis)).as_quat()
        diagnostics[side]={'elbow_limited_samples':int(np.sum(bounded!=flexion)),'elbow_max_off_hinge_deg':float(off_plane.max()),'rest_elbow_bend_deg':float(np.degrees(rest_bend)),
                           'forearm_roll_bridged_samples':int(np.sum(~inside)),'forearm_max_requested_roll_deg':float(np.degrees(np.abs(theta).max())),
                           'wrist_limited_samples':int(np.sum((flex_bounded!=flex)|(dev_bounded!=dev))),'wrist_max_removed_roll_deg':float(np.degrees(np.abs(wrist_roll).max()))}
    bundle.metadata['arm_limit_diagnostics']=diagnostics
    return bundle
