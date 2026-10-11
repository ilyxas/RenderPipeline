"""Distribute arm axial roll over the skinned twist bones.

Xandra's arm skin is weighted to twist bones, not to upperarm/lowerarm, and the
asset has no constraints or drivers for them. A twist bone with an identity
delta turns rigidly with its parent, so the whole forearm roll lands next to
the elbow and the skin wraps there. Each twist bone keeps the share of its
parent's roll given by its rest position along the segment; the remainder is
counter-rotated. Joint positions do not change.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from xms.animation.fk import quat_matrix

CHAINS=[('upperarm','lowerarm',('upperarm_twist_01','upperarm_twist_02')),('lowerarm','hand',('lowerarm_twist_01','lowerarm_twist_02'))]


def roll(q,axis):
    """Unwrapped twist about a rest-local axis, using q = swing * twist."""
    return np.unwrap(2*np.arctan2(q[:,:3]@axis,q[:,3]))


def distribute(bundle):
    a=bundle.arrays;ids={s:i for i,s in enumerate(bundle.metadata['joint_names'])};diagnostics={}
    for side in ('l','r'):
        for bone,child,twists in CHAINS:
            if bone+'_'+side not in ids:continue
            # A child's rest translation is expressed in the parent's delta frame.
            segment=a['rest_translation'][ids[child+'_'+side]];length=float(np.linalg.norm(segment));axis=segment/length
            theta=roll(a['local_rotation_delta'][:,ids[bone+'_'+side]],axis);shares={}
            for name in twists:
                if name+'_'+side not in ids:continue
                t=ids[name+'_'+side];share=float(np.clip(a['rest_translation'][t]@axis/length,0,1))
                local_axis=quat_matrix(a['rest_rotation'][t]).T@axis
                a['local_rotation_delta'][:,t]=Rotation.from_rotvec(local_axis[None]*(-(1-share)*theta)[:,None]).as_quat()
                shares[name+'_'+side]=share
            diagnostics[bone+'_'+side]={'max_abs_roll_deg':float(np.degrees(np.abs(theta).max())),'twist_shares':shares}
    bundle.metadata['twist_diagnostics']=diagnostics
    return bundle
