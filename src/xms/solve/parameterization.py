"""Rotation-vector trajectory parameterization for the Stage 8 window solve."""
import numpy as np
from scipy.spatial.transform import Rotation


def quaternions_to_rotvec(q):
    q=np.asarray(q,dtype=float)
    q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    # SciPy uses the same XYZW convention as AnimationBundle.
    return Rotation.from_quat(q).as_rotvec()


def rotvec_to_quaternions(v):
    q=Rotation.from_rotvec(np.asarray(v,dtype=float)).as_quat()
    # Keep a deterministic continuous representative for the immutable bundle.
    for i in range(1,len(q)):
        if np.dot(q[i-1],q[i])<0:q[i]*=-1
    return q


def cap_rotvec(v,cap_degrees):
    v=np.asarray(v,dtype=float);length=np.linalg.norm(v,axis=-1,keepdims=True)
    cap=np.deg2rad(cap_degrees)
    return v*np.minimum(1,cap/np.maximum(length,1e-12))
