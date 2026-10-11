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


def two_bone(shoulder,target,hint,l1,l2,reach_margin=.005):
    delta=target-shoulder;distance=np.linalg.norm(delta);axis=delta/max(distance,1e-8)
    distance=np.clip(distance,abs(l1-l2)+reach_margin,l1+l2-reach_margin);target=shoulder+axis*distance
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


def foot_orientation(forward,rest_forward,rest_rotation,planted=False):
    """Heel-to-toe direction is not the downward-sloping ankle-to-ball bone."""
    forward=np.array(forward,dtype=float,copy=True)
    if planted:forward[1]=0
    lateral=np.cross([0.,1,0],forward)
    reference=np.array(rest_forward,dtype=float,copy=True);reference[1]=0
    reference_lateral=np.cross([0.,1,0],reference)
    return frame(lateral,np.cross(forward,lateral))@frame(reference_lateral,[0.,1,0]).T@rest_rotation


def support_root_drop(hips,anchors,lengths):
    """One shared vertical correction makes all planted legs reachable."""
    available=np.sqrt(np.maximum(0,np.asarray(lengths)**2-np.sum((hips[:,[0,2]]-anchors[:,[0,2]])**2,axis=1)))
    return float(np.clip(np.max(hips[:,1]-anchors[:,1]-available),0,.12))


def solve(observations,timeline,profile,rig):
    from .image_body import solve as retarget
    return retarget(observations,timeline,profile,rig)
