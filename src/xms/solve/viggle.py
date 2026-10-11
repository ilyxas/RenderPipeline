"""External mocap backend: Viggle MetaHuman-body rotations retargeted onto Xandra.

Rotations, not points. Both skeletons share names and world axes (Y up, Z
forward, +X left) but not bone-local axes: Xandra's world bone frame is the
Viggle frame times one constant offset. Each mapped bone takes Viggle's world
orientation, so Viggle's own rest (a T-pose) becomes a T-posed Xandra; unmapped
bones (correctives, extra twist, fingers for channels=body) keep their rest
local transform. Local channels are derived once, down the Xandra hierarchy.
"""
import copy,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from xms.animation.bundle import neutral
from xms.animation.fk import evaluate,quat_matrix
from xms.animation.gltf import Skeleton
from xms.animation.io import file_hash
from xms.animation.validate import validate

SPEC=Path(__file__).resolve().parents[3]/'profiles/characters/xandra/viggle_metahuman_body.v1.json'


def continuous(q):
    q=q.copy()
    for i in range(1,len(q)):
        if q[i]@q[i-1]<0:q[i]=-q[i]
    return q


def retarget(world,pelvis,rest_pelvis,times,source_times,profile,rig,spec):
    """world: {joint: (n,3,3) Viggle world rotations}; pelvis: (n,3) Viggle world pelvis; rest_pelvis: (3,)."""
    if spec['asset_sha256']!=profile['asset_sha256']:raise ValueError('Viggle retarget spec is registered for a different character asset')
    b=neutral(profile,rig,times,source_times);a=b.arrays;names=profile['joint_names'];n=len(times)
    offset=Rotation.from_quat(spec['axis_offset']['xyzw']).as_matrix()
    armature=a['armature_transform'][:3,:3]/np.linalg.norm(a['armature_transform'][:3,:3],axis=0);actual=[]
    for j,name in enumerate(names):
        parent=int(a['parent_indices'][j]);parent_world=np.broadcast_to(armature,(n,3,3)) if parent<0 else actual[parent];rest=quat_matrix(a['rest_rotation'][j])
        if name in world:
            desired=world[name]@offset
            a['local_rotation_delta'][:,j]=continuous(Rotation.from_matrix(rest.T@np.swapaxes(parent_world,1,2)@desired).as_quat())
            a['rotation_validity'][:,j]=True;a['rotation_confidence'][:,j]=.6;a['rotation_provenance'][:,j]=1;actual.append(desired)
        else:actual.append(parent_world@rest)
    # Root: Viggle pelvis scaled by rest pelvis height, as a world offset of Xandra's pelvis.
    rest_pose=evaluate(neutral(profile,rig,times[:1]),0);root=profile['root_carrier_index']
    scale=float(rest_pose[root,1,3]/rest_pelvis[1]);a['root_translation'][:]=scale*np.asarray(pelvis)-rest_pose[root,:3,3]
    a['root_validity'][:]=True;a['root_confidence'][:]=.6;a['root_provenance'][:]=1
    # One floor shift per clip: the median lowest sole proxy goes to Y=0.
    from .grounding import sole_proxies,sole_positions
    points=sole_proxies(b,rest_pose);lowest=np.array([sole_positions(evaluate(b,i),points)[:,:,1].min() for i in range(n)]);shift=-float(np.median(lowest))
    a['root_translation'][:,1]+=shift
    return b,{'scale':scale,'floor_shift_m':shift,'lowest_sole_after_m':[float((lowest+shift).min()),float((lowest+shift).max())]}


def solve(glb,timeline,profile,rig,spec_path=SPEC):
    spec=json.loads(Path(spec_path).read_text());skeleton=Skeleton(glb)
    missing=[x for x in spec['body_joints'] if x not in skeleton.names]
    if missing:raise ValueError('Viggle GLB lacks mapped joints: '+', '.join(missing[:5]))
    times=np.array(timeline['output_times_s']);source=np.array(timeline['source_time_map'])
    rotations,positions=skeleton.world(source);_,rest_positions=skeleton.world()
    index={x:skeleton.names.index(x) for x in spec['body_joints']};pelvis=index['pelvis']
    world={x:rotations[:,i] for x,i in index.items()}
    b,root=retarget(world,positions[:,pelvis],rest_positions[0,pelvis],times,source,profile,rig,spec)
    from .twist import distribute
    from .limits import apply as arm_limits
    # Twist bones are derived from the mapped arm roll (body prior), not observed.
    a=b.arrays;names=profile['joint_names']
    for side in ('l','r'):
        for name in ('upperarm_twist_01','upperarm_twist_02','lowerarm_twist_01','lowerarm_twist_02'):
            j=names.index(name+'_'+side)
            if not a['rotation_validity'][0,j]:a['rotation_validity'][:,j]=True;a['rotation_confidence'][:,j]=.5;a['rotation_provenance'][:,j]=2
    distribute(b);probe=arm_limits(copy.deepcopy(b),profile).metadata.get('arm_limit_diagnostics',{})
    keys=max((len(t) for t,_ in skeleton.channels.values()),default=0);fps=(keys-1)/skeleton.duration if skeleton.duration else None
    b.metadata.update(solver={'backend':'viggle_metahuman_body','version':'1','channels':'body'},
        calibration={'schema_version':'xms.calibration.v1','policy':'external_viggle_metahuman_body','source_glb_sha256':file_hash(glb),'image_plane_metres_per_normalized_height':None,'root_scale':root['scale'],'floor_shift_m':root['floor_shift_m']},
        viggle_diagnostics={'source_glb':str(glb),'source_glb_sha256':file_hash(glb),'source_joints':len(skeleton.names),'mapped_joints':len(index),'source_fps':fps,'source_duration_s':skeleton.duration,
            'clip_end_s':float(source[-1]),'clip_beyond_source_s':float(max(0.,source[-1]-skeleton.duration)),**root,
            'arm_limits_not_applied_would_trigger':{s:{k:v for k,v in probe.get(s,{}).items() if k.endswith('samples')} for s in ('l','r')},'quality_accepted':False},
        limitations=['External Viggle extraction (MetaHuman body); fingers, face and hands are not retargeted with channels=body.','Root height uses one floor shift per clip; contacts are not enforced.','Arm anatomical limits are reported, not applied, in this comparison backend.'])
    return validate(b)
