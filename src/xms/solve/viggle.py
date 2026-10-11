"""External mocap backend: Viggle MetaHuman-body rotations retargeted onto Xandra.

Rotations, not points. Both skeletons share names and world axes (Y up, Z
forward, +X left) but not bone-local axes: Xandra's world bone frame is the
Viggle frame times one constant offset. Each mapped bone takes Viggle's world
orientation, so Viggle's own rest (a T-pose) becomes a T-posed Xandra; unmapped
bones (correctives, extra twist, fingers for channels=body) keep their rest
local transform. Local channels are derived once, down the Xandra hierarchy.
"""
import copy,json,re
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
    # Viggle's world origin is not the subject's spot (about 1.6 m behind in practice):
    # horizontal travel is kept relative to the first sample, height stays absolute.
    a['root_translation'][:,[0,2]]=scale*(np.asarray(pelvis)[:,[0,2]]-np.asarray(pelvis)[0,[0,2]])
    a['root_validity'][:]=True;a['root_confidence'][:]=.6;a['root_provenance'][:]=1
    # Per-frame support: the lowest sole proxy goes to Y=0 each sample, lightly smoothed.
    # This absorbs camera-induced root drift; airborne phases (jumps) are not modelled.
    from scipy.ndimage import gaussian_filter1d
    from .grounding import sole_proxies,sole_positions
    points=sole_proxies(b,rest_pose);lowest=np.array([sole_positions(evaluate(b,i),points)[:,:,1].min() for i in range(n)])
    fps=1/float(np.median(np.diff(times))) if n>1 else 24.;shift=gaussian_filter1d(-lowest,max(.05*fps,.01),mode='nearest')
    a['root_translation'][:,1]+=shift;after=lowest+shift
    return b,{'scale':scale,'floor_policy':'per-frame lowest sole at Y=0, 50 ms smoothing, no flight','floor_shift_range_m':[float(shift.min()),float(shift.max())],'lowest_sole_after_m':[float(after.min()),float(after.max())]}


FINGER=re.compile(r'^(thumb|index|middle|ring|pinky)_(metacarpal|\d\d)_[lr]$')


def solve(glb,timeline,profile,rig,source_sha256=None,fingers=False,spec_path=SPEC):
    spec=json.loads(Path(spec_path).read_text());skeleton=Skeleton(glb)
    if fingers:spec['body_joints']=spec['body_joints']+[x for x in skeleton.names if FINGER.match(x) and x in profile['joint_names']]
    missing=[x for x in spec['body_joints'] if x not in skeleton.names]
    if missing:raise ValueError('Viggle GLB lacks mapped joints: '+', '.join(missing[:5]))
    times=np.array(timeline['output_times_s']);source=np.array(timeline['source_time_map'])
    # Viggle samples the whole file from its first frame; timeline source times are stream-absolute.
    viggle_times=source-float(timeline.get('video_start_s',0.))
    rotations,positions=skeleton.world(viggle_times);_,rest_positions=skeleton.world()
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
    if source_sha256:b.metadata['observations_source_sha256']=source_sha256
    b.metadata.update(solver={'backend':'viggle_metahuman_body','version':'1','channels':'body_fingers' if fingers else 'body'},
        calibration={'schema_version':'xms.calibration.v1','policy':'external_viggle_metahuman_body','source_glb_sha256':file_hash(glb),'image_plane_metres_per_normalized_height':None,'root_scale':root['scale'],'floor_policy':root['floor_policy']},
        viggle_diagnostics={'source_glb':str(glb),'source_glb_sha256':file_hash(glb),'source_joints':len(skeleton.names),'mapped_joints':len(index),'source_fps':fps,'source_duration_s':skeleton.duration,
            'clip_end_s':float(viggle_times[-1]),'clip_beyond_source_s':float(max(0.,viggle_times[-1]-skeleton.duration)),**root,
            'arm_limits_not_applied_would_trigger':{s:{k:v for k,v in probe.get(s,{}).items() if k.endswith('samples')} for s in ('l','r')},'quality_accepted':False},
        limitations=['External Viggle extraction (MetaHuman body); '+('fingers are retargeted from Viggle.' if fingers else 'fingers, face and hands are not retargeted with channels=body.'),'Root height keeps the lowest sole on the floor every sample; jumps and horizontal foot locking are not modelled.','Arm anatomical limits are reported, not applied, in this comparison backend.'])
    return validate(b)


def solve_full(glb,face_obs,timeline,profile,rig,face_map,audio=None,source_sha256=None):
    """Viggle body, head and fingers; ARKit face morphs (and audio lip cues) from the video face pipeline."""
    from xms.animation.compose import compose
    from .face import solve_face
    b=solve(glb,timeline,profile,rig,source_sha256,fingers=True);face=solve_face(face_obs,timeline,profile,face_map)
    if audio is not None:
        from .lips import fuse
        face=fuse(face,audio,profile['face_channel_names'])
    # The Viggle head is kept: the compositor sees no face-driven head samples.
    head={'index':profile['joint_names'].index('head'),'face_validity':np.zeros(len(timeline['output_times_s']),bool)};hands={'indices':[]}
    result=compose(b,face,head,hands)
    result.metadata.update(face_diagnostics=face['diagnostics'],lip_diagnostics=face.get('lip_diagnostics',{}))
    result.metadata['solver']['channels']='viggle_body_head_fingers_video_face'
    result.metadata['limitations']=result.metadata['limitations']+['Face morphs come from the video face tracker; head orientation comes from Viggle.']
    return validate(result)
