"""Practical full video reconstruction; baseline remains a separate fallback backend."""
from xms.animation.compose import compose
from xms.animation.validate import validate
from .face import solve_face
from .head import solve_head
from .hands import solve_hands,palm_correction


def solve_full(body_obs,face_obs,hand_obs,timeline,profile,rig,face_map,solver='video',audio=None):
    if solver=='temporal':
        from .windows import solve
    else:
        from .video_body import solve
    body=solve(body_obs,timeline,profile,rig)
    if solver=='temporal':
        from .stability import stabilize_bundle
        stabilize_bundle(body)
    face=solve_face(face_obs,timeline,profile,face_map)
    if audio is not None:
        from .lips import fuse
        face=fuse(face,audio,profile['face_channel_names'])
    head=solve_head(body,face_obs,timeline,stable=True);hands=solve_hands(body,hand_obs,timeline);result=compose(body,face,head,hands)
    try:palm_correction(result)
    except ImportError:result.metadata['palm_diagnostics']={'status':'unavailable','reason':'Optional SciPy missing'}
    from .contacts import constraints
    from xms.qa.contacts import measure
    constraints(result);result.metadata['contact_diagnostics']=measure(result)
    result.metadata.update(face_diagnostics=face['diagnostics'],lip_diagnostics=face.get('lip_diagnostics',{}),hand_diagnostics=hands['diagnostics'],quality='development_candidate',limitations=body.metadata.get('limitations',[])+['Single visible upright person; monocular depth/floor are estimates.','Missing channels use bounded smooth transitions; long occlusions remain uncertain.','Temporal optimizer, palm constraints and audio lip cues are experimental; quality acceptance pending.'])
    if profile.get('collision_proxies'):
        from xms.qa.collision_proxy import measure as proxy_measure
        result.metadata['collision_proxy_diagnostics']=proxy_measure(result,profile['collision_proxies'])
    result.metadata['solver']['channels']='video_face_head_temporal_hands';return validate(result)
