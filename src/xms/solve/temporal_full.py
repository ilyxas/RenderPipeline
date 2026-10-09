"""Stage 8 body backend with unchanged Stage 6 face/head/hand ownership."""
from .windows import solve as solve_body
from .temporal_body import LIMITATIONS
from .baseline_face import solve_face
from .head import solve_head
from .baseline_hands import solve_hands
from xms.animation.compose import compose


def solve_full(body_observations,face_observations,hand_observations,timeline,profile,rig,face_map,calibration=None,max_nfev=80):
    body=solve_body(body_observations,timeline,profile,rig,calibration,max_nfev)
    face=solve_face(face_observations,timeline,profile,face_map);head=solve_head(body,face_observations,timeline);hands=solve_hands(body,hand_observations,timeline)
    result=compose(body,face,head,hands);result.metadata['limitations']=LIMITATIONS+['Face/head/hands retain the Stage 6 baseline backends; Stage 8 optimizes body only.'];result.metadata['solver']['channels']='temporal_body_with_baseline_head_face_hands';return result
