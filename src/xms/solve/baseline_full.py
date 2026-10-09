from .baseline_body import solve as solve_body,LIMITATIONS
from .baseline_face import solve_face
from .head import solve_head
from .baseline_hands import solve_hands
from xms.animation.compose import compose

FULL_LIMITATIONS=[x for x in LIMITATIONS if not x.startswith(('Body-only','Face, gaze'))]+['Body directions remain an unfiltered baseline.','Facial expressions and head are video-only; no audio articulation fusion or lip constraints.','Face confidence uses availability/ROI size; hand confidence uses handedness/body-wrist association proxies.','Missing face frames use neutral expression/body head prior; missing hands/fingers use rest. No gap interpolation.','Hand association is frame-local; temporal L/R identity recovery, contact and finger collisions are not implemented.']


def solve_full(body_observations,face_observations,hand_observations,timeline,profile,rig,face_map,calibration=None):
    body=solve_body(body_observations,timeline,profile,rig,calibration)
    face=solve_face(face_observations,timeline,profile,face_map);head=solve_head(body,face_observations,timeline);hands=solve_hands(body,hand_observations,timeline)
    result=compose(body,face,head,hands);result.metadata['limitations']=FULL_LIMITATIONS;result.metadata['solver']['channels']='body_head_face_hands';return result
