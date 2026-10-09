import itertools
import numpy as np
from .details_io import write
from xms.animation.io import file_hash


def assign_sides(wrists,body,body_valid):
    """Frame-local anatomical association. Mirroring both image sets preserves sides."""
    count=len(wrists)
    if not count:return []
    choices=[]
    for sides in itertools.permutations(range(2),count):
        distances=[np.linalg.norm(wrists[j]-body[side]) if body_valid[side] else 1. for j,side in enumerate(sides)]
        choices.append((sum(distances),sides,distances))
    _,sides,distances=min(choices,key=lambda x:x[0])
    return [(side,d<=.2) for side,d in zip(sides,distances)]


def observe_hands(frames,timeline,transform,body,model,out,source_hash):
    import mediapipe as mp
    n=len(frames);image=np.zeros((n,2,21,3));world=np.zeros_like(image);valid=np.zeros((n,2),bool);confidence=np.zeros((n,2));raw=np.full((n,2),-1,dtype=np.int8);association=np.zeros((n,2),dtype=np.uint8)
    options=mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),running_mode=mp.tasks.vision.RunningMode.VIDEO,num_hands=2)
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        for i,frame in enumerate(frames):
            result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(frame)),round((timeline['source_times_s'][i]-timeline['source_times_s'][0])*1000))
            wrists=np.array([[h[0].x,h[0].y] for h in result.hand_landmarks]);assignment=assign_sides(wrists,body['image_landmarks_raw'][i,[15,16],:2],body['landmark_validity'][i,[15,16]])
            for j,(side,matched) in enumerate(assignment):
                if not matched:continue
                category=result.handedness[j][0];raw[i,side]=0 if category.category_name.lower()=='left' else 1
                image[i,side]=[[p.x,p.y,p.z] for p in result.hand_landmarks[j]];world[i,side]=[[p.x,p.y,p.z] for p in result.hand_world_landmarks[j]]
                valid[i,side]=True;confidence[i,side]=min(float(category.score),float(body['visibility'][i,[15,16][side]]));association[i,side]=1
    a={'source_pts':np.array(timeline['source_pts'],dtype=np.int64),'source_times_s':np.array(timeline['source_times_s']),'crop_transforms':np.tile(transform,(n,1,1)),'image_landmarks_raw':image,'world_landmarks_raw':world,'validity':valid,'confidence':confidence,'raw_handedness':raw,'association_source':association}
    m={'schema_version':'xms.hand_observations.v1','sides':['left','right'],'landmark_names':['wrist']+[f'{finger}_{part}' for finger in ['thumb','index','middle','ring','pinky'] for part in ['base','proximal','distal','tip']],'source_sha256':source_hash,'model_sha256':file_hash(model),'backend_version':mp.__version__,'mirrored':False,'subject_id':'single_person_0','coordinate_system':'tracker camera x-right,y-down,z-away; hip/wrist-relative estimates, not metric truth','association_policy':'one-to-one nearest reliable body wrist within 0.2 normalized image distance; ambiguous/unassociated detections invalid; no temporal identity recovery','raw_handedness_codes':{'unavailable':-1,'left':0,'right':1},'association_codes':{'unobserved':0,'body_wrist':1},'confidence_policy':'min handedness classification score and body wrist visibility; baseline proxy, not landmark confidence','crop_transform_semantics':'normalized source image -> display-oriented full frame','raw_vs_cleaned':'raw only'}
    write(out,m,a);return m,a
