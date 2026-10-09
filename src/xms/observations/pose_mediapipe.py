"""One sequential pose worker. Tracker-native observations only, never rig data."""
import json,time
from pathlib import Path
import numpy as np
from .contract import LANDMARKS
from .io import write_observations
from xms.ingest.probe import probe
from xms.ingest.timeline import make_timeline
from xms.ingest.decode import decode
from xms.animation.io import file_hash


def pack_detection(result):
    image=np.zeros((33,3));world=np.zeros((33,3));visibility=np.zeros(33);presence=np.zeros(33)
    if not result.pose_landmarks:return image,world,visibility,presence,False
    if len(result.pose_landmarks)!=1:raise ValueError('Ambiguous multi-person frame; subject selection required')
    for i,(a,b) in enumerate(zip(result.pose_landmarks[0],result.pose_world_landmarks[0])):
        image[i]=[a.x,a.y,a.z];world[i]=[b.x,b.y,b.z];visibility[i]=a.visibility;presence[i]=a.presence
    return image,world,visibility,presence,True


def observe(video,start,end,out,model,ffmpeg,ffprobe,fps=None):
    import mediapipe as mp
    from PIL import Image,ImageDraw
    out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    info=probe(video,ffprobe);timeline=make_timeline(info,start,end,fps)
    (out/'timeline.json').write_text(json.dumps(timeline,indent=2));(out/'probe.json').write_text(json.dumps(info,indent=2))
    frames,transform=decode(video,info,timeline,ffmpeg,out/'decode.log')
    n=len(frames);image=np.zeros((n,33,3));world=np.zeros_like(image);vis=np.zeros((n,33));pres=np.zeros_like(vis);valid=np.zeros(n,dtype=bool)
    options=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),running_mode=mp.tasks.vision.RunningMode.VIDEO,num_poses=2)
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
        last=-1
        for i,frame in enumerate(frames):
            timestamp=round((timeline['source_times_s'][i]-timeline['source_times_s'][0])*1000)
            if timestamp<=last:raise ValueError('PTS cannot be represented as unique MediaPipe milliseconds')
            result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(frame)),timestamp)
            image[i],world[i],vis[i],pres[i],valid[i]=pack_detection(result);last=timestamp
    arrays={'image_landmarks_raw':image,'world_landmarks_raw':world,'visibility':vis,'presence':pres,'landmark_validity':(vis>=.5)&(pres>=.5)&valid[:,None],'frame_validity':valid,'source_pts':np.array(timeline['source_pts'],dtype=np.int64),'source_times_s':np.array(timeline['source_times_s']),'crop_transforms':np.tile(transform,(n,1,1))}
    metadata={'schema_version':'xms.observations.v1','landmark_names':LANDMARKS,'backend':'mediapipe_pose','backend_version':mp.__version__,'model_sha256':file_hash(model),'source_sha256':file_hash(video),'source_time_base':timeline['source_time_base'],'subject_id':'single_person_0','identity_policy':'single-person-only; reject multiple detections; no temporal re-identification','mirrored':False,'coordinate_systems':{'image':'display-oriented normalized x-right/y-down; z tracker-relative','world':'MediaPipe hip-centred estimate x-right/y-down/z-away; not metric ground truth'},'crop_transform_semantics':'normalized source image -> display-oriented full frame','raw_vs_cleaned':'raw only; no cleaning performed','width':frames.shape[2],'height':frames.shape[1],'sample_aspect_ratio':info['sample_aspect_ratio']}
    write_observations(out/'body',metadata,arrays)
    overlays=out/'overlays';overlays.mkdir()
    for i in sorted(set([0,n//2,n-1])):
        im=Image.fromarray(frames[i]);d=ImageDraw.Draw(im);w,h=im.size
        for side,color,chain in [('LEFT',(30,235,100),[11,13,15,23,25,27,31]),('RIGHT',(255,90,50),[12,14,16,24,26,28,32])]:
            for s,e in [(chain[0],chain[1]),(chain[1],chain[2]),(chain[0],chain[3]),(chain[3],chain[4]),(chain[4],chain[5]),(chain[5],chain[6])]:
                if arrays['landmark_validity'][i,s] and arrays['landmark_validity'][i,e]:d.line([(image[i,s,0]*w,image[i,s,1]*h),(image[i,e,0]*w,image[i,e,1]*h)],fill=color,width=5)
            j=chain[0];d.text((image[i,j,0]*w,image[i,j,1]*h),side,fill=color)
        d.rectangle((0,0,w,30),fill='black');d.text((8,8),f"PTS {timeline['source_pts'][i]} | {timeline['source_times_s'][i]:.6f}s | LEFT green / RIGHT orange",fill='white');im.save(overlays/f'{i:04d}.png')
    manifest={'stage':3,'status':'passed','exit_status':0,'seconds':time.monotonic()-started,'parameters':{'start_s':start,'end_s':end,'fps':timeline['fps']},'source_sha256':metadata['source_sha256'],'model_sha256':metadata['model_sha256'],'numpy':np.__version__,'mediapipe':mp.__version__,'frames':n,'valid_frames':int(valid.sum()),'arrays_sha256':file_hash(out/'body/arrays.npz'),'metadata_sha256':file_hash(out/'body/metadata.json'),'decoded_pts_verified':True}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return out
