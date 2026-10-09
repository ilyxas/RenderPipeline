import numpy as np
from .details_io import write
from xms.animation.io import file_hash


def observe_face(frames,timeline,transform,model,out,source_hash,body):
    import mediapipe as mp
    n=len(frames);landmarks=np.zeros((n,478,3));matrices=np.tile(np.eye(4),(n,1,1));valid=np.zeros(n,bool);confidence=np.zeros(n);roi=np.zeros((n,2));scores=[];names=None;transforms=np.tile(transform,(n,1,1))
    options=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),running_mode=mp.tasks.vision.RunningMode.VIDEO,num_faces=1,output_face_blendshapes=True,output_facial_transformation_matrixes=True)
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        for i,frame in enumerate(frames):
            h,w=frame.shape[:2];points=body['image_landmarks_raw'][i,:,:2]*[w,h]
            usable=body['landmark_validity'][i,[0,2,5,7,8]]
            if usable.any():
                center=np.mean(points[[0,2,5,7,8]][usable],axis=0)
                size=max(96.,2.6*np.linalg.norm(points[7]-points[8]),.9*np.linalg.norm(points[11]-points[12]))
                x0=max(0,int(center[0]-size/2));y0=max(0,int(center[1]-size/2));x1=min(w,int(center[0]+size/2));y1=min(h,int(center[1]+size/2))
            else:x0,y0,x1,y1=0,0,w,h
            crop=frame[y0:y1,x0:x1];cw,ch=x1-x0,y1-y0
            transforms[i]=np.array([[w/cw,0,-x0/cw],[0,h/ch,-y0/ch],[0,0,1.]])@transform
            result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(crop)),round((timeline['source_times_s'][i]-timeline['source_times_s'][0])*1000))
            if not result.face_landmarks:scores.append(None);continue
            observed={c.category_name:float(c.score) for c in result.face_blendshapes[0]}
            if names is None:names=sorted(observed)
            if set(observed)!=set(names):raise ValueError('Face channel set changed during extraction')
            landmarks[i]=[[p.x,p.y,p.z] for p in result.face_landmarks[0]];matrices[i]=np.array(result.facial_transformation_matrixes[0])
            landmarks[i,:,0]=(landmarks[i,:,0]*cw+x0)/w;landmarks[i,:,1]=(landmarks[i,:,1]*ch+y0)/h;landmarks[i,:,2]*=cw/w
            roi[i]=np.ptp(landmarks[i,:,:2],axis=0)*[frame.shape[1],frame.shape[0]]
            # Availability and projected ROI size only; expression intensity is not confidence.
            valid[i]=min(roi[i])>=32;confidence[i]=min(.9,min(roi[i])/160) if valid[i] else 0;scores.append(observed)
    if names is None:names=[]
    weights=np.array([[0. if row is None else row[k] for k in names] for row in scores],dtype=float).reshape(n,len(names))
    a={'source_pts':np.array(timeline['source_pts'],dtype=np.int64),'source_times_s':np.array(timeline['source_times_s']),'crop_transforms':transforms,'landmarks_raw':landmarks,'blendshapes_raw':weights,'face_matrix_raw':matrices,'validity':valid,'confidence':confidence,'roi_size_px':roi}
    m={'schema_version':'xms.face_observations.v1','blendshape_names':names,'landmark_names':[f'face_{i}' for i in range(478)],'source_sha256':source_hash,'model_sha256':file_hash(model),'backend_version':mp.__version__,'mirrored':False,'subject_id':'single_person_0','confidence_policy':'availability and min ROI side / 160 px capped at .9, minimum 32px; not detection probability and never blendshape intensity','matrix_convention':'MediaPipe canonical face to camera metric transform; x-right, y-up, z-toward-camera. Scale/translation retained raw.','crop_transform_semantics':'normalized source image -> display-oriented full frame','raw_vs_cleaned':'raw only'}
    m['crop_transform_semantics']='normalized original source -> tracker head ROI; output landmarks mapped back to full display-oriented image'
    m['roi_policy']='single pass; center of reliable nose/eyes/ears; size max(96px,2.6 ear span,0.9 shoulder span); detector thresholds unchanged'
    write(out,m,a);return m,a
