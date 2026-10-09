import numpy as np

LANDMARKS=['nose','left_eye_inner','left_eye','left_eye_outer','right_eye_inner','right_eye','right_eye_outer','left_ear','right_ear','mouth_left','mouth_right','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_pinky','right_pinky','left_index','right_index','left_thumb','right_thumb','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_heel','right_heel','left_foot_index','right_foot_index']


def validate(metadata,a):
    if metadata['schema_version']!='xms.observations.v1':raise ValueError('Unsupported observations version')
    if metadata['landmark_names']!=LANDMARKS:raise ValueError('Unexpected landmark names')
    n=len(a['source_pts'])
    shapes={'image_landmarks_raw':(n,33,3),'world_landmarks_raw':(n,33,3),'visibility':(n,33),'presence':(n,33),'landmark_validity':(n,33),'frame_validity':(n,),'source_pts':(n,),'source_times_s':(n,),'crop_transforms':(n,3,3)}
    if set(a)!=set(shapes):raise ValueError('Unexpected observation arrays')
    for key,shape in shapes.items():
        dtype='b' if key.endswith('validity') else 'iu' if key=='source_pts' else 'f'
        if a[key].shape!=shape or a[key].dtype.kind not in dtype or not np.all(np.isfinite(a[key])):raise ValueError('Invalid '+key)
    if n==0 or np.any(np.diff(a['source_pts'])<=0) or np.any(np.diff(a['source_times_s'])<=0):raise ValueError('Invalid observation times')
    for key in ('visibility','presence'):
        if np.any((a[key]<0)|(a[key]>1)):raise ValueError('Invalid '+key)
    if np.any(a['landmark_validity'][~a['frame_validity']]):raise ValueError('Invalid frame has valid landmarks')
    if np.any(a['visibility'][~a['frame_validity']]) or np.any(a['presence'][~a['frame_validity']]):raise ValueError('Missing detection has confidence')
    return metadata,a
