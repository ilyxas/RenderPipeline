"""Named face/hand observations; independent of tracker runtime and target rig."""
import json
from pathlib import Path
import numpy as np


def validate(m,a):
    n=len(a['source_pts']);kind=m['schema_version']
    if kind=='xms.face_observations.v1':
        names=m['blendshape_names']
        if len(set(names))!=len(names):raise ValueError('Duplicate face observation channels')
        shapes={'source_pts':(n,),'source_times_s':(n,),'crop_transforms':(n,3,3),'landmarks_raw':(n,478,3),'blendshapes_raw':(n,len(names)),'face_matrix_raw':(n,4,4),'validity':(n,),'confidence':(n,),'roi_size_px':(n,2)}
    elif kind=='xms.hand_observations.v1':
        if m['sides']!=['left','right']:raise ValueError('Hand side contract mismatch')
        shapes={'source_pts':(n,),'source_times_s':(n,),'crop_transforms':(n,3,3),'image_landmarks_raw':(n,2,21,3),'world_landmarks_raw':(n,2,21,3),'validity':(n,2),'confidence':(n,2),'raw_handedness':(n,2),'association_source':(n,2)}
    else:raise ValueError('Unsupported detail observations')
    if set(a)!=set(shapes):raise ValueError('Unexpected detail arrays')
    for k,shape in shapes.items():
        dtype='b' if k=='validity' else 'iu' if k in ('source_pts','raw_handedness','association_source') else 'f'
        if a[k].shape!=shape or a[k].dtype.kind not in dtype or not np.isfinite(a[k]).all():raise ValueError('Invalid detail array '+k)
    if not n or np.any(np.diff(a['source_pts'])<=0) or np.any(np.diff(a['source_times_s'])<=0):raise ValueError('Invalid detail timeline')
    if np.any((a['confidence']<0)|(a['confidence']>1)) or np.any(a['confidence'][~a['validity']]!=0):raise ValueError('Invalid detail confidence')
    return m,a


def write(path,m,a):
    validate(m,a);p=Path(path);p.mkdir(parents=True,exist_ok=False);(p/'metadata.json').write_text(json.dumps(m,indent=2));np.savez_compressed(p/'arrays.npz',**a)


def read(path):
    p=Path(path)
    with np.load(p/'arrays.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    return validate(json.loads((p/'metadata.json').read_text()),a)
