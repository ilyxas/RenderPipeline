import numpy as np


def measure(bundle):
    names=bundle.metadata['joint_names'];ids=[i for i,n in enumerate(names) if n.startswith(('hand_','thumb_','index_','middle_','ring_','pinky_')) and ('metacarpal' not in n)]
    a=bundle.arrays;q=a['local_rotation_delta'][:,ids];angles=np.degrees(2*np.arccos(np.clip(abs(q[:,:,3]),0,1)))
    return {'max_local_rotation_degrees':float(angles.max()),'channel_coverage':float(a['rotation_validity'][:,ids].mean()),'palm':bundle.metadata.get('palm_diagnostics',{}),'quality_accepted':False}
