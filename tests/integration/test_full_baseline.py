import json,sys
from pathlib import Path
import numpy as np
from xms.animation.io import read_bundle,bundle_hash,file_hash
from xms.observations.details_io import read


def verify(path):
    p=Path(path);m=json.loads((p/'manifest.json').read_text());b=read_bundle(p/'animation');fm,fa=read(p/'observations/face');hm,ha=read(p/'observations/hands')
    assert 'mediapipe' not in sys.modules,'Reader imports tracker runtime'
    assert b.metadata['face_observations_sha256']==file_hash(p/'observations/face/arrays.npz')
    assert b.metadata['hand_observations_sha256']==file_hash(p/'observations/hands/arrays.npz')
    assert np.ptp(b.arrays['face_weights'],axis=0).max()>0
    for side in ('l','r'):
        finger=[i for i,n in enumerate(b.metadata['joint_names']) if n in [f'{f}_{k:02d}_{side}' for f in ['thumb','index','middle','ring','pinky'] for k in [1,2,3]]]
        assert b.arrays['rotation_validity'][:,finger].any(),side+' no fingers observed'
        assert np.ptp(b.arrays['local_rotation_delta'][:,finger],axis=0).max()>0,side+' static fingers'
    assert 'face' in b.metadata['head_source_by_sample']
    for view,artifact in m['diagnostic_videos'].items():
        assert artifact['bundle_hash']==bundle_hash(p/'animation')
        assert Path(artifact['path']).exists() and artifact['verification']['full_decode_exit_status']==0
    np.testing.assert_array_equal(fa['source_pts'],ha['source_pts'])
    assert m['verification']['full_decode_exit_status']==0
    print('PASS: full bundle motion, named detail readers without tracker, same-bundle diagnostic MP4s')


if __name__=='__main__':verify(sys.argv[1])
