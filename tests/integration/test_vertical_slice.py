"""Verify the real run's independent stage evidence, without repeating rendering."""
import json,sys
from pathlib import Path
import numpy as np
from xms.animation.io import read_bundle,bundle_hash,file_hash
from xms.observations.io import read_observations


def verify(path):
    p=Path(path);m=json.loads((p/'manifest.json').read_text());b=read_bundle(p/'animation');om,oa=read_observations(p/'observations/body')
    assert m['exit_status']==0
    assert bundle_hash(p/'animation')==m['bundle_hash']==m['stages']['render']['bundle_hash']
    assert b.metadata['observations_arrays_sha256']==file_hash(p/'observations/body/arrays.npz')
    assert b.metadata['observations_source_sha256']==om['source_sha256']==m['input_hashes']['video']
    assert m['verification']['full_decode_exit_status']==0 and m['verification']['frame_count']==len(b.arrays['sample_times_s'])
    assert not b.arrays['face_validity'].any() and not b.arrays['face_weights'].any()
    hands=[i for i,n in enumerate(b.metadata['joint_names']) if n.startswith(('hand_','thumb_','index_','middle_','ring_','pinky_'))]
    assert not b.arrays['rotation_validity'][:,hands].any()
    assert np.max(np.ptp(b.arrays['local_rotation_delta'],axis=0))>.01,'No actual motion in bundle'
    assert b.metadata['contacts']==[] and b.metadata['events']==[]
    assert file_hash(p/'outputs/video.mp4')==m['output_sha256']
    print('PASS: new observations -> bundle -> render provenance; neutral face/hands; timing; real motion')


if __name__=='__main__':verify(sys.argv[1])
