import json,sys
from pathlib import Path
import numpy as np
from xms.observations.io import read_observations


def verify(path):
    p=Path(path);m,a=read_observations(p/'body');t=json.loads((p/'timeline.json').read_text())
    assert 'mediapipe' not in sys.modules, 'Reader must not import tracker runtime'
    np.testing.assert_array_equal(a['source_pts'],t['source_pts'])
    np.testing.assert_allclose(a['source_times_s'],t['source_times_s'],atol=1e-10)
    assert a['frame_validity'].any()
    assert len(list((p/'overlays').glob('*.png')))==3
    print('PASS: independent observations reader, explicit PTS, valid detections, overlays')


if __name__=='__main__':verify(sys.argv[1])
