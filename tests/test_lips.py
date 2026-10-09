import unittest
import numpy as np
from xms.solve.lips import fuse
from xms.audio.envelope import extract

class Lips(unittest.TestCase):
    def test_good_video_and_silence(self):
        face={'weights':np.array([[.4],[.4]]),'confidence':np.ones((2,1))*.9,'validity':np.ones((2,1),bool),'provenance':np.ones((2,1),np.uint8)}
        audio={'envelope':np.ones(2),'confidence':np.ones(2)*.15,'times':np.arange(2)/24}
        result=fuse(face,audio,['jawOpen']);self.assertLess(abs(result['weights'][0,0]-.4),.001)
        silent=extract(np.zeros(16000),16000,np.arange(2)/24);audio.update(silent);np.testing.assert_array_equal(fuse(face,audio,['jawOpen'])['weights'],face['weights'])
