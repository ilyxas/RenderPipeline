import unittest
import numpy as np
from xms.observations.hand_identity import assign

class Identity(unittest.TestCase):
    def test_crossing_and_mirroring(self):
        body=np.array([[.7,.5],[.3,.5]]);wrists=np.array([[.31,.5],[.69,.5]])
        self.assertEqual(assign(wrists,body,[True,True]),[(1,True),(0,True)])
        self.assertEqual(assign(wrists,body,[True,True],mirrored=True),[(1,True),(0,True)])
    def test_lost_body_and_expired_history(self):
        self.assertEqual(assign([[.31,.5]],[[0,0],[0,0]],[False,False],np.array([[.3,.5],[.8,.5]])),[(0,True)])
        self.assertFalse(assign([[.31,.5]],[[0,0],[0,0]],[False,False])[0][1])
