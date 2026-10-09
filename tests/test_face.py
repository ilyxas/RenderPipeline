import unittest
import numpy as np
from xms.solve.face import fill_short
from xms.solve.head import rotation

class Face(unittest.TestCase):
    def test_short_long_and_edge_gaps(self):
        t=np.arange(12)/24;v=np.ones((12,1),bool);v[2]=False;v[5:10]=False;v[0]=False
        w=np.arange(12.)[:,None];out,mask,c,e=fill_short(w,v,v.astype(float),t)
        self.assertTrue(mask[2,0]);self.assertAlmostEqual(out[2,0],2);self.assertFalse(mask[0,0]);self.assertFalse(mask[6,0]);self.assertEqual(c[2,0],.5)
    def test_head_removes_scale(self):
        r=rotation(np.diag([2.,3.,4.,1.]));np.testing.assert_allclose(r,np.eye(3))
