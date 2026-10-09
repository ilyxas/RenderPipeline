import unittest
import numpy as np
from xms.geometry.surface_distance import inside,watertight
from xms.qa.surface_collision import assess

class Surface(unittest.TestCase):
    def test_tetrahedron_sign_and_open_rejection(self):
        v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1.]]);f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
        self.assertTrue(inside([.1,.1,.1],v,f));self.assertFalse(inside([2,2,2],v,f))
        with self.assertRaises(ValueError):inside([.1,.1,.1],v,f[:-1])
    def test_partial_is_not_accepted(self):
        report=assess([{'candidate_depths_m':[],'signed_interior_available':False}],[0,1],[0]);self.assertEqual(report['status'],'unavailable');self.assertFalse(report['surface_quality_accepted'])
