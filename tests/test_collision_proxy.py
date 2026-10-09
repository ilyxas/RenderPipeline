import unittest
import numpy as np
from xms.geometry.proxies import segment_distance,capsule_depth,plane_depth

class Proxies(unittest.TestCase):
    def test_crossing_parallel_and_points(self):
        self.assertAlmostEqual(segment_distance([0,0,0],[1,0,0],[.5,-1,0],[.5,1,0]),0)
        self.assertAlmostEqual(segment_distance([0,0,0],[1,0,0],[0,2,0],[1,2,0]),2)
        self.assertAlmostEqual(segment_distance([0,0,0],[0,0,0],[1,0,0],[1,0,0]),1)
        self.assertAlmostEqual(capsule_depth([0,0,0],[1,0,0],.1,[0,.15,0],[1,.15,0],.1),.05)
        np.testing.assert_allclose(plane_depth([[0,-.01,0]],[0,1,0],0),[.01])
