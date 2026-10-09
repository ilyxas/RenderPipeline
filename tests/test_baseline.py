import unittest
import numpy as np
from xms.solve.baseline_body import align


class BaselineMath(unittest.TestCase):
    def test_direction_rotation_and_caps(self):
        np.testing.assert_allclose(align([1,0,0],[0,1,0],180)@np.array([1,0,0]),[0,1,0],atol=1e-10)
        np.testing.assert_allclose(align([1,0,0],[-1,0,0],180)@np.array([1,0,0]),[-1,0,0],atol=1e-10)
        r=align([1,0,0],[0,1,0],30);self.assertAlmostEqual(float(np.arccos((np.trace(r)-1)/2)),np.pi/6)
