import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from xms.solve.image_hands import continuous_twist

class HandTwist(unittest.TestCase):
    def test_half_turn_branch_and_quaternion_sign_do_not_reverse_pronation(self):
        degrees=np.arange(0.,221.,2.)
        q=Rotation.from_rotvec(np.deg2rad(degrees)[:,None]*[0,1,0]).as_quat()
        q[::3]*=-1
        recovered=continuous_twist(q,np.tile([0,1,0],(len(q),1)))
        np.testing.assert_allclose(np.degrees(recovered),degrees,atol=1e-8)
