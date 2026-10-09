import unittest
import numpy as np
from xms.qa.pose import project
from xms.qa.kinematics import rotation_velocity

class Metrics(unittest.TestCase):
    def test_known_orthographic_projection(self):
        camera={'world_anchor_m':[0,1,0],'world_to_camera_rotation':np.eye(3).tolist(),'metres_per_normalized_image_height':2,'image_anchor':[.5,.5],'display_aspect':2}
        np.testing.assert_allclose(project(np.array([[1,2,7],[0,1,-3]]),camera),[[.75,0],[.5,.5]])
    def test_constant_angular_speed_and_quaternion_sign(self):
        times=np.array([0.,.1,.2]);angle=times*2
        q=np.zeros((3,1,4));q[:,0,2]=np.sin(angle/2);q[:,0,3]=np.cos(angle/2);q[1]*=-1
        np.testing.assert_allclose(rotation_velocity(q,times),np.array([[[0,0,2]],[[0,0,2]]]),atol=1e-12)
