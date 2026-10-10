import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from xms.solve.stability import rotations,angle
from xms.solve.video_body import two_bone,clearance,foot_orientation,support_root_drop


class MotionStability(unittest.TestCase):
    def test_planted_foot_keeps_sole_level_and_rest_ankle_offset(self):
        # A yawed heel-to-toe observation must not pitch the ankle-to-ball bone
        # (which slopes down to the toe) onto the horizontal sole direction.
        rest=np.array([.02,-.065,.13]);forward=np.array([.6,.1,.8])
        r=foot_orientation(forward,rest,np.eye(3),planted=True)
        self.assertAlmostEqual((r@rest)[1],rest[1])
        self.assertTrue(np.allclose(r@[0,1,0],[0,1,0]))
        delta=support_root_drop(np.array([[0,.92,0],[.1,.90,0]]),np.array([[0,.075,0],[.1,.075,0]]),[.82,.82])
        self.assertAlmostEqual(delta,.025)
        self.assertEqual(support_root_drop(np.array([[0,1.3,0]]),np.array([[0,.075,0]]),[.82]),.12)

    def test_isolated_orientation_spike_is_repaired(self):
        times=np.arange(7)/24;q=Rotation.from_euler('x',np.array([0,0,0,170,0,0,0])[:,None],degrees=True).as_quat()
        repaired,v,c,estimated=rotations(q,np.ones(7,bool),np.ones(7),times)
        self.assertTrue(estimated[3]);self.assertLess(angle(repaired[2],repaired[3]),np.deg2rad(1))

    def test_coherent_fast_gesture_retains_amplitude_and_timing(self):
        times=np.arange(21)/24;degrees=np.r_[np.zeros(5),np.linspace(0,90,6),np.full(10,90)]
        q=Rotation.from_euler('z',degrees[:,None],degrees=True).as_quat();out,_,_,estimated=rotations(q,np.ones(21,bool),np.ones(21),times)
        self.assertFalse(estimated.any());self.assertLess(angle(q[-1],out[-1]),np.deg2rad(1))
        expected=np.flatnonzero(degrees>=45)[0];actual=np.flatnonzero(Rotation.from_quat(out).as_euler('xyz',degrees=True)[:,2]>=45)[0]
        self.assertLessEqual(abs(expected-actual),1)

    def test_long_dropout_transitions_without_identity_snap(self):
        times=np.arange(40)/24;q=Rotation.from_euler('z',np.full((40,1),80),degrees=True).as_quat();v=np.ones(40,bool);v[8:30]=False;q[~v]=[0,0,0,1]
        out,valid,confidence,estimated=rotations(q,v,v.astype(float),times)
        steps=[angle(a,b) for a,b in zip(out[:-1],out[1:])]
        self.assertLess(max(steps),np.deg2rad(15));self.assertTrue(estimated.any());self.assertTrue((confidence[~valid]==0).all())

    def test_two_bone_reach_and_torso_clearance(self):
        s=np.array([.16,1.35,0]);w=clearance(np.array([0.,1.1,-.08]),np.array([0.,.9,0]),np.array([0.,1.45,0]),.2)
        self.assertGreaterEqual(w[2],.2)
        e,w=two_bone(s,w,np.array([.3,1.1,.1]),.27,.25)
        self.assertAlmostEqual(np.linalg.norm(e-s),.27,places=7);self.assertAlmostEqual(np.linalg.norm(w-e),.25,places=7)
