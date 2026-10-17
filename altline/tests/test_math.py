import unittest

import numpy as np

from altline.quat import (mat_to_quat, quat_angle_between, quat_from_axis_angle, quat_mul, quat_to_mat,
                          rot_between, slerp, swing_twist, twist_angle, is_rotation)
from altline.skeleton import reference_skeleton
from altline.tests.synthetic import random_pose


class QuatTests(unittest.TestCase):
    def test_matrix_roundtrip(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            back = mat_to_quat(quat_to_mat(q))
            self.assertLess(quat_angle_between(q, back), 1e-9)
            self.assertAlmostEqual(np.linalg.norm(back), 1.0, places=12)

    def test_swing_twist_recomposes(self):
        rng = np.random.default_rng(2)
        y = np.array([0.0, 1.0, 0.0])
        for _ in range(100):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            swing, twist = swing_twist(q, y)
            self.assertLess(quat_angle_between(quat_mul(swing, twist), q), 1e-7)
            self.assertAlmostEqual(swing[2], 0.0, places=9)  # swing has no component about Y

    def test_twist_angle_sign(self):
        y = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(np.degrees(twist_angle(quat_from_axis_angle(y, np.radians(70)), y)), 70.0)
        self.assertAlmostEqual(np.degrees(twist_angle(quat_from_axis_angle(y, np.radians(-70)), y)), -70.0)

    def test_rot_between_antiparallel(self):
        a = np.array([0.0, 1.0, 0.0])
        r = rot_between(a, -a)
        self.assertTrue(is_rotation(r))
        np.testing.assert_allclose(r @ a, -a, atol=1e-12)

    def test_slerp_endpoints(self):
        a = quat_from_axis_angle([1, 0, 0], 0.3)
        b = -quat_from_axis_angle([0, 1, 0], 1.2)  # opposite hemisphere on purpose
        self.assertLess(quat_angle_between(slerp(a, b, 0.0), a), 1e-9)
        self.assertLess(quat_angle_between(slerp(a, b, 1.0), b), 1e-9)


class SkeletonTests(unittest.TestCase):
    def test_rest_fk_reproduces_rest(self):
        skel = reference_skeleton()
        _, heads, tails = skel.fk(np.repeat(np.eye(3)[None], len(skel.names), axis=0))
        np.testing.assert_allclose(heads, skel.heads, atol=1e-12)
        np.testing.assert_allclose(tails, skel.tails, atol=1e-12)

    def test_rest_frames_are_rotations(self):
        skel = reference_skeleton()
        for r in skel.rest:
            self.assertTrue(is_rotation(r))

    def test_lengths_constant_under_any_pose(self):
        skel = reference_skeleton()
        rng = np.random.default_rng(3)
        for _ in range(20):
            _, heads, tails = skel.fk(random_pose(skel, rng))
            np.testing.assert_allclose(np.linalg.norm(tails - heads, axis=1), skel.lengths, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
