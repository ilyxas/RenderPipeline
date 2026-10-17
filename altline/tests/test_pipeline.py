import tempfile
import unittest
from pathlib import Path

import numpy as np

from altline import animation, calibration, limits, solver
from altline.calibration import derived_joints
from altline.config import LimitsConfig, SolverConfig
from altline.io_utils import read_json, write_json
from altline.quat import (quat_angle_between, quat_from_axis_angle, quat_mul, quat_to_mat, mat_to_quat,
                          rot_axis_angle)
from altline.skeleton import Skeleton, frame_from_hint, reference_skeleton
from altline.tests.synthetic import (NEUTRAL_PROFILE, X, Y, Z, landmarks_from_pose, q, random_pose,
                                     to_mediapipe_frame)

FPS = 24.0


def solve_landmarks(skel, j, profile=NEUTRAL_PROFILE):
    joints = {k: v for k, v in j.items() if not k.startswith("_")}
    vis = {k: 1.0 for k in joints}
    derived_joints(joints, vis)
    targets, _ = solver.bone_targets(joints, vis, profile, SolverConfig())
    return solver.solve_frame(skel, targets)


class SolverRoundTrip(unittest.TestCase):
    def test_rest_pose_gives_identity(self):
        skel = reference_skeleton()
        ident = np.repeat(np.eye(3)[None], len(skel.names), axis=0)
        basis, _, _ = solve_landmarks(skel, landmarks_from_pose(skel, ident))
        for name, b in zip(skel.names, basis):
            self.assertLess(np.degrees(quat_angle_between(mat_to_quat(b), [1, 0, 0, 0])), 1e-6, name)

    def test_random_poses_recovered_exactly(self):
        skel = reference_skeleton()
        rng = np.random.default_rng(7)
        for _ in range(30):
            pose = random_pose(skel, rng)
            basis, world, _ = solve_landmarks(skel, landmarks_from_pose(skel, pose))
            world_ref, _, _ = skel.fk(pose)
            for i, name in enumerate(skel.names):
                err = np.degrees(quat_angle_between(mat_to_quat(world[i]), mat_to_quat(world_ref[i])))
                self.assertLess(err, 1e-5, f"{name}: {err:.2e} deg")

    def test_single_shoulder_rotation_moves_only_that_arm(self):
        skel = reference_skeleton()
        pose = np.repeat(np.eye(3)[None], len(skel.names), axis=0)
        pose[skel.index["upper_arm_l"]] = rot_axis_angle(Z, np.radians(-50))  # abduct further
        basis, _, _ = solve_landmarks(skel, landmarks_from_pose(skel, pose))
        for name, b in zip(skel.names, basis):
            angle = np.degrees(quat_angle_between(mat_to_quat(b), [1, 0, 0, 0]))
            expected = 50.0 if name == "upper_arm_l" else 0.0
            self.assertAlmostEqual(angle, expected, places=5, msg=name)

    def test_hinge_debias_keeps_chord(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            r = rng.normal(size=3)
            m = r + rng.normal(size=3) * 0.3
            e = m + rng.normal(size=3) * 0.3
            prox, dist, _, raw, fixed = solver._hinge(r, m, e, 25.0)
            a, b = np.linalg.norm(m - r), np.linalg.norm(e - m)
            end = r + a * prox / np.linalg.norm(prox) + b * dist / np.linalg.norm(dist)
            self.assertLess(solver.angle_deg(end - r, e - r), 1e-3)
            self.assertAlmostEqual(solver.angle_deg(prox, dist), fixed, places=3)


class LimitTests(unittest.TestCase):
    cfg = LimitsConfig()

    def angles(self, name, quat):
        out, ch = limits.clamp_bone(name, quat, self.cfg)
        return out, limits.clamp_bone(name, out, self.cfg)[1]

    def test_elbow_hyperflexion_clamped(self):
        _, ch = self.angles("forearm_l", q(X, 170))
        self.assertAlmostEqual(ch["flex_deg"], 140.0, places=6)

    def test_elbow_hyperextension_clamped(self):
        _, ch = self.angles("forearm_r", q(X, -30))
        self.assertAlmostEqual(ch["flex_deg"], 0.0, places=6)

    def test_elbow_off_axis_removed(self):
        _, ch = self.angles("forearm_l", quat_mul(q(X, 60), q(Z, 20)))
        self.assertAlmostEqual(ch["off_axis_deg"], 0.0, places=6)

    def test_forearm_roll_clamped(self):
        _, ch = self.angles("forearm_l", quat_mul(q(X, 60), q(Y, 120)))
        self.assertAlmostEqual(ch["twist_deg"], 90.0, places=6)
        self.assertAlmostEqual(ch["flex_deg"], 60.0, places=6)

    def test_wrist_clamped(self):
        _, ch = self.angles("hand_l", quat_mul(q(X, 30), q(Y, -130)))
        self.assertAlmostEqual(ch["twist_deg"], -90.0, places=6)

    def test_knee_flexes_backward_only(self):
        _, ch = self.angles("shin_l", q(X, -60))
        self.assertAlmostEqual(ch["flex_deg"], 60.0, places=6)
        _, ch = self.angles("shin_l", q(X, 40))  # knee bending forward is impossible
        self.assertAlmostEqual(ch["flex_deg"], 0.0, places=6)

    def test_flip_spike_bridged_without_dropping_frames(self):
        n = 21
        bones = ["forearm_l"]
        frames = []
        for t in range(n):
            angle = 40 + 2 * t
            quat = q(X, angle) if t != 10 else quat_mul(q(X, angle), q(Y, 170))  # one flipped frame
            frames.append({"frame_index": t, "timestamp_s": t / FPS, "valid": True,
                           "rotations": [quat.tolist()], "confidence": [1.0]})
        clean, report = limits.apply_limits({"meta": {"bones": bones}, "frames": frames})
        self.assertEqual(len(clean["frames"]), n)
        self.assertIn(10, report["bones"]["forearm_l"]["anomaly_frames"])
        got = np.array(clean["frames"][10]["rotations"][0])
        self.assertLess(np.degrees(quat_angle_between(got, q(X, 60))), 0.5)

    def test_missing_frames_bridged(self):
        frames = []
        for t in range(10):
            valid = t not in (4, 5)
            quat = q(X, 10 * t) if valid else np.array([1.0, 0, 0, 0])
            frames.append({"frame_index": t, "timestamp_s": t / FPS, "valid": valid,
                           "rotations": [quat.tolist()], "confidence": [1.0 if valid else 0.0]})
        clean, _ = limits.apply_limits({"meta": {"bones": ["upper_arm_l"]}, "frames": frames})
        got = np.array(clean["frames"][5]["rotations"][0])
        self.assertLess(np.degrees(quat_angle_between(got, q(X, 50))), 1e-6)


class EndToEnd(unittest.TestCase):
    """Synthetic MediaPipe JSON -> calibration -> solver -> limits -> bundle -> FK."""

    def make_clip(self, skel, rng, n_rest=12, n_move=24):
        rest = np.repeat(np.eye(3)[None], len(skel.names), axis=0)
        target = random_pose(skel, rng)
        poses = [rest] * n_rest
        rest_q = [mat_to_quat(m) for m in rest]
        tgt_q = [mat_to_quat(m) for m in target]
        from altline.quat import slerp
        for k in range(1, n_move + 1):
            poses.append(np.array([quat_to_mat(slerp(a, b, k / n_move)) for a, b in zip(rest_q, tgt_q)]))
        frames = [to_mediapipe_frame(landmarks_from_pose(skel, p), i, FPS) for i, p in enumerate(poses)]
        return poses, {"meta": {"fps": FPS, "video": "synthetic"}, "frames": frames}

    def test_end_to_end_recovers_motion(self):
        skel = reference_skeleton()
        poses, raw = self.make_clip(skel, np.random.default_rng(11))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_json(tmp / "raw_pose.json", raw)
            prof_path, cal_path = calibration.calibrate(tmp / "raw_pose.json", tmp / "cal", skel)
            prof = read_json(prof_path)
            self.assertTrue(prof["baseline"]["rest_pose_found"])
            self.assertLess(prof["baseline"]["end"], 12)
            self.assertLess(prof["rest_alignment_angle_deg"], 1e-6)
            for k, v in prof["hinge_bias_deg"].items():
                self.assertLess(v, 1e-3, k)
            solver.solve(cal_path, prof_path, tmp / "raw.json", skel)
            clean, report = limits.apply_limits(read_json(tmp / "raw.json"))
            bundle = animation.build_bundle(clean, skel)
            self.assertEqual(len(bundle["frames"]), len(poses))
            for f, pose in zip(bundle["frames"], poses):
                basis = np.array([quat_to_mat(qq) for qq in f["q"]])
                _, h, t = skel.fk(basis)
                _, h_ref, t_ref = skel.fk(pose)
                # <= a few mm: while an elbow/knee is only slightly bent its bend plane gets
                # partial twist weight, and the hinge projection in limits removes the rest
                np.testing.assert_allclose(t, t_ref, atol=5e-3)
                np.testing.assert_allclose(np.linalg.norm(t - h, axis=1), skel.lengths, atol=1e-6)

    def test_rig_with_other_bone_rolls(self):
        """A rig whose bones are rolled differently must give the same motion via rig_offset."""
        ref = reference_skeleton()
        rng = np.random.default_rng(5)
        rolls = [rot_axis_angle(Y, rng.uniform(-np.pi, np.pi)) for _ in ref.names]
        rolled_rest = np.array([r @ roll for r, roll in zip(ref.rest, rolls)])
        offset = np.einsum("nji,njk->nik", ref.rest, rolled_rest)
        rig = Skeleton(list(ref.names), ref.parents.copy(), ref.heads.copy(), ref.tails.copy(), ref.rest.copy(),
                       rig_offset=offset)
        rolled = Skeleton(list(ref.names), ref.parents.copy(), ref.heads.copy(), ref.tails.copy(), rolled_rest)
        pose = random_pose(ref, rng)
        clean = {"meta": {"bones": ref.names, "fps": FPS},
                 "frames": [{"frame_index": 0, "timestamp_s": 0.0,
                             "rotations": [mat_to_quat(m).tolist() for m in pose]}]}
        bundle = animation.build_bundle(clean, rig)
        basis = np.array([quat_to_mat(qq) for qq in bundle["frames"][0]["q"]])
        _, _, tails_rolled = rolled.fk(basis)
        _, _, tails_ref = ref.fk(pose)
        np.testing.assert_allclose(tails_rolled, tails_ref, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
