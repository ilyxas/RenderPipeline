"""Phase 2 stages: Butterworth filter, flip guard, temporal solver, damper."""

import unittest

import numpy as np

from altline import damper, filters, flip_guard, solver_temporal
from altline.config import ButterworthConfig, DamperConfig, FlipGuardConfig, TemporalSolverConfig
from altline.limits import make_continuous
from altline.quat import quat_angle_between, quat_from_axis_angle

FPS = 24.0


class ButterworthTests(unittest.TestCase):
    def test_zero_phase_keeps_motion_and_removes_noise(self):
        t = np.arange(144) / FPS
        motion = 0.2 * np.sin(2 * np.pi * 1.0 * t)
        noise = 0.02 * np.sin(2 * np.pi * 9.0 * t)
        out = filters.butter_filtfilt((motion + noise)[:, None], FPS, 4.0)[:, 0]
        inner = slice(12, -12)
        # no lag and no amplitude loss at 1 Hz, the 9 Hz component is gone
        self.assertLess(np.abs(out - motion)[inner].max(), 0.004)
        self.assertLess(np.abs(np.argmax(out[:24]) - np.argmax(motion[:24])), 1)

    def test_cutoff_stays_in_band(self):
        rng = np.random.default_rng(1)
        t = np.arange(144) / FPS
        x = np.column_stack([np.sin(2 * np.pi * 0.8 * t), np.cos(2 * np.pi * 0.5 * t)]) + rng.normal(0, 0.02, (144, 2))
        fc = filters.winter_cutoff(x, FPS, 2, (2.0, 5.0))
        self.assertGreaterEqual(fc, 2.0)
        self.assertLessEqual(fc, 5.0)


def _calibrated(forearm_dirs):
    """Calibrated-pose dict whose left forearm points along forearm_dirs[t]; the rest is static."""
    frames = []
    for i, d in enumerate(forearm_dirs):
        j = {}
        for side, sx in (("left", 1.0), ("right", -1.0)):
            j[f"{side}_shoulder"] = [0.18 * sx, 1.4, 0.0]
            j[f"{side}_elbow"] = [0.22 * sx, 1.12, 0.0]
            j[f"{side}_hip"] = [0.1 * sx, 0.95, 0.0]
            j[f"{side}_knee"] = [0.1 * sx, 0.5, 0.02]
            j[f"{side}_ankle"] = [0.1 * sx, 0.08, 0.0]
            j[f"{side}_heel"] = [0.1 * sx, 0.05, -0.05]
            j[f"{side}_foot_index"] = [0.1 * sx, 0.02, 0.15]
        for side, sx in (("left", 1.0), ("right", -1.0)):
            e = np.array(j[f"{side}_elbow"])
            dd = np.asarray(d, float) if side == "left" else np.array([0.0, -1.0, 0.0])
            w = e + 0.25 * dd / np.linalg.norm(dd)
            j[f"{side}_wrist"] = w.tolist()
            j[f"{side}_index"] = (w + [0.0, -0.08, 0.02]).tolist()
            j[f"{side}_pinky"] = (w + [0.0, -0.07, -0.02]).tolist()
            j[f"{side}_thumb"] = (w + [0.0, -0.04, 0.04]).tolist()
            j[f"{side}_hand_mid"] = (w + [0.0, -0.075, 0.0]).tolist()
        frames.append({"frame_index": i, "timestamp_s": i / FPS, "valid": True,
                       "joints": {k: v + [1.0] for k, v in j.items()}})
    return {"meta": {"fps": FPS}, "frames": frames}


def _dir(deg):
    a = np.radians(deg)
    return [np.sin(a), -np.cos(a), 0.0]


class FlipGuardTests(unittest.TestCase):
    def forearm(self, guarded, t):
        j = guarded["frames"][t]["joints"]
        return np.array(j["left_wrist"][:3]) - np.array(j["left_elbow"][:3])

    def test_two_frame_flip_is_bridged(self):
        angles = list(np.linspace(0, 30, 20))
        dirs = [_dir(a) for a in angles]
        dirs[9] = dirs[10] = _dir(angles[9] + 150)  # forearm turned inside out for 2 frames
        guarded, report = flip_guard.guard(_calibrated(dirs), FlipGuardConfig())
        self.assertEqual(report["limbs"]["left_arm"]["bridged_runs"], [[9, 10]])
        for t in (9, 10):
            got = self.forearm(guarded, t)
            want = 0.25 * np.array(_dir(angles[t]))
            self.assertLess(np.linalg.norm(got - want), 0.01)
        self.assertEqual(report["limbs"]["right_arm"]["frames_bridged"], 0)

    def test_fast_but_possible_motion_is_kept(self):
        dirs = [_dir(a) for a in np.linspace(0, 160, 6)]  # 32 deg per frame
        cal = _calibrated(dirs)
        guarded, report = flip_guard.guard(cal, FlipGuardConfig())
        self.assertEqual(report["limbs"]["left_arm"]["frames_bridged"], 0)
        self.assertEqual(guarded["frames"][3]["joints"]["left_wrist"], cal["frames"][3]["joints"]["left_wrist"])

    def test_lasting_jump_is_not_bridged(self):
        dirs = [_dir(0)] * 5 + [_dir(120)] * 15
        _, report = flip_guard.guard(_calibrated(dirs), FlipGuardConfig())
        self.assertEqual(report["limbs"]["left_arm"]["frames_bridged"], 0)
        self.assertTrue(report["limbs"]["left_arm"]["kept_runs_over_threshold"])


def _rot_sequence(angles_deg, axis=(1.0, 0.0, 0.0)):
    return np.array([quat_from_axis_angle(np.array(axis), np.radians(a)) for a in angles_deg])


class TemporalSolverTests(unittest.TestCase):
    def run_one(self, qs, valid=None):
        n = len(qs)
        valid = np.ones(n, bool) if valid is None else valid
        return solver_temporal.smooth_bone(qs, np.ones(n), valid, TemporalSolverConfig())

    def test_constant_velocity_is_unchanged(self):
        qs = _rot_sequence(np.linspace(0, 90, 30))
        out, change = self.run_one(qs)
        self.assertLess(change.max(), 1e-6)

    def test_noise_is_reduced_without_bias(self):
        rng = np.random.default_rng(3)
        true = np.linspace(0, 60, 48)
        qs = _rot_sequence(true + rng.normal(0, 2.0, 48))
        out, _ = self.run_one(qs)
        acc_in = damper.angular_acceleration(qs, FPS)
        acc_out = damper.angular_acceleration(out, FPS)
        self.assertLess(np.median(acc_out[2:-2]), 0.6 * np.median(acc_in[2:-2]))
        err = [np.degrees(quat_angle_between(o, t)) for o, t in zip(out, _rot_sequence(true))]
        self.assertLess(np.mean(err), 2.0)

    def test_invalid_frames_are_left_alone(self):
        qs = _rot_sequence(np.linspace(0, 30, 10))
        valid = np.ones(10, bool)
        valid[4] = False
        qs[4] = [1.0, 0.0, 0.0, 0.0]
        out, _ = self.run_one(qs, valid)
        np.testing.assert_allclose(out[4], qs[4])
        # the identity placeholder does not pull its neighbours
        self.assertLess(np.degrees(quat_angle_between(out[3], qs[3])), 0.5)


class DamperTests(unittest.TestCase):
    GATE = 3000.0

    def damp(self, qs):
        return damper.damp_bone(qs, FPS, self.GATE, DamperConfig())

    def test_smooth_motion_passes_unchanged(self):
        t = np.arange(48) / FPS
        qs = _rot_sequence(20 * np.sin(2 * np.pi * 0.5 * t))  # peak acc ~790 deg/s^2
        out, rep = self.damp(qs)
        self.assertEqual(rep["frames_damped"], 0)
        self.assertLess(rep["max_deviation_deg"], 1e-6)

    def test_spike_is_damped_under_the_gate_without_ringing(self):
        angles = np.zeros(48)
        angles[20] = 15.0  # one-frame 15 deg jitter
        out, rep = self.damp(_rot_sequence(angles))
        acc = damper.angular_acceleration(make_continuous(out), FPS)
        self.assertLessEqual(acc.max(), self.GATE * 1.001)
        # the pose is back on the input a few frames later (no ringing, no drift)
        tail = [np.degrees(quat_angle_between(q, [1, 0, 0, 0])) for q in out[30:]]
        self.assertLess(max(tail), 0.1)

    def test_step_is_followed_not_frozen(self):
        angles = np.r_[np.zeros(20), np.full(40, 40.0)]
        qs = _rot_sequence(angles)
        out, rep = self.damp(qs)
        self.assertGreater(rep["frames_damped"], 0)
        self.assertLess(np.degrees(quat_angle_between(out[-1], qs[-1])), 0.1)
        # the joint moves on every frame of the transition: no frame-to-frame stall
        mid = [np.degrees(quat_angle_between(out[t], out[t - 1])) for t in range(18, 23)]
        self.assertTrue(all(m > 0.5 for m in mid))


if __name__ == "__main__":
    unittest.main()
