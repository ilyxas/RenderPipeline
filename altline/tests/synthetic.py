"""Synthetic landmarks from a known pose on the reference rig.

The landmarks are placed so that every solver reference is exact for the generating pose
(see solver.bone_targets), which makes the solver's round trip checkable to numerical
precision: what goes in as local rotations must come back out.
"""

from __future__ import annotations

import numpy as np

from altline.quat import quat_exp, quat_from_axis_angle, quat_to_mat
from altline.skeleton import Skeleton

X, Y, Z = np.eye(3)


def landmarks_from_pose(skel: Skeleton, basis: np.ndarray) -> dict[str, np.ndarray]:
    """Canonical-space landmarks (MediaPipe names) for an FK pose of the reference rig."""
    world, heads, tails = skel.fk(basis)
    i = skel.index
    j: dict[str, np.ndarray] = {}
    hip_mid = heads[i["pelvis"]]
    for s, side in (("l", "left"), ("r", "right")):
        j[f"{side}_shoulder"] = heads[i[f"upper_arm_{s}"]]
        j[f"{side}_elbow"] = heads[i[f"forearm_{s}"]]
        j[f"{side}_wrist"] = heads[i[f"hand_{s}"]]
        hz = world[i[f"hand_{s}"]][:, 2]
        j[f"{side}_index"] = tails[i[f"hand_{s}"]] + 0.02 * hz
        j[f"{side}_pinky"] = tails[i[f"hand_{s}"]] - 0.02 * hz
        j[f"{side}_thumb"] = heads[i[f"hand_{s}"]] + 0.03 * hz
        j[f"{side}_hip"] = heads[i[f"thigh_{s}"]]
        j[f"{side}_knee"] = heads[i[f"shin_{s}"]]
        j[f"{side}_ankle"] = heads[i[f"foot_{s}"]]
        j[f"{side}_foot_index"] = tails[i[f"foot_{s}"]]
        j[f"{side}_heel"] = j[f"{side}_foot_index"] - 0.2 * world[i[f"shin_{s}"]][:, 2]
    sh_mid = 0.5 * (j["left_shoulder"] + j["right_shoulder"])
    neck_w, head_w = world[i["neck"]], world[i["head"]]
    ear_mid = sh_mid + 0.15 * neck_w[:, 1]
    j["left_ear"] = ear_mid + 0.07 * head_w[:, 0]
    j["right_ear"] = ear_mid - 0.07 * head_w[:, 0]
    j["nose"] = ear_mid + 0.10 * head_w[:, 2]
    for side, sign in (("left", 1), ("right", -1)):
        for k, off in (("eye_inner", 0.015), ("eye", 0.03), ("eye_outer", 0.045)):
            j[f"{side}_{k}"] = j["nose"] + 0.03 * head_w[:, 1] + sign * off * head_w[:, 0]
        j[f"mouth_{side}"] = j["nose"] - 0.03 * head_w[:, 1] + sign * 0.02 * head_w[:, 0]
    j["_hip_mid"] = hip_mid
    return j


def to_mediapipe_frame(j: dict[str, np.ndarray], index: int, fps: float) -> dict:
    hip = 0.5 * (j["left_hip"] + j["right_hip"])
    joints = {k: [float(v[0] - hip[0]), float(-(v[1] - hip[1])), float(-(v[2] - hip[2])), 1.0]
              for k, v in j.items() if not k.startswith("_")}
    return {"frame_index": index, "timestamp_s": index / fps, "detected": True,
            "joints": joints, "image_joints": {}}


def q(axis, deg) -> np.ndarray:
    return quat_from_axis_angle(np.asarray(axis, dtype=float), np.radians(deg))


def random_pose(skel: Skeleton, rng: np.random.Generator) -> np.ndarray:
    """Local basis matrices inside the solver's observable set (see module docstring)."""
    def rnd(max_deg):
        axis = rng.normal(size=3)
        return q(axis, rng.uniform(0, max_deg))
    quats = {n: np.array([1.0, 0, 0, 0]) for n in skel.names}
    quats["pelvis"] = rnd(40)
    quats["neck"] = rnd(30)
    quats["head"] = q(X, rng.uniform(-30, 30))
    for s in ("l", "r"):
        quats[f"upper_arm_{s}"] = rnd(120)
        quats[f"forearm_{s}"] = q(X, rng.uniform(35, 130))
        quats[f"hand_{s}"] = q(X, rng.uniform(-50, 50))
        quats[f"thigh_{s}"] = rnd(70)
        quats[f"shin_{s}"] = q(X, -rng.uniform(35, 120))
        # the foot has no twist reference, so only a swing (rotation axis in the XZ plane) is observable
        quats[f"foot_{s}"] = quat_exp(np.radians([rng.uniform(-30, 30), 0.0, rng.uniform(-15, 15)]))
    return np.array([quat_to_mat(quats[n]) for n in skel.names])


NEUTRAL_PROFILE = {
    "reference_pitch_offsets_rad": {"head_forward": 0.0, "neck_up": 0.0},
    "hinge_bias_deg": {"elbow_l": 0.0, "elbow_r": 0.0, "knee_l": 0.0, "knee_r": 0.0},
}
