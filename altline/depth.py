"""Depth reconstruction from 2D image points and the actor's constant segment lengths.

MediaPipe's world landmarks flatten limbs that point at the camera (a forearm seen end-on
can come out at half its length, so the solver aims it in the wrong direction). The image
points are much more reliable than the world depth. Under a weak-perspective camera a
segment of constant length L whose image projection measures d (in meters, after a
global pixel scale) has |dz| = sqrt(L^2 - d^2); the sign comes from the tracker's own
world depth, which is usually right even when its magnitude is not.

The skeleton is rebuilt outward from the hip midpoint along TREE, so each joint's image
position is kept exactly and only its depth changes. Points the solver does not aim along
(eyes, mouth) keep their tracker offset from the nose.
"""

from __future__ import annotations

import numpy as np

SIDES = ("left", "right")

# (parent joint, child joint, length key, rebuild depth from the length?)
# Only segments that lie in the image plane in a rest pose get their depth from the
# length: their rest length is then measured reliably. Segments that point forward even at
# rest (ear->nose, ankle->toe, ankle->heel, wrist->fingers) keep the tracker's depth, with
# their image-plane position from the 2D points.
TREE: list[tuple[str, str, str, bool]] = [
    ("hip_mid", "left_hip", "half_hip_width", True),
    ("hip_mid", "right_hip", "half_hip_width", True),
    ("hip_mid", "shoulder_mid", "trunk", True),
    ("shoulder_mid", "left_shoulder", "clavicle_l", True),
    ("shoulder_mid", "right_shoulder", "clavicle_r", True),
    ("shoulder_mid", "ear_mid", "neck", True),
    ("ear_mid", "left_ear", "half_ear_width", True),
    ("ear_mid", "right_ear", "half_ear_width", True),
    ("ear_mid", "nose", "nose", False),
]
for _s, _side in (("l", "left"), ("r", "right")):
    TREE += [
        (f"{_side}_shoulder", f"{_side}_elbow", f"upper_arm_{_s}", True),
        (f"{_side}_elbow", f"{_side}_wrist", f"forearm_{_s}", True),
        (f"{_side}_wrist", f"{_side}_index", f"index_{_s}", False),
        (f"{_side}_wrist", f"{_side}_pinky", f"pinky_{_s}", False),
        (f"{_side}_wrist", f"{_side}_thumb", f"thumb_{_s}", False),
        (f"{_side}_hip", f"{_side}_knee", f"thigh_{_s}", True),
        (f"{_side}_knee", f"{_side}_ankle", f"shin_{_s}", True),
        (f"{_side}_ankle", f"{_side}_heel", f"heel_{_s}", False),
        (f"{_side}_ankle", f"{_side}_foot_index", f"foot_{_s}", False),
    ]

# projection/length ratio above which the tracker depth is kept, and below which the
# length-based depth is used in full (cos 14 deg and cos 32 deg)
PLANAR_TRACKER_RATIO = 0.97
PLANAR_FULL_LENGTH_RATIO = 0.85


def _smoothstep(x: float, lo: float, hi: float) -> float:
    t = min(max((x - lo) / (hi - lo), 0.0), 1.0)
    return t * t * (3 - 2 * t)


FACE_POINTS = ["left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
               "right_eye_outer", "mouth_left", "mouth_right"]

# Segments that lie close to the image plane in an A/T rest pose: used for the pixel scale.
SCALE_SEGMENTS = [("hip_mid", "shoulder_mid", "trunk"),
                  ("left_hip", "left_knee", "thigh_l"), ("right_hip", "right_knee", "thigh_r"),
                  ("left_knee", "left_ankle", "shin_l"), ("right_knee", "right_ankle", "shin_r")]


def image_points(image_joints: dict[str, list[float]], width: int, height: int) -> dict[str, np.ndarray]:
    """Pixel coordinates with +y up (canonical orientation), including midpoints."""
    pts = {k: np.array([v[0] * width, -v[1] * height]) for k, v in image_joints.items()}
    for name, a, b in (("hip_mid", "left_hip", "right_hip"), ("shoulder_mid", "left_shoulder", "right_shoulder"),
                       ("ear_mid", "left_ear", "right_ear")):
        pts[name] = 0.5 * (pts[a] + pts[b])
    return pts


def pixel_scale(frames_2d: list[dict[str, np.ndarray]], lengths: dict[str, float]) -> float:
    """Meters per pixel, the median over rest-baseline frames of L / projected length."""
    ratios = []
    for pts in frames_2d:
        for a, b, key in SCALE_SEGMENTS:
            d = np.linalg.norm(pts[b] - pts[a])
            if d > 1e-6 and lengths.get(key):
                ratios.append(lengths[key] / d)
    if not ratios:
        raise RuntimeError("no segments to estimate the pixel scale")
    return float(np.median(ratios))


def planar_lengths(frames_2d: list[dict[str, np.ndarray] | None], idx: list[int], scale: float,
                   percentile: float = 98.0) -> dict[str, float]:
    """Segment lengths for the depth rebuild, measured in the image, not by the tracker.

    A segment's projection is never longer than the segment (weak perspective), so its
    length is the larger of the rest-baseline projection (where these segments lie in the
    image plane) and a high percentile of its projection over the whole clip. The tracker's
    own 3D lengths are not used here: they carry its depth noise, and the rebuild is very
    sensitive to L near the image plane (L 10 % too long on a planar forearm already adds
    ~11 cm of false depth).
    """
    out = {}
    for parent, child, key, from_length in TREE:
        if not from_length or key in out:
            continue
        rest = [np.linalg.norm(frames_2d[i][child] - frames_2d[i][parent]) for i in idx]
        clip = [np.linalg.norm(p[child] - p[parent]) for p in frames_2d if p is not None]
        out[key] = scale * float(max(np.median(rest), np.percentile(clip, percentile)))
    return out


def reconstruct(world: dict[str, np.ndarray], pts: dict[str, np.ndarray], scale: float,
                lengths: dict[str, float]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Rebuild canonical joint positions; returns (joints, per-segment depth change in m).

    `world` holds canonical tracker positions (including midpoints); its depth signs are used.
    """
    out = {"hip_mid": np.zeros(3)}
    dz_change = {}
    for parent, child, key, from_length in TREE:
        d2 = (pts[child] - pts[parent]) * scale
        tracker = world[child] - world[parent]
        L = lengths.get(key)
        if L is None or not from_length:
            dz = tracker[2]
        else:
            ratio = float(np.linalg.norm(d2)) / L
            mag = np.sqrt(max(L * L - float(d2 @ d2), 0.0))
            sign = 1.0 if tracker[2] >= 0 else -1.0
            # Near the image plane sqrt(L^2 - d^2) is ill-conditioned (a 3 % shorter
            # projection already means 14 deg of depth), so there the tracker's own depth is
            # kept, capped at what the length allows; clearly foreshortened segments take
            # the length-based depth.
            w = _smoothstep(ratio, PLANAR_FULL_LENGTH_RATIO, PLANAR_TRACKER_RATIO)
            dz = w * float(np.clip(tracker[2], -mag, mag)) + (1.0 - w) * sign * mag
        out[child] = out[parent] + np.array([d2[0], d2[1], dz])
        dz_change[f"{parent}->{child}"] = float(dz - tracker[2])
    for name in FACE_POINTS:
        if name in world:
            out[name] = out["nose"] + (world[name] - world["nose"])
    return out, dz_change
