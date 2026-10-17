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

# Weight of a depth jump against tracker disagreement in choose_signs. A rigid segment can
# only change its depth sign by passing through the image plane (dz = 0), where switching is
# free; a switch at large |dz| is never motion, only the repair of an earlier wrong choice.
# With weight 1 (v4 first run) a 41 cm elbow jump in one frame (0663, frame 31) was cheaper
# than disagreeing with the tracker for 5 frames. At 10 the tracker must disagree with the
# chosen sign for roughly 10x the jump, in L units, before a mid-air switch is taken.
JUMP_WEIGHT = 10.0


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


def planar_lengths(frames_2d: list[dict[str, np.ndarray] | None], idx: list[int], scale: float) -> dict[str, float]:
    """Segment lengths for the depth rebuild: their projection in the rest baseline, where
    these segments lie in the image plane. Not the tracker's 3D lengths (they carry its
    depth noise, and the rebuild is very sensitive to L near the image plane), and not a
    high percentile over the clip (the camera is perspective: a limb moved toward it looks
    longer, which v2 turned into false depth at rest).
    """
    out = {}
    for parent, child, key, from_length in TREE:
        if not from_length or key in out:
            continue
        out[key] = scale * float(np.median([np.linalg.norm(frames_2d[i][child] - frames_2d[i][parent]) for i in idx]))
    return out


def _segment_dz(world: dict[str, np.ndarray], pts: dict[str, np.ndarray], scale: float,
                parent: str, child: str, L: float) -> tuple[float, float, float, np.ndarray]:
    """(tracker dz, length-based |dz|, planar blend weight, scaled 2D vector) of one segment."""
    d2 = (pts[child] - pts[parent]) * scale
    tracker = float(world[child][2] - world[parent][2])
    mag = float(np.sqrt(max(L * L - float(d2 @ d2), 0.0)))
    w = _smoothstep(float(np.linalg.norm(d2)) / L, PLANAR_FULL_LENGTH_RATIO, PLANAR_TRACKER_RATIO)
    return tracker, mag, w, d2


def _blend(tracker: float, mag: float, w: float, sign: float) -> float:
    # Near the image plane sqrt(L^2 - d^2) is ill-conditioned (a 3 % shorter projection
    # already means 14 deg of depth), so there the tracker's own depth is kept, capped at
    # what the length allows; clearly foreshortened segments take the length-based depth.
    return w * float(np.clip(tracker, -mag, mag)) + (1.0 - w) * sign * mag


def choose_signs(worlds: list[dict | None], pts: list[dict | None], scale: float,
                 lengths: dict[str, float], jump_weight: float = JUMP_WEIGHT) -> list[dict[str, float] | None]:
    """Depth sign of every length-rebuilt segment over the whole clip (Viterbi per segment).

    Each frame has two candidates, in front (+) or behind (-). Cost of a candidate: how
    strongly the tracker's own depth points the other way (|dz| / L), plus, between
    consecutive frames, the jump in rebuilt depth (jump_weight * |dz_t - dz_t-1| / L). A real crossing of
    the image plane is free (both candidates meet at dz = 0 there), while a flip the tracker
    reports for a frame or two costs more than it explains and is rejected. v3 decided per
    frame and flipped whenever the tracker was "confident", which made 36-47 cm jumps.
    """
    n = len(worlds)
    signs: list[dict[str, float] | None] = [None if w is None else {} for w in worlds]
    for parent, child, key, from_length in TREE:
        L = lengths.get(key)
        if not from_length or L is None:
            continue
        seg = f"{parent}->{child}"
        cost = {+1.0: 0.0, -1.0: 0.0}
        back: list[dict[float, float]] = []
        prev_dz: dict[float, float] | None = None
        frames: list[int] = []
        for t in range(n):
            if worlds[t] is None:
                continue
            tracker, mag, w, _ = _segment_dz(worlds[t], pts[t], scale, parent, child, L)
            dz = {s: _blend(tracker, mag, w, s) for s in (1.0, -1.0)}
            unary = {s: max(0.0, -s * tracker) / L for s in (1.0, -1.0)}
            if prev_dz is None:
                new = {s: unary[s] for s in dz}
                back.append({1.0: 1.0, -1.0: -1.0})
            else:
                new, ptr = {}, {}
                for s in dz:
                    options = {p: cost[p] + jump_weight * abs(dz[s] - prev_dz[p]) / L for p in (1.0, -1.0)}
                    best = min(options, key=options.get)
                    new[s], ptr[s] = options[best] + unary[s], best
                back.append(ptr)
            cost, prev_dz = new, dz
            frames.append(t)
        if not frames:
            continue
        s = min(cost, key=cost.get)
        for k in range(len(frames) - 1, -1, -1):
            signs[frames[k]][seg] = s
            s = back[k][s]
    return signs


def reconstruct(world: dict[str, np.ndarray], pts: dict[str, np.ndarray], scale: float,
                lengths: dict[str, float], signs: dict[str, float] | None = None
                ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Rebuild canonical joint positions; returns (joints, per-segment depth change in m).

    `world` holds canonical tracker positions (including midpoints). `signs` (from
    choose_signs) gives each length-rebuilt segment's depth sign; without it the tracker's
    sign is used.
    """
    out = {"hip_mid": np.zeros(3)}
    dz_change = {}
    for parent, child, key, from_length in TREE:
        seg = f"{parent}->{child}"
        L = lengths.get(key)
        if L is None or not from_length:
            d2 = (pts[child] - pts[parent]) * scale
            tracker = float(world[child][2] - world[parent][2])
            dz = tracker
        else:
            tracker, mag, w, d2 = _segment_dz(world, pts, scale, parent, child, L)
            sign = (signs or {}).get(seg, 1.0 if tracker >= 0 else -1.0)
            dz = _blend(tracker, mag, w, sign)
        out[child] = out[parent] + np.array([d2[0], d2[1], dz])
        dz_change[seg] = float(dz - tracker)
    for name in FACE_POINTS:
        if name in world:
            out[name] = out["nose"] + (world[name] - world["nose"])
    return out, dz_change
