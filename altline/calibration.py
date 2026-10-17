"""Task 2: relative neutral-pose calibration.

Input: <run>/observations/raw_pose.json
Output: <run>/calibration/actor_profile.json and <run>/calibration/calibrated_pose.json

1. Convert MediaPipe world landmarks to canonical space (+X actor left, +Y up, +Z actor
   forward): canon = (x, -y, -z).
2. Find a baseline interval where the actor is closest to a static rest pose (A/T pose):
   elbows and knees straight-ish, arms abducted symmetrically between the configured
   bounds, minimal joint speed. Measure the actor's segment lengths there (median).
3. Rebuild landmark depth from the 2D image points and constant segment lengths
   (depth.py), then smooth all joint trajectories with a zero-phase One-Euro filter
   (filters.py).
4. From the baseline of the rebuilt landmarks: a rest alignment rotation that makes the
   actor stand upright and face +Z (removes camera tilt / yaw); the rest biases of
   landmark references with no anatomical ground truth (head frame, neck pitch, foot
   pitch); the elbow/knee bend the tracker reports for straight rest limbs (hinge bias).
5. calibrated_pose.json holds the per-frame aligned joints, plus each segment's tracker
   length ratio to the actor's constant length (a diagnostic of tracker depth loss). The
   solver uses only unit bone directions, so actor lengths never reach the rig.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import CalibrationConfig
from .depth import choose_signs, image_points, pixel_scale, planar_lengths, reconstruct
from .filters import one_euro_zero_phase
from .io_utils import read_json, run_dir, write_json
from .skeleton import Skeleton, load_skeleton

SIDES = {"l": "left", "r": "right"}

# segment name -> (from joint, to joint). "*_mid" joints are derived in `derived_joints`.
SEGMENTS = {
    "trunk": ("hip_mid", "shoulder_mid"),
    "neck": ("shoulder_mid", "ear_mid"),
    "hip_width": ("right_hip", "left_hip"),
    "shoulder_width": ("right_shoulder", "left_shoulder"),
    "ear_width": ("right_ear", "left_ear"),
    "nose": ("ear_mid", "nose"),
}
for s, side in SIDES.items():
    SEGMENTS.update({
        f"clavicle_{s}": ("shoulder_mid", f"{side}_shoulder"),
        f"upper_arm_{s}": (f"{side}_shoulder", f"{side}_elbow"),
        f"forearm_{s}": (f"{side}_elbow", f"{side}_wrist"),
        f"hand_{s}": (f"{side}_wrist", f"{side}_hand_mid"),
        f"thigh_{s}": (f"{side}_hip", f"{side}_knee"),
        f"shin_{s}": (f"{side}_knee", f"{side}_ankle"),
        f"foot_{s}": (f"{side}_ankle", f"{side}_foot_index"),
        f"heel_{s}": (f"{side}_ankle", f"{side}_heel"),
        f"index_{s}": (f"{side}_wrist", f"{side}_index"),
        f"pinky_{s}": (f"{side}_wrist", f"{side}_pinky"),
        f"thumb_{s}": (f"{side}_wrist", f"{side}_thumb"),
    })


def to_canonical(p) -> np.ndarray:
    return np.array([p[0], -p[1], -p[2]], dtype=float)


def derived_joints(joints: dict[str, np.ndarray], vis: dict[str, float]) -> None:
    """Add midpoints in place; visibility of a midpoint is the min of its inputs."""
    def mid(name, a, b):
        if a in joints and b in joints:
            joints[name] = 0.5 * (joints[a] + joints[b])
            vis[name] = min(vis[a], vis[b])
    mid("hip_mid", "left_hip", "right_hip")
    mid("shoulder_mid", "left_shoulder", "right_shoulder")
    mid("ear_mid", "left_ear", "right_ear")
    mid("ankle_mid", "left_ankle", "right_ankle")
    for side in SIDES.values():
        mid(f"{side}_hand_mid", f"{side}_index", f"{side}_pinky")


def load_frames(raw: dict) -> list[tuple[dict, dict] | None]:
    out = []
    for f in raw["frames"]:
        if not f["detected"]:
            out.append(None)
            continue
        joints = {k: to_canonical(v) for k, v in f["joints"].items()}
        vis = {k: float(v[3]) for k, v in f["joints"].items()}
        derived_joints(joints, vis)
        out.append((joints, vis))
    return out


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))


def rest_features(joints: dict[str, np.ndarray]) -> dict[str, float]:
    up = joints["shoulder_mid"] - joints["hip_mid"]
    f = {}
    for side in SIDES.values():
        upper = joints[f"{side}_elbow"] - joints[f"{side}_shoulder"]
        fore = joints[f"{side}_wrist"] - joints[f"{side}_elbow"]
        thigh = joints[f"{side}_knee"] - joints[f"{side}_hip"]
        shin = joints[f"{side}_ankle"] - joints[f"{side}_knee"]
        f[f"{side}_elbow_bend"] = _angle(upper, fore)
        f[f"{side}_knee_bend"] = _angle(thigh, shin)
        f[f"{side}_abduction"] = _angle(upper, -up)
    f["asymmetry"] = abs(f["left_abduction"] - f["right_abduction"])
    return f


def find_baseline(frames, cfg: CalibrationConfig) -> dict:
    n = len(frames)
    keys = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
            "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    eligible = np.zeros(n, dtype=bool)
    asym = np.full(n, np.inf)
    speed = np.full(n, np.inf)
    for i, fr in enumerate(frames):
        if fr is None:
            continue
        joints, vis = fr
        if min(vis[k] for k in keys) < cfg.min_visibility:
            continue
        feat = rest_features(joints)
        asym[i] = feat["asymmetry"]
        eligible[i] = (
            feat["left_elbow_bend"] <= cfg.max_elbow_bend_deg
            and feat["right_elbow_bend"] <= cfg.max_elbow_bend_deg
            and feat["left_knee_bend"] <= cfg.max_knee_bend_deg
            and feat["right_knee_bend"] <= cfg.max_knee_bend_deg
            and cfg.min_arm_abduction_deg <= feat["left_abduction"] <= cfg.max_arm_abduction_deg
            and cfg.min_arm_abduction_deg <= feat["right_abduction"] <= cfg.max_arm_abduction_deg
        )
        prev = frames[i - 1] if i > 0 else None
        nxt = frames[i + 1] if i + 1 < n else None
        moves = [np.mean([np.linalg.norm(joints[k] - other[0][k]) for k in keys]) for other in (prev, nxt) if other]
        speed[i] = float(np.mean(moves)) if moves else np.inf

    w = min(cfg.window_frames, n)
    best, best_score, found = None, np.inf, False
    for require_eligible in (True, False):
        for s in range(0, n - w + 1):
            sl = slice(s, s + w)
            if not np.all(np.isfinite(speed[sl])):
                continue
            if require_eligible and not eligible[sl].all():
                continue
            # speed in meters/frame, asymmetry in degrees: 1 deg of asymmetry ~ 1 mm/frame of motion
            score = float(np.mean(speed[sl]) + 0.001 * np.mean(asym[sl]))
            if score < best_score:
                best, best_score = s, score
        if best is not None:
            found = require_eligible
            break
    if best is None:
        raise RuntimeError("no frame window with a detected person; cannot calibrate")
    return {"start": best, "end": best + w - 1, "frames": list(range(best, best + w)),
            "rest_pose_found": found, "score": best_score,
            "n_eligible_frames": int(eligible.sum())}


def rest_alignment(frames, idx: list[int]) -> np.ndarray:
    """Rotation taking the camera-derived canonical frame to an upright actor facing +Z."""
    up = np.mean([frames[i][0]["shoulder_mid"] - frames[i][0]["ankle_mid"] for i in idx], axis=0)
    lateral = np.mean([frames[i][0]["left_hip"] - frames[i][0]["right_hip"] for i in idx], axis=0)
    up = up / np.linalg.norm(up)
    fwd = np.cross(lateral, up)
    fwd /= np.linalg.norm(fwd)
    lateral = np.cross(up, fwd)
    return np.vstack([lateral, up, fwd])  # rows: new axes expressed in the old frame


def target_lengths(skel: Skeleton) -> dict[str, float]:
    L = dict(zip(skel.names, skel.lengths))
    idx = skel.index
    out = {
        "trunk": L["pelvis"] + L["spine"] + L["chest"],
        "neck": L["neck"],
        "hip_width": float(np.linalg.norm(skel.heads[idx["thigh_l"]] - skel.heads[idx["thigh_r"]])),
        "shoulder_width": float(np.linalg.norm(skel.heads[idx["upper_arm_l"]] - skel.heads[idx["upper_arm_r"]])),
    }
    for name in SEGMENTS:
        if name in L:
            out[name] = float(L[name])
    return out


def orthonormal_mean(mats: list[np.ndarray]) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.mean(mats, axis=0))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def head_frame(j: dict[str, np.ndarray]) -> np.ndarray:
    """Head frame from landmarks: X = right->left ear, Z = ear centre->nose, Y = Z x X."""
    lat = j["left_ear"] - j["right_ear"]
    fwd = j["nose"] - j["ear_mid"]
    up = np.cross(fwd, lat)
    x = lat / np.linalg.norm(lat)
    y = up / np.linalg.norm(up)
    return np.column_stack([x, y, np.cross(x, y)])


def foot_pitch(v: np.ndarray) -> float:
    return float(np.arctan2(v[1], np.linalg.norm(v[[0, 2]])))


def rest_references(frames, idx: list[int], skel: Skeleton) -> dict:
    """Biases of landmark references that have no anatomical ground truth, measured in the
    (already aligned) rest baseline, assuming the actor holds head and feet neutral there.

      head_rest_bias: the landmark head frame at rest (ears/nose sit off the anatomical
                      axes: nose below the ear line, head turned a few degrees); the solver
                      uses H_measured @ bias^T.
      neck_up:        pitch (rad, about +X) of (ear_mid - shoulder_mid) from vertical.
      foot_pitch_l/r: rig rest foot pitch minus the measured ankle->toe pitch (rad).
    """
    heads = [head_frame(frames[i][0]) for i in idx]
    neck = []
    foot = {"l": [], "r": []}
    for i in idx:
        j = frames[i][0]
        up = j["ear_mid"] - j["shoulder_mid"]
        neck.append(np.arctan2(-up[2], up[1]))
        for s, side in (("l", "left"), ("r", "right")):
            foot[s].append(foot_pitch(j[f"{side}_foot_index"] - j[f"{side}_ankle"]))
    out = {"head_rest_bias": orthonormal_mean(heads).tolist(), "neck_up": float(np.median(neck))}
    for s in ("l", "r"):
        b = skel.index[f"foot_{s}"]
        out[f"foot_pitch_{s}"] = foot_pitch(skel.tails[b] - skel.heads[b]) - float(np.median(foot[s]))
    return out


def hinge_bias(frames, idx: list[int], enabled: bool, cap_deg: float = 30.0) -> dict[str, float]:
    """Median elbow/knee bend in the rest baseline (degrees); zeros when disabled."""
    keys = {"elbow_l": "left_elbow_bend", "elbow_r": "right_elbow_bend",
            "knee_l": "left_knee_bend", "knee_r": "right_knee_bend"}
    if not enabled:
        return {k: 0.0 for k in keys}
    feats = [rest_features(frames[i][0]) for i in idx]
    return {k: min(cap_deg, float(np.median([f[v] for f in feats]))) for k, v in keys.items()}


def _actor_lengths(frames, idx, min_vis: float) -> dict[str, float | None]:
    actor = {}
    for seg, (a, b) in SEGMENTS.items():
        vals = [np.linalg.norm(frames[i][0][b] - frames[i][0][a]) for i in idx
                if min(frames[i][1][a], frames[i][1][b]) >= min_vis]
        actor[seg] = float(np.median(vals)) if vals else None
    for half, full in (("half_hip_width", "hip_width"), ("half_ear_width", "ear_width")):
        actor[half] = actor[full] / 2 if actor[full] else None
    return actor


def _rebuild_depth(raw: dict, frames, idx, actor) -> tuple[list, dict]:
    meta = raw["meta"]
    pts = [image_points(f["image_joints"], meta["width"], meta["height"]) if f["detected"] else None
           for f in raw["frames"]]
    scale = pixel_scale([pts[i] for i in idx], actor)
    lengths = {**actor, **planar_lengths(pts, idx, scale)}
    signs = choose_signs([None if fr is None else fr[0] for fr in frames], pts, scale, lengths)
    out, max_dz = [], []
    for fr, p, sg in zip(frames, pts, signs):
        if fr is None:
            out.append(None)
            max_dz.append(None)
            continue
        joints, vis = fr
        rec, dz = reconstruct(joints, p, scale, lengths, sg)
        vis = dict(vis)
        derived_joints(rec, vis)
        out.append((rec, vis))
        max_dz.append(max(abs(v) for v in dz.values()))
    info = {"pixel_scale_m": scale, "lengths_m": {k: lengths[k] for k in planar_lengths(pts, idx, scale)},
            "max_depth_change_m_per_frame": max_dz}
    return out, info


def _filter(frames, times: np.ndarray, cfg: CalibrationConfig) -> list:
    names = next(fr[0].keys() for fr in frames if fr is not None)
    out = [None if fr is None else ({}, fr[1]) for fr in frames]
    for name in names:
        x = np.array([fr[0][name] if fr is not None else [np.nan] * 3 for fr in frames], dtype=float)
        y = one_euro_zero_phase(x, times, cfg.filter_min_cutoff_hz, cfg.filter_beta)
        for i, fr in enumerate(out):
            if fr is not None:
                fr[0][name] = y[i]
    return out


def calibrate(raw_path: Path, out_dir: Path, skel: Skeleton, cfg: CalibrationConfig | None = None) -> tuple[Path, Path]:
    cfg = cfg or CalibrationConfig()
    raw = read_json(raw_path)
    tracker = load_frames(raw)                     # canonical tracker landmarks, unaligned
    base = find_baseline(tracker, cfg)
    idx = base["frames"]
    actor = _actor_lengths(tracker, idx, cfg.min_visibility)

    frames, depth_info = tracker, None
    if cfg.depth_from_2d and all(f["image_joints"] for f in raw["frames"] if f["detected"]):
        frames, depth_info = _rebuild_depth(raw, tracker, idx, actor)
    if cfg.filter_enabled:
        frames = _filter(frames, np.array([f["timestamp_s"] for f in raw["frames"]]), cfg)

    align = rest_alignment(frames, idx)
    frames = [None if fr is None else ({k: align @ v for k, v in fr[0].items()}, fr[1]) for fr in frames]

    target = target_lengths(skel)
    ratios = {k: (target[k] / v if v and k in target else None) for k, v in actor.items()}
    leg_trunk = ["trunk", "thigh_l", "shin_l", "thigh_r", "shin_r"]
    global_scale = float(sum(target[k] for k in leg_trunk) / sum(actor[k] for k in leg_trunk))

    profile = {
        "source": str(raw_path),
        "baseline": base,
        "canonical_axes": "+X actor left, +Y up, +Z actor forward (meters)",
        "depth_from_2d": None if depth_info is None else {k: depth_info[k] for k in ("pixel_scale_m", "lengths_m")},
        "filter": {"enabled": cfg.filter_enabled, "min_cutoff_hz": cfg.filter_min_cutoff_hz, "beta": cfg.filter_beta},
        "rest_alignment": align.tolist(),
        "rest_alignment_angle_deg": float(np.degrees(np.arccos(np.clip((np.trace(align) - 1) / 2, -1, 1)))),
        "rest_references": rest_references(frames, idx, skel),
        "hinge_bias_deg": hinge_bias(frames, idx, base["rest_pose_found"] and cfg.hinge_bias_correction,
                                     cfg.max_hinge_bias_deg),
        "actor_lengths_m": actor,
        "target_lengths_m": target,
        "target_over_actor": ratios,
        "global_scale": global_scale,
    }

    out_frames = []
    for i, fr in enumerate(frames):
        rec = {"frame_index": raw["frames"][i]["frame_index"], "timestamp_s": raw["frames"][i]["timestamp_s"]}
        if fr is None:
            rec.update(valid=False, joints={}, length_ratio={})
        else:
            joints, vis = fr
            tj = tracker[i][0]
            # length ratio of the *tracker's* world landmarks: depth loss shows up here
            lr = {seg: (float(np.linalg.norm(tj[b] - tj[a]) / actor[seg]) if actor[seg] else None)
                  for seg, (a, b) in SEGMENTS.items()}
            rec.update(valid=True, joints={k: v.tolist() + [vis[k]] for k, v in joints.items()}, length_ratio=lr)
            if depth_info is not None:
                rec["max_depth_change_m"] = depth_info["max_depth_change_m_per_frame"][i]
        out_frames.append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = write_json(out_dir / "actor_profile.json", profile)
    p2 = write_json(out_dir / "calibrated_pose.json", {"meta": {**raw["meta"], "profile": "actor_profile.json"},
                                                        "frames": out_frames})
    return p1, p2


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="altline calibration")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--rig", type=Path, help="rig.npz; default: altline reference rig")
    args = ap.parse_args(argv)
    rd = run_dir(args.run_id)
    for p in calibrate(rd / "observations" / "raw_pose.json", rd / "calibration", load_skeleton(args.rig)):
        print(p)


if __name__ == "__main__":
    main()
