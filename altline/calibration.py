"""Task 2: relative neutral-pose calibration.

Input: <run>/observations/raw_pose.json
Output: <run>/calibration/actor_profile.json and <run>/calibration/calibrated_pose.json

1. Convert MediaPipe world landmarks to canonical space (+X actor left, +Y up, +Z actor
   forward): canon = (x, -y, -z).
2. Find a baseline interval where the actor is closest to a static rest pose (A/T pose):
   elbows and knees straight, arms abducted symmetrically between the configured bounds,
   minimal joint speed.
3. From the baseline: the actor's constant segment lengths (median), the ratio to the
   target rig's lengths, and a rest alignment rotation that makes the actor stand upright
   and face +Z (removes camera tilt / yaw). Also the head-reference pitch offset, so a
   neutral head in the baseline is solved as a neutral head, and the elbow/knee bend the
   tracker reports for visibly straight rest limbs (hinge bias).
4. calibrated_pose.json holds per-frame aligned joints plus each segment's length ratio
   to the actor's constant length. The solver uses only unit bone directions, so length
   variance (depth errors, foreshortening noise) cannot leak into the target's lengths;
   the ratios are kept as a diagnostic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import CalibrationConfig
from .io_utils import read_json, run_dir, write_json
from .skeleton import Skeleton, load_skeleton

SIDES = {"l": "left", "r": "right"}

# segment name -> (from joint, to joint). "*_mid" joints are derived in `derived_joints`.
SEGMENTS = {
    "trunk": ("hip_mid", "shoulder_mid"),
    "neck": ("shoulder_mid", "ear_mid"),
    "hip_width": ("right_hip", "left_hip"),
    "shoulder_width": ("right_shoulder", "left_shoulder"),
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


def reference_pitch_offsets(frames, idx: list[int], align: np.ndarray) -> dict[str, float]:
    """Pitch (radians, about +X) of the face/neck references in the baseline.

    Landmarks give no true neck axis or head-forward axis: ear->nose points down and the
    ears sit in front of the shoulders even in a neutral stance. The solver rotates these
    references back by the baseline pitch, assuming the actor holds the head neutral there.
      head_forward: pitch of (nose - ear_mid) from horizontal
      neck_up:      pitch of (ear_mid - shoulder_mid) from vertical
    """
    head, neck = [], []
    for i in idx:
        j = frames[i][0]
        fwd = align @ (j["nose"] - j["ear_mid"])
        up = align @ (j["ear_mid"] - j["shoulder_mid"])
        head.append(np.arctan2(fwd[1], fwd[2]))
        neck.append(np.arctan2(-up[2], up[1]))
    return {"head_forward": float(np.median(head)), "neck_up": float(np.median(neck))}


def hinge_bias(frames, idx: list[int], enabled: bool) -> dict[str, float]:
    """Median elbow/knee bend in the rest baseline (degrees); zeros when disabled."""
    keys = {"elbow_l": "left_elbow_bend", "elbow_r": "right_elbow_bend",
            "knee_l": "left_knee_bend", "knee_r": "right_knee_bend"}
    if not enabled:
        return {k: 0.0 for k in keys}
    feats = [rest_features(frames[i][0]) for i in idx]
    return {k: float(np.median([f[v] for f in feats])) for k, v in keys.items()}


def calibrate(raw_path: Path, out_dir: Path, skel: Skeleton, cfg: CalibrationConfig | None = None) -> tuple[Path, Path]:
    cfg = cfg or CalibrationConfig()
    raw = read_json(raw_path)
    frames = load_frames(raw)
    base = find_baseline(frames, cfg)
    idx = base["frames"]
    align = rest_alignment(frames, idx)

    actor = {}
    for seg, (a, b) in SEGMENTS.items():
        vals = [np.linalg.norm(frames[i][0][b] - frames[i][0][a]) for i in idx
                if min(frames[i][1][a], frames[i][1][b]) >= cfg.min_visibility]
        actor[seg] = float(np.median(vals)) if vals else None
    target = target_lengths(skel)
    ratios = {k: (target[k] / v if v else None) for k, v in actor.items()}
    leg_trunk = ["trunk", "thigh_l", "shin_l", "thigh_r", "shin_r"]
    global_scale = float(sum(target[k] for k in leg_trunk) / sum(actor[k] for k in leg_trunk))

    profile = {
        "source": str(raw_path),
        "baseline": base,
        "canonical_axes": "+X actor left, +Y up, +Z actor forward (meters)",
        "rest_alignment": align.tolist(),
        "rest_alignment_angle_deg": float(np.degrees(np.arccos(np.clip((np.trace(align) - 1) / 2, -1, 1)))),
        "reference_pitch_offsets_rad": reference_pitch_offsets(frames, idx, align),
        "hinge_bias_deg": hinge_bias(frames, idx, base["rest_pose_found"] and cfg.hinge_bias_correction),
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
            aligned = {k: (align @ v).tolist() + [vis[k]] for k, v in joints.items()}
            lr = {seg: (float(np.linalg.norm(joints[b] - joints[a]) / actor[seg]) if actor[seg] else None)
                  for seg, (a, b) in SEGMENTS.items()}
            rec.update(valid=True, joints=aligned, length_ratio=lr)
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
