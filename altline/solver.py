"""Task 3: analytical inverse-orientation solver.

Input: <run>/calibration/calibrated_pose.json + actor_profile.json, and the target rig.
Output: <run>/solver/raw_rotations.json — per frame, per bone local basis quaternions [w,x,y,z].

Per frame, every bone gets a target direction (unit vector between landmarks, so the
actor's segment lengths never reach the rig) and, where the landmarks define one, a twist
reference: the direction its local +Z should face.

    upper arm  dir shoulder->elbow     twist: elbow bend plane (forearm, perpendicular part)
    forearm    dir elbow->wrist        twist: pinky->index across the hand
    hand       dir wrist->hand centre  twist: pinky->index
    thigh      dir hip->knee           twist: knee bend plane (shin points to -Z)
    shin       dir knee->ankle         twist: heel->toe
    foot       dir ankle->toe          (inherits twist)
    pelvis / spine / chest: trunk up; facing from the hip line / both lines / shoulder line
    neck / head: shoulder centre->ear centre; head frame from ears and nose

Bones are solved root to leaf. Each bone starts from where its parent carries it with a
zero local rotation; a minimal swing aligns its Y axis to the target direction and a twist
about Y aligns +Z to the twist reference (weighted by how well the reference is defined:
a straight limb has no bend plane, so its twist falls back to the inherited one).
local basis = rest_local^T @ parent_world^T @ world. Only rotations are produced; bone
lengths are the rig's constants (FK in skeleton.Skeleton.fk).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import SolverConfig
from .io_utils import read_json, run_dir, write_json
from .quat import mat_to_quat, normalize, rot_axis_angle, rot_between, signed_angle_about
from .skeleton import Skeleton, load_skeleton

SIDES = {"l": "left", "r": "right"}


def smoothstep(x: float, lo: float, hi: float) -> float:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return float(t * t * (3 - 2 * t))


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))


def debias_bend(theta: float, bias: float) -> float:
    """Remap a hinge bend [bias, 180] -> [0, 180]; bends below the rest bias become 0."""
    if bias <= 0:
        return theta
    return max(0.0, (theta - bias) * 180.0 / (180.0 - bias))


def perp(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    a = normalize(axis)
    return v - np.dot(v, a) * a


def _perp_weight(v: np.ndarray, axis: np.ndarray) -> float:
    """How well v defines a direction around axis: 0 when parallel, 1 when well off-axis."""
    n = np.linalg.norm(v)
    if n < 1e-9:
        return 0.0
    return smoothstep(np.linalg.norm(perp(v, axis)) / n, 0.2, 0.5)


def _hinge(root: np.ndarray, mid: np.ndarray, end: np.ndarray, bias: float
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """De-bias a two-segment limb while keeping its chord (root -> end) and bend plane.

    Returns (proximal dir, distal dir, rotation applied to the distal segment,
    raw bend deg, corrected bend deg). The end effector (wrist / ankle) is the landmark the
    tracker places best, so the chord direction is preserved and only the bend is reduced;
    segment proportions are this frame's measured ones.
    """
    prox, dist = mid - root, end - mid
    theta = angle_deg(prox, dist)
    target = debias_bend(theta, bias)
    n = np.cross(prox, dist)
    if np.linalg.norm(n) < 1e-9 or abs(target - theta) < 1e-9:
        return prox, dist, np.eye(3), theta, theta
    a, b = np.linalg.norm(prox), np.linalg.norm(dist)
    chord = end - root
    inner = np.radians(180.0 - target)                       # interior angle at the middle joint
    d = np.sqrt(a * a + b * b - 2 * a * b * np.cos(inner))
    alpha = np.arccos(np.clip((a * a + d * d - b * b) / (2 * a * d), -1.0, 1.0))  # proximal angle off the chord
    sign = np.sign(signed_angle_about(chord, prox, n)) or 1.0
    new_prox = rot_axis_angle(n, sign * alpha) @ normalize(chord)
    new_dist = rot_axis_angle(n, np.radians(target)) @ new_prox  # n = prox x dist: bend is +theta about n
    return new_prox, new_dist, rot_between(dist, new_dist), theta, target


def bone_targets(j: dict[str, np.ndarray], vis: dict[str, float], profile: dict,
                 cfg: SolverConfig) -> tuple[dict, dict]:
    """Per-bone (direction, twist reference, twist weight, confidence) for one frame."""
    offs = profile["reference_pitch_offsets_rad"]
    bias = profile["hinge_bias_deg"]
    t: dict[str, tuple] = {}
    diag: dict[str, float] = {}

    def conf(*names):
        return float(min(vis[n] for n in names))

    up = normalize(j["shoulder_mid"] - j["hip_mid"])
    fwd_hips = normalize(np.cross(j["left_hip"] - j["right_hip"], up))
    sh_lat = j["left_shoulder"] - j["right_shoulder"]
    fwd_sh = normalize(np.cross(sh_lat, up))
    c_trunk = conf("left_hip", "right_hip", "left_shoulder", "right_shoulder")
    t["pelvis"] = (up, fwd_hips, 1.0, c_trunk)
    t["spine"] = (up, normalize(fwd_hips + fwd_sh), 1.0, c_trunk)
    t["chest"] = (up, fwd_sh, 1.0, c_trunk)

    ear_lat = j["left_ear"] - j["right_ear"]
    head_fwd = rot_axis_angle(ear_lat, offs["head_forward"]) @ (j["nose"] - j["ear_mid"])
    head_up = normalize(np.cross(head_fwd, ear_lat))
    neck_up = rot_axis_angle(sh_lat, offs["neck_up"]) @ (j["ear_mid"] - j["shoulder_mid"])
    c_head = conf("nose", "left_ear", "right_ear")
    t["neck"] = (neck_up, head_fwd, 1.0, min(c_head, c_trunk))
    t["head"] = (head_up, head_fwd, 1.0, c_head)

    for s, side in SIDES.items():
        sh, el, wr = j[f"{side}_shoulder"], j[f"{side}_elbow"], j[f"{side}_wrist"]
        upper, fore, corr, raw, fixed = _hinge(sh, el, wr, bias[f"elbow_{s}"])
        diag[f"elbow_{s}_bend_raw"], diag[f"elbow_{s}_bend"] = raw, fixed
        hand_dir = corr @ (j[f"{side}_hand_mid"] - wr)
        hand_ref = corr @ (j[f"{side}_index"] - j[f"{side}_pinky"])
        w_bend = smoothstep(fixed, cfg.bend_twist_zero_deg, cfg.bend_twist_full_deg)
        c_hand = conf(f"{side}_wrist", f"{side}_index", f"{side}_pinky")
        t[f"clavicle_{s}"] = (j[f"{side}_shoulder"] - j["shoulder_mid"], None, 0.0, c_trunk)
        t[f"upper_arm_{s}"] = (upper, perp(fore, upper), w_bend, conf(f"{side}_shoulder", f"{side}_elbow"))
        t[f"forearm_{s}"] = (fore, hand_ref, _perp_weight(hand_ref, fore) * min(1.0, c_hand / 0.5),
                             conf(f"{side}_elbow", f"{side}_wrist"))
        t[f"hand_{s}"] = (hand_dir, hand_ref, _perp_weight(hand_ref, hand_dir), c_hand)

        hip, kn, an = j[f"{side}_hip"], j[f"{side}_knee"], j[f"{side}_ankle"]
        thigh, shin, corr, raw, fixed = _hinge(hip, kn, an, bias[f"knee_{s}"])
        diag[f"knee_{s}_bend_raw"], diag[f"knee_{s}_bend"] = raw, fixed
        foot_dir = corr @ (j[f"{side}_foot_index"] - an)
        foot_ref = corr @ (j[f"{side}_foot_index"] - j[f"{side}_heel"])
        w_bend = smoothstep(fixed, cfg.bend_twist_zero_deg, cfg.bend_twist_full_deg)
        c_foot = conf(f"{side}_ankle", f"{side}_heel", f"{side}_foot_index")
        t[f"thigh_{s}"] = (thigh, -perp(shin, thigh), w_bend, conf(f"{side}_hip", f"{side}_knee"))
        t[f"shin_{s}"] = (shin, foot_ref, _perp_weight(foot_ref, shin), conf(f"{side}_knee", f"{side}_ankle"))
        t[f"foot_{s}"] = (foot_dir, None, 0.0, c_foot)
    return t, diag


def solve_frame(skel: Skeleton, targets: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (local basis matrices (N,3,3), world rotations (N,3,3), per-bone twist applied deg)."""
    n = len(skel.names)
    world = np.zeros((n, 3, 3))
    basis = np.zeros((n, 3, 3))
    twist_deg = {}
    for i, name in enumerate(skel.names):
        p = skel.parents[i]
        rest_local = skel.rest_local(i)
        carried = skel.rest[i] if p < 0 else world[p] @ rest_local
        tgt = targets.get(name)
        if tgt is None:
            world[i] = carried
        else:
            direction, ref, weight, _ = tgt
            w = rot_between(carried[:, 1], direction) @ carried
            if ref is not None and weight > 0 and np.linalg.norm(perp(ref, w[:, 1])) > 1e-9:
                ang = signed_angle_about(w[:, 2], ref, w[:, 1]) * weight
                w = rot_axis_angle(w[:, 1], ang) @ w
                twist_deg[name] = float(np.degrees(ang))
            world[i] = w
        parent_world = np.eye(3) if p < 0 else world[p]
        basis[i] = rest_local.T @ parent_world.T @ world[i]
    return basis, world, twist_deg


def solve(calibrated_path: Path, profile_path: Path, out_path: Path, skel: Skeleton,
          cfg: SolverConfig | None = None) -> Path:
    cfg = cfg or SolverConfig()
    cal = read_json(calibrated_path)
    profile = read_json(profile_path)
    frames_out = []
    ident = [1.0, 0.0, 0.0, 0.0]
    for f in cal["frames"]:
        rec = {"frame_index": f["frame_index"], "timestamp_s": f["timestamp_s"], "valid": f["valid"]}
        if not f["valid"]:
            rec.update(rotations=[ident] * len(skel.names), confidence=[0.0] * len(skel.names), diag={})
        else:
            j = {k: np.array(v[:3]) for k, v in f["joints"].items()}
            vis = {k: float(v[3]) for k, v in f["joints"].items()}
            targets, diag = bone_targets(j, vis, profile, cfg)
            basis, world, twist = solve_frame(skel, targets)
            # direction residual: angle between solved bone Y and target direction (should be ~0)
            resid = {name: angle_deg(world[i][:, 1], targets[name][0])
                     for i, name in enumerate(skel.names) if name in targets}
            diag.update(max_direction_residual_deg=max(resid.values()))
            rec.update(
                rotations=[mat_to_quat(b).tolist() for b in basis],
                confidence=[float(targets[nm][3]) if nm in targets else 1.0 for nm in skel.names],
                diag=diag,
            )
        frames_out.append(rec)
    meta = {
        "bones": skel.names,
        "fps": cal["meta"]["fps"],
        "rotation": "local basis quaternion [w,x,y,z] in the bone's rest frame (Blender rotation_quaternion)",
        "rig_lengths_m": dict(zip(skel.names, skel.lengths.tolist())),
        "solver_config": cfg.__dict__,
    }
    return write_json(out_path, {"meta": meta, "frames": frames_out})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="altline solver")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--rig", type=Path)
    args = ap.parse_args(argv)
    rd = run_dir(args.run_id)
    print(solve(rd / "calibration" / "calibrated_pose.json", rd / "calibration" / "actor_profile.json",
                rd / "solver" / "raw_rotations.json", load_skeleton(args.rig)))


if __name__ == "__main__":
    main()
