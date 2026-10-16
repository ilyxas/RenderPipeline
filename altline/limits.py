"""Task 4: biomechanical constraint gate.

Input: <run>/solver/raw_rotations.json
Output: <run>/limits/clean_rotations.json and <run>/limits/limits_report.json

Limits are expressed on local basis quaternions in the anatomical bone frames of
skeleton.py (Y along the bone, Z anterior, X = Y x Z), via swing-twist decomposition about
the bone's Y axis:
    elbow (forearm bone): swing reduced to one hinge axis (+X), flexion in [0, 140] deg;
                          twist = forearm roll, clamped to +-90 deg
    wrist (hand bone):    twist clamped to +-90 deg; swing X and Z components each +-90 deg
    knee (shin bone):     swing reduced to the hinge axis (-X is flexion), [0, 150] deg;
                          twist clamped to +-30 deg

Anomalies are bridged, not dropped: a frame is anomalous for a bone when the frame had no
detection, its confidence is low, the clamp had to move it further than
`anomaly_clamp_deg` (a flipped or inside-out joint), or it is a single-frame spike (it
departs from the slerp of its neighbours by more than `spike_deg`). Runs of anomalous
frames up to `max_bridge_frames` long are replaced by slerp between the nearest good
frames on both sides (held at the clip edges). Every output frame keeps its timestamp.

Clamping is hierarchical: bones are clamped root to leaf and each child is re-expressed so
it keeps the world orientation it had before its parent was clamped. Twist beyond the
forearm-roll / knee-twist limit is first handed to the upper arm / thigh while that hinge
is nearly straight, instead of being cut off.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import LimitsConfig
from .io_utils import read_json, run_dir, write_json
from .quat import (mat_to_quat, quat_angle_between, quat_exp, quat_from_axis_angle, quat_log, quat_mul,
                   quat_normalize, quat_to_mat, slerp, swing_twist, twist_angle)
from .skeleton import Skeleton, load_skeleton, reference_skeleton

X, Y, Z = np.eye(3)
MIN_CONFIDENCE = 0.3


def _hinge(q: np.ndarray, flex_sign: float, flex_range_deg: tuple[float, float], twist_deg: float) -> tuple[np.ndarray, dict]:
    swing, twist = swing_twist(q, Y)
    phi = twist_angle(twist, Y)
    rv = quat_log(swing)
    flex = np.degrees(flex_sign * rv[0])
    lo, hi = flex_range_deg
    flex_c = float(np.clip(flex, lo, hi))
    phi_c = float(np.clip(np.degrees(phi), -twist_deg, twist_deg))
    out = quat_mul(quat_from_axis_angle(X, np.radians(flex_sign * flex_c)), quat_from_axis_angle(Y, np.radians(phi_c)))
    return out, {"flex_deg": float(flex), "twist_deg": float(np.degrees(phi)),
                 "off_axis_deg": float(np.degrees(abs(rv[2])))}


def _wrist(q: np.ndarray, lim_deg: float) -> tuple[np.ndarray, dict]:
    swing, twist = swing_twist(q, Y)
    phi = np.degrees(twist_angle(twist, Y))
    rv = np.degrees(quat_log(swing))
    rv_c = np.array([np.clip(rv[0], -lim_deg, lim_deg), 0.0, np.clip(rv[2], -lim_deg, lim_deg)])
    phi_c = float(np.clip(phi, -lim_deg, lim_deg))
    out = quat_mul(quat_exp(np.radians(rv_c)), quat_from_axis_angle(Y, np.radians(phi_c)))
    return out, {"swing_x_deg": float(rv[0]), "swing_z_deg": float(rv[2]), "twist_deg": float(phi)}


def clamp_bone(name: str, q: np.ndarray, cfg: LimitsConfig) -> tuple[np.ndarray, dict | None]:
    if name.startswith("forearm_"):
        return _hinge(q, +1.0, cfg.elbow_flex_deg, cfg.forearm_roll_deg)
    if name.startswith("hand_"):
        return _wrist(q, cfg.wrist_deg)
    if name.startswith("shin_"):
        return _hinge(q, -1.0, cfg.knee_flex_deg, cfg.knee_twist_deg)
    return q, None


def make_continuous(qs: np.ndarray) -> np.ndarray:
    out = qs.copy()
    for t in range(1, len(out)):
        if np.dot(out[t], out[t - 1]) < 0:
            out[t] = -out[t]
    return out


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for t, m in enumerate(mask):
        if m and start is None:
            start = t
        if not m and start is not None:
            runs.append((start, t - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def bridge(qs: np.ndarray, mask: np.ndarray, max_len: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    out = qs.copy()
    bridged = []
    n = len(qs)
    if mask.all():
        return out, bridged
    for s, e in find_runs(mask):
        if e - s + 1 > max_len:
            continue
        a, b = s - 1, e + 1
        for t in range(s, e + 1):
            if a < 0:
                out[t] = qs[b]
            elif b >= n:
                out[t] = qs[a]
            else:
                out[t] = slerp(qs[a], qs[b], (t - a) / (b - a))
        bridged.append((s, e))
    return out, bridged


def _twist_transfer(local: np.ndarray, names: list[str], idx: dict, cfg: LimitsConfig) -> np.ndarray:
    """Move twist beyond the forearm-roll / knee-twist limit into the parent bone while the
    hinge is nearly straight (a straight limb's twist is shared by both segments anyway).
    Returns new local quaternions; the child's world orientation is restored later."""
    out = local.copy()
    pairs = [(f"upper_arm_{s}", f"forearm_{s}", cfg.forearm_roll_deg, +1.0) for s in "lr"] + \
            [(f"thigh_{s}", f"shin_{s}", cfg.knee_twist_deg, -1.0) for s in "lr"]
    for parent, child, lim, flex_sign in pairs:
        if parent not in idx or child not in idx:
            continue
        swing, twist = swing_twist(out[idx[child]], Y)
        phi = np.degrees(twist_angle(twist, Y))
        flex = np.degrees(flex_sign * quat_log(swing)[0])
        excess = phi - np.clip(phi, -lim, lim)
        if excess != 0.0 and flex < cfg.twist_transfer_max_flex_deg:
            out[idx[parent]] = quat_mul(out[idx[parent]], quat_from_axis_angle(Y, np.radians(excess)))
    return out


def _clamp_frame(local: np.ndarray, skel: Skeleton, cfg: LimitsConfig) -> tuple[np.ndarray, np.ndarray, list]:
    """Clamp bones root to leaf; every child is re-expressed so its world orientation is the
    one it had before its parent was clamped (a clamped elbow does not swing the hand)."""
    n = len(skel.names)
    want = np.array([quat_to_mat(q) for q in local])
    want_world, _, _ = skel.fk(want)
    world = np.zeros((n, 3, 3))
    out = np.zeros((n, 4))
    correction = np.zeros(n)
    channels = []
    for b, name in enumerate(skel.names):
        p = skel.parents[b]
        rl = skel.rest_local(b)
        parent_world = np.eye(3) if p < 0 else world[p]
        q_in = mat_to_quat(rl.T @ parent_world.T @ want_world[b])
        q_out, ch = clamp_bone(name, q_in, cfg)
        out[b] = quat_normalize(q_out)
        correction[b] = np.degrees(quat_angle_between(q_in, out[b]))
        channels.append(ch)
        world[b] = parent_world @ rl @ quat_to_mat(out[b])
    return out, correction, channels


def apply_limits(raw: dict, cfg: LimitsConfig | None = None, skel: Skeleton | None = None) -> tuple[dict, dict]:
    cfg = cfg or LimitsConfig()
    bones = raw["meta"]["bones"]
    skel = skel or reference_skeleton()
    if skel.names != bones:
        skel = _sub_skeleton(skel, bones)
    idx = {n: i for i, n in enumerate(bones)}
    frames = raw["frames"]
    n_t = len(frames)
    rot = np.array([[f["rotations"][b] for b in range(len(bones))] for f in frames])  # (T, N, 4)
    conf = np.array([f["confidence"] for f in frames])
    valid = np.array([f["valid"] for f in frames])

    clamped = np.zeros_like(rot)
    correction = np.zeros((n_t, len(bones)))
    channels = []
    for t in range(n_t):
        local = _twist_transfer(rot[t], bones, idx, cfg)
        clamped[t], correction[t], ch = _clamp_frame(local, skel, cfg)
        channels.append(ch)

    out = np.zeros_like(rot)
    report = {"config": cfg.__dict__, "bones": {}}
    for b, name in enumerate(bones):
        qs = make_continuous(clamped[:, b])
        spike = np.zeros(n_t)
        for t in range(1, n_t - 1):
            spike[t] = np.degrees(quat_angle_between(qs[t], slerp(qs[t - 1], qs[t + 1], 0.5)))
        mask = (~valid) | (conf[:, b] < MIN_CONFIDENCE) | (correction[:, b] > cfg.anomaly_clamp_deg) | (spike > cfg.spike_deg)
        fixed, bridged = bridge(qs, mask, cfg.max_bridge_frames)
        # interpolation between two in-range poses can leave the hinge plane; clamp again
        out[:, b] = make_continuous(np.array([quat_normalize(clamp_bone(name, q, cfg)[0]) for q in fixed]))

        entry = {
            "frames_clamped": int((correction[:, b] > 0.5).sum()),
            "max_clamp_correction_deg": float(correction[:, b].max()),
            "anomaly_frames": [int(t) for t in np.flatnonzero(mask)],
            "bridged_runs": bridged,
            "unbridged_runs": [r for r in find_runs(mask) if tuple(r) not in {tuple(x) for x in bridged}],
            "max_spike_deg": float(spike.max()),
        }
        if channels[0][b] is not None:
            keys = channels[0][b].keys()
            entry["raw_channel_range_deg"] = {k: [float(min(c[b][k] for c in channels)), float(max(c[b][k] for c in channels))]
                                              for k in keys}
        report["bones"][name] = entry

    clean = {"meta": {**raw["meta"], "limits_config": cfg.__dict__},
             "frames": [{"frame_index": f["frame_index"], "timestamp_s": f["timestamp_s"],
                         "rotations": out[t].tolist()} for t, f in enumerate(frames)]}
    return clean, report


def _sub_skeleton(skel: Skeleton, bones: list[str]) -> Skeleton:
    """The rig restricted to `bones` (tests use a single bone); parents outside become roots."""
    keep = [skel.index[b] for b in bones]
    pos = {old: new for new, old in enumerate(keep)}
    parents = np.array([pos.get(int(skel.parents[i]), -1) for i in keep])
    return Skeleton(list(bones), parents, skel.heads[keep], skel.tails[keep], skel.rest[keep])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="altline joint limits")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--rig", type=Path)
    args = ap.parse_args(argv)
    rd = run_dir(args.run_id)
    ap_rig = load_skeleton(args.rig)
    clean, report = apply_limits(read_json(rd / "solver" / "raw_rotations.json"), skel=ap_rig)
    print(write_json(rd / "limits" / "clean_rotations.json", clean))
    print(write_json(rd / "limits" / "limits_report.json", report))


if __name__ == "__main__":
    main()
