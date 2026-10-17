"""Reviewer: v1 vs v2 metrics on the same clip, from the motion bundles and the source 2D points.

    python3 -m altline.review.compare --pair v1_0663 v2_0663 [--pair ...] [--ablate]

All references are independent of the pipeline version: angular acceleration of the bundle,
2D (image-plane) direction / end-effector errors of the rig against the tracker's *image*
points (raw_pose.json image_joints, identical for v1 and v2), a 2D head-yaw estimate from
ear spacing + nose offset, out-of-plane angles of limbs that do not move, trunk yaw, and
clip-specific semantic checks (hand on shoulder, hand on face, kick direction, ...).
`--ablate` re-runs v2 calibration -> solver -> limits in a scratch dir with the depth
rebuild or the filter switched off, to attribute effects (nothing in runs/ is touched).

Writes reports/<v2>/plots/cmp_*.png and returns a dict that audit_v2 merges into
dev_audit.json["v1_vs_v2"].
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from altline.animation import load_bundle
from altline.config import Config
from altline.depth import image_points, pixel_scale, planar_lengths
from altline.io_utils import REPORTS_ROOT, read_json, run_dir
from altline.quat import quat_conj, quat_log, quat_mul, quat_to_mat
from altline.skeleton import reference_skeleton

GATE = Config().stability_gate_deg_s2
SKEL = reference_skeleton()
I = SKEL.index
SIDES = {"l": "left", "r": "right"}

# clip hash -> what is moving / static, hold window, semantic checks (review-side only)
CLIPS = {
    "061c": {"moving": ["arm_r"], "static": ["arm_l", "leg_l", "leg_r"], "hold": (100, 144), "trunk_static": True},
    "0663": {"moving": ["arm_l", "arm_r"], "static": ["leg_l", "leg_r"], "hold": (60, 144), "trunk_static": True,
             "head_static": True},
    "5313": {"moving": ["head"], "static": ["arm_l", "arm_r", "leg_l", "leg_r"], "hold": (60, 144), "trunk_static": False},
    "619b": {"moving": ["leg_r"], "static": ["leg_l"], "hold": (70, 110), "trunk_static": True, "head_static": True},
    "690b": {"moving": ["arm_l"], "static": ["arm_r", "leg_l", "leg_r"], "hold": (37, 110), "trunk_static": True},
    "85a9": {"moving": ["arm_l", "arm_r"], "static": ["leg_l", "leg_r"], "hold": (45, 144), "trunk_static": True},
    "9645": {"moving": ["head"], "static": ["arm_l", "arm_r", "leg_l", "leg_r"], "hold": (60, 144), "trunk_static": False},
    "df1e": {"moving": ["leg_l"], "static": ["leg_r"], "hold": (100, 144), "trunk_static": True, "head_static": True},
}
LIMB_BONES = {"arm_l": ["upper_arm_l", "forearm_l"], "arm_r": ["upper_arm_r", "forearm_r"],
              "leg_l": ["thigh_l", "shin_l"], "leg_r": ["thigh_r", "shin_r"]}
BONE_2D = {f"upper_arm_{s}": (f"{d}_shoulder", f"{d}_elbow") for s, d in SIDES.items()}
BONE_2D.update({f"forearm_{s}": (f"{d}_elbow", f"{d}_wrist") for s, d in SIDES.items()})
BONE_2D.update({f"thigh_{s}": (f"{d}_hip", f"{d}_knee") for s, d in SIDES.items()})
BONE_2D.update({f"shin_{s}": (f"{d}_knee", f"{d}_ankle") for s, d in SIDES.items()})
BONE_2D["chest"] = ("hip_mid", "shoulder_mid")
LEN_KEY = {b: b for b in BONE_2D}
LEN_KEY["chest"] = "trunk"


def ang(a, b):
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))


def ang2(a, b):
    return float(np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)))


def fk_all(quats):
    W, H, T = [], [], []
    for qs in quats:
        w, h, t = SKEL.fk(np.array([quat_to_mat(q) for q in qs]))
        W.append(w), H.append(h), T.append(t)
    return np.array(W), np.array(H), np.array(T)


def ang_acc(quats, fps):
    T, N, _ = quats.shape
    w = np.zeros((T - 1, N, 3))
    for t in range(1, T):
        for b in range(N):
            w[t - 1, b] = quat_log(quat_mul(quat_conj(quats[t - 1, b]), quats[t, b])) * fps
    a = np.zeros((T, N))
    a[1:-1] = np.degrees(np.linalg.norm(np.diff(w, axis=0), axis=2)) * fps
    return a


def _q(x, n=1):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def stats(v, n=1):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return None
    return {"median": _q(np.median(v), n), "p95": _q(np.percentile(v, 95), n), "max": _q(v.max(), n), "n": int(len(v))}


class Run:
    def __init__(self, run_id: str, quats: np.ndarray | None = None, profile: dict | None = None):
        rd = run_dir(run_id) if quats is None else None
        self.id = run_id
        if quats is None:
            bones, _, Q, fps = load_bundle(rd / "animation" / "motion_bundle.json")
            assert bones == SKEL.names
            self.profile = read_json(rd / "calibration" / "actor_profile.json")
            raw = read_json(rd / "observations" / "raw_pose.json")
        else:
            Q, fps = quats, 24.0
            self.profile = profile
            raw = read_json(run_dir(profile["_raw_run"]) / "observations" / "raw_pose.json")
        self.Q, self.fps = Q, fps
        self.W, self.H, self.T = fk_all(Q)
        self.align = np.asarray(self.profile["rest_alignment"])
        self.base = self.profile["baseline"]["frames"]
        meta = raw["meta"]
        self.pts = [image_points(f["image_joints"], meta["width"], meta["height"]) if f["detected"] else None
                    for f in raw["frames"]]
        self.n = len(Q)

    def img(self, v):
        """Rig vector (aligned canonical) -> camera image plane (x right, y up), weak perspective."""
        return (self.align.T @ v)[:2]


def reference_lengths(v2_profile: dict, pts) -> tuple[float, dict]:
    """Pixel scale and planar segment lengths from the v2 profile (same raw data for v1 and v2)."""
    d = v2_profile["depth_from_2d"]
    return d["pixel_scale_m"], d["lengths_m"]


def head_yaw_2d(pts, base):
    """Signed head yaw (deg, + toward actor left = image right) from the image only:
    |yaw| = acos(ear span / rest ear span), sign from the nose offset to the ear midpoint."""
    span = np.array([np.linalg.norm(p["left_ear"] - p["right_ear"]) if p else np.nan for p in pts])
    off = np.array([(p["nose"][0] - p["ear_mid"][0]) if p else np.nan for p in pts])
    s0, o0 = np.nanmedian(span[base]), np.nanmedian(off[base])
    mag = np.degrees(np.arccos(np.clip(span / s0, 0, 1)))
    return np.sign(off - o0) * mag, span / s0, (off - o0) / s0


def metrics(run: Run, scale: float, L: dict, clip: dict) -> dict:
    W, H, T = run.W, run.H, run.T
    n = run.n
    acc = ang_acc(run.Q, run.fps)
    out = {"angular_acceleration": {}}
    for b, name in enumerate(SKEL.names):
        a = acc[:, b]
        out["angular_acceleration"][name] = {"max": _q(a.max(), 0), "argmax": int(a.argmax()),
                                             "p99": _q(np.percentile(a, 99), 0), "p95": _q(np.percentile(a, 95), 0),
                                             "frames_above_gate": int((a > GATE).sum())}
    out["frames_above_gate_total"] = int((acc > GATE).sum())
    out["_acc"] = acc

    # 2D direction error of each bone vs image points (only where the source segment is not
    # strongly foreshortened: projected >= 30 % of its length)
    d2 = {}
    for bone, (a, b) in BONE_2D.items():
        e = []
        for t in range(n):
            p = run.pts[t]
            if p is None:
                e.append(None)
                continue
            src = (p[b] - p[a]) * scale
            if np.linalg.norm(src) < 0.3 * L[LEN_KEY[bone]]:
                e.append(None)
                continue
            rig = run.img(T[t, I[bone]] - H[t, I[bone]]) if bone != "chest" else run.img(H[t, I["upper_arm_l"]] + H[t, I["upper_arm_r"]] - 2 * H[t, I["pelvis"]])
            e.append(abs(ang2(rig, src)))
        d2[bone] = {**(stats(e) or {}), "per_frame": [_q(x) for x in e]}
    out["direction_error_2d_deg"] = d2

    # end effector (wrist rel. shoulder, ankle rel. hip) in the image plane, source scaled to rig
    eff = {}
    for s, d in SIDES.items():
        for lab, root, end, b1, b2 in (("wrist", "shoulder", "wrist", f"upper_arm_{s}", f"forearm_{s}"),
                                       ("ankle", "hip", "ankle", f"thigh_{s}", f"shin_{s}")):
            k = (SKEL.lengths[I[b1]] + SKEL.lengths[I[b2]]) / (L[b1] + L[b2])
            e = []
            for t in range(n):
                p = run.pts[t]
                if p is None:
                    e.append(None)
                    continue
                src = (p[f"{d}_{end}"] - p[f"{d}_{root}"]) * scale * k
                rig = run.img(T[t, I[b2]] - H[t, I[b1]])
                e.append(100 * float(np.linalg.norm(rig - src)))
            eff[f"{lab}_{s}"] = {**(stats(e) or {}), "per_frame_cm": [_q(x) for x in e]}
    out["end_effector_error_2d_cm"] = eff

    # head yaw vs image estimate
    yaw2d, ratio, noff = head_yaw_2d(run.pts, run.base)
    yaw_rig = np.array([np.degrees(np.arctan2(W[t, I["head"]][0, 2], W[t, I["head"]][2, 2])) for t in range(n)])
    yaw_rig_rel = yaw_rig - np.median(yaw_rig[run.base])
    h0, h1 = clip["hold"]
    hold = slice(h0, h1 + 1)
    big = np.abs(yaw2d) > 20
    out["head_yaw"] = {
        "rig_rest_deg": _q(np.median(yaw_rig[run.base])),
        "rig_rel_rest_deg": [_q(x) for x in yaw_rig_rel], "image_estimate_deg": [_q(x) for x in yaw2d],
        "rig_hold_median_deg": _q(np.median(yaw_rig_rel[hold])), "image_hold_median_deg": _q(np.nanmedian(yaw2d[hold])),
        "rig_abs_max_deg": _q(np.abs(yaw_rig_rel).max()), "rig_abs_argmax": int(np.abs(yaw_rig_rel).argmax()),
        "abs_error_where_image_yaw_gt20": stats(np.abs(yaw_rig_rel - yaw2d)[big]) if big.any() else None,
        "abs_error_hold": stats(np.abs(yaw_rig_rel - yaw2d)[hold]),
        "image_ear_span_ratio_min_max": [_q(np.nanmin(ratio), 3), _q(np.nanmax(ratio), 3)],
    }
    out["_yaw"] = (yaw_rig_rel, yaw2d)

    # out-of-plane angle of limbs that should not move + at the rest baseline
    oop = {}
    for limb in clip["static"]:
        for bone in LIMB_BONES[limb]:
            fwd = np.array([np.degrees(np.arcsin(np.clip((T[t, I[bone]] - H[t, I[bone]])[2] / SKEL.lengths[I[bone]], -1, 1)))
                            for t in range(n)])
            oop[bone] = {"median": _q(np.median(fwd)), "abs_max": _q(np.abs(fwd).max()), "argmax": int(np.abs(fwd).argmax()),
                         "range": _q(fwd.max() - fwd.min())}
    out["static_limb_forward_angle_deg(+ = toward camera)"] = oop
    rest_oop = {}
    for bone in ("upper_arm_l", "forearm_l", "upper_arm_r", "forearm_r", "thigh_l", "shin_l", "thigh_r", "shin_r"):
        fwd = [np.degrees(np.arcsin(np.clip((T[t, I[bone]] - H[t, I[bone]])[2] / SKEL.lengths[I[bone]], -1, 1))) for t in run.base]
        rest_oop[bone] = _q(np.median(fwd))
    out["rest_baseline_forward_angle_deg"] = rest_oop

    chest_yaw = np.array([np.degrees(np.arctan2(W[t, I["chest"]][0, 2], W[t, I["chest"]][2, 2])) for t in range(n)])
    chest_pitch = np.array([np.degrees(np.arctan2(-W[t, I["chest"]][2, 1], W[t, I["chest"]][1, 1])) for t in range(n)])
    out["chest_yaw_deg"] = {"min": _q(chest_yaw.min()), "max": _q(chest_yaw.max()), "abs_argmax": int(np.abs(chest_yaw).argmax())}
    out["chest_pitch_deg(+ back)"] = {"min": _q(chest_pitch.min()), "max": _q(chest_pitch.max()), "argmax": int(chest_pitch.argmax())}
    out["_trunk"] = (chest_yaw, chest_pitch)
    out["semantic"] = semantic(run, clip)
    return out


def semantic(run: Run, clip_id_info: dict) -> dict:
    W, H, T, n = run.W, run.H, run.T, run.n
    h0, h1 = clip_id_info["hold"]
    hold = range(h0, h1 + 1)
    cid = clip_id_info["id"]
    s = {}
    elbow = {sd: np.array([np.degrees(quat_log(run.Q[t, I[f"forearm_{sd}"]])[0]) for t in range(n)]) for sd in "lr"}
    knee = {sd: np.array([-np.degrees(quat_log(run.Q[t, I[f"shin_{sd}"]])[0]) for t in range(n)]) for sd in "lr"}
    if cid == "0663":
        for sd, other in (("l", "r"), ("r", "l")):
            dist = [100 * np.linalg.norm(T[t, I[f"hand_{sd}"]] - H[t, I[f"upper_arm_{other}"]]) for t in hold]
            s[f"hand_{sd}_tip_to_opposite_shoulder_cm(hold)"] = stats(dist)
            s[f"hand_{sd}_tip_forward_of_chest_cm(hold median, z)"] = _q(np.median([100 * (T[t, I[f"hand_{sd}"]][2]) for t in hold]))
            s[f"elbow_{sd}_flex_hold_median_deg"] = _q(np.median(elbow[sd][list(hold)]))
        dd = [100 * seg_min(H[t, I["forearm_l"]], T[t, I["hand_l"]], H[t, I["forearm_r"]], T[t, I["hand_r"]]) for t in range(n)]
        s["forearm_l_r_min_distance_cm"] = {"min": _q(min(dd)), "frames_below_5cm": int(sum(x < 5 for x in dd))}
    if cid == "85a9":
        for sd in "lr":
            face = [H[t, I["head"]] + 0.08 * W[t, I["head"]][:, 1] + 0.09 * W[t, I["head"]][:, 2] for t in range(n)]
            dist = [100 * np.linalg.norm(T[t, I[f"hand_{sd}"]] - face[t]) for t in hold]
            s[f"hand_{sd}_tip_to_face_cm(hold)"] = stats(dist)
            s[f"elbow_{sd}_flex_hold_median_deg"] = _q(np.median(elbow[sd][list(hold)]))
            s[f"hand_{sd}_inside_head_sphere_frames(<9cm from head centre)"] = int(sum(
                np.linalg.norm(T[t, I[f"hand_{sd}"]] - (0.5 * (H[t, I["head"]] + T[t, I["head"]]) + 0.01 * W[t, I["head"]][:, 2])) < 0.09
                for t in range(n)))
    if cid == "df1e":
        th = np.array([T[t, I["thigh_l"]] - H[t, I["thigh_l"]] for t in range(n)])
        th /= np.linalg.norm(th, axis=1)[:, None]
        sag = np.degrees(np.arctan2(th[:, 2], -th[:, 1]))     # 0 = down, 90 = forward horizontal, 180 = up
        leg = np.array([T[t, I["shin_l"]] - H[t, I["thigh_l"]] for t in range(n)])
        s["thigh_l_sagittal_angle_deg(0 down, 90 forward, 180 up, >180 behind)"] = {
            "hold_median": _q(np.median(sag[list(hold)])), "max": _q(sag.max()), "argmax": int(sag.argmax())}
        s["thigh_l_forward_component_hold_median(z of unit dir)"] = _q(np.median(th[list(hold), 2]), 3)
        s["thigh_l_lateral_component_hold_median(x of unit dir)"] = _q(np.median(th[list(hold), 0]), 3)
        s["ankle_l_forward_of_hip_cm(hold median)"] = _q(100 * np.median(leg[list(hold), 2]))
        s["ankle_l_height_above_hip_cm(hold median)"] = _q(100 * np.median(leg[list(hold), 1]))
        s["knee_l_flex_hold_median_deg"] = _q(np.median(knee["l"][list(hold)]))
        s["thigh_l_twist_range_hold_deg"] = None
        s["_sag"] = sag
    if cid == "619b":
        th = np.array([T[t, I["thigh_r"]] - H[t, I["thigh_r"]] for t in range(n)])
        th /= np.linalg.norm(th, axis=1)[:, None]
        sag = np.degrees(np.arctan2(th[:, 2], -th[:, 1]))
        s["thigh_r_sagittal_angle_deg(hold median / max)"] = [_q(np.median(sag[list(hold)])), _q(sag.max())]
        s["knee_r_flex_max_deg"] = _q(knee["r"].max())
    if cid in ("690b", "061c"):
        sd = "l" if cid == "690b" else "r"
        hand = np.array([T[t, I[f"hand_{sd}"]] - H[t, I[f"upper_arm_{sd}"]] for t in range(n)])
        s[f"hand_{sd}_forward_of_shoulder_cm(hold median)"] = _q(100 * np.median(hand[list(hold), 2]))
        s[f"upper_arm_{sd}_elevation_max_deg"] = _q(max(ang(T[t, I[f"upper_arm_{sd}"]] - H[t, I[f"upper_arm_{sd}"]], [0, -1, 0]) for t in range(n)))
        s[f"elbow_{sd}_flex_hold_median_deg"] = _q(np.median(elbow[sd][list(hold)]))
    return s


def seg_min(p1, q1, p2, q2):
    t = np.linspace(0, 1, 11)
    a = p1[None] + t[:, None] * (q1 - p1)[None]
    b = p2[None] + t[:, None] * (q2 - p2)[None]
    return float(np.min(np.linalg.norm(a[:, None] - b[None], axis=2)))


def chord_angle_series(run: Run, sd: str, limb: str, scale: float):
    """Image-plane angle (deg) of the limb chord for the source 2D points and for the rig."""
    d = SIDES[sd]
    if limb == "arm":
        a, e, b1, b2 = f"{d}_shoulder", f"{d}_wrist", f"upper_arm_{sd}", f"forearm_{sd}"
    else:
        a, e, b1, b2 = f"{d}_hip", f"{d}_ankle", f"thigh_{sd}", f"shin_{sd}"
    src, rig = [], []
    for t in range(run.n):
        p = run.pts[t]
        v = p[e] - p[a]
        src.append(np.degrees(np.arctan2(v[0], -v[1])))
        r = run.img(run.T[t, I[b2]] - run.H[t, I[b1]])
        rig.append(np.degrees(np.arctan2(r[0], -r[1])))
    return np.unwrap(np.radians(src)) * 180 / np.pi, np.unwrap(np.radians(rig)) * 180 / np.pi


def lag_frames(src: np.ndarray, rig: np.ndarray, max_lag: int = 6) -> float:
    """Lag (frames, + = rig late) maximising the correlation of the velocities."""
    vs, vr = np.diff(src), np.diff(rig)
    best, cs = None, []
    for L_ in range(-max_lag, max_lag + 1):
        if L_ >= 0:
            c = np.dot(vs[: len(vs) - L_], vr[L_:])
        else:
            c = np.dot(vs[-L_:], vr[: len(vr) + L_])
        cs.append(c)
    cs = np.array(cs)
    k = int(cs.argmax())
    if 0 < k < len(cs) - 1:  # parabolic sub-frame
        den = cs[k - 1] - 2 * cs[k] + cs[k + 1]
        off = 0.5 * (cs[k - 1] - cs[k + 1]) / den if den != 0 else 0.0
    else:
        off = 0.0
    return round(k - max_lag + off, 2)


def lag_check(v1: Run, v2: Run, cid: str, scale: float) -> dict:
    out = {}
    for limb_id in CLIPS[cid]["moving"]:
        if limb_id == "head":
            continue
        limb, sd = limb_id.split("_")
        src, r1 = chord_angle_series(v1, sd, limb, scale)
        _, r2 = chord_angle_series(v2, sd, limb, scale)
        vs = np.abs(np.diff(src)) * v2.fps
        tpk = int(vs.argmax()) + 1
        out[f"{limb_id}_chord_image_angle"] = {
            "lag_frames_v1(+ = late)": lag_frames(src, r1), "lag_frames_v2(+ = late)": lag_frames(src, r2),
            "peak_speed_deg_s": {"source_2d": _q(vs.max()), "v1": _q(np.abs(np.diff(r1)).max() * v2.fps),
                                 "v2": _q(np.abs(np.diff(r2)).max() * v2.fps)},
            "frame_of_source_peak_speed": tpk,
            "excursion_max_deg(abs from rest)": {k: _q(np.abs(x - np.median(x[v2.base])).max()) for k, x in
                                                 (("source_2d", src), ("v1", r1), ("v2", r2))},
            "abs_error_vs_source_deg": {"v1": stats(np.abs(r1 - src)), "v2": stats(np.abs(r2 - src))},
            "_series": (src, r1, r2),
        }
    return out


def ablation(v2_id: str) -> dict:
    """Re-run v2 calibration/solver/limits in a temp dir with depth or filter off."""
    from altline.calibration import calibrate
    from altline.config import CalibrationConfig
    from altline.limits import apply_limits
    from altline.solver import solve
    rd = run_dir(v2_id)
    out = {}
    for label, cfg in (("v2_reproduced", CalibrationConfig()),
                       ("v2_no_depth_rebuild", CalibrationConfig(depth_from_2d=False)),
                       ("v2_no_filter", CalibrationConfig(filter_enabled=False))):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            calibrate(rd / "observations" / "raw_pose.json", td / "cal", SKEL, cfg)
            solve(td / "cal" / "calibrated_pose.json", td / "cal" / "actor_profile.json", td / "raw.json", SKEL)
            clean, rep = apply_limits(read_json(td / "raw.json"), skel=SKEL)
            prof = read_json(td / "cal" / "actor_profile.json")
            cal = read_json(td / "cal" / "calibrated_pose.json")
        prof["_raw_run"] = v2_id
        Q = np.array([f["rotations"] for f in clean["frames"]], dtype=float)
        out[label] = (Q, prof, cal, rep)
    return out


def calibrated_length_stats(cal: dict, L: dict) -> dict:
    """Segment length of the landmarks the solver actually sees, relative to the planar length."""
    seg = {"upper_arm_l": ("left_shoulder", "left_elbow"), "forearm_l": ("left_elbow", "left_wrist"),
           "upper_arm_r": ("right_shoulder", "right_elbow"), "forearm_r": ("right_elbow", "right_wrist"),
           "thigh_l": ("left_hip", "left_knee"), "shin_l": ("left_knee", "left_ankle"),
           "thigh_r": ("right_hip", "right_knee"), "shin_r": ("right_knee", "right_ankle")}
    out = {}
    for k, (a, b) in seg.items():
        v = np.array([np.linalg.norm(np.array(f["joints"][b][:3]) - np.array(f["joints"][a][:3])) / L[k]
                      for f in cal["frames"] if f["valid"]])
        out[k] = {"min": _q(v.min(), 3), "argmin": int(v.argmin()), "max": _q(v.max(), 3),
                  "frames_off_15pct": int((np.abs(v - 1) > 0.15).sum())}
    return out


def verdict(v1, v2, lower_better=True, tol=0.1, abs_tol=0.0):
    if v1 is None or v2 is None:
        return "n/a"
    if abs(v2 - v1) <= max(abs_tol, tol * abs(v1)):
        return "same"
    better = (v2 < v1) if lower_better else (v2 > v1)
    return "better" if better else "WORSE"


def compare(v1_id: str, v2_id: str, ablate: bool = False) -> dict:
    cid = v2_id.split("_")[1]
    clip = {**CLIPS[cid], "id": cid}
    r1, r2 = Run(v1_id), Run(v2_id)
    scale, L = reference_lengths(r2.profile, r2.pts)
    m1, m2 = metrics(r1, scale, L, clip), metrics(r2, scale, L, clip)
    res = {"reference": "same raw_pose.json; 2D references = image_joints scaled by v2 pixel_scale_m and planar lengths",
           "clip": cid}

    acc = {}
    for b in SKEL.names:
        a1, a2 = m1["angular_acceleration"][b], m2["angular_acceleration"][b]
        if max(a1["max"], a2["max"]) < GATE:
            continue
        acc[b] = {"v1": {k: a1[k] for k in ("max", "argmax", "p99", "p95", "frames_above_gate")},
                  "v2": {k: a2[k] for k in ("max", "argmax", "p99", "p95", "frames_above_gate")},
                  "verdict_peak": verdict(a1["max"], a2["max"]),
                  "verdict_frames_above_gate": verdict(a1["frames_above_gate"], a2["frames_above_gate"], abs_tol=1)}
    res["angular_acceleration_per_bone(bones above gate in v1 or v2)"] = acc
    res["frames_above_gate_total"] = {"v1": m1["frames_above_gate_total"], "v2": m2["frames_above_gate_total"],
                                      "verdict": verdict(m1["frames_above_gate_total"], m2["frames_above_gate_total"], abs_tol=1)}
    d = {}
    for b in BONE_2D:
        a1, a2 = m1["direction_error_2d_deg"][b], m2["direction_error_2d_deg"][b]
        if "p95" not in a1:
            continue
        d[b] = {"v1": {k: a1[k] for k in ("median", "p95", "max", "n")}, "v2": {k: a2[k] for k in ("median", "p95", "max", "n")},
                "verdict_p95": verdict(a1["p95"], a2["p95"], abs_tol=1.5)}
    res["direction_error_2d_vs_image_landmarks_deg"] = d
    e = {}
    for k in m1["end_effector_error_2d_cm"]:
        a1, a2 = m1["end_effector_error_2d_cm"][k], m2["end_effector_error_2d_cm"][k]
        e[k] = {"v1": {x: a1[x] for x in ("median", "p95", "max")}, "v2": {x: a2[x] for x in ("median", "p95", "max")},
                "verdict_p95": verdict(a1["p95"], a2["p95"], abs_tol=1.0)}
    res["hand_foot_position_2d_vs_source_cm(rig scale)"] = e
    hy = {}
    for k in ("rig_rest_deg", "rig_hold_median_deg", "image_hold_median_deg", "rig_abs_max_deg", "rig_abs_argmax",
              "abs_error_hold", "abs_error_where_image_yaw_gt20", "image_ear_span_ratio_min_max"):
        hy[k] = {"v1": m1["head_yaw"][k], "v2": m2["head_yaw"][k]}
    e1, e2 = m1["head_yaw"]["abs_error_hold"], m2["head_yaw"]["abs_error_hold"]
    hy["verdict_hold_error_median"] = verdict(e1["median"], e2["median"], abs_tol=2)
    res["head_yaw_vs_2d_estimate_deg"] = hy
    res["static_limb_forward_angle_deg"] = {"v1": m1["static_limb_forward_angle_deg(+ = toward camera)"],
                                            "v2": m2["static_limb_forward_angle_deg(+ = toward camera)"],
                                            "verdict_abs_max": {b: verdict(m1["static_limb_forward_angle_deg(+ = toward camera)"][b]["abs_max"],
                                                                           m2["static_limb_forward_angle_deg(+ = toward camera)"][b]["abs_max"], abs_tol=3)
                                                                for b in m1["static_limb_forward_angle_deg(+ = toward camera)"]}}
    res["rest_baseline_forward_angle_deg"] = {"v1": m1["rest_baseline_forward_angle_deg"], "v2": m2["rest_baseline_forward_angle_deg"]}
    res["chest_yaw_deg"] = {"v1": m1["chest_yaw_deg"], "v2": m2["chest_yaw_deg"],
                            "verdict_abs_max": verdict(max(abs(m1["chest_yaw_deg"]["min"]), abs(m1["chest_yaw_deg"]["max"])),
                                                       max(abs(m2["chest_yaw_deg"]["min"]), abs(m2["chest_yaw_deg"]["max"])), abs_tol=3)}
    res["chest_pitch_deg(+ back)"] = {"v1": m1["chest_pitch_deg(+ back)"], "v2": m2["chest_pitch_deg(+ back)"],
                                      "verdict_max": verdict(m1["chest_pitch_deg(+ back)"]["max"], m2["chest_pitch_deg(+ back)"]["max"], abs_tol=2)}
    s1, s2 = m1["semantic"], m2["semantic"]
    res["semantic"] = {k: {"v1": s1[k], "v2": s2[k]} for k in s1 if not k.startswith("_")}
    v2cal = read_json(run_dir(v2_id) / "calibration" / "calibrated_pose.json")
    v1cal = read_json(run_dir(v1_id) / "calibration" / "calibrated_pose.json")
    res["bone_length"] = {
        "rig_fk": "constant by construction in v1 and v2 (checked in dev_audit.bone_length_drift)",
        "solver_input_segment_length_over_planar_length": {"v1(tracker, aligned)": calibrated_length_stats(v1cal, L),
                                                           "v2(rebuilt+filtered)": calibrated_length_stats(v2cal, L)}}
    res["filter_lag"] = lag_check(r1, r2, cid, scale)

    abl = {}
    if ablate:
        A = ablation(v2_id)
        Qrep = A["v2_reproduced"][0]
        abl["v2_reproduced_max_quat_diff_vs_bundle"] = float(np.abs(np.abs(np.einsum("tbi,tbi->tb", Qrep, r2.Q)) - 1).max())
        for label in ("v2_no_depth_rebuild", "v2_no_filter"):
            Q, prof, cal, rep = A[label]
            ra = Run(label, Q, prof)
            ma = metrics(ra, scale, L, clip)
            abl[label] = {
                "frames_above_gate_total": ma["frames_above_gate_total"],
                "acc_max_per_bone(above gate)": {b: ma["angular_acceleration"][b]["max"] for b in SKEL.names
                                                 if ma["angular_acceleration"][b]["max"] > GATE},
                "head_yaw_hold_median": ma["head_yaw"]["rig_hold_median_deg"],
                "head_yaw_abs_max": ma["head_yaw"]["rig_abs_max_deg"],
                "chest_yaw": ma["chest_yaw_deg"], "chest_pitch": ma["chest_pitch_deg(+ back)"],
                "hinge_bias_deg": prof["hinge_bias_deg"],
                "static_limb_forward_angle_abs_max": {b: v["abs_max"] for b, v in ma["static_limb_forward_angle_deg(+ = toward camera)"].items()},
                "semantic": {k: v for k, v in ma["semantic"].items() if not k.startswith("_")},
            }
            if CLIPS[cid]["moving"] and CLIPS[cid]["moving"][0] != "head":
                limb, sd = CLIPS[cid]["moving"][0].split("_")
                src, rr = chord_angle_series(ra, sd, limb, scale)
                abl[label]["chord_lag_frames"] = lag_frames(src, rr)
                abl[label]["chord_excursion_max_deg"] = _q(np.abs(rr - np.median(rr[ra.base])).max())
    res["ablation(v2 code, one feature off)"] = abl

    plots(v2_id, cid, m1, m2, res, r2)
    for v in res["filter_lag"].values():
        v.pop("_series", None)
    return res


def plots(v2_id, cid, m1, m2, res, r2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pd = REPORTS_ROOT / v2_id / "plots"
    pd.mkdir(parents=True, exist_ok=True)
    fr = np.arange(r2.n)
    # head yaw
    y1, img = m1["_yaw"]
    y2, _ = m2["_yaw"]
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=110)
    ax.plot(fr, img, color="#9a9a9a", lw=1.2, label="2D image estimate (ears+nose)")
    ax.plot(fr, y1, color="#7f8c8d", lw=1.2, ls="--", label="v1 rig")
    ax.plot(fr, y2, color="#c0392b", lw=1.6, label="v2 rig")
    ax.set_title(f"{v2_id}: head world yaw relative to rest (+ = actor left)", fontsize=10)
    ax.set_xlabel("frame"); ax.set_ylabel("deg"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(pd / "cmp_head_yaw.png"); plt.close(fig)
    # trunk
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=110)
    for m, lab, st in ((m1, "v1", "--"), (m2, "v2", "-")):
        cy, cp = m["_trunk"]
        ax.plot(fr, cy, ls=st, color="#2980b9", lw=1.3, label=f"{lab} chest yaw")
        ax.plot(fr, cp, ls=st, color="#d35400", lw=1.3, label=f"{lab} chest pitch (+ back)")
    ax.set_title(f"{v2_id}: chest yaw / pitch", fontsize=10); ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)
    ax.set_xlabel("frame"); ax.set_ylabel("deg")
    fig.tight_layout(); fig.savefig(pd / "cmp_trunk.png"); plt.close(fig)
    # acceleration totals
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=110)
    ax.plot(fr, m1["_acc"].max(axis=1), color="#7f8c8d", lw=1, label="v1 max over bones")
    ax.plot(fr, m2["_acc"].max(axis=1), color="#c0392b", lw=1.4, label="v2 max over bones")
    ax.axhline(GATE, color="#2c3e50", ls="--", lw=1, label="gate")
    ax.set_yscale("log"); ax.set_ylim(10, 2e5)
    ax.set_title(f"{v2_id}: max angular acceleration over all bones, v1 vs v2", fontsize=10)
    ax.set_xlabel("frame"); ax.set_ylabel("deg/s²"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(pd / "cmp_acc_v1_v2.png"); plt.close(fig)
    # lag
    for k, v in res["filter_lag"].items():
        src, a, b = v["_series"]
        fig, ax = plt.subplots(figsize=(9, 3.2), dpi=110)
        ax.plot(fr, src, color="#9a9a9a", lw=2.2, label="source 2D")
        ax.plot(fr, a, color="#7f8c8d", ls="--", lw=1.2, label="v1 rig (image projection)")
        ax.plot(fr, b, color="#c0392b", lw=1.4, label="v2 rig (image projection)")
        ax.set_title(f"{v2_id}: {k} (lag v1 {v['lag_frames_v1(+ = late)']}, v2 {v['lag_frames_v2(+ = late)']} frames)", fontsize=10)
        ax.set_xlabel("frame"); ax.set_ylabel("deg"); ax.grid(alpha=.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(pd / f"cmp_lag_{k}.png"); plt.close(fig)
    if cid == "df1e":
        fig, ax = plt.subplots(figsize=(9, 3.2), dpi=110)
        ax.plot(fr, m1["semantic"]["_sag"], color="#7f8c8d", ls="--", label="v1")
        ax.plot(fr, m2["semantic"]["_sag"], color="#c0392b", label="v2")
        ax.axhline(90, color="#bbb"); ax.axhline(180, color="#bbb")
        ax.set_title("df1e: left thigh sagittal angle (0 down, 90 forward-horizontal, 180 straight up, >180 behind)", fontsize=9)
        ax.set_xlabel("frame"); ax.set_ylabel("deg"); ax.grid(alpha=.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(pd / "cmp_kick_thigh_sagittal.png"); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, action="append", required=True)
    ap.add_argument("--ablate", action="store_true")
    a = ap.parse_args(argv)
    for v1, v2 in a.pair:
        res = compare(v1, v2, a.ablate)
        print(json.dumps(res, indent=1, default=str)[:20000])


if __name__ == "__main__":
    main()


def depth_diagnostics(v2_id: str) -> dict:
    """Re-run the v2 depth rebuild (unfiltered) and report, per rebuilt segment: the rest
    projection ratio d/L (< 0.97 means depth is invented at rest), frames in the
    length-based zone, sign flips of the rebuilt depth between consecutive frames while the
    segment is clearly out of plane, and the largest frame-to-frame depth jump."""
    from altline.calibration import CalibrationConfig, _actor_lengths, find_baseline, load_frames
    from altline.depth import TREE, PLANAR_FULL_LENGTH_RATIO, PLANAR_TRACKER_RATIO, reconstruct
    rd = run_dir(v2_id)
    raw = read_json(rd / "observations" / "raw_pose.json")
    cfg = CalibrationConfig()
    tracker = load_frames(raw)
    base = find_baseline(tracker, cfg)["frames"]
    actor = _actor_lengths(tracker, base, cfg.min_visibility)
    meta = raw["meta"]
    pts = [image_points(f["image_joints"], meta["width"], meta["height"]) if f["detected"] else None for f in raw["frames"]]
    scale = pixel_scale([pts[i] for i in base], actor)
    pl = planar_lengths(pts, base, scale)
    lengths = {**actor, **pl}
    recs = [reconstruct(fr[0], p, scale, lengths)[0] if fr else None for fr, p in zip(tracker, pts)]
    out = {"pixel_scale_m": scale}
    for parent, child, key, from_len in TREE:
        if not from_len:
            continue
        L = lengths[key]
        ratio = np.array([np.linalg.norm(pts[t][child] - pts[t][parent]) * scale / L for t in range(len(pts))])
        dz = np.array([recs[t][child][2] - recs[t][parent][2] for t in range(len(pts))])
        tz = np.array([tracker[t][0][child][2] - tracker[t][0][parent][2] for t in range(len(pts))])
        big = np.abs(dz) > 0.3 * L
        flips = [int(t) for t in range(1, len(dz)) if np.sign(dz[t]) != np.sign(dz[t - 1]) and (big[t] or big[t - 1])]
        jump = np.abs(np.diff(dz))
        rest_ratio = float(np.median(ratio[base]))
        rest_dz = float(np.median(dz[base]))
        out[f"{parent}->{child}"] = {
            "L_m": round(L, 4), "rest_projection_ratio": round(rest_ratio, 3),
            "rest_out_of_plane_deg(rebuilt)": round(float(np.degrees(np.arcsin(np.clip(abs(rest_dz) / L, 0, 1)))), 1),
            "rest_out_of_plane_deg(tracker)": round(float(np.degrees(np.arcsin(np.clip(abs(np.median(tz[base])) / L, 0, 1)))), 1),
            "frames_length_zone(ratio<0.97)": int((ratio < PLANAR_TRACKER_RATIO).sum()),
            "frames_full_length_zone(ratio<0.85)": int((ratio < PLANAR_FULL_LENGTH_RATIO).sum()),
            "sign_flips_while_out_of_plane": flips[:30], "n_sign_flips": len(flips),
            "max_depth_jump_cm_per_frame": round(float(jump.max()) * 100, 1), "argmax_jump_frame": int(jump.argmax()) + 1,
        }
    return out
