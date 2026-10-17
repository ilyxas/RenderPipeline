"""Reviewer audit of the v4 (Phase 2) runs: before (v4raw) vs after (v4), stage attribution.

    python3 -m altline.review.audit_v4 --clip df1e [--clip ...]   # or --all

Read-only on altline/runs; writes altline/reports/v4_<clip>/dev_audit.json and plots/*.png.

Stages compared (local basis rotations, the stability-gate measure of damper.angular_acceleration):
    before      v4raw_<clip> damper/damped_rotations.json (no Butterworth, no flip guard,
                per-frame solve, limits, damper off) = what the BEFORE character does
    before_solve v4raw_<clip> solver/frame_rotations.json (raw tracking solved per frame, no limits)
    frame       v4_<clip> solver/frame_rotations.json   (after Butterworth + depth + flip guard)
    temporal    v4_<clip> solver/raw_rotations.json      (sliding-window solver)
    limits      v4_<clip> limits/clean_rotations.json    (= damper input)
    damped      v4_<clip> damper/damped_rotations.json   (= motion_bundle)
Landmark stages (world bone frames built from landmarks, attribution only):
    raw_pose -> filtered_pose -> calibrated_pose -> guarded_pose (all of v4_<clip>), and the
    2D image points of raw_pose (no depth at all).

A bone frame from landmarks: Y = segment direction, Z = the twist reference of the solver
(index - pinky for forearm and hand, knee - ankle for the foot) made perpendicular to Y.
Its world angular acceleration is invariant to the constant axis change between MediaPipe
and canonical space, so raw and calibrated landmarks are directly comparable.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from altline.config import Config
from altline.damper import angular_acceleration
from altline.io_utils import REPORTS_ROOT, read_json, run_dir, write_json
from altline.quat import quat_angle_between, quat_conj, quat_log, quat_mul, quat_to_mat, mat_to_quat
from altline.skeleton import reference_skeleton

GATE = Config().stability_gate_deg_s2
SKEL = reference_skeleton()
IDX = SKEL.index
CLIPS = ["061c", "0663", "5313", "619b", "690b", "85a9", "9645", "df1e"]
MOTION = {"061c": "правая рука вверх", "0663": "скрестить руки", "5313": "голова вправо", "619b": "колено вверх",
          "690b": "левая рука вверх", "85a9": "закрыть лицо", "9645": "голова влево", "df1e": "мах ногой"}
JOINTS = {  # reviewer joint -> bone whose local rotation is that joint
    "elbow_l": "forearm_l", "elbow_r": "forearm_r", "wrist_l": "hand_l", "wrist_r": "hand_r",
    "ankle_l": "foot_l", "ankle_r": "foot_r"}
SIDE = {"l": "left", "r": "right"}
STAGES = ["before_solve", "before", "frame", "temporal", "limits", "damped"]
STAGE_FILES = {"before_solve": ("v4raw", "solver/frame_rotations.json"),
               "before": ("v4raw", "damper/damped_rotations.json"),
               "frame": ("v4", "solver/frame_rotations.json"),
               "temporal": ("v4", "solver/raw_rotations.json"),
               "limits": ("v4", "limits/clean_rotations.json"),
               "damped": ("v4", "damper/damped_rotations.json")}
LM_STAGES = ["raw", "filtered", "calibrated", "guarded"]
LM_FILES = {"raw": "observations/raw_pose.json", "filtered": "observations/filtered_pose.json",
            "calibrated": "calibration/calibrated_pose.json", "guarded": "calibration/guarded_pose.json"}
ROTATION_LANDMARKS = {  # bone -> (from, to, twist ref a, twist ref b): ref = b - a
    "forearm": ("elbow", "wrist", "pinky", "index"),
    "hand": ("wrist", "hand_mid", "pinky", "index"),
    "foot": ("ankle", "foot_index", "ankle", "knee"),
}


def q(x, n=1):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


# ------------------------------------------------------------------ loading
def load_rot(run_id: str, rel: str) -> tuple[np.ndarray, float]:
    d = read_json(run_dir(run_id) / rel)
    return np.array([f["rotations"] for f in d["frames"]], dtype=float), float(d["meta"]["fps"])


def load_stages(clip: str) -> tuple[dict, float]:
    out, fps = {}, 24.0
    for s, (pre, rel) in STAGE_FILES.items():
        out[s], fps = load_rot(f"{pre}_{clip}", rel)
    return out, fps


def local_acc(Q: np.ndarray, fps: float) -> np.ndarray:
    """(T, N) deg/s^2, exactly damper.angular_acceleration per bone."""
    return np.stack([angular_acceleration(Q[:, b], fps) for b in range(Q.shape[1])], axis=1)


def world_rots(Q: np.ndarray) -> np.ndarray:
    W, H, T = [], [], []
    for qs in Q:
        w, h, t = SKEL.fk(np.array([quat_to_mat(x) for x in qs]))
        W.append(w), H.append(h), T.append(t)
    return np.array(W), np.array(H), np.array(T)


def mat_seq_acc(R: np.ndarray, fps: float) -> np.ndarray:
    """Angular acceleration (deg/s^2) of a sequence of world frames (T,3,3), body-frame velocity."""
    qs = np.array([mat_to_quat(r) for r in R])
    for t in range(1, len(qs)):
        if np.dot(qs[t], qs[t - 1]) < 0:
            qs[t] = -qs[t]
    return angular_acceleration(qs, fps)


def dir_acc(u: np.ndarray, fps: float) -> np.ndarray:
    """Angular acceleration of a unit-direction sequence (T,3): |Δω|·fps, ω = axis·angle·fps."""
    w = np.zeros_like(u)
    for t in range(1, len(u)):
        c = np.cross(u[t - 1], u[t])
        s, co = np.linalg.norm(c), float(np.dot(u[t - 1], u[t]))
        w[t] = (c / s if s > 1e-12 else 0 * c) * np.arctan2(s, co) * fps
    a = np.zeros(len(u))
    a[1:-1] = np.degrees(np.linalg.norm(np.diff(w[1:], axis=0), axis=1)) * fps
    return a


def step_deg(u: np.ndarray) -> np.ndarray:
    s = np.zeros(len(u))
    for t in range(1, len(u)):
        s[t] = np.degrees(np.arctan2(np.linalg.norm(np.cross(u[t - 1], u[t])), np.dot(u[t - 1], u[t])))
    return s


def rot_deg(A, B):
    return float(np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1))))


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def landmark_series(d: dict) -> dict:
    """name -> (T,3) world positions (+ hand_mid if missing), and 'vis' -> name -> (T,)."""
    frames = d["frames"]
    names = list(frames[0]["joints"].keys())
    P = {n: np.array([f["joints"][n][:3] if f.get("joints") else [np.nan] * 3 for f in frames]) for n in names}
    V = {n: np.array([f["joints"][n][3] if f.get("joints") else 0 for f in frames]) for n in names}
    for s in ("left", "right"):
        if f"{s}_hand_mid" not in P:
            P[f"{s}_hand_mid"] = 0.5 * (P[f"{s}_index"] + P[f"{s}_pinky"])
    return {"P": P, "V": V}


def image_series(d: dict) -> dict:
    w, h = d["meta"]["width"], d["meta"]["height"]
    frames = d["frames"]
    names = list(frames[0]["image_joints"].keys())
    P = {n: np.array([[f["image_joints"][n][0] * w, f["image_joints"][n][1] * h] for f in frames]) for n in names}
    for s in ("left", "right"):
        P[f"{s}_hand_mid"] = 0.5 * (P[f"{s}_index"] + P[f"{s}_pinky"])
    return P


def landmark_frames(P: dict, bone: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World bone frames (T,3,3) from landmarks, Y direction (T,3), perp fraction of the ref."""
    kind, sd = bone.rsplit("_", 1)
    a, b, ra, rb = ROTATION_LANDMARKS[kind]
    s = SIDE[sd]
    y = unit(P[f"{s}_{b}"] - P[f"{s}_{a}"])
    ref = P[f"{s}_{rb}"] - P[f"{s}_{ra}"]
    rp = ref - np.sum(ref * y, axis=1, keepdims=True) * y
    frac = np.linalg.norm(rp, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1e-12)
    z = unit(rp)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=2), y, frac


def angle2d(v):
    return np.unwrap(np.arctan2(v[:, 1], v[:, 0]))


def acc2d(theta, fps):
    a = np.zeros(len(theta))
    a[1:-1] = np.degrees(np.abs(np.diff(theta, 2))) * fps * fps
    return a


def pos_acc(p, fps):
    a = np.zeros(len(p))
    a[1:-1] = np.linalg.norm(np.diff(p, 2, axis=0), axis=1) * fps * fps
    return a


def prof(a: np.ndarray) -> dict:
    return {"max": q(a.max(), 0), "argmax": int(a.argmax()), "p95": q(np.percentile(a, 95), 0),
            "frames_over_gate": int((a > GATE).sum())}


def local_maxima(x, thr):
    return [t for t in range(1, len(x) - 1) if x[t] > thr and x[t] >= x[t - 1] and x[t] >= x[t + 1]]


# ------------------------------------------------------------------ analysis
def analyse(clip: str) -> dict:
    st, fps = load_stages(clip)
    n = len(st["damped"])
    acc = {s: local_acc(Q, fps) for s, Q in st.items()}
    wr = {s: world_rots(st[s]) for s in ("before", "frame", "temporal", "limits", "damped")}
    wacc = {s: {b: mat_seq_acc(wr[s][0][:, IDX[b]], fps) for b in JOINTS.values()} for s in wr}
    v4 = run_dir(f"v4_{clip}")
    raw = read_json(v4 / LM_FILES["raw"])
    lm = {s: landmark_series(read_json(v4 / LM_FILES[s])) for s in LM_STAGES}
    lm_v4raw_cal = landmark_series(read_json(run_dir(f"v4raw_{clip}") / LM_FILES["calibrated"]))
    img = image_series(raw)
    dmp = read_json(v4 / "damper" / "damper_report.json")["bones"]
    flip = read_json(v4 / "calibration" / "flip_report.json")["limbs"]
    temporal = read_json(v4 / "solver" / "temporal_report.json")["bones"]
    lim = read_json(v4 / "limits" / "limits_report.json")

    out = {"clip": clip, "motion": MOTION[clip], "fps": fps, "n_frames": n, "gate_deg_s2": GATE,
           "honesty_note": "damped (= motion_bundle) is under the gate BY CONSTRUCTION of damper.py; "
                           "the evidence is the damper input ('limits'), frames damped and the deviation."}
    # totals over all 20 bones per stage
    out["all_bones_frames_over_gate"] = {s: int((a > GATE).sum()) for s, a in acc.items()}
    out["all_bones_max"] = {s: {"max": q(a.max(), 0), "bone": SKEL.names[int(np.unravel_index(a.argmax(), a.shape)[1])],
                                "frame": int(np.unravel_index(a.argmax(), a.shape)[0])} for s, a in acc.items()}
    out["all_bones_over_gate_per_bone"] = {s: {b: int((acc[s][:, i] > GATE).sum()) for i, b in enumerate(SKEL.names)
                                                if (acc[s][:, i] > GATE).any()} for s in STAGES}
    # damper: deviation series per bone, hand/foot tip displacement in cm
    _, Hc, Tc = wr["limits"]
    _, Hd, Td = wr["damped"]
    dev = np.array([[np.degrees(quat_angle_between(a, b)) for a, b in zip(st["limits"][:, i], st["damped"][:, i])]
                    for i in range(len(SKEL.names))]).T
    out["damper"] = {
        "bones_damped": {b: {"frames_damped": r["frames_damped"], "max_deviation_deg": q(r["max_deviation_deg"]),
                             "max_deviation_frame": r["max_deviation_frame"],
                             "frames_deviation_over_10deg": int((dev[:, IDX[b]] > 10).sum()),
                             "input_max_acc": q(r["input_max_acc_deg_s2"], 0), "input_frames_over_gate": r["input_frames_over_gate"],
                             "output_max_acc": q(r["output_max_acc_deg_s2"], 0)}
                         for b, r in dmp.items() if r["frames_damped"] > 0},
        "total_frames_damped(bone-frames)": int(sum(r["frames_damped"] for r in dmp.values())),
        "max_deviation_deg": q(dev.max()), "max_deviation_bone": SKEL.names[int(np.unravel_index(dev.argmax(), dev.shape)[1])],
        "max_deviation_frame": int(np.unravel_index(dev.argmax(), dev.shape)[0]),
        "end_effector_displacement_cm(damped vs damper input, world FK)": {
            nm: {"max": q(100 * np.linalg.norm(Td[:, IDX[nm]] - Tc[:, IDX[nm]], axis=1).max()),
                 "argmax": int(np.linalg.norm(Td[:, IDX[nm]] - Tc[:, IDX[nm]], axis=1).argmax())}
            for nm in ("hand_l", "hand_r", "foot_l", "foot_r")},
        "bundle_check_output_frames_over_gate": int((acc["damped"] > GATE).sum()),
    }
    out["_dev"] = dev
    out["flip_guard"] = {k: {"bridged_runs": v["bridged_runs"], "kept_runs_over_threshold": v["kept_runs_over_threshold"],
                             "frames_bridged": v["frames_bridged"], "max_step_deg": q(v["max_step_deg"])} for k, v in flip.items()}
    out["temporal_solver_max_change_deg"] = {b: {"max": q(temporal[b]["max_change_deg"]), "frame": temporal[b]["max_change_frame"]}
                                             for b in JOINTS.values()}

    # six joints
    joints = {}
    series = {}
    for jn, bone in JOINTS.items():
        i = IDX[bone]
        kind, sd = bone.rsplit("_", 1)
        s = SIDE[sd]
        J = {"bone": bone, "local_acc_by_stage": {st_: prof(acc[st_][:, i]) for st_ in STAGES},
             "world_acc_by_stage": {st_: prof(wacc[st_][bone]) for st_ in wacc}}
        lmacc, lmdir, lmfrac, lmstep = {}, {}, {}, {}
        for ls in LM_STAGES:
            R, y, frac = landmark_frames(lm[ls]["P"], bone)
            lmacc[ls] = mat_seq_acc(R, fps)
            lmdir[ls] = dir_acc(y, fps)
            lmfrac[ls] = frac
            lmstep[ls] = step_deg(R[:, :, 2])
        Rv, yv, _ = landmark_frames(lm_v4raw_cal["P"], bone)
        lmacc["v4raw_calibrated"] = mat_seq_acc(Rv, fps)
        lmdir["v4raw_calibrated"] = dir_acc(yv, fps)
        J["landmark_frame_world_acc"] = {k: prof(v) for k, v in lmacc.items()}
        J["landmark_direction_only_acc"] = {k: prof(v) for k, v in lmdir.items()}
        a, b_, ra, rb = ROTATION_LANDMARKS[kind]
        th_dir = angle2d(img[f"{s}_{b_}"] - img[f"{s}_{a}"])
        th_ref = angle2d(img[f"{s}_{rb}"] - img[f"{s}_{ra}"])
        a2d_dir, a2d_ref = acc2d(th_dir, fps), acc2d(th_ref, fps)
        J["image_2d_angle_acc"] = {"direction": prof(a2d_dir), "twist_ref_vector": prof(a2d_ref)}
        # landmark position acceleration (m/s^2) of the joint's own landmark, raw vs filtered
        lmk = {"elbow": "elbow", "wrist": "wrist", "ankle": "ankle"}[jn.split("_")[0]]
        pr, pf = pos_acc(lm["raw"]["P"][f"{s}_{lmk}"], fps), pos_acc(lm["filtered"]["P"][f"{s}_{lmk}"], fps)
        J[f"landmark_{s}_{lmk}_position_acc_m_s2"] = {
            "raw": {"max": q(pr.max(), 1), "p95": q(np.percentile(pr, 95), 1), "median": q(np.median(pr), 2)},
            "filtered": {"max": q(pf.max(), 1), "p95": q(np.percentile(pf, 95), 1), "median": q(np.median(pf), 2)}}
        Wf = wr["frame"][0][:, i]
        wdep = {x: np.array([rot_deg(Wf[t], wr[x][0][t, i]) for t in range(n)]) for x in ("temporal", "limits", "damped")}
        J["world_orientation_departure_from_per_frame_solve_deg"] = {
            x: {"max": q(v.max()), "argmax": int(v.argmax()), "frames_over_30deg": int((v > 30).sum())} for x, v in wdep.items()}
        J["damper"] = {k: dmp[bone][k] for k in ("frames_damped", "max_deviation_deg", "max_deviation_frame", "mean_deviation_deg")}
        J["damper"]["damped_frames"] = dmp[bone]["damped_frames"]
        # events = local maxima above gate in any pre-damper stage (before, frame, temporal, limits)
        pre = np.max(np.stack([acc[x][:, i] for x in ("before", "frame", "temporal", "limits")]), axis=0)
        cand = sorted(local_maxima(pre, GATE), key=lambda t: -pre[t])
        events = []
        for t in cand:
            if any(abs(t - e) <= 3 for e in events):
                continue
            events.append(t)
        ev_out = []
        for t in sorted(events):
            lo, hi = max(0, t - 3), min(n, t + 4)
            w = slice(lo, hi)
            e = {"frame": t, "window": [lo, hi - 1]}
            e["local_acc_max_in_window"] = {x: q(acc[x][w, i].max(), 0) for x in STAGES}
            e["world_acc_max_in_window"] = {x: q(wacc[x][bone][w].max(), 0) for x in wacc}
            e["landmark_frame_acc_max_in_window"] = {x: q(lmacc[x][w].max(), 0) for x in lmacc}
            e["landmark_direction_acc_max_in_window"] = {x: q(lmdir[x][w].max(), 0) for x in lmdir}
            e["landmark_twist_ref_step_max_deg"] = {x: q(lmstep[x][w].max()) for x in LM_STAGES}
            e["twist_ref_perp_fraction_min(raw/guarded)"] = [q(lmfrac["raw"][w].min(), 2), q(lmfrac["guarded"][w].min(), 2)]
            e["image_2d_acc_max_in_window"] = {"direction": q(a2d_dir[w].max(), 0), "twist_ref_vector": q(a2d_ref[w].max(), 0)}
            e["raw_visibility_min"] = q(min(lm["raw"]["V"][f"{s}_{x}"][w].min() for x in (a, b_) if f"{s}_{x}" in lm["raw"]["V"]), 2)
            e["damper_deviation_max_deg"] = q(dev[w, i].max())
            e["world_departure_from_per_frame_solve_max_deg"] = {x: q(v[w].max()) for x, v in wdep.items()}
            e["damper_frames_in_window"] = [int(x) for x in dmp[bone]["damped_frames"] if lo <= x < hi]
            e["attribution"] = attribute(e)
            ev_out.append(e)
        J["events_over_gate(any pre-damper stage, ±3 frames)"] = ev_out
        joints[jn] = J
        series[jn] = {"acc": {x: acc[x][:, i] for x in STAGES}, "lm": lmacc, "dev": dev[:, i], "events": ev_out, "wdep": wdep}
    out["joints"] = joints
    out["_series"] = series
    return out


CHAIN = [("raw", "landmark_frame_acc_max_in_window"), ("filtered", "landmark_frame_acc_max_in_window"),
         ("calibrated", "landmark_frame_acc_max_in_window"), ("guarded", "landmark_frame_acc_max_in_window"),
         ("frame", "local_acc_max_in_window"), ("temporal", "local_acc_max_in_window"),
         ("limits", "local_acc_max_in_window")]
STAGE_LABEL = {"raw": "tracking data (raw tracker landmarks)", "filtered": "Butterworth filter",
               "calibrated": "internal: calibration / depth rebuild", "guarded": "internal: flip guard",
               "frame": "internal: per-frame solver (twist split between parent and child / aim)",
               "temporal": "internal: temporal solver", "limits": "internal: limits (clamp / twist transfer / bridge)"}
AMP = 2.0  # a stage that multiplies the acceleration by >= 2x is where the peak is made


def attribute(e: dict) -> dict:
    """Stated rule (fixed before looking at results, not tuned per clip).

    Chain of maxima in the event window: landmark world bone frame raw -> filtered ->
    calibrated -> guarded (solver input) -> solved local rotation frame -> temporal -> limits
    (damper input). The peak is MADE by the stage that multiplies the previous stage's value
    by >= 2x and lifts it over the gate (largest such factor). If no stage does, and the raw
    tracker is already over the gate, the peak is tracking data passed through. Landmark
    world frames vs local rotations are not the same quantity: a local value far above the
    solved bone's own WORLD acceleration means the parent/child twist split made it, so that
    comparison is reported too ("local_over_world_frame_stage").
    """
    vals = [(k, float(e[src][k] or 0.0)) for k, src in CHAIN]
    final = vals[-1][1]
    res = {"chain_max_deg_s2": {k: round(v) for k, v in vals}}
    w = e["world_acc_max_in_window"].get("frame") or 1.0
    res["local_over_world_frame_stage"] = round(vals[4][1] / max(w, 1.0), 2)
    if final <= GATE:
        res["status"] = "under the gate before the damper"
        if e["local_acc_max_in_window"]["before"] and e["local_acc_max_in_window"]["before"] > GATE:
            fixed = [k for (k, v), (_, v0) in zip(vals[1:], vals[:-1]) if v0 > GATE >= v]
            res["origin"] = "only in BEFORE (v4raw); Phase 2 removed it" + (f" (dropped under the gate at: {', '.join(fixed)})" if fixed else "")
        else:
            res["origin"] = "none (pre-damper stages over the gate only transiently)"
        return res
    res["status"] = "OVER the gate at the damper input (damper hides it)"
    best, bestf = None, AMP
    for (k0, v0), (k, v) in zip(vals[:-1], vals[1:]):
        f = v / max(v0, 1.0)
        if v > GATE and f >= bestf:
            best, bestf = k, f
    if best is None:
        res["origin"] = STAGE_LABEL["raw"] if vals[0][1] > GATE else "accumulated (no single stage >= 2x)"
    else:
        res["origin"] = STAGE_LABEL[best]
        res["amplification"] = round(bestf, 1)
    res["raw_tracker_over_gate_too"] = vals[0][1] > GATE
    return res


# ------------------------------------------------------------------ v3 -> v4 (and v4raw) comparison
SEGS = {f"{k}_{sd}": (f"{SIDE[sd]}_{a}", f"{SIDE[sd]}_{b}") for sd in "lr" for k, a, b in
        (("upper_arm", "shoulder", "elbow"), ("forearm", "elbow", "wrist"), ("thigh", "hip", "knee"), ("shin", "knee", "ankle"))}


def depth_jumps(run_id: str) -> dict:
    """Solver input (guarded, else calibrated) per limb segment: largest frame-to-frame change of
    the segment's depth offset (cm), depth sign flips while |dz| > 30 % of the length, and frames
    where the depth was flattened: |dz| < 0.5 cm while the segment is stretched past its rest
    projection length L (projection >= L, so the rebuild clips depth to 0: C4)."""
    rd = run_dir(run_id)
    path = rd / "calibration" / "guarded_pose.json"
    P = landmark_series(read_json(path if path.exists() else rd / "calibration" / "calibrated_pose.json"))["P"]
    Ls = read_json(rd / "calibration" / "actor_profile.json").get("depth_from_2d", {}).get("lengths_m", {})
    out = {}
    for seg, (a, b) in SEGS.items():
        v = P[b] - P[a]
        n = np.linalg.norm(v, axis=1)
        L = Ls.get(seg, float(np.nanmedian(n)))
        dz = v[:, 2]
        jump = np.abs(np.diff(dz))
        big = np.abs(dz) > 0.3 * L
        flips = [int(t) for t in range(1, len(dz)) if np.sign(dz[t]) != np.sign(dz[t - 1]) and (big[t] or big[t - 1])]
        flat = (np.abs(dz) < 0.005) & (n > 1.02 * L)
        out[seg] = {"max_depth_jump_cm": q(100 * jump.max()), "argmax_frame": int(jump.argmax()) + 1,
                    "sign_flips_out_of_plane": flips[:20], "frames_flattened_and_stretched": int(flat.sum()),
                    "first_flattened_frame": int(np.flatnonzero(flat)[0]) if flat.any() else None,
                    "max_length_over_L": q(n.max() / L, 3)}
    return out


def versions(clip: str) -> dict:
    from altline.review import compare3
    res, ms = compare3.run(clip, ["v3", "v4raw", "v4"], ablate=False)
    keep = {}
    for pair in ("v3_vs_v4", "v4raw_vs_v4"):
        s = res[pair]
        keep[pair] = {k: s[k] for k in ("frames_above_gate_total", "chest_yaw_deg", "chest_pitch_deg(+ back)", "semantic",
                                        "static_limb_forward_angle_deg", "direction_error_2d_vs_image_landmarks_deg",
                                        "hand_foot_position_2d_vs_source_cm(rig scale)")}
        keep[pair]["head_yaw"] = {k: s["head_yaw_vs_2d_estimate_deg"][k] for k in
                                  ("rig_hold_median_deg", "image_hold_median_deg", "abs_error_hold", "verdict_hold_error_median")}
        keep[pair]["angular_acceleration_per_bone"] = s["angular_acceleration_per_bone(bones above gate in either)"]
    keep["filter_lag_and_peak_speed"] = res["filter_lag"]
    keep["depth_jumps(solver input)"] = {v: depth_jumps(f"{v}_{clip}") for v in ("v3", "v4raw", "v4")}
    return keep


# ------------------------------------------------------------------ plots
def plot_joints(clip: str, A: dict, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 2, figsize=(13, 9.5), dpi=100, sharex=True)
    for ax, jn in zip(axs.flat, JOINTS):
        s = A["_series"][jn]
        fr = np.arange(len(s["acc"]["damped"]))
        ax.plot(fr, np.maximum(s["acc"]["before"], 10), color="#9a9a9a", lw=1.1, label="BEFORE: raw tracking, per-frame + limits")
        ax.plot(fr, np.maximum(s["acc"]["limits"], 10), color="#e08e0b", lw=1.1, ls="--", label="v4 before damper (limits)")
        ax.plot(fr, np.maximum(s["acc"]["damped"], 10), color="#c0392b", lw=1.8, label="AFTER: v4 final (damped)")
        ax.axhline(GATE, color="#2c3e50", ls=":", lw=1.2)
        ax.text(1, GATE * 1.15, "gate 3000", fontsize=7, color="#2c3e50")
        for ev in s["events"]:
            t = ev["frame"]
            y = max(ev["local_acc_max_in_window"]["before"] or 10, ev["local_acc_max_in_window"]["limits"] or 10)
            ax.plot([t], [y], "v", color="#333", ms=5)
            ax.annotate(f"f{t}", (t, y), textcoords="offset points", xytext=(2, 3), fontsize=7)
        ax.set_yscale("log"); ax.set_ylim(10, 3e5)
        dev = s["dev"].max()
        ax.set_title(f"{jn} ({JOINTS[jn]}) — damper deviation max {dev:.1f}°", fontsize=9)
        ax.grid(alpha=.3)
    for ax in axs[-1]:
        ax.set_xlabel("frame (24 fps)")
    for ax in axs[:, 0]:
        ax.set_ylabel("deg/s² (log)")
    axs[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{clip}: local angular acceleration, before vs after (▼ = event over the gate in any pre-damper stage)", fontsize=11)
    fig.tight_layout()
    p = out_dir / "acc_before_after_joints.png"
    fig.savefig(p); plt.close(fig)
    return p.name


def plot_stages(clip: str, A: dict, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), dpi=100)
    labels = STAGES
    cols = ["#bdbdbd", "#7f7f7f", "#5b9bd5", "#2e75b6", "#e08e0b", "#c0392b"]
    x = np.arange(len(JOINTS))
    wbar = 0.13
    for k, s in enumerate(labels):
        v = [A["joints"][j]["local_acc_by_stage"][s]["frames_over_gate"] for j in JOINTS]
        axs[0].bar(x + (k - 2.5) * wbar, v, wbar, color=cols[k], label=s)
    axs[0].set_xticks(x); axs[0].set_xticklabels(list(JOINTS), fontsize=8)
    axs[0].set_ylabel("frames over 3000 deg/s²"); axs[0].legend(fontsize=7)
    axs[0].set_title("frames over the gate per stage (6 joints)", fontsize=10)
    lmv = ["raw", "filtered", "calibrated", "guarded"]
    lcols = ["#bdbdbd", "#8fbc8f", "#3c8d3c", "#1e5631"]
    for k, s in enumerate(lmv):
        v = [A["joints"][j]["landmark_frame_world_acc"][s]["frames_over_gate"] for j in JOINTS]
        axs[1].bar(x + (k - 1.5) * 0.2, v, 0.2, color=lcols[k], label=f"landmarks: {s}")
    v = [A["joints"][j]["world_acc_by_stage"]["frame"]["frames_over_gate"] for j in JOINTS]
    axs[1].plot(x, v, "o", color="#2e75b6", label="solved bone, world (frame stage)")
    axs[1].set_xticks(x); axs[1].set_xticklabels(list(JOINTS), fontsize=8)
    axs[1].set_ylabel("frames over 3000 deg/s²"); axs[1].legend(fontsize=7)
    axs[1].set_title("solver input: world bone frame built from landmarks", fontsize=10)
    fig.suptitle(f"{clip}: where the acceleration comes from (stage attribution)", fontsize=11)
    fig.tight_layout()
    p = out_dir / "stage_attribution.png"
    fig.savefig(p); plt.close(fig)
    return p.name


def plot_event_trace(clip: str, A: dict, jn: str, ev: dict, out_dir):
    """One event: every stage's acceleration around it, landmarks and rotations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = A["_series"][jn]
    t = ev["frame"]
    lo, hi = max(0, t - 15), min(len(s["dev"]), t + 16)
    fr = np.arange(lo, hi)
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), dpi=100, sharex=True)
    for nm, c in zip(("raw", "filtered", "calibrated", "guarded"), ("#d4b106", "#8fbc8f", "#3c8d3c", "#1e5631")):
        axs[0].plot(fr, np.maximum(s["lm"][nm][lo:hi], 10), color=c, lw=1.4, label=f"landmark frame: {nm}")
    for nm, c, ls in (("before", "#7f7f7f", "-"), ("frame", "#5b9bd5", "-"), ("temporal", "#2e75b6", "--"),
                      ("limits", "#e08e0b", "-"), ("damped", "#c0392b", "-")):
        axs[0].plot(fr, np.maximum(s["acc"][nm][lo:hi], 10), color=c, ls=ls, lw=1.6 if nm == "damped" else 1.1,
                    label=f"rotation (local): {nm}")
    axs[0].axhline(GATE, color="#2c3e50", ls=":", lw=1.2)
    axs[0].set_yscale("log"); axs[0].set_ylim(10, 5e5); axs[0].set_ylabel("deg/s²")
    axs[0].legend(fontsize=7, ncol=2); axs[0].grid(alpha=.3)
    axs[0].set_title(f"{clip} {jn}: event at frame {t} — {ev['attribution']['origin']}", fontsize=9)
    axs[1].plot(fr, s["dev"][lo:hi], color="#c0392b")
    axs[1].set_ylabel("damper deviation, deg"); axs[1].set_xlabel("frame"); axs[1].grid(alpha=.3)
    fig.tight_layout()
    p = out_dir / f"event_{jn}_f{t:03d}.png"
    fig.savefig(p); plt.close(fig)
    return p.name


def preview_frames(clip: str, frames: list[int], out_dir) -> list[str]:
    """v4 preview (top) and v3 preview (bottom) at the given frames, one PNG per frame."""
    import cv2
    want = set(frames)

    def grab(run):
        cap = cv2.VideoCapture(str(run_dir(run) / "preview" / "preview.mp4"))
        got, t = {}, 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if t in want:
                got[t] = img
            t += 1
        return got
    a, b = grab(f"v4_{clip}"), grab(f"v3_{clip}")
    names = []
    for f in frames:
        if f not in a:
            continue
        top = a[f]
        w = top.shape[1]
        rows = []
        lab = np.full((40, w, 3), 255, np.uint8)
        cv2.putText(lab, f"{clip} frame {f}: top v4 final, bottom v3 (source|front|side|top)", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        rows += [lab, top]
        if f in b:
            rows.append(cv2.resize(b[f], (w, int(b[f].shape[0] * w / b[f].shape[1]))))
        img = np.vstack(rows)
        img = cv2.resize(img, (int(img.shape[1] * 0.6), int(img.shape[0] * 0.6)), interpolation=cv2.INTER_AREA)
        p = out_dir / f"preview_f{f:03d}.png"
        cv2.imwrite(str(p), img)
        names.append(p.name)
    return names


def zoom_frames(clip: str, frames: list[int], out_dir, runs=("v4", "v3")) -> list[str]:
    """Full-resolution crop of source + front + side panels, one row per run, one PNG per frame."""
    import cv2
    want = set(frames)
    got = {}
    for r in runs:
        cap = cv2.VideoCapture(str(run_dir(f"{r}_{clip}") / "preview" / "preview.mp4"))
        t, g = 0, {}
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if t in want:
                g[t] = img[:, :1050]
            t += 1
        got[r] = g
    names = []
    for f in frames:
        rows = []
        for r in runs:
            if f not in got[r]:
                continue
            lab = np.full((36, 1050, 3), 255, np.uint8)
            cv2.putText(lab, f"{clip} frame {f}: {r}" + (" (final bundle)" if r == "v4" else ""), (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            rows += [lab, got[r][f]]
        if rows:
            p = out_dir / f"zoom_f{f:03d}.png"
            cv2.imwrite(str(p), np.vstack(rows))
            names.append(p.name)
    return names


def run(clip: str, extra_frames: list[int] | None = None) -> dict:
    out_dir = REPORTS_ROOT / f"v4_{clip}"
    out_dir.mkdir(parents=True, exist_ok=True)
    A = analyse(clip)
    plots = [plot_joints(clip, A, out_dir), plot_stages(clip, A, out_dir)]
    # event traces: worst 3 events by pre-damper acceleration or damper deviation
    evs = [(jn, e) for jn in JOINTS for e in A["_series"][jn]["events"]]
    evs.sort(key=lambda je: -max(je[1]["local_acc_max_in_window"]["limits"] or 0, 50 * (je[1]["damper_deviation_max_deg"] or 0)))
    for jn, e in evs[:3]:
        plots.append(plot_event_trace(clip, A, jn, e, out_dir))
    pf = sorted(set([e["frame"] for _, e in evs[:3]] + (extra_frames or [])))
    # plus the frame of the largest damper deviation and of the largest limits world departure (> 20 deg)
    i_dev = int(np.unravel_index(A["_dev"].argmax(), A["_dev"].shape)[0])
    if A["_dev"].max() > 5:
        pf.append(i_dev)
    for jn in JOINTS:
        w = A["_series"][jn]["wdep"]["damped"]
        if w.max() > 20:
            pf.append(int(w.argmax()))
    pf = sorted(set(pf))
    A["preview_frames_checked"] = preview_frames(clip, pf, out_dir) + zoom_frames(clip, pf, out_dir)
    A["plots"] = plots
    res = {k: v for k, v in A.items() if not k.startswith("_")}
    res["versions_v3_v4raw_v4"] = versions(clip)
    manual = out_dir / "manual.json"
    if manual.exists():
        res.update(read_json(manual))
    write_json(out_dir / "dev_audit.json", json.loads(json.dumps(res, default=float)))
    return A


# hand-picked review frames (hold poses for C4/C6 and the df1e limits departure), review-side only
EXTRA_FRAMES = {"df1e": [104, 110], "85a9": [17, 100], "0663": [24, 100], "061c": [89, 120], "690b": [18, 78]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)
    for c in (CLIPS if a.all else a.clip):
        run(c, EXTRA_FRAMES.get(c))
        print("ok", c)


if __name__ == "__main__":
    main()
