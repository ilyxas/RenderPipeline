"""Reviewer audit of one altline run (read-only on the run; writes altline/reports/<run_id>/).

    python3 -m altline.review.audit --run-id v1_061c [--run-id ...]

Produces reports/<run_id>/dev_audit.json (machine sections; visual_findings and
action_items are merged in from reports/<run_id>/manual.json when that file exists) and
reports/<run_id>/plots/*.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from altline import solver as S
from altline.animation import load_bundle
from altline.config import Config
from altline.io_utils import REPORTS_ROOT, read_json, run_dir, write_json
from altline.quat import (mat_to_quat, normalize, quat_angle_between, quat_conj, quat_log, quat_mul,
                          quat_to_mat, signed_angle_about, swing_twist, twist_angle)
from altline.skeleton import reference_skeleton

CFG = Config()
GATE = CFG.stability_gate_deg_s2
Yax = np.array([0.0, 1.0, 0.0])
LIMB = {  # bone -> (from landmark, to landmark) for raw (pre-debias) direction checks
    **{f"upper_arm_{s}": (f"{d}_shoulder", f"{d}_elbow") for s, d in S.SIDES.items()},
    **{f"forearm_{s}": (f"{d}_elbow", f"{d}_wrist") for s, d in S.SIDES.items()},
    **{f"hand_{s}": (f"{d}_wrist", f"{d}_hand_mid") for s, d in S.SIDES.items()},
    **{f"thigh_{s}": (f"{d}_hip", f"{d}_knee") for s, d in S.SIDES.items()},
    **{f"shin_{s}": (f"{d}_knee", f"{d}_ankle") for s, d in S.SIDES.items()},
    **{f"foot_{s}": (f"{d}_ankle", f"{d}_foot_index") for s, d in S.SIDES.items()},
}
# rough segment mass fractions (Dempster/Winter-like) for a CoM estimate
MASS = {"pelvis": .142, "spine": .10, "chest": .20, "neck": .02, "head": .07, "clavicle": .0,
        "upper_arm": .028, "forearm": .016, "hand": .006, "thigh": .10, "shin": .0465, "foot": .0145}


def r(x, n=2):
    return [round(float(v), n) for v in x]


def ang(a, b):
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))


def seg_dist(p1, q1, p2, q2):
    """Minimum distance between segments p1-q1 and p2-q2 (sampled, robust enough here)."""
    t = np.linspace(0, 1, 11)
    a = p1[None] + t[:, None] * (q1 - p1)[None]
    b = p2[None] + t[:, None] * (q2 - p2)[None]
    return float(np.min(np.linalg.norm(a[:, None] - b[None], axis=2)))


def _dist_to_hull(p, pts):
    """Distance from 2D point p to the convex hull of pts (0 inside)."""
    import matplotlib.path as mpath
    P = sorted(map(tuple, pts))
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for q in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    for q in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], q) <= 0:
            up.pop()
        up.append(q)
    hull = np.array(lo[:-1] + up[:-1])
    if mpath.Path(hull).contains_point(p):
        return 0.0
    best = np.inf
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        ab = b - a
        u = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
        best = min(best, float(np.linalg.norm(p - (a + u * ab))))
    return best


def fk_all(skel, quats):
    W, H, T = [], [], []
    for qs in quats:
        w, h, t = skel.fk(np.array([quat_to_mat(q) for q in qs]))
        W.append(w), H.append(h), T.append(t)
    return np.array(W), np.array(H), np.array(T)


def instrumented_solve(skel, cal, profile):
    """Re-run solver.bone_targets + a copy of solve_frame that records degeneracy data."""
    names = skel.names
    out = {k: [] for k in ("basis", "targets", "swing_deg", "perp_frac", "weight", "twist_raw_deg", "diag")}
    for f in cal["frames"]:
        if not f["valid"]:
            for k in out:
                out[k].append(None)
            continue
        j = {k: np.array(v[:3]) for k, v in f["joints"].items()}
        vis = {k: float(v[3]) for k, v in f["joints"].items()}
        tg, diag = S.bone_targets(j, vis, profile, CFG.solver)
        n = len(names)
        world = np.zeros((n, 3, 3)); basis = np.zeros((n, 3, 3))
        sw, pf, wt, tw = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n), np.full(n, np.nan)
        for i, name in enumerate(names):
            p = skel.parents[i]
            rl = skel.rest_local(i)
            carried = skel.rest[i] if p < 0 else world[p] @ rl
            direction, ref, weight, _ = tg[name]
            sw[i] = ang(carried[:, 1], direction)
            from altline.quat import rot_between, rot_axis_angle
            w = rot_between(carried[:, 1], direction) @ carried
            if ref is not None:
                pf[i] = np.linalg.norm(S.perp(ref, w[:, 1])) / max(np.linalg.norm(ref), 1e-12)
            if ref is not None and weight > 0 and np.linalg.norm(S.perp(ref, w[:, 1])) > 1e-9:
                a = signed_angle_about(w[:, 2], ref, w[:, 1])
                tw[i] = np.degrees(a)
                wt[i] = weight
                w = rot_axis_angle(w[:, 1], a * weight) @ w
            world[i] = w
            pw = np.eye(3) if p < 0 else world[p]
            basis[i] = rl.T @ pw.T @ w
        out["basis"].append(basis); out["targets"].append(tg); out["swing_deg"].append(sw)
        out["perp_frac"].append(pf); out["weight"].append(wt); out["twist_raw_deg"].append(tw); out["diag"].append(diag)
    return out


def ang_vel_acc(quats, fps):
    """Local-frame angular velocity vectors from consecutive quaternions; |dω|/dt as acceleration."""
    T, N, _ = quats.shape
    w = np.zeros((T - 1, N, 3))
    for t in range(1, T):
        for b in range(N):
            rel = quat_mul(quat_conj(quats[t - 1, b]), quats[t, b])
            w[t - 1, b] = quat_log(rel) * fps
    speed = np.degrees(np.linalg.norm(w, axis=2))           # (T-1, N), value at t-1/2
    acc = np.degrees(np.linalg.norm(np.diff(w, axis=0), axis=2)) * fps  # (T-2, N), value at frame t (1..T-2)
    vel = np.zeros((T, N)); vel[1:] = speed                 # vel[t] = between t-1 and t
    a = np.zeros((T, N)); a[1:-1] = acc
    return vel, a


def local_maxima(x, thr, k=6):
    idx = [t for t in range(len(x)) if x[t] > thr and (t == 0 or x[t] >= x[t - 1]) and (t == len(x) - 1 or x[t] >= x[t + 1])]
    return sorted(idx, key=lambda t: -x[t])[:k]


def plot_bone(path, bone, acc, acc_raw, bridged, gate, peaks, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=110)
    fr = np.arange(len(acc))
    for s, e in bridged:
        ax.axvspan(s - 0.5, e + 0.5, color="#f3d9a4", alpha=0.6, lw=0)
    ax.plot(fr, acc_raw, color="#9a9a9a", lw=1, label="solver raw (before limits)")
    ax.plot(fr, acc, color="#c0392b", lw=1.6, label="motion_bundle (final)")
    ax.axhline(gate, color="#2c3e50", ls="--", lw=1, label=f"gate {gate:.0f} deg/s²")
    for p in peaks:
        ax.plot([p], [acc[p]], "o", color="#c0392b", ms=4)
        ax.annotate(f"f{p}\n{acc[p]:.0f}", (p, acc[p]), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_title(f"{bone}: local angular acceleration (shaded = bridged by limits)", fontsize=10)
    ax.set_xlabel(f"frame ({fps:.0f} fps)"); ax.set_ylabel("deg/s²")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_heatmap(path, acc, bones, gate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=110)
    im = ax.imshow(np.log10(np.maximum(acc.T, 10)), aspect="auto", cmap="magma", vmin=1, vmax=5, interpolation="nearest")
    ys, xs = np.nonzero(acc.T > gate)
    ax.scatter(xs, ys, s=6, facecolors="none", edgecolors="#00e5ff", lw=0.7)
    ax.set_yticks(range(len(bones))); ax.set_yticklabels(bones, fontsize=7)
    ax.set_xlabel("frame"); ax.set_title(f"log10 angular acceleration (deg/s²), cyan = above gate {gate:.0f}", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def audit(run_id: str) -> dict:
    rd = run_dir(run_id)
    out_dir = REPORTS_ROOT / run_id
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    skel = reference_skeleton()
    names = skel.names
    N = len(names)
    manifest = read_json(rd / "manifest.json")
    cal = read_json(rd / "calibration" / "calibrated_pose.json")
    profile = read_json(rd / "calibration" / "actor_profile.json")
    raw = read_json(rd / "solver" / "raw_rotations.json")
    clean = read_json(rd / "limits" / "clean_rotations.json")
    lrep = read_json(rd / "limits" / "limits_report.json")
    bones, times, Q, fps = load_bundle(rd / "animation" / "motion_bundle.json")
    assert bones == names
    T = len(times)
    Qraw = np.array([f["rotations"] for f in raw["frames"]], dtype=float)
    Qclean = np.array([f["rotations"] for f in clean["frames"]], dtype=float)
    valid = np.array([f["valid"] for f in cal["frames"]])
    base = profile["baseline"]["frames"]

    inst = instrumented_solve(skel, cal, profile)
    repro = max(float(np.degrees(quat_angle_between(mat_to_quat(inst["basis"][t][b]), Qraw[t, b])))
                for t in range(T) if valid[t] for b in range(N))

    Wb, Hb, Tb = fk_all(skel, Q)
    Wr, Hr, Tr = fk_all(skel, Qraw)

    # ---------------- math anomalies ----------------
    math = {"solver_reproduced_max_deg": repro}
    for label, arr in (("raw_rotations", Qraw), ("clean_rotations", Qclean), ("motion_bundle", Q)):
        nerr = np.abs(np.linalg.norm(arr, axis=2) - 1)
        math[f"{label}_norm"] = {"max_abs_norm_minus_1": float(nerr.max()),
                                 "n_over_1e-6": int((nerr > 1e-6).sum())}
    orth = []
    for t in range(T):
        for b in range(N):
            q = Q[t, b]
            w, x, y, z = q  # unnormalised matrix exposes norm errors
            m = np.array([[w*w+x*x-y*y-z*z, 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), w*w-x*x+y*y-z*z, 2*(y*z-w*x)],
                          [2*(x*z-w*y), 2*(y*z+w*x), w*w-x*x-y*y+z*z]])
            orth.append((np.abs(m.T @ m - np.eye(3)).max(), abs(np.linalg.det(m) - 1)))
    orth = np.array(orth)
    fko = max(np.abs(Wb[t, b].T @ Wb[t, b] - np.eye(3)).max() for t in range(T) for b in range(N))
    math["rotation_matrices"] = {"bundle_max_orthonormality_err": float(orth[:, 0].max()),
                                 "bundle_max_det_err": float(orth[:, 1].max()),
                                 "fk_world_max_orthonormality_err": float(fko)}
    flips, jumps = {}, {}
    for label, arr in (("raw_rotations", Qraw), ("motion_bundle", Q)):
        fl, jp = {}, {}
        for b, n in enumerate(names):
            d = np.einsum("ti,ti->t", arr[1:, b], arr[:-1, b])
            f_ = [int(t + 1) for t in np.flatnonzero(d < 0)]
            g = np.array([np.degrees(quat_angle_between(arr[t - 1, b], arr[t, b])) for t in range(1, T)])
            j_ = [{"frame": int(t + 1), "deg": round(float(g[t]), 1)} for t in np.flatnonzero(g > 20)]
            if f_:
                fl[n] = f_
            if j_:
                jp[n] = j_
        flips[label], jumps[label] = fl, jp
    math["hemisphere_sign_flips_dot_lt_0"] = flips
    math["geodesic_jumps_over_20deg_per_frame"] = jumps

    sw = np.array([s if s is not None else np.full(N, np.nan) for s in inst["swing_deg"]])
    pf = np.array([s if s is not None else np.full(N, np.nan) for s in inst["perp_frac"]])
    wt = np.array([s if s is not None else np.zeros(N) for s in inst["weight"]])
    tw = np.array([s if s is not None else np.full(N, np.nan) for s in inst["twist_raw_deg"]])
    degen = {}
    for b, n in enumerate(names):
        e = {}
        near180 = np.flatnonzero(sw[:, b] > 160)
        if len(near180):
            e["rot_between_near_180deg_frames"] = [int(t) for t in near180]
        if not np.all(np.isnan(pf[:, b])):
            par = np.flatnonzero(pf[:, b] < 0.2)
            blend = np.flatnonzero((pf[:, b] >= 0.2) & (pf[:, b] < 0.5))
            e["twist_ref_parallel_to_bone_frames(perp<0.2, twist dropped)"] = [int(t) for t in par]
            e["twist_ref_blend_zone_frames(0.2<=perp<0.5)"] = [int(t) for t in blend]
            e["twist_ref_perp_fraction_min"] = round(float(np.nanmin(pf[:, b])), 3)
        partial = np.flatnonzero((wt[:, b] > 0.01) & (wt[:, b] < 0.99))
        if len(partial):
            e["partial_twist_weight_frames"] = [int(t) for t in partial]
        t_ = tw[:, b]
        if not np.all(np.isnan(t_)):
            big = np.flatnonzero(np.abs(t_) > 120)
            dt = np.abs(np.diff(t_))
            wrap = np.flatnonzero(dt > 90)
            e["twist_correction_raw_deg_range"] = r([np.nanmin(t_), np.nanmax(t_)], 1)
            if len(big):
                e["twist_correction_over_120deg_frames"] = [int(t) for t in big]
            if len(wrap):
                e["twist_correction_jump_over_90deg_frames"] = [int(t + 1) for t in wrap]
        # swing-twist singularity of the final local rotation (swing ~ 180)
        st = []
        for t in range(T):
            s, _ = swing_twist(Q[t, b], Yax)
            st.append(np.degrees(2 * np.arccos(np.clip(abs(s[0]), 0, 1))))
        st = np.array(st)
        e["local_swing_max_deg"] = round(float(st.max()), 1)
        if (st > 150).any():
            e["local_swing_over_150deg_frames(swing-twist ill-conditioned)"] = [int(t) for t in np.flatnonzero(st > 150)]
        degen[n] = e
    math["degeneracy"] = degen

    # ---------------- coordinate space ----------------
    J = [{k: np.array(v[:3]) for k, v in f["joints"].items()} if f["valid"] else None for f in cal["frames"]]
    rawpose = read_json(rd / "observations" / "raw_pose.json")
    chk = {"lshoulder_x_gt_rshoulder_x": [], "lhip_x_gt_rhip_x": [], "nose_ahead_of_ears_z": [],
           "ltoe_ahead_of_lankle_z": [], "rtoe_ahead_of_rankle_z": [], "image_lshoulder_right_of_rshoulder": []}
    for t, j in enumerate(J):
        if j is None:
            continue
        chk["lshoulder_x_gt_rshoulder_x"].append(j["left_shoulder"][0] > j["right_shoulder"][0])
        chk["lhip_x_gt_rhip_x"].append(j["left_hip"][0] > j["right_hip"][0])
        chk["nose_ahead_of_ears_z"].append(j["nose"][2] > 0.5 * (j["left_ear"][2] + j["right_ear"][2]))
        chk["ltoe_ahead_of_lankle_z"].append(j["left_foot_index"][2] > j["left_ankle"][2])
        chk["rtoe_ahead_of_rankle_z"].append(j["right_foot_index"][2] > j["right_ankle"][2])
        ij = rawpose["frames"][t]["image_joints"]
        chk["image_lshoulder_right_of_rshoulder"].append(ij["left_shoulder"][0] > ij["right_shoulder"][0])
    lr_checks = {k: {"fraction_true": round(float(np.mean(v)), 3),
                     "failing_frames": [int(t) for t, ok in enumerate(v) if not ok][:40]} for k, v in chk.items()}
    i = skel.index
    fkchk = {
        "upper_arm_l_tail_x_positive_at_baseline": bool(np.all(Tb[base, i["upper_arm_l"], 0] > 0)),
        "upper_arm_r_tail_x_negative_at_baseline": bool(np.all(Tb[base, i["upper_arm_r"], 0] < 0)),
        "foot_l_points_forward_at_baseline": bool(np.all(Tb[base, i["foot_l"], 2] > Hb[base, i["foot_l"], 2])),
        "foot_r_points_forward_at_baseline": bool(np.all(Tb[base, i["foot_r"], 2] > Hb[base, i["foot_r"], 2])),
        "head_Z_forward_at_baseline": bool(np.all(Wb[base, i["head"], 2, 2] > 0.9)),
    }
    rest_dev = {n: round(float(np.median([np.degrees(quat_angle_between(Q[t, b], [1, 0, 0, 0])) for t in base])), 2)
                for b, n in enumerate(names)}
    fk_vs_target, fk_vs_landmark, twist_err = {}, {}, {}
    for b, n in enumerate(names):
        e1, e2, e3 = [], [], []
        for t in range(T):
            if not valid[t]:
                e1.append(None), e2.append(None), e3.append(None)
                continue
            tg = inst["targets"][t][n]
            e1.append(round(ang(Wb[t, b][:, 1], tg[0]), 2))
            if n in LIMB:
                a_, b_ = LIMB[n]
                e2.append(round(ang(Wb[t, b][:, 1], J[t][b_] - J[t][a_]), 2))
            ref, w_ = tg[1], tg[2]
            if ref is not None and w_ >= 0.99 and np.linalg.norm(S.perp(ref, Wb[t, b][:, 1])) > 1e-6:
                e3.append(round(abs(np.degrees(signed_angle_about(Wb[t, b][:, 2], ref, Wb[t, b][:, 1]))), 2))
            else:
                e3.append(None)
        def summ(e):
            v = np.array([x for x in e if x is not None])
            if not len(v):
                return None
            return {"max": round(float(v.max()), 2), "p95": round(float(np.percentile(v, 95)), 2),
                    "frames_over_10deg": [int(t) for t, x in enumerate(e) if x is not None and x > 10], "per_frame": e}
        fk_vs_target[n] = summ(e1)
        if n in LIMB:
            fk_vs_landmark[n] = summ(e2)
        if any(x is not None for x in e3):
            twist_err[n] = summ(e3)
    # end effectors relative to the shoulder/hip, tracker scaled per limb to rig length
    eff = {}
    for s, d in S.SIDES.items():
        for lab, root, mid, end, b1, b2 in (("wrist", "shoulder", "elbow", "wrist", f"upper_arm_{s}", f"forearm_{s}"),
                                            ("ankle", "hip", "knee", "ankle", f"thigh_{s}", f"shin_{s}")):
            act = profile["actor_lengths_m"]
            k = (skel.lengths[i[b1]] + skel.lengths[i[b2]]) / (act[b1] + act[b2])
            err, dz = [], []
            for t in range(T):
                if not valid[t]:
                    err.append(None), dz.append(None)
                    continue
                trk = (J[t][f"{d}_{end}"] - J[t][f"{d}_{root}"]) * k
                fk = Tb[t, i[b2]] - Hb[t, i[b1]]
                err.append(round(float(np.linalg.norm(fk - trk)) * 100, 1))
                dz.append(round(float(fk[2] - trk[2]) * 100, 1))
            v = np.array([x for x in err if x is not None])
            eff[f"{lab}_{s}"] = {"max_cm": float(v.max()), "argmax_frame": int(np.nanargmax([x if x is not None else -1 for x in err])),
                                 "per_frame_cm": err, "depth_component_cm(fk_z-tracker_z)": dz}
    coord = {
        "mapping": "canon = (x, -y, -z) from MediaPipe world; det = +1 (no mirror)",
        "rest_alignment_angle_deg": profile["rest_alignment_angle_deg"],
        "left_right_forward_checks_calibrated": lr_checks,
        "fk_bundle_checks": fkchk,
        "baseline_frames": base,
        "rest_local_rotation_from_identity_median_deg": rest_dev,
        "fk_vs_solver_target_direction_deg": fk_vs_target,
        "fk_vs_raw_landmark_direction_deg(before hinge debias)": fk_vs_landmark,
        "fk_twist_vs_twist_reference_deg(weight>=0.99)": twist_err,
        "end_effector_error_vs_tracker_scaled_per_limb": eff,
    }

    # ---------------- bone length drift ----------------
    L = np.linalg.norm(Tb - Hb, axis=2)
    gaps = {}
    for b, n in enumerate(names):
        p = skel.parents[b]
        if p >= 0:
            g = np.linalg.norm(Hb[:, b] - Tb[:, p], axis=1)
            gaps[n] = round(float(g.max() - g.min()) * 1000, 4)
    lr_stats = {}
    segs = cal["frames"][0]["length_ratio"].keys() if valid[0] else next(f for f in cal["frames"] if f["valid"])["length_ratio"].keys()
    for sname in segs:
        v = np.array([f["length_ratio"][sname] if f["valid"] and f["length_ratio"][sname] is not None else np.nan for f in cal["frames"]])
        lr_stats[sname] = {"mean": round(float(np.nanmean(v)), 3), "std": round(float(np.nanstd(v)), 3),
                           "min": round(float(np.nanmin(v)), 3), "argmin": int(np.nanargmin(v)),
                           "max": round(float(np.nanmax(v)), 3), "argmax": int(np.nanargmax(v)),
                           "frames_off_by_more_than_15pct": int(np.sum(np.abs(v - 1) > 0.15))}
    lengths = {"fk_max_abs_dev_m_per_bone": {n: float(np.abs(L[:, b] - skel.lengths[b]).max()) for b, n in enumerate(names)},
               "fk_joint_gap_variation_mm(child head vs parent tail)": gaps,
               "actor_length_ratio_from_calibrated_pose": lr_stats}

    # ---------------- angular velocity / acceleration ----------------
    vel, acc = ang_vel_acc(Q, fps)
    vel_r, acc_r = ang_vel_acc(Qraw, fps)
    flagged = [{"bone": n, "frame": int(t), "acc_deg_s2": round(float(acc[t, b]), 0),
                "raw_solver_acc_deg_s2": round(float(acc_r[t, b]), 0)}
               for b, n in enumerate(names) for t in range(T) if acc[t, b] > GATE]
    angvel = {"definition": "vel[t] = |log(q[t-1]^-1 q[t])| * fps (deg/s), vel[0]=0",
              "per_bone": {n: {"max": round(float(vel[:, b].max()), 1), "argmax": int(vel[:, b].argmax()),
                               "per_frame": r(vel[:, b], 1)} for b, n in enumerate(names)}}
    angacc = {"definition": "acc[t] = |omega(t+1/2) - omega(t-1/2)| * fps, omega = local rotation vector rate (deg/s^2)",
              "gate_deg_s2": GATE,
              "n_flagged_bone_frames": len(flagged),
              "n_flagged_raw_solver_bone_frames": int((acc_r > GATE).sum()),
              "flagged": flagged,
              "per_bone": {n: {"max": round(float(acc[:, b].max()), 0), "argmax": int(acc[:, b].argmax()),
                               "raw_solver_max": round(float(acc_r[:, b].max()), 0),
                               "per_frame": r(acc[:, b], 0)} for b, n in enumerate(names)}}

    # ---------------- plots ----------------
    plots = []
    plot_heatmap(out_dir / "plots" / "acc_heatmap.png", acc, names, GATE)
    for b, n in enumerate(names):
        if acc[:, b].max() > GATE:
            peaks = local_maxima(acc[:, b], GATE)
            p = out_dir / "plots" / f"acc_{n}.png"
            plot_bone(p, n, acc[:, b], acc_r[:, b], lrep["bones"][n]["bridged_runs"], GATE, peaks, fps)
            plots.append({"bone": n, "path": f"plots/acc_{n}.png",
                          "peaks": [{"frame": int(t), "acc_deg_s2": round(float(acc[t, b]), 0)} for t in peaks]})
    jitter = {"heatmap": "plots/acc_heatmap.png", "bone_plots": plots}

    # ---------------- limits behaviour ----------------
    lim = {}
    for b, n in enumerate(names):
        e = dict(lrep["bones"][n])
        e["n_anomaly_frames"] = len(e.pop("anomaly_frames"))
        dirchg = np.array([ang(Wr[t, b][:, 1], Wb[t, b][:, 1]) for t in range(T)])
        e["fk_direction_change_raw_to_final_deg"] = {"max": round(float(dirchg.max()), 1), "argmax": int(dirchg.argmax()),
                                                    "frames_over_5deg": [int(t) for t in np.flatnonzero(dirchg > 5)]}
        if n.split("_")[0] in ("upper", "thigh", "neck", "head", "clavicle", "spine", "chest", "pelvis", "foot"):
            twv = np.array([np.degrees(twist_angle(swing_twist(Q[t, b], Yax)[1], Yax)) for t in range(T)])
            e["unlimited_twist_about_bone_Y_deg"] = {"min": round(float(twv.min()), 1), "argmin": int(twv.argmin()),
                                                     "max": round(float(twv.max()), 1), "argmax": int(twv.argmax())}
        lim[n] = e

    # ---------------- physical plausibility (automated) ----------------
    phys = {"floor": [], "legs_cross": [], "arm_in_torso": [], "hand_in_head": [], "forearm_forearm_min_dist_cm": [],
            "com_offset_from_support_cm": []}
    for t in range(T):
        feet = [Hb[t, i["foot_l"]], Tb[t, i["foot_l"]], Hb[t, i["foot_r"]], Tb[t, i["foot_r"]]]
        ly = min(Hb[t, i["foot_l"], 1], Tb[t, i["foot_l"], 1]); ry = min(Hb[t, i["foot_r"], 1], Tb[t, i["foot_r"], 1])
        phys["floor"].append([round(float(ly) * 100, 1), round(float(ry) * 100, 1)])
        phys["legs_cross"].append(bool(Hb[t, i["foot_l"], 0] < Hb[t, i["foot_r"], 0] or Tb[t, i["shin_l"], 0] < Tb[t, i["shin_r"], 0]))
        Wc = Wb[t, i["chest"]]; o = Hb[t, i["spine"]]
        pen = []
        for s in "lr":
            for bn in (f"forearm_{s}", f"hand_{s}"):
                for u in np.linspace(0, 1, 6):
                    pt = Hb[t, i[bn]] + u * (Tb[t, i[bn]] - Hb[t, i[bn]])
                    lp = Wc.T @ (pt - o)
                    if -0.12 < lp[1] < 0.30 and (lp[0] / 0.13) ** 2 + (lp[2] / 0.09) ** 2 < 1:
                        pen.append(bn)
                        break
        phys["arm_in_torso"].append(sorted(set(pen)))
        hc = 0.5 * (Hb[t, i["head"]] + Tb[t, i["head"]]) + 0.01 * Wb[t, i["head"]][:, 2]
        hh = []
        for s in "lr":
            for bn in (f"hand_{s}",):
                for u in np.linspace(0, 1, 5):
                    pt = Hb[t, i[bn]] + u * (Tb[t, i[bn]] - Hb[t, i[bn]])
                    if np.linalg.norm(pt - hc) < 0.09:
                        hh.append(bn); break
        phys["hand_in_head"].append(hh)
        phys["forearm_forearm_min_dist_cm"].append(round(100 * min(
            seg_dist(Hb[t, i[f"forearm_l"]], Tb[t, i[f"hand_l"]], Hb[t, i[f"forearm_r"]], Tb[t, i[f"hand_r"]]), 99), 1))
        com = np.zeros(3); mtot = 0
        for b, n in enumerate(names):
            m = MASS[n.rsplit("_", 1)[0] if n.endswith(("_l", "_r")) else n]
            com += m * 0.5 * (Hb[t, b] + Tb[t, b]); mtot += m
        com /= mtot
        pts = []
        for s, fy in (("l", ly), ("r", ry)):
            if fy < min(ly, ry) + 0.05:  # foot counts as support if within 5 cm of the lowest foot
                a_, t_ = Hb[t, i[f"foot_{s}"]], Tb[t, i[f"foot_{s}"]]
                fd = (t_ - a_)[[0, 2]]; fd = fd / (np.linalg.norm(fd) + 1e-9); lat = np.array([-fd[1], fd[0]])
                heel = a_[[0, 2]] - 0.06 * fd
                for c in (heel, t_[[0, 2]]):
                    pts += [c + 0.04 * lat, c - 0.04 * lat]
        phys["com_offset_from_support_cm"].append(round(_dist_to_hull(com[[0, 2]], np.array(pts)) * 100, 1))
        face = Hb[t, i["head"]] + 0.08 * Wb[t, i["head"]][:, 1] + 0.09 * Wb[t, i["head"]][:, 2]
        phys.setdefault("hand_to_face_cm", []).append([round(float(np.linalg.norm(Tb[t, i[f"hand_{s}"]] - face)) * 100, 1) for s in "lr"])
    fl = np.array(phys["floor"])
    stance = np.minimum(fl[:, 0], fl[:, 1])
    trunk_tilt = [round(ang(Wb[t, i["chest"]][:, 1], Yax), 1) for t in range(T)]
    head_yaw = [round(float(np.degrees(np.arctan2(Wb[t, i["head"]][0, 2], Wb[t, i["head"]][2, 2]))), 1) for t in range(T)]
    head_pitch = [round(float(np.degrees(np.arcsin(np.clip(Wb[t, i["head"]][1, 2], -1, 1)))), 1) for t in range(T)]
    head_roll = [round(float(np.degrees(np.arcsin(np.clip(-Wb[t, i["head"]][1, 0], -1, 1)))), 1) for t in range(T)]
    elbow_flex = {s: [round(float(np.degrees(quat_log(Q[t, i[f"forearm_{s}"]])[0])), 1) for t in range(T)] for s in "lr"}
    knee_flex = {s: [round(float(-np.degrees(quat_log(Q[t, i[f"shin_{s}"]])[0])), 1) for t in range(T)] for s in "lr"}
    hip_flex = {s: [round(ang(Wb[t, i[f"thigh_{s}"]][:, 1], -Yax) * np.sign(Wb[t, i[f"thigh_{s}"]][2, 1] or 1), 1) for t in range(T)] for s in "lr"}
    plaus = {
        "note": "root has no translation (pelvis fixed at rig rest height); floor = y 0",
        "lowest_foot_point_cm_per_frame[l,r]": phys["floor"],
        "frames_any_foot_below_floor(<-1cm)": [int(t) for t in np.flatnonzero(stance < -1)],
        "frames_no_foot_on_floor(>5cm)": [int(t) for t in np.flatnonzero(stance > 5)],
        "rest_lowest_foot_point_cm(median over baseline)": round(float(np.median(stance[base])), 1),
        "frames_legs_crossed": [int(t) for t, v in enumerate(phys["legs_cross"]) if v],
        "frames_arm_inside_torso": {int(t): v for t, v in enumerate(phys["arm_in_torso"]) if v},
        "frames_hand_inside_head": {int(t): v for t, v in enumerate(phys["hand_in_head"]) if v},
        "forearm+hand_l_vs_r_min_distance_cm": phys["forearm_forearm_min_dist_cm"],
        "frames_forearms_interpenetrating(<5cm)": [int(t) for t, v in enumerate(phys["forearm_forearm_min_dist_cm"]) if v < 5],
        "com_offset_from_support_cm": phys["com_offset_from_support_cm"],
        "frames_com_outside_support(>10cm)": [int(t) for t, v in enumerate(phys["com_offset_from_support_cm"]) if v > 10],
        "hand_tip_to_face_front_cm[l,r]": phys["hand_to_face_cm"],
        "chest_world_yaw_deg": [round(float(np.degrees(np.arctan2(Wb[t, i["chest"]][0, 2], Wb[t, i["chest"]][2, 2]))), 1) for t in range(T)],
        "chest_pitch_deg(+ = leaning back)": [round(float(np.degrees(np.arctan2(-Wb[t, i["chest"]][2, 1], Wb[t, i["chest"]][1, 1]))), 1) for t in range(T)],
        "upper_arm_elevation_from_down_deg": {s: [round(ang(Wb[t, i[f"upper_arm_{s}"]][:, 1], -Yax), 1) for t in range(T)] for s in "lr"},
        "chest_tilt_from_vertical_deg": trunk_tilt,
        "head_world_yaw_deg(+ = toward actor left)": head_yaw,
        "head_world_pitch_deg(+ = up)": head_pitch,
        "head_world_roll_deg": head_roll,
        "elbow_flex_deg": elbow_flex, "knee_flex_deg": knee_flex,
        "thigh_angle_from_down_deg(+ = forward)": hip_flex,
        "solver_diag_raw_vs_debiased_bend": {k: r([d[k] if d else np.nan for d in inst["diag"]], 1)
                                            for k in ("elbow_l_bend_raw", "elbow_l_bend", "elbow_r_bend_raw", "elbow_r_bend",
                                                      "knee_l_bend_raw", "knee_l_bend", "knee_r_bend_raw", "knee_r_bend")},
    }

    rep = {
        "run_id": run_id,
        "video": manifest["video"],
        "fps": fps, "n_frames": T, "n_valid_frames": int(valid.sum()),
        "baseline": profile["baseline"],
        "hinge_bias_deg": profile["hinge_bias_deg"],
        "math_anomalies": math,
        "coordinate_space": coord,
        "bone_length_drift": lengths,
        "angular_velocity": angvel,
        "angular_acceleration": angacc,
        "jitter_plots": jitter,
        "limits_behaviour": lim,
        "physical_plausibility_auto": plaus,
    }
    manual = out_dir / "manual.json"
    if manual.exists():
        rep.update(json.loads(manual.read_text()))
    write_json(out_dir / "dev_audit.json", rep)
    return rep


def brief(rep: dict) -> None:
    """Print the essentials for the reviewer."""
    run = rep["run_id"]
    m = rep["math_anomalies"]
    print(f"=== {run}  valid {rep['n_valid_frames']}/{rep['n_frames']}  baseline {rep['baseline']['start']}  repro {m['solver_reproduced_max_deg']:.1e}")
    print(" norms", {k: v["max_abs_norm_minus_1"] for k, v in m.items() if k.endswith("_norm")}, m["rotation_matrices"])
    print(" flips", m["hemisphere_sign_flips_dot_lt_0"])
    print(" jumps", {k: {b: [(j['frame'], j['deg']) for j in v] for b, v in d.items()} for k, d in m["geodesic_jumps_over_20deg_per_frame"].items()})
    for b, e in m["degeneracy"].items():
        s = {k: (v if not isinstance(v, list) or len(v) < 12 else f"{len(v)} frames {v[0]}..{v[-1]}") for k, v in e.items()}
        print("  degen", b, s)
    c = rep["coordinate_space"]
    print(" lr", {k: (v["fraction_true"], v["failing_frames"][:8]) for k, v in c["left_right_forward_checks_calibrated"].items()})
    print(" fkchk", c["fk_bundle_checks"])
    print(" rest dev", c["rest_local_rotation_from_identity_median_deg"])
    print(" fk-target max", {b: (v["max"], len(v["frames_over_10deg"])) for b, v in c["fk_vs_solver_target_direction_deg"].items() if v and v["max"] > 1})
    print(" fk-landmark max", {b: (v["max"], v["p95"]) for b, v in c["fk_vs_raw_landmark_direction_deg(before hinge debias)"].items() if v})
    print(" twist err", {b: (v["max"], v["p95"]) for b, v in c["fk_twist_vs_twist_reference_deg(weight>=0.99)"].items() if v})
    print(" eff", {k: (v["max_cm"], v["argmax_frame"]) for k, v in c["end_effector_error_vs_tracker_scaled_per_limb"].items()})
    lr = rep["bone_length_drift"]
    print(" fk len dev", max(lr["fk_max_abs_dev_m_per_bone"].values()), "gap var mm", max(lr["fk_joint_gap_variation_mm(child head vs parent tail)"].values()))
    print(" length ratio", {k: (v["min"], v["argmin"], v["max"], v["argmax"], v["std"]) for k, v in lr["actor_length_ratio_from_calibrated_pose"].items()})
    a = rep["angular_acceleration"]
    print(" acc flagged", a["n_flagged_bone_frames"], "raw", a["n_flagged_raw_solver_bone_frames"])
    print(" acc max", {b: (v["max"], v["argmax"], v["raw_solver_max"]) for b, v in a["per_bone"].items() if v["max"] > GATE})
    print(" vel max", {b: (v["max"], v["argmax"]) for b, v in rep["angular_velocity"]["per_bone"].items() if v["max"] > 150})
    for b, e in rep["limits_behaviour"].items():
        if e["frames_clamped"] or e["bridged_runs"] or e["unbridged_runs"] or e["fk_direction_change_raw_to_final_deg"]["max"] > 3 or "unlimited_twist_about_bone_Y_deg" in e:
            print("  lim", b, {k: v for k, v in e.items() if k not in ("config",)})
    p = rep["physical_plausibility_auto"]
    for k in ("frames_any_foot_below_floor(<-1cm)", "frames_no_foot_on_floor(>5cm)", "frames_legs_crossed",
              "frames_arm_inside_torso", "frames_hand_inside_head", "frames_forearms_interpenetrating(<5cm)", "frames_com_outside_support(>10cm)"):
        v = p[k]
        print(" ", k, v if len(v) < 15 else f"{len(v)}: {list(v)[:3]}..{list(v)[-3:]}")
    def mm(x):
        x = np.array(x, dtype=float); return (float(np.nanmin(x)), int(np.nanargmin(x)), float(np.nanmax(x)), int(np.nanargmax(x)))
    print("  com max", mm(p["com_offset_from_support_cm"]), "tilt", mm(p["chest_tilt_from_vertical_deg"]), "chest pitch", mm(p["chest_pitch_deg(+ = leaning back)"]), "chest yaw", mm(p["chest_world_yaw_deg"]), "rest foot", p["rest_lowest_foot_point_cm(median over baseline)"])
    hf = np.array(p["hand_tip_to_face_front_cm[l,r]"]); print("  hand-face min", mm(hf[:, 0]), mm(hf[:, 1]))
    print("  arm elev", {s: mm(v) for s, v in p["upper_arm_elevation_from_down_deg"].items()})
    print("  head yaw", mm(p["head_world_yaw_deg(+ = toward actor left)"]), "pitch", mm(p["head_world_pitch_deg(+ = up)"]), "roll", mm(p["head_world_roll_deg"]))
    print("  elbow", {s: mm(v) for s, v in p["elbow_flex_deg"].items()}, "knee", {s: mm(v) for s, v in p["knee_flex_deg"].items()},
          "thigh", {s: mm(v) for s, v in p["thigh_angle_from_down_deg(+ = forward)"].items()})
    print("  forearm dist", mm(p["forearm+hand_l_vs_r_min_distance_cm"]))
    print("  bend raw/deb", {k: mm(v) for k, v in p["solver_diag_raw_vs_debiased_bend"].items()})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", action="append", required=True)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    for rid in a.run_id:
        rep = audit(rid)
        if not a.quiet:
            brief(rep)


if __name__ == "__main__":
    main()
