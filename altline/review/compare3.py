"""Reviewer: N-version comparison (e.g. v1, v2, v3) of one clip; reuses compare.py metrics.

    python3 -m altline.review.compare3 --clip df1e --versions v1 v2 v3

All runs are judged against the same version-independent references (source 2D points,
2D head-yaw estimate, semantic checks), with the newest version's rest-projection lengths
as the 2D scale. Pairwise sections: "<prev>_vs_<new>" for every older version.
`ablation` re-runs the current pipeline code with the depth rebuild / the filter off.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from altline.io_utils import REPORTS_ROOT, read_json, run_dir
from altline.review import compare as C
from altline.review.compare import (BONE_2D, CLIPS, GATE, SKEL, Run, _q, calibrated_length_stats, chord_angle_series,
                                    lag_frames, metrics, stats, verdict)

COLORS = {"v1": ("#95a5a6", ":"), "v2": ("#2980b9", "--"), "v3": ("#c0392b", "-")}


def pair_section(la, ma, lb, mb, cal_a, cal_b, L) -> dict:
    res = {}
    acc = {}
    for b in SKEL.names:
        a1, a2 = ma["angular_acceleration"][b], mb["angular_acceleration"][b]
        if max(a1["max"], a2["max"]) < GATE:
            continue
        acc[b] = {la: {k: a1[k] for k in ("max", "argmax", "p99", "p95", "frames_above_gate")},
                  lb: {k: a2[k] for k in ("max", "argmax", "p99", "p95", "frames_above_gate")},
                  "verdict_peak": verdict(a1["max"], a2["max"]),
                  "verdict_frames_above_gate": verdict(a1["frames_above_gate"], a2["frames_above_gate"], abs_tol=1)}
    res["angular_acceleration_per_bone(bones above gate in either)"] = acc
    res["frames_above_gate_total"] = {la: ma["frames_above_gate_total"], lb: mb["frames_above_gate_total"],
                                      "verdict": verdict(ma["frames_above_gate_total"], mb["frames_above_gate_total"], abs_tol=1)}
    d = {}
    for b in BONE_2D:
        a1, a2 = ma["direction_error_2d_deg"][b], mb["direction_error_2d_deg"][b]
        if "p95" not in a1 or "p95" not in a2:
            continue
        d[b] = {la: {k: a1[k] for k in ("median", "p95", "max", "n")}, lb: {k: a2[k] for k in ("median", "p95", "max", "n")},
                "verdict_p95": verdict(a1["p95"], a2["p95"], abs_tol=1.5)}
    res["direction_error_2d_vs_image_landmarks_deg"] = d
    e = {}
    for k in ma["end_effector_error_2d_cm"]:
        a1, a2 = ma["end_effector_error_2d_cm"][k], mb["end_effector_error_2d_cm"][k]
        e[k] = {la: {x: a1[x] for x in ("median", "p95", "max")}, lb: {x: a2[x] for x in ("median", "p95", "max")},
                "verdict_p95": verdict(a1["p95"], a2["p95"], abs_tol=1.0)}
    res["hand_foot_position_2d_vs_source_cm(rig scale)"] = e
    hy = {k: {la: ma["head_yaw"][k], lb: mb["head_yaw"][k]} for k in
          ("rig_rest_deg", "rig_hold_median_deg", "image_hold_median_deg", "rig_abs_max_deg", "rig_abs_argmax",
           "abs_error_hold", "abs_error_where_image_yaw_gt20")}
    hy["verdict_hold_error_median"] = verdict(ma["head_yaw"]["abs_error_hold"]["median"], mb["head_yaw"]["abs_error_hold"]["median"], abs_tol=2)
    res["head_yaw_vs_2d_estimate_deg"] = hy
    sk = "static_limb_forward_angle_deg(+ = toward camera)"
    res["static_limb_forward_angle_deg"] = {la: ma[sk], lb: mb[sk],
                                            "verdict_abs_max": {b: verdict(ma[sk][b]["abs_max"], mb[sk][b]["abs_max"], abs_tol=3) for b in ma[sk]}}
    res["rest_baseline_forward_angle_deg"] = {la: ma["rest_baseline_forward_angle_deg"], lb: mb["rest_baseline_forward_angle_deg"]}
    absmax = lambda m: max(abs(m["chest_yaw_deg"]["min"]), abs(m["chest_yaw_deg"]["max"]))
    res["chest_yaw_deg"] = {la: ma["chest_yaw_deg"], lb: mb["chest_yaw_deg"], "verdict_abs_max": verdict(absmax(ma), absmax(mb), abs_tol=3)}
    pk = "chest_pitch_deg(+ back)"
    res[pk] = {la: ma[pk], lb: mb[pk], "verdict_max": verdict(ma[pk]["max"], mb[pk]["max"], abs_tol=2)}
    res["semantic"] = {k: {la: ma["semantic"][k], lb: mb["semantic"].get(k)} for k in ma["semantic"] if not k.startswith("_")}
    res["solver_input_segment_length_over_rest_length"] = {la: calibrated_length_stats(cal_a, L), lb: calibrated_length_stats(cal_b, L)}
    return res


def lag_section(runs: dict, cid: str, scale: float) -> dict:
    out = {}
    for limb_id in CLIPS[cid]["moving"]:
        if limb_id == "head":
            continue
        limb, sd = limb_id.split("_")
        ser = {}
        src = None
        for lab, r in runs.items():
            src, ser[lab] = chord_angle_series(r, sd, limb, scale)
        base = list(runs.values())[-1].base
        fps = list(runs.values())[-1].fps
        out[f"{limb_id}_chord_image_angle"] = {
            "lag_frames(+ = late)": {lab: lag_frames(src, s) for lab, s in ser.items()},
            "peak_speed_deg_s": {"source_2d": _q(np.abs(np.diff(src)).max() * fps),
                                 **{lab: _q(np.abs(np.diff(s)).max() * fps) for lab, s in ser.items()}},
            "excursion_max_deg(abs from rest)": {"source_2d": _q(np.abs(src - np.median(src[base])).max()),
                                                 **{lab: _q(np.abs(s - np.median(s[base])).max()) for lab, s in ser.items()}},
            "abs_error_vs_source_deg": {lab: stats(np.abs(s - src)) for lab, s in ser.items()},
            "_series": (src, ser),
        }
    return out


def depth_diagnostics(run_id: str) -> dict:
    """Re-run the *current* depth rebuild (unfiltered, with sign continuity) and report per
    rebuilt segment: rest projection ratio, rest out-of-plane angle, frames in the length zone,
    projection longer than L (d > L: depth clipped to 0, segment stretched), sign flips while
    clearly out of plane, largest frame-to-frame depth jump."""
    from altline.calibration import CalibrationConfig, _actor_lengths, find_baseline, load_frames
    from altline.depth import (PLANAR_FULL_LENGTH_RATIO, PLANAR_TRACKER_RATIO, TREE, image_points, pixel_scale,
                               planar_lengths, reconstruct)
    rd = run_dir(run_id)
    raw = read_json(rd / "observations" / "raw_pose.json")
    cfg = CalibrationConfig()
    tracker = load_frames(raw)
    base = find_baseline(tracker, cfg)["frames"]
    actor = _actor_lengths(tracker, base, cfg.min_visibility)
    meta = raw["meta"]
    pts = [image_points(f["image_joints"], meta["width"], meta["height"]) if f["detected"] else None for f in raw["frames"]]
    scale = pixel_scale([pts[i] for i in base], actor)
    lengths = {**actor, **planar_lengths(pts, base, scale)}
    recs, prev = [], None
    for fr, p in zip(tracker, pts):
        rec, _, prev = reconstruct(fr[0], p, scale, lengths, prev)
        recs.append(rec)
    out = {"pixel_scale_m": scale, "code": "current altline/depth.py"}
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
        out[f"{parent}->{child}"] = {
            "L_m": round(L, 4), "rest_projection_ratio": round(float(np.median(ratio[base])), 3),
            "rest_out_of_plane_deg(rebuilt)": round(float(np.degrees(np.arcsin(np.clip(abs(np.median(dz[base])) / L, 0, 1)))), 1),
            "rest_out_of_plane_deg(tracker)": round(float(np.degrees(np.arcsin(np.clip(abs(np.median(tz[base])) / L, 0, 1)))), 1),
            "frames_length_zone(ratio<0.97)": int((ratio < PLANAR_TRACKER_RATIO).sum()),
            "frames_full_length_zone(ratio<0.85)": int((ratio < PLANAR_FULL_LENGTH_RATIO).sum()),
            "frames_projection_longer_than_L(>1.03)": int((ratio > 1.03).sum()), "max_projection_ratio": round(float(ratio.max()), 3),
            "sign_flips_while_out_of_plane": flips[:30], "n_sign_flips": len(flips),
            "max_depth_jump_cm_per_frame": round(float(jump.max()) * 100, 1), "argmax_jump_frame": int(jump.argmax()) + 1,
        }
    return out


def ablation(run_id: str, scale, L, clip) -> dict:
    A = C.ablation(run_id)  # labels say v2_ but it runs the current code
    r = Run(run_id)
    out = {"reproduced_max_quat_diff_vs_bundle": float(np.abs(np.abs(np.einsum("tbi,tbi->tb", A["v2_reproduced"][0], r.Q)) - 1).max())}
    for lab, new in (("v2_no_depth_rebuild", "no_depth_rebuild"), ("v2_no_filter", "no_filter")):
        Q, prof, cal, rep = A[lab]
        ra = Run(new, Q, prof)
        ma = metrics(ra, scale, L, clip)
        out[new] = {"frames_above_gate_total": ma["frames_above_gate_total"],
                    "acc_max_per_bone(above gate)": {b: ma["angular_acceleration"][b]["max"] for b in SKEL.names
                                                     if ma["angular_acceleration"][b]["max"] > GATE},
                    "head_yaw_hold_median": ma["head_yaw"]["rig_hold_median_deg"], "head_yaw_abs_max": ma["head_yaw"]["rig_abs_max_deg"],
                    "chest_yaw": ma["chest_yaw_deg"], "chest_pitch": ma["chest_pitch_deg(+ back)"],
                    "hinge_bias_deg": prof["hinge_bias_deg"],
                    "semantic": {k: v for k, v in ma["semantic"].items() if not k.startswith("_")}}
    return out


def run(clip: str, versions: list[str], ablate: bool = True) -> tuple[dict, dict]:
    cinfo = {**CLIPS[clip], "id": clip}
    ids = {v: f"{v}_{clip}" for v in versions}
    runs = {v: Run(i) for v, i in ids.items()}
    newest = versions[-1]
    scale, L = C.reference_lengths(runs[newest].profile, runs[newest].pts)
    ms = {v: metrics(r, scale, L, cinfo) for v, r in runs.items()}
    cals = {v: read_json(run_dir(i) / "calibration" / "calibrated_pose.json") for v, i in ids.items()}
    out = {"reference": f"same raw_pose.json; 2D scale = {newest} pixel_scale_m and rest-projection lengths", "clip": clip}
    for v in versions[:-1]:
        out[f"{v}_vs_{newest}"] = pair_section(v, ms[v], newest, ms[newest], cals[v], cals[newest], L)
    lag = lag_section(runs, clip, scale)
    out["filter_lag"] = {k: {a: b for a, b in v.items() if a != "_series"} for k, v in lag.items()}
    if ablate:
        out[f"ablation({newest} code, one feature off)"] = ablation(ids[newest], scale, L, cinfo)
    plots(ids[newest], clip, ms, lag, runs[newest])
    return out, ms


def plots(rid, clip, ms, lag, rn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pd = REPORTS_ROOT / rid / "plots"
    pd.mkdir(parents=True, exist_ok=True)
    fr = np.arange(rn.n)
    labs = "_".join(ms)

    def styled(ax, lab, y, **kw):
        c, ls = COLORS.get(lab, ("#333", "-"))
        ax.plot(fr, y, color=c, ls=ls, lw=1.6 if lab == list(ms)[-1] else 1.1, label=f"{lab}{kw.get('suffix', '')}")
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=110)
    ax.plot(fr, list(ms.values())[-1]["_yaw"][1], color="#000", lw=2.2, alpha=.35, label="2D image estimate")
    for lab, m in ms.items():
        styled(ax, lab, m["_yaw"][0])
    ax.set_title(f"{rid}: head world yaw relative to rest (+ = actor left)", fontsize=10)
    ax.set_xlabel("frame"); ax.set_ylabel("deg"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(pd / f"cmp_head_yaw_{labs}.png"); plt.close(fig)
    fig, axs = plt.subplots(2, 1, figsize=(9, 5), dpi=110, sharex=True)
    for lab, m in ms.items():
        styled(axs[0], lab, m["_trunk"][0]); styled(axs[1], lab, m["_trunk"][1])
    axs[0].set_title(f"{rid}: chest yaw (top) and chest pitch, + = leaning back (bottom)", fontsize=10)
    for a in axs:
        a.grid(alpha=.3); a.legend(fontsize=7); a.set_ylabel("deg")
    axs[1].set_xlabel("frame")
    fig.tight_layout(); fig.savefig(pd / f"cmp_trunk_{labs}.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=110)
    for lab, m in ms.items():
        styled(ax, lab, np.maximum(m["_acc"].max(axis=1), 10), suffix=f" (frames>gate {m['frames_above_gate_total']})")
    ax.axhline(GATE, color="#2c3e50", ls="--", lw=1, label="gate")
    ax.set_yscale("log"); ax.set_ylim(10, 2e5)
    ax.set_title(f"{rid}: max angular acceleration over all bones", fontsize=10)
    ax.set_xlabel("frame"); ax.set_ylabel("deg/s²"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(pd / f"cmp_acc_{labs}.png"); plt.close(fig)
    for k, v in lag.items():
        src, ser = v["_series"]
        fig, ax = plt.subplots(figsize=(9, 3.2), dpi=110)
        ax.plot(fr, src, color="#000", alpha=.35, lw=2.4, label="source 2D")
        for lab, s in ser.items():
            styled(ax, lab, s)
        ax.set_title(f"{rid}: {k} lag {v['lag_frames(+ = late)']}", fontsize=9)
        ax.set_xlabel("frame"); ax.set_ylabel("deg"); ax.grid(alpha=.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(pd / f"cmp_lag_{k}_{labs}.png"); plt.close(fig)
    if clip == "df1e":
        fig, axs = plt.subplots(2, 1, figsize=(9, 5), dpi=110, sharex=True)
        for lab, m in ms.items():
            styled(axs[0], lab, m["semantic"]["_sag"])
            knee = np.array([-np.degrees(C.quat_log(r)[0]) for r in Run(f"{lab}_df1e").Q[:, SKEL.index["shin_l"]]])
            styled(axs[1], lab, knee)
        axs[0].set_title("df1e: left thigh sagittal angle (0 down, 90 fwd, 180 up) / left knee flex (bottom)", fontsize=9)
        for a in axs:
            a.grid(alpha=.3); a.legend(fontsize=7); a.set_ylabel("deg")
        fig.tight_layout(); fig.savefig(pd / f"cmp_kick_leg_{labs}.png"); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", required=True)
    ap.add_argument("--versions", nargs="+", default=["v1", "v2", "v3"])
    a = ap.parse_args(argv)
    for c in a.clip:
        res, _ = run(c, a.versions)
        print(json.dumps(res, default=str)[:3000])


if __name__ == "__main__":
    main()
