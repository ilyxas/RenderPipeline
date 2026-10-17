"""Task 7: palm / limb flip guard with cubic Hermite bridging.

Input: <run>/calibration/calibrated_pose.json (aligned canonical landmarks)
Output: <run>/calibration/guarded_pose.json (same format) and flip_report.json

A tracker "palm flip" (or a depth-sign flip of a limb) shows up as a limb vector turning
further in one frame than a human limb can: more than `max_step_deg` (45 deg; at 24 fps
that is > 1000 deg/s). Monitored per limb:
    arm: upper arm, forearm, hand direction, palm normal (pinky x index around the wrist)
    leg: thigh, shin, foot (ankle->toe), heel
(the directive names forearm and wrist; legs get the same guard because the depth flips
the v3 review found hit shins as well as forearms).

Each frame is compared with the last clean frame. Frames that jump away are corrupted
until the limb comes back within the threshold; such a run is replaced by a cubic Hermite
spline through up to `neighbors` clean frames on each side, applied to the limb's joint
offsets from their parents (so the limb root stays where the trunk puts it). A run longer
than `max_run_frames`, or one that reaches the end of the clip, is real motion (or cannot
be bridged) and is kept. No frame is dropped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import FlipGuardConfig
from .io_utils import read_json, run_dir, write_json

# limb -> chain of (parent, child) joints whose offsets are bridged, and monitored vectors
LIMBS: dict[str, dict] = {}
for _side in ("left", "right"):
    LIMBS[f"{_side}_arm"] = {
        "chain": [(f"{_side}_shoulder", f"{_side}_elbow"), (f"{_side}_elbow", f"{_side}_wrist"),
                  (f"{_side}_wrist", f"{_side}_index"), (f"{_side}_wrist", f"{_side}_pinky"),
                  (f"{_side}_wrist", f"{_side}_thumb")],
        "vectors": {"upper_arm": (f"{_side}_shoulder", f"{_side}_elbow"),
                    "forearm": (f"{_side}_elbow", f"{_side}_wrist"),
                    "hand": (f"{_side}_wrist", f"{_side}_hand_mid"),
                    "palm_normal": None},
        "palm": (f"{_side}_wrist", f"{_side}_pinky", f"{_side}_index"),
        "derived": (f"{_side}_hand_mid", f"{_side}_index", f"{_side}_pinky"),
    }
    LIMBS[f"{_side}_leg"] = {
        "chain": [(f"{_side}_hip", f"{_side}_knee"), (f"{_side}_knee", f"{_side}_ankle"),
                  (f"{_side}_ankle", f"{_side}_heel"), (f"{_side}_ankle", f"{_side}_foot_index")],
        "vectors": {"thigh": (f"{_side}_hip", f"{_side}_knee"), "shin": (f"{_side}_knee", f"{_side}_ankle"),
                    "foot": (f"{_side}_ankle", f"{_side}_foot_index"), "heel": (f"{_side}_ankle", f"{_side}_heel")},
    }


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))


def limb_vectors(j: dict[str, np.ndarray], limb: dict) -> dict[str, np.ndarray]:
    out = {}
    for name, ends in limb["vectors"].items():
        if ends is None:
            w, p, i = limb["palm"]
            out[name] = np.cross(j[p] - j[w], j[i] - j[w])
        else:
            out[name] = j[ends[1]] - j[ends[0]]
    return out


def detect_runs(vecs: list[dict[str, np.ndarray] | None], cfg: FlipGuardConfig) -> tuple[list, list, np.ndarray]:
    """Return (bridgeable corrupt runs, kept runs, per-frame max step vs previous frame)."""
    n = len(vecs)
    step = np.zeros(n)
    for t in range(1, n):
        if vecs[t] is not None and vecs[t - 1] is not None:
            step[t] = max(_angle(vecs[t][k], vecs[t - 1][k]) for k in vecs[t])

    def jump(t, ref):
        return max(_angle(vecs[t][k], vecs[ref][k]) for k in vecs[t])

    bridge, kept = [], []
    ref = next((t for t in range(n) if vecs[t] is not None), None)
    t = (ref or 0) + 1
    while ref is not None and t < n:
        if vecs[t] is None:
            t += 1
            continue
        if jump(t, ref) <= cfg.max_step_deg:
            ref = t
            t += 1
            continue
        k = t
        while k < n and (vecs[k] is None or jump(k, ref) > cfg.max_step_deg):
            k += 1
        run = (t, k - 1)
        if k < n and (k - t) <= cfg.max_run_frames:
            bridge.append(run)
            ref, t = k, k + 1
        else:  # long or unterminated: real motion (or not bridgeable); follow it
            kept.append(run)
            ref, t = t, t + 1
    return bridge, kept, step


def hermite_bridge(times: np.ndarray, values: np.ndarray, clean: np.ndarray, run: tuple[int, int],
                   neighbors: int) -> np.ndarray:
    """Cubic Hermite spline through up to `neighbors` clean samples on each side of `run`;
    slopes are finite differences of those samples. Returns values for the run's frames."""
    from scipy.interpolate import CubicHermiteSpline

    s, e = run
    before = [i for i in range(s - 1, -1, -1) if clean[i]][:neighbors][::-1]
    after = [i for i in range(e + 1, len(times)) if clean[i]][:neighbors]
    idx = np.array(before + after)
    if len(idx) < 2:
        return values[s:e + 1]
    x, y = times[idx], values[idx]
    dydx = np.gradient(y, x, axis=0) if len(idx) > 2 else np.repeat(((y[1] - y[0]) / (x[1] - x[0]))[None], 2, axis=0)
    return CubicHermiteSpline(x, y, dydx, axis=0)(times[s:e + 1])


def guard(cal: dict, cfg: FlipGuardConfig | None = None) -> tuple[dict, dict]:
    cfg = cfg or FlipGuardConfig()
    frames = cal["frames"]
    n = len(frames)
    times = np.array([f["timestamp_s"] for f in frames])
    joints = [None if not f["valid"] else {k: np.array(v[:3]) for k, v in f["joints"].items()} for f in frames]
    report = {"config": cfg.__dict__, "limbs": {}}

    for limb_name, limb in LIMBS.items():
        vecs = [None if j is None else limb_vectors(j, limb) for j in joints]
        bridge, kept, step = detect_runs(vecs, cfg)
        corrupt = np.zeros(n, dtype=bool)
        for s, e in bridge:
            corrupt[s:e + 1] = True
        clean = np.array([j is not None for j in joints]) & ~corrupt
        if bridge:
            offsets = {}
            for parent, child in limb["chain"]:
                vals = np.array([j[child] - j[parent] if j is not None else [np.nan] * 3 for j in joints])
                offsets[(parent, child)] = vals
            for run in bridge:
                for key, vals in offsets.items():
                    vals[run[0]:run[1] + 1] = hermite_bridge(times, vals, clean, run, cfg.neighbors)
            for t in np.flatnonzero(corrupt):
                j = joints[t]
                for parent, child in limb["chain"]:  # parents precede children in the chain
                    j[child] = j[parent] + offsets[(parent, child)][t]
                if "derived" in limb:
                    mid, a, b = limb["derived"]
                    j[mid] = 0.5 * (j[a] + j[b])
        report["limbs"][limb_name] = {
            "bridged_runs": [list(r) for r in bridge],
            "kept_runs_over_threshold": [list(r) for r in kept],
            "frames_bridged": int(corrupt.sum()),
            "max_step_deg": float(step.max()),
            "frames_step_over_threshold": [int(t) for t in np.flatnonzero(step > cfg.max_step_deg)],
        }

    out_frames = []
    for f, j in zip(frames, joints):
        if j is None:
            out_frames.append(f)
            continue
        out_frames.append({**f, "joints": {k: list(map(float, j[k])) + [f["joints"][k][3]] for k in f["joints"]}})
    return {"meta": {**cal["meta"], "flip_guard": cfg.__dict__}, "frames": out_frames}, report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="altline flip guard")
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args(argv)
    rd = run_dir(args.run_id)
    guarded, report = guard(read_json(rd / "calibration" / "calibrated_pose.json"))
    print(write_json(rd / "calibration" / "guarded_pose.json", guarded))
    print(write_json(rd / "calibration" / "flip_report.json", report))


if __name__ == "__main__":
    main()
