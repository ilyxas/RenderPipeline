"""Task 8: sliding-window temporal solver.

Input: <run>/solver/frame_rotations.json (the per-frame analytic solve, solver.py)
Output: <run>/solver/raw_rotations.json (same format; limits.py reads it)

The per-frame solver answers every frame on its own, so tracker noise goes straight into
the rotations. Here every bone's rotation at frame t is re-estimated from the window
t-2 .. t+2 (`window` frames). In the tangent space at the per-frame solution q_t, the
neighbours are y_k = log(q_t^-1 q_{t+k}). The window's corrected trajectory x_k minimises

    sum_k c_k |x_k - y_k|^2  +  lambda * sum_k |x_{k-1} - 2 x_k + x_{k+1}|^2

with c_k the per-frame solver confidence: the first term keeps the fit to the observations,
the second penalises changes of the angular velocity vector (x_{k+1} - x_k) inside the
window. It is a 5x5 linear system per axis; the output is q_t * exp(x_0). Only the centre
of each window is kept (a sliding window, not a global smoother), so the result depends
only on t-2 .. t+2. Frames without a detection keep the per-frame result (limits.py bridges
them) and carry zero weight in their neighbours' windows.
"""

from __future__ import annotations

import argparse

import numpy as np

from .config import TemporalSolverConfig
from .io_utils import read_json, run_dir, write_json
from .quat import quat_conj, quat_exp, quat_log, quat_mul, quat_normalize

MIN_WEIGHT = 0.05  # a valid frame with near-zero visibility still anchors its own pose a little


def window_matrix(weights: np.ndarray, smoothness: float) -> np.ndarray:
    """Normal matrix of the window cost: diag(c) + lambda * D2^T D2."""
    m = len(weights)
    a = np.diag(weights.astype(float))
    if m >= 3:
        d2 = np.zeros((m - 2, m))
        for i in range(m - 2):
            d2[i, i:i + 3] = (1.0, -2.0, 1.0)
        a = a + smoothness * d2.T @ d2
    return a


def smooth_bone(qs: np.ndarray, conf: np.ndarray, valid: np.ndarray, cfg: TemporalSolverConfig
                ) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window estimate of one bone's (T, 4) quaternions; returns (rotations, change in deg)."""
    n = len(qs)
    half = cfg.window // 2
    out = qs.copy()
    change = np.zeros(n)
    w_all = np.where(valid, np.maximum(conf, MIN_WEIGHT), 0.0)
    for t in range(n):
        if not valid[t]:
            continue
        ks = np.arange(max(0, t - half), min(n, t + half + 1))
        inv = quat_conj(qs[t])
        y = np.array([quat_log(quat_mul(inv, qs[k])) if valid[k] else np.zeros(3) for k in ks])
        w = w_all[ks]
        a = window_matrix(w, cfg.smoothness)
        x = np.linalg.solve(a, w[:, None] * y)
        x0 = x[ks == t][0]
        out[t] = quat_normalize(quat_mul(qs[t], quat_exp(x0)))
        change[t] = float(np.degrees(np.linalg.norm(x0)))
    return out, change


def temporal_solve(frame_rot: dict, cfg: TemporalSolverConfig | None = None) -> tuple[dict, dict]:
    cfg = cfg or TemporalSolverConfig()
    bones = frame_rot["meta"]["bones"]
    frames = frame_rot["frames"]
    rot = np.array([f["rotations"] for f in frames], dtype=float)  # (T, N, 4)
    conf = np.array([f["confidence"] for f in frames], dtype=float)
    valid = np.array([f["valid"] for f in frames])
    out = rot.copy()
    report = {"config": cfg.__dict__, "bones": {}}
    for b, name in enumerate(bones):
        out[:, b], change = smooth_bone(rot[:, b], conf[:, b], valid, cfg)
        report["bones"][name] = {"mean_change_deg": float(change[valid].mean()) if valid.any() else 0.0,
                                 "max_change_deg": float(change.max()),
                                 "max_change_frame": int(change.argmax())}
    result = {"meta": {**frame_rot["meta"], "temporal_config": cfg.__dict__},
              "frames": [{**f, "rotations": out[t].tolist()} for t, f in enumerate(frames)]}
    return result, report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="altline temporal solver")
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args(argv)
    rd = run_dir(args.run_id)
    result, report = temporal_solve(read_json(rd / "solver" / "frame_rotations.json"))
    print(write_json(rd / "solver" / "raw_rotations.json", result))
    print(write_json(rd / "solver" / "temporal_report.json", report))


if __name__ == "__main__":
    main()
