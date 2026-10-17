"""Task 9: angular-acceleration damper on the final local rotations.

Input: <run>/limits/clean_rotations.json
Output: <run>/damper/damped_rotations.json (same format; animation.py reads it) and
        <run>/damper/damper_report.json

Per bone, angular velocity w_t = log(q_{t-1}^-1 q_t) * fps (local frame) and acceleration
a_t = (w_t - w_{t-1}) * fps, the same measure the review gate uses. A follower walks the
clip: at every frame it asks for the input's velocity plus a braked correction toward the
input pose (see follow) and the acceleration that takes. Up to
`onset_ratio` x gate that acceleration is used as is, so smooth motion passes through
unchanged. Above it the magnitude is soft-saturated,

    |a'| = onset + (gate - onset) * (1 - exp(-(|a| - onset) / (gate - onset)))

an exponential damping multiplier |a'|/|a| that approaches the gate but never reaches it and
never drops the requested change to zero, so the joint keeps moving toward the input (no
freezing); it only arrives a few frames later. A one-way follower lags, so the clip is
damped forward and backward and the two are averaged (slerp 0.5), which cancels the lag.

By construction the output stays (close to) under the gate. That makes "the gate passes"
meaningless as a quality claim on its own: the report therefore keeps the input
acceleration, how many frames needed damping and how far the output departs from the input.
"""

from __future__ import annotations

import argparse

import numpy as np

from .config import DamperConfig
from .io_utils import read_json, run_dir, write_json
from .quat import quat_angle_between, quat_conj, quat_exp, quat_log, quat_mul, quat_normalize, slerp


def angular_velocity(qs: np.ndarray, fps: float) -> np.ndarray:
    """(T, 3) local angular velocity in rad/s; row t is between t-1 and t (row 0 = 0)."""
    w = np.zeros((len(qs), 3))
    for t in range(1, len(qs)):
        w[t] = quat_log(quat_mul(quat_conj(qs[t - 1]), qs[t])) * fps
    return w


def angular_acceleration(qs: np.ndarray, fps: float) -> np.ndarray:
    """(T,) |angular acceleration| in deg/s^2 at frame t (between w_t and w_t+1); edges 0."""
    w = angular_velocity(qs, fps)
    a = np.zeros(len(qs))
    if len(qs) > 2:
        a[1:-1] = np.degrees(np.linalg.norm(np.diff(w[1:], axis=0), axis=1)) * fps
    return a


def soft_cap(mag: float, onset: float, gate: float) -> float:
    if mag <= onset:
        return mag
    span = gate - onset
    return onset + span * (1.0 - np.exp(-(mag - onset) / span))


def follow(qs: np.ndarray, fps: float, gate_rad: float, onset_ratio: float, brake_ratio: float
           ) -> tuple[np.ndarray, np.ndarray]:
    """One-way acceleration-limited follower; returns (rotations, damping multiplier per frame).

    The velocity it asks for is the input's own velocity plus a correction toward the input
    pose. The correction is capped at sqrt(2 * brake * gate * |error|), the speed from which
    the remaining error can still be braked to zero within the gate; asking for the full
    error in one frame (the first version) overshot and rang for 30 frames after one spike.
    """
    n = len(qs)
    out = qs.copy()
    mult = np.ones(n)
    onset = onset_ratio * gate_rad
    for t in range(2, n):
        w_prev = quat_log(quat_mul(quat_conj(out[t - 2]), out[t - 1])) * fps
        w_in = quat_log(quat_mul(quat_conj(qs[t - 1]), qs[t])) * fps
        err = quat_log(quat_mul(quat_conj(out[t - 1]), qs[t - 1]))
        e = float(np.linalg.norm(err))
        w_des = w_in
        if e > 1e-9:
            w_des = w_in + err / e * min(e * fps, np.sqrt(2.0 * brake_ratio * gate_rad * e))
        acc = (w_des - w_prev) * fps
        mag = float(np.linalg.norm(acc))
        if mag > onset:
            m = soft_cap(mag, onset, gate_rad) / mag
            acc = acc * m
            mult[t] = m
        out[t] = quat_normalize(quat_mul(out[t - 1], quat_exp((w_prev + acc / fps) / fps)))
    return out, mult


def damp_bone(qs: np.ndarray, fps: float, gate_deg: float, cfg: DamperConfig) -> tuple[np.ndarray, dict]:
    g = np.radians(gate_deg)
    fwd, mf = follow(qs, fps, g, cfg.onset_ratio, cfg.brake_ratio)
    bwd, mb = follow(qs[::-1].copy(), fps, g, cfg.onset_ratio, cfg.brake_ratio)
    bwd, mb = bwd[::-1], mb[::-1]
    out = np.array([quat_normalize(slerp(a, b, 0.5)) for a, b in zip(fwd, bwd)])
    for t in range(1, len(out)):
        if np.dot(out[t], out[t - 1]) < 0:
            out[t] = -out[t]
    dev = np.array([np.degrees(quat_angle_between(a, b)) for a, b in zip(out, qs)])
    damped = (mf < 1.0) | (mb < 1.0)
    a_in, a_out = angular_acceleration(qs, fps), angular_acceleration(out, fps)
    return out, {
        "frames_damped": int(damped.sum()),
        "damped_frames": [int(t) for t in np.flatnonzero(damped)],
        "min_multiplier": float(min(mf.min(), mb.min())),
        "max_deviation_deg": float(dev.max()),
        "max_deviation_frame": int(dev.argmax()),
        "mean_deviation_deg": float(dev.mean()),
        "input_max_acc_deg_s2": float(a_in.max()),
        "input_frames_over_gate": int((a_in > gate_deg).sum()),
        "output_max_acc_deg_s2": float(a_out.max()),
        "output_frames_over_gate": int((a_out > gate_deg).sum()),
    }


def damp(clean: dict, gate_deg: float, cfg: DamperConfig | None = None) -> tuple[dict, dict]:
    cfg = cfg or DamperConfig()
    fps = float(clean["meta"]["fps"])
    bones = clean["meta"]["bones"]
    frames = clean["frames"]
    rot = np.array([f["rotations"] for f in frames], dtype=float)
    out = rot.copy()
    report = {"config": cfg.__dict__, "gate_deg_s2": gate_deg, "bones": {}}
    if cfg.enabled:
        for b, name in enumerate(bones):
            out[:, b], report["bones"][name] = damp_bone(rot[:, b], fps, gate_deg, cfg)
    result = {"meta": {**clean["meta"], "damper_config": cfg.__dict__, "gate_deg_s2": gate_deg},
              "frames": [{**f, "rotations": out[t].tolist()} for t, f in enumerate(frames)]}
    return result, report


def main(argv: list[str] | None = None) -> None:
    from .config import Config

    ap = argparse.ArgumentParser(description="altline acceleration damper")
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args(argv)
    rd = run_dir(args.run_id)
    cfg = Config()
    result, report = damp(read_json(rd / "limits" / "clean_rotations.json"), cfg.stability_gate_deg_s2, cfg.damper)
    print(write_json(rd / "damper" / "damped_rotations.json", result))
    print(write_json(rd / "damper" / "damper_report.json", report))


if __name__ == "__main__":
    main()
