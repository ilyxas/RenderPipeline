"""One-Euro filter (Casiez et al. 2012) for landmark trajectories.

Adaptive low-pass: a low cutoff while a point is slow (kills jitter), rising with speed so
fast motion is not lagged. Run on canonical joint positions before the solver, so the
geometry the solver sees stays consistent across joints.
"""

from __future__ import annotations

import numpy as np


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2 * np.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


def one_euro(x: np.ndarray, t: np.ndarray, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> np.ndarray:
    """Filter a (T, D) trajectory sampled at times t (seconds). NaN rows reset the filter."""
    out = np.full_like(x, np.nan, dtype=float)
    prev = None
    dprev = np.zeros(x.shape[1])
    tprev = None
    for i in range(len(x)):
        if np.isnan(x[i]).any():
            prev = None
            continue
        if prev is None:
            prev, dprev, tprev = x[i].astype(float), np.zeros(x.shape[1]), t[i]
            out[i] = prev
            continue
        dt = max(t[i] - tprev, 1e-6)
        dx = (x[i] - prev) / dt
        dprev = dprev + _alpha(d_cutoff, dt) * (dx - dprev)
        cutoff = min_cutoff + beta * np.linalg.norm(dprev)
        prev = prev + _alpha(cutoff, dt) * (x[i] - prev)
        tprev = t[i]
        out[i] = prev
    return out


def one_euro_zero_phase(x: np.ndarray, t: np.ndarray, min_cutoff: float, beta: float,
                        d_cutoff: float = 1.0) -> np.ndarray:
    """Forward-backward pass: offline clips can afford it, and it cancels the filter's lag."""
    fwd = one_euro(x, t, min_cutoff, beta, d_cutoff)
    back = one_euro(fwd[::-1], t[-1] - t[::-1], min_cutoff, beta, d_cutoff)[::-1]
    return back


# ---------------------------------------------------------------------------------------
# Task 6: zero-phase Butterworth low-pass with per-joint adaptive cutoff
# ---------------------------------------------------------------------------------------

DISTAL = ("wrist", "index", "pinky", "thumb", "ankle", "heel", "foot_index")
PROXIMAL = ("elbow", "knee")
FILTFILT_CORRECTION = 0.802  # Winter: a two-pass Butterworth cuts at fc*0.802 of the design


def joint_group(name: str) -> str:
    if name.endswith(DISTAL):
        return "distal"
    if name.endswith(PROXIMAL):
        return "proximal"
    return "core"


def butter_filtfilt(x: np.ndarray, fps: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    """Zero-phase low-pass of a (T, D) array. The design cutoff is raised by Winter's
    two-pass correction so the effective -3 dB point is `cutoff_hz`."""
    from scipy.signal import butter, filtfilt

    nyq = 0.5 * fps
    wn = min(cutoff_hz / FILTFILT_CORRECTION / nyq, 0.99)
    b, a = butter(order, wn)
    padlen = min(3 * max(len(a), len(b)), len(x) - 1)
    return filtfilt(b, a, x, axis=0, padlen=padlen)


def winter_cutoff(x: np.ndarray, fps: float, order: int = 2, band: tuple[float, float] = (1.0, 8.0)) -> float:
    """Winter's residual analysis: the residual between signal and filtered signal falls
    with the cutoff until only noise remains; a line fitted to that noise-only tail meets
    the residual axis at the noise RMS, and the cutoff whose residual equals it balances
    noise removed against signal distorted. Returned clamped to `band`."""
    nyq = 0.5 * fps
    cut = np.linspace(0.5, 0.9 * nyq, 40)
    resid = np.array([np.sqrt(np.mean((x - butter_filtfilt(x, fps, c, order)) ** 2)) for c in cut])
    tail = cut >= 0.55 * nyq
    slope, intercept = np.polyfit(cut[tail], resid[tail], 1)
    below = np.flatnonzero(resid <= max(intercept, 0.0))
    fc = float(cut[below[0]]) if len(below) else float(cut[-1])
    return float(np.clip(fc, *band))


def _fill_gaps(x: np.ndarray) -> np.ndarray:
    """Linear fill of NaN rows (frames without a detection) so the filter can run."""
    out = x.copy()
    t = np.arange(len(x))
    good = ~np.isnan(x).any(axis=1)
    if good.sum() < 2:
        return out
    for d in range(x.shape[1]):
        out[~good, d] = np.interp(t[~good], t[good], x[good, d])
    return out


def butterworth_pose(raw: dict, cfg) -> tuple[dict, dict]:
    """Filter every joint of a raw_pose.json (world xyz and image xy) with its own cutoff.

    The cutoff is chosen on the image track in pixels (the tracker's cleanest signal) and
    applied to both tracks of that joint. Visibility is not filtered; frames without a
    detection stay without one. Returns (filtered raw_pose-shaped dict, report)."""
    fps = float(raw["meta"]["fps"])
    frames = raw["frames"]
    w, h = raw["meta"].get("width", 1), raw["meta"].get("height", 1)
    detected = np.array([f["detected"] for f in frames])
    names = next(f["joints"].keys() for f in frames if f["detected"])
    bands = {"core": cfg.band_core_hz, "proximal": cfg.band_proximal_hz, "distal": cfg.band_distal_hz}
    out = [{**f, "joints": dict(f["joints"]), "image_joints": dict(f["image_joints"])} for f in frames]
    report = {"order_per_pass": cfg.order, "cutoff_hz": {}, "group": {}}
    if detected.sum() < 8:
        return {"meta": {**raw["meta"], "filter": report}, "frames": out}, report
    for name in names:
        world = np.array([f["joints"][name][:3] if f["detected"] else [np.nan] * 3 for f in frames])
        has_image = all(f["image_joints"] for f in frames if f["detected"])
        image = np.array([[f["image_joints"][name][0] * w, f["image_joints"][name][1] * h]
                          if f["detected"] else [np.nan] * 2 for f in frames]) if has_image else None
        group = joint_group(name)
        probe = _fill_gaps(image if image is not None else world)
        fc = winter_cutoff(probe, fps, cfg.order, bands[group])
        world_f = butter_filtfilt(_fill_gaps(world), fps, fc, cfg.order)
        image_f = butter_filtfilt(_fill_gaps(image), fps, fc, cfg.order) if image is not None else None
        for i, f in enumerate(out):
            if not detected[i]:
                continue
            f["joints"][name] = world_f[i].tolist() + [frames[i]["joints"][name][3]]
            if image_f is not None:
                src = frames[i]["image_joints"][name]
                f["image_joints"][name] = [image_f[i, 0] / w, image_f[i, 1] / h] + list(src[2:])
        report["cutoff_hz"][name] = fc
        report["group"][name] = group
    return {"meta": {**raw["meta"], "filter": report}, "frames": out}, report
