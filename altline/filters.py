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
