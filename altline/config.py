"""All altline thresholds in one place. Values are generic (no per-clip constants)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class CalibrationConfig:
    window_frames: int = 8            # length of the rest-pose baseline interval
    min_visibility: float = 0.5       # joints below this are ignored for lengths
    # Rest pose: limbs close to straight. MediaPipe world landmarks report ~20-35 deg of
    # bend for visibly straight standing limbs (depth bias), hence the generous bounds.
    max_elbow_bend_deg: float = 45.0
    max_knee_bend_deg: float = 35.0
    # The hinge bend measured in a found rest baseline is treated as tracking bias and
    # removed by remapping [bias, 180] -> [0, 180] (see solver.debias_bend).
    hinge_bias_correction: bool = True
    max_hinge_bias_deg: float = 30.0  # a larger "rest bend" is not tracker bias; cap it
    min_arm_abduction_deg: float = 10.0   # A-pose .. T-pose band, measured from the trunk
    max_arm_abduction_deg: float = 100.0
    # Rebuild landmark depth from the 2D image points and constant actor lengths (depth.py).
    depth_from_2d: bool = True
    # One-Euro filter on canonical joint positions (filters.py), forward-backward. Off since
    # Phase 2: the Butterworth stage (ButterworthConfig) smooths the observations instead.
    filter_enabled: bool = False
    filter_min_cutoff_hz: float = 1.5
    filter_beta: float = 2.0          # cutoff rises by beta Hz per m/s of joint speed


@dataclass
class SolverConfig:
    min_visibility: float = 0.3
    # Twist from the elbow/knee bend plane is only trusted once the limb is bent this much.
    # v4 tried 5..20 deg (v3 review C2: the twist snaps on small elbow wobble). It cut frames
    # over the gate but made the pose worse (df1e left forearm 2D error p95 10 -> 24 deg,
    # 690b 6 -> 10 deg; reports/v4_overview.md), so it is back to 2..10 deg; C2 stays open.
    bend_twist_full_deg: float = 10.0
    bend_twist_zero_deg: float = 2.0
    # Thigh: bend-plane twist fades in over this (de-biased) knee bend; below it the foot
    # heading drives the thigh twist.
    knee_twist_zero_deg: float = 10.0
    knee_twist_full_deg: float = 30.0


@dataclass
class LimitsConfig:
    elbow_flex_deg: tuple[float, float] = (0.0, 140.0)
    forearm_roll_deg: float = 90.0
    wrist_deg: float = 90.0           # wrist twist and each wrist swing component
    knee_flex_deg: tuple[float, float] = (0.0, 150.0)
    knee_twist_deg: float = 30.0
    # Excess forearm roll / knee twist goes to the parent bone while the hinge flex is below this.
    twist_transfer_max_flex_deg: float = 20.0
    # A frame is a spike if it departs from the slerp of its neighbours by more than this.
    spike_deg: float = 25.0
    # A clamp that has to move a joint further than this marks the frame as a tracking
    # anomaly: it is bridged by interpolation instead of being kept as a clamped pose.
    anomaly_clamp_deg: float = 30.0
    max_bridge_frames: int = 12


@dataclass
class ButterworthConfig:
    """Task 6: zero-phase Butterworth low-pass on the raw observations (filters.butterworth_pose)."""
    enabled: bool = True
    order: int = 2                    # per pass; filtfilt doubles it
    # Per-joint cutoff by Winter's residual analysis, clamped to the joint group's band.
    # Distal joints (wrists, fingers, feet) carry the most tracker noise, so their band is lowest.
    band_core_hz: tuple[float, float] = (2.0, 6.0)       # face, shoulders, hips
    band_proximal_hz: tuple[float, float] = (2.5, 6.0)   # elbows, knees
    band_distal_hz: tuple[float, float] = (2.0, 5.0)     # wrists, fingers, ankles, heels, toes


@dataclass
class FlipGuardConfig:
    """Task 7: palm / limb flip guard on the calibrated landmarks (flip_guard.py)."""
    enabled: bool = True
    max_step_deg: float = 45.0        # a limb vector turning more than this in one frame is a flip
    neighbors: int = 3                # clean frames on each side for the Hermite spline
    max_run_frames: int = 8           # a longer "flip" is real motion: accept it


@dataclass
class TemporalSolverConfig:
    """Task 8: sliding-window optimisation of the per-frame solution (solver_temporal.py)."""
    window: int = 5                   # frames t-2 .. t+2
    smoothness: float = 2.0           # weight of the velocity-change (2nd difference) term


@dataclass
class DamperConfig:
    """Task 9: acceleration damper on the final local rotations (damper.py)."""
    enabled: bool = True
    onset_ratio: float = 0.7          # damping starts at this fraction of the stability gate
    brake_ratio: float = 0.5          # catch-up speed is planned with this fraction of the gate


@dataclass
class Config:
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    butterworth: ButterworthConfig = field(default_factory=ButterworthConfig)
    flip_guard: FlipGuardConfig = field(default_factory=FlipGuardConfig)
    temporal: TemporalSolverConfig = field(default_factory=TemporalSolverConfig)
    damper: DamperConfig = field(default_factory=DamperConfig)
    # Review gate: angular acceleration of any bone's local rotation above this is jitter.
    stability_gate_deg_s2: float = 3000.0

    def to_dict(self) -> dict:
        return asdict(self)
