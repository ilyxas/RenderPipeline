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
    # One-Euro filter on canonical joint positions (filters.py), forward-backward.
    filter_enabled: bool = True
    filter_min_cutoff_hz: float = 1.5
    filter_beta: float = 2.0          # cutoff rises by beta Hz per m/s of joint speed


@dataclass
class SolverConfig:
    min_visibility: float = 0.3
    # Twist from the elbow/knee bend plane is only trusted once the limb is bent this much.
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
class Config:
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    # Review gate: angular acceleration of any bone's local rotation above this is jitter.
    stability_gate_deg_s2: float = 3000.0

    def to_dict(self) -> dict:
        return asdict(self)
