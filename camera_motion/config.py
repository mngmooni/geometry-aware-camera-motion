"""Frozen Phase-1 configuration: data ranges, mode probabilities, classifier thresholds.

Everything here is fixed **before** any model sample is inspected. Changing a
classifier threshold after reading generated results invalidates mode metrics.
``FROZEN`` is written next to every generated dataset as ``config_frozen.json``.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

# Category order used everywhere (index = integer label). 4 valid modes + unassigned.
MODE_NAMES = (
    "short_ccw+near",
    "short_ccw+far",
    "long_cw+near",
    "long_cw+far",
)
UNASSIGNED = "unassigned"
CATEGORIES = MODE_NAMES + (UNASSIGNED,)
UNASSIGNED_ID = len(MODE_NAMES)  # 4


@dataclass(frozen=True)
class ModeSpec:
    p_short: float = 0.70  # short_ccw vs long_cw
    p_near: float = 0.60  # near vs far
    sweep_short: float = math.pi / 2  # +90 deg, CCW
    sweep_long: float = -3 * math.pi / 2  # -270 deg, CW (same final bearing)

    def joint(self) -> tuple:
        ps, pn = self.p_short, self.p_near
        return (ps * pn, ps * (1 - pn), (1 - ps) * pn, (1 - ps) * (1 - pn))

    def target5(self) -> tuple:
        return self.joint() + (0.0,)


@dataclass(frozen=True)
class ConditionRanges:
    """Public-condition sampling ranges (metres / radians)."""
    subject_xy_train: tuple = (-3.0, 3.0)  # square, in-distribution
    subject_xy_ood_annulus: tuple = (5.0, 7.0)  # OOD-position: |xy| in this range
    subject_z: tuple = (0.0, 1.0)
    radius: tuple = (3.0, 5.0)  # start / end radius from subject (independent)
    height: tuple = (0.5, 2.0)  # camera height above subject (constant along E1 path)
    start_bearing: tuple = (0.0, 2 * math.pi)


@dataclass(frozen=True)
class LatentRanges:
    """Hidden, non-public trajectory variables."""
    radial_bump_abs: tuple = (0.30, 0.45)  # |relative radial bump|; near = -, far = +
    time_warp: tuple = (-0.5, 0.5)  # tau(u) = u + w sin(2 pi u)/(2 pi), monotone for |w|<1
    # E1b orientation offsets (continuous latent)
    offset_amp_deg: tuple = (20.0, 60.0)
    offset_peak_time: tuple = (0.3, 0.7)
    offset_axes: tuple = ("pan", "roll")  # camera-local y (pan) or z (roll)


@dataclass(frozen=True)
class ClassifierSpec:
    """Frozen E1 deterministic classifier thresholds (label-blind).

    Version history (all changes made BEFORE any model sample was inspected):
      e1-v1.0  planarity = max |z - median z| <= 0.5 m
      e1-v1.1  planarity = RMS(z - median z) <= 0.25 m. The robustness probe showed
               the max statistic rejected 36% of GT paths under i.i.d. z-jitter of
               sigma = 0.2 m (no mode errors, but geometry rejections unrelated to
               mode). RMS is robust to isolated jitter and still rejects cranes.
    """
    version: str = "e1-v1.1"
    families: tuple = ("E1a", "E1b")
    r_min: float = 0.5  # any frame closer than this to the subject axis -> invalid
    planarity_rms_tol: float = 0.25  # RMS(z - median z) in metres; E1 is planar by construction
    max_step: float = math.pi - math.pi / 3  # |delta_i| must stay below this (aliasing guard)
    angle_tol: float = math.pi / 3  # +-60 deg around each designed sweep
    n_exclude_end: int = 2  # frames excluded at each end for the radial statistic
    tau_near: float = -0.08  # rho <= tau_near -> near
    tau_far: float = 0.08  # rho >= tau_far  -> far ; gray zone in between
    endpoint_off_task: float | None = None  # not used (predeclared: disabled)


METRICS_VERSION = "m2"  # m1: Phase-2 gate-2 report; m2 adds conditional-calibration dispersion,
#                        E1b support-violation rates, matched oracle W1 floors, 200 E1b GT refs


@dataclass(frozen=True)
class SplitSpec:
    L: int = 32
    n_train_e1a: int = 1000
    n_train_e1b: int = 500
    paths_per_train_condition: int = 1
    n_audit_conditions_e1a: int = 4
    n_audit_conditions_e1b: int = 2
    audit_group_size: int = 100  # exact 42/28/18/12 allocation
    n_eval_id_conditions: int = 50
    n_eval_ood_conditions: int = 20
    eval_gt_ref_per_condition: int = 20  # distributional reference only, never paired
    # metrics-v2 (registered after fm_abs_pose gate 2, before any rel_* run): E1b
    # orientation distributions need a larger GT reference per condition to have power.
    eval_gt_ref_per_condition_e1b: int = 200
    n_e0_per_kind: int = 20
    n_e2_stub: int = 50


@dataclass(frozen=True)
class Config:
    seed: int = 20261104
    modes: ModeSpec = field(default_factory=ModeSpec)
    cond: ConditionRanges = field(default_factory=ConditionRanges)
    latent: LatentRanges = field(default_factory=LatentRanges)
    clf: ClassifierSpec = field(default_factory=ClassifierSpec)
    split: SplitSpec = field(default_factory=SplitSpec)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mode_names"] = list(MODE_NAMES)
        d["categories"] = list(CATEGORIES)
        d["target5"] = list(self.modes.target5())
        d["conventions"] = {
            "world": "right-handed, z-up",
            "camera": "OpenCV: x right, y down, z forward",
            "pose": "camera-to-world (R columns = camera axes in world, p = centre)",
            "offset": "R = R_lookat @ R_offset (camera-local, right-multiplied)",
            "increments": "body frame: dR=R_i^T R_{i+1}, dp=R_i^T(p_{i+1}-p_i)",
            "time": "frame index i=0..L-1, dt = 1/(L-1)",
            "azimuth": "CCW positive viewed from +z",
            "radial_statistic": "interior radial ratio offset: rho = mean_interior(r_i / r_lin_i) - 1, r_lin = index-linear interp of r_0, r_{L-1}",
        }
        return d


FROZEN = Config()
