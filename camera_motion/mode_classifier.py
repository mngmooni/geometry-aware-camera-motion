"""Deterministic, label-blind E1 mode classifier (E1a / E1b ONLY).

Input is positions + subject centre + frozen thresholds. It never sees latent mode
labels, model IDs or seeds, and is applied identically to GT and generated paths.

Decision rule (config.ClassifierSpec):
1. Validity: finite coordinates, planar (RMS of z about its median <= ``planarity_rms_tol``),
   every frame at least ``r_min`` from the subject's vertical axis -> else ``invalid_geometry``.
2. Unwrapped azimuth: sum of consecutive signed steps
   ``delta_i = atan2(u_perp_i . u_{i+1}, u_i . u_{i+1})``, CCW positive from +z.
   Any |delta_i| >= max_step -> ``angular_ambiguous`` (aliasing guard).
3. Angular branch: within ``angle_tol`` of +pi/2 -> short_ccw; of -3pi/2 -> long_cw;
   else ``angular_ambiguous``.
4. Radial branch: the *interior radial ratio offset*
   rho = mean_{i in interior} (r_i / r_lin_i) - 1, where r_i is the horizontal distance to
   the subject axis and r_lin is the index-linear interpolation between the path's OWN
   first and last radius (so one threshold works for every condition, and endpoint
   accuracy does not enter the mode label; endpoint error is reported separately).
   rho <= tau_near -> near; rho >= tau_far -> far; gray zone -> ``radial_ambiguous``.
5. Assigned only if both sub-labels pass; otherwise ``unassigned`` + reason code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CATEGORIES, FROZEN, MODE_NAMES, UNASSIGNED_ID, ClassifierSpec


class ClassifierScopeError(ValueError):
    """Raised when the E1 classifier is applied outside E1a/E1b."""


REASONS = ("ok", "invalid_geometry", "angular_ambiguous", "radial_ambiguous")


@dataclass
class ModeResult:
    label: int  # 0..3 valid mode, 4 = unassigned
    reason: str
    delta_phi: float
    rho: float
    max_step: float

    @property
    def name(self) -> str:
        return CATEGORIES[self.label]


def _check_scope(family: str, spec: ClassifierSpec) -> None:
    if family not in spec.families:
        raise ClassifierScopeError(
            f"E1 orbital/radial classifier is not valid for family '{family}'. "
            "Use generic metrics, or define and GT-validate a family-specific classifier."
        )


def unwrapped_azimuth(positions: np.ndarray, subject: np.ndarray):
    u = positions[:, :2] - subject[None, :2]
    a, b = u[:-1], u[1:]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]  # u_perp . u_next with u_perp = (-uy, ux)
    dot = np.sum(a * b, axis=-1)
    steps = np.arctan2(cross, dot)
    return steps, float(np.sum(steps))


def radial_statistic(positions: np.ndarray, subject: np.ndarray, n_exclude: int) -> float:
    r = np.linalg.norm(positions[:, :2] - subject[None, :2], axis=-1)
    L = len(r)
    r_lin = r[0] + (r[-1] - r[0]) * np.linspace(0, 1, L)
    sl = slice(n_exclude, L - n_exclude)
    if L - 2 * n_exclude <= 0:
        return float("nan")  # too few frames -> radial branch cannot pass
    return float(np.mean(r[sl] / r_lin[sl]) - 1.0)


def classify_path(positions, subject, family: str, spec: ClassifierSpec = FROZEN.clf,
                  sweeps=(FROZEN.modes.sweep_short, FROZEN.modes.sweep_long)) -> ModeResult:
    _check_scope(family, spec)
    P = np.asarray(positions, dtype=float)
    s = np.asarray(subject, dtype=float)
    nan = float("nan")
    if P.ndim != 2 or P.shape[1] != 3 or not np.all(np.isfinite(P)) or not np.all(np.isfinite(s)):
        return ModeResult(UNASSIGNED_ID, "invalid_geometry", nan, nan, nan)
    if np.sqrt(np.mean((P[:, 2] - np.median(P[:, 2])) ** 2)) > spec.planarity_rms_tol:
        return ModeResult(UNASSIGNED_ID, "invalid_geometry", nan, nan, nan)
    r = np.linalg.norm(P[:, :2] - s[None, :2], axis=-1)
    if np.min(r) < spec.r_min:
        return ModeResult(UNASSIGNED_ID, "invalid_geometry", nan, nan, nan)

    steps, dphi = unwrapped_azimuth(P, s)
    mstep = float(np.max(np.abs(steps)))
    rho = radial_statistic(P, s, spec.n_exclude_end)
    if mstep >= spec.max_step:
        return ModeResult(UNASSIGNED_ID, "angular_ambiguous", dphi, rho, mstep)

    if abs(dphi - sweeps[0]) <= spec.angle_tol:
        ang = 0
    elif abs(dphi - sweeps[1]) <= spec.angle_tol:
        ang = 1
    else:
        return ModeResult(UNASSIGNED_ID, "angular_ambiguous", dphi, rho, mstep)

    if rho <= spec.tau_near:
        rad = 0
    elif rho >= spec.tau_far:
        rad = 1
    else:
        return ModeResult(UNASSIGNED_ID, "radial_ambiguous", dphi, rho, mstep)
    return ModeResult(2 * ang + rad, "ok", dphi, rho, mstep)


def classify_batch(positions, subjects, family: str, spec: ClassifierSpec = FROZEN.clf):
    """positions (N,L,3), subjects (N,3) -> dict of arrays."""
    _check_scope(family, spec)
    res = [classify_path(P, s, family, spec) for P, s in zip(positions, subjects)]
    return {
        "label": np.array([r.label for r in res], dtype=int),
        "reason": np.array([r.reason for r in res]),
        "delta_phi": np.array([r.delta_phi for r in res]),
        "rho": np.array([r.rho for r in res]),
        "max_step": np.array([r.max_step for r in res]),
    }


def classify_dataset(ds: dict, spec: ClassifierSpec = FROZEN.clf):
    """Classify a generated/GT dataset dict. Uses only positions, subject and family."""
    fam = np.unique(ds["family"])
    if len(fam) != 1:
        raise ClassifierScopeError(f"mixed families in one call: {fam}")
    return classify_batch(ds["p"], ds["subject"], str(fam[0]), spec)


__all__ = ["classify_path", "classify_batch", "classify_dataset", "ClassifierScopeError",
           "ModeResult", "MODE_NAMES", "CATEGORIES", "REASONS"]
