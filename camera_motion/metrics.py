"""Phase-1 evaluators (no learning required).

Mode metrics (E1a/E1b only)
    * valid-mode coverage: how many of the 4 target modes appear among K samples
    * 5-category frequency over ALL K samples (4 modes + unassigned), never renormalised
    * TV distance to the 5-way target (0.42, 0.28, 0.18, 0.12, 0.0)
    * per-mode error, minority-mode error, unassigned rate
    * conditional-on-assignment frequencies (SECONDARY, labelled as such)

Endpoint metrics: translation distance and SO(3) geodesic angle, in world frame.
Smoothness: translational acceleration/jerk and angular velocity/acceleration with
dt = 1/(L-1). Report as ratios against GT, never as "lower is better" alone.
E1b orientation: continuous latent -> per-condition DISTRIBUTION comparison of
excursion features with 1D Wasserstein. No GT/model sample-index pairing anywhere.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats

from . import geometry as G
from .config import CATEGORIES, FROZEN, MODE_NAMES, UNASSIGNED_ID

TARGET5 = np.array(FROZEN.modes.target5())
MINORITY_MODE = int(np.argmin(TARGET5[:4]))  # long_cw+far (0.12)


# ---------------------------------------------------------------------------
# Mode coverage / calibration
# ---------------------------------------------------------------------------
def frequency5(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        raise ValueError("no samples")
    if labels.min() < 0 or labels.max() > UNASSIGNED_ID:
        raise ValueError("labels must be in 0..4")
    return np.bincount(labels, minlength=5)[:5] / labels.size


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(np.asarray(p) - np.asarray(q))))


def mode_report(labels: np.ndarray, target5: np.ndarray = TARGET5, min_k: int = 50) -> dict:
    """Metrics for the K samples of ONE condition."""
    labels = np.asarray(labels, dtype=int)
    K = labels.size
    f = frequency5(labels)
    counts = np.bincount(labels, minlength=5)[:5]
    assigned = counts[:4].sum()
    cond_assigned = (counts[:4] / assigned).tolist() if assigned > 0 else [float("nan")] * 4
    return {
        "K": int(K),
        "sample_count_ok": bool(K >= min_k),
        "counts5": counts.tolist(),
        "freq5": f.tolist(),
        "coverage4": int(np.sum(counts[:4] > 0)),
        "tv5": total_variation(f, target5),
        "per_mode_error": (f - target5).tolist(),
        "minority_mode": CATEGORIES[MINORITY_MODE],
        "minority_error": float(f[MINORITY_MODE] - target5[MINORITY_MODE]),
        "unassigned_rate": float(f[UNASSIGNED_ID]),
        "secondary_freq_given_assigned": cond_assigned,
    }


def aggregate_condition_reports(reports: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """Pool per-condition reports. Uncertainty = bootstrap over CONDITIONS."""
    ids = sorted(reports)
    tv = np.array([reports[c]["tv5"] for c in ids])
    cov = np.array([reports[c]["coverage4"] for c in ids])
    un = np.array([reports[c]["unassigned_rate"] for c in ids])
    mino = np.array([reports[c]["minority_error"] for c in ids])
    freq = np.array([reports[c]["freq5"] for c in ids])
    rng = np.random.default_rng(seed)

    def ci(x):
        if len(x) < 2:
            return [float(x.mean()), float(x.mean())]
        bs = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
        return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]

    return {
        "n_conditions": len(ids),
        "tv5_mean": float(tv.mean()), "tv5_ci95": ci(tv),
        "coverage4_mean": float(cov.mean()), "full_coverage_rate": float(np.mean(cov == 4)),
        "unassigned_rate_mean": float(un.mean()), "unassigned_ci95": ci(un),
        "minority_error_mean": float(mino.mean()), "minority_error_ci95": ci(mino),
        "pooled_freq5": freq.mean(axis=0).tolist(),
        "any_condition_under_min_k": bool(any(not reports[c]["sample_count_ok"] for c in ids)),
    }


def confusion_matrix(true_modes: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    """rows = true mode 0..3, cols = predicted 0..4 (incl. unassigned)."""
    M = np.zeros((4, 5), dtype=int)
    for t, p in zip(true_modes, pred_labels):
        M[int(t), int(p)] += 1
    return M


# ---------------------------------------------------------------------------
# Model-evaluation contract (unit-testable without any trained model)
# ---------------------------------------------------------------------------
SampleFn = Callable[[dict, int, np.random.Generator], tuple]


def evaluate_mode_contract(sample_fn: SampleFn, public_conds: dict, family: str, K: int,
                           seed: int = 0, min_k: int = 50) -> dict:
    """For each held-out public condition draw K samples and score them.

    ``sample_fn(public_condition_row, K, rng) -> (R (K,L,3,3), p (K,L,3))`` in WORLD frame.
    The function receives ONLY public fields (start/end pose, subject, shot).
    """
    from .mode_classifier import classify_batch

    rng = np.random.default_rng(seed)
    reports, endpoint = {}, []
    n = len(public_conds["condition_id"])
    for i in range(n):
        row = {k: public_conds[k][i] for k in public_conds}
        R, p = sample_fn(row, K, rng)
        subj = np.broadcast_to(row["subject"], (K, 3))
        lab = classify_batch(p, subj, family)["label"]
        reports[str(row["condition_id"])] = mode_report(lab, min_k=min_k)
        ep = endpoint_errors(R, p, row["end_pose"])
        endpoint.append(ep)
    e_p = np.concatenate([e["e_p"] for e in endpoint])
    e_R = np.concatenate([e["e_R_deg"] for e in endpoint])
    return {
        "per_condition": reports,
        "pooled": aggregate_condition_reports(reports),
        "endpoint": {"e_p_mean": float(e_p.mean()), "e_p_median": float(np.median(e_p)),
                     "e_R_deg_mean": float(e_R.mean()), "e_R_deg_median": float(np.median(e_R))},
    }


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
def endpoint_errors(R_hat: np.ndarray, p_hat: np.ndarray, end_pose: np.ndarray) -> dict:
    """R_hat (N,L,3,3), p_hat (N,L,3) WORLD frame; end_pose (4,4) or (N,4,4)."""
    R_star, p_star = G.split_pose(end_pose)
    e_p = np.linalg.norm(p_hat[:, -1] - p_star, axis=-1)
    e_R = G.geodesic_angle(R_star, R_hat[:, -1])
    return {"e_p": e_p, "e_R_deg": np.degrees(e_R)}


def start_errors(R_hat, p_hat, start_pose) -> dict:
    R0, p0 = G.split_pose(start_pose)
    return {"e_p": np.linalg.norm(p_hat[:, 0] - p0, axis=-1),
            "e_R_deg": np.degrees(G.geodesic_angle(R0, R_hat[:, 0]))}


# ---------------------------------------------------------------------------
# Smoothness
# ---------------------------------------------------------------------------
def smoothness(R: np.ndarray, p: np.ndarray, dt: float | None = None) -> dict:
    """Per-path mean magnitudes. R (N,L,3,3), p (N,L,3)."""
    L = p.shape[-2]
    dt = dt or 1.0 / (L - 1)
    v = np.diff(p, axis=-2) / dt
    a = np.diff(v, axis=-2) / dt
    j = np.diff(a, axis=-2) / dt
    w = G.angular_velocity_body(R, dt)
    alpha = np.diff(w, axis=-2) / dt
    return {
        "trans_acc": np.linalg.norm(a, axis=-1).mean(axis=-1),
        "trans_jerk": np.linalg.norm(j, axis=-1).mean(axis=-1),
        "ang_vel": np.linalg.norm(w, axis=-1).mean(axis=-1),
        "ang_acc": np.linalg.norm(alpha, axis=-1).mean(axis=-1),
    }


def jerk_profile(p: np.ndarray, dt: float | None = None) -> np.ndarray:
    """Median |translational jerk| at each jerk index (L-3 values). Locates where excess
    jerk lives, e.g. at the anchored start (index 0 involves frames 0..3)."""
    L = p.shape[-2]
    dt = dt or 1.0 / (L - 1)
    j = np.diff(p, n=3, axis=-2) / dt**3
    return np.median(np.linalg.norm(j, axis=-1), axis=0)


def smoothness_ratio(gen: dict, ref: dict) -> dict:
    """median(gen)/median(ref) per statistic; ~1 means GT-like smoothness."""
    return {k: float(np.median(gen[k]) / max(np.median(ref[k]), 1e-12)) for k in gen}


# ---------------------------------------------------------------------------
# E1b continuous orientation excursions
# ---------------------------------------------------------------------------
EXCURSION_FEATURES = ("max_excursion_deg", "peak_time", "signed_pan_peak_deg", "signed_roll_peak_deg")


def orientation_excursion_features(R: np.ndarray, p: np.ndarray, subject: np.ndarray) -> dict:
    """Features of R relative to the look-at rotation implied by the path's OWN positions.

    Uses only generated poses + public subject (no latent parameters).
    R (N,L,3,3), p (N,L,3), subject (N,3) or (3,).
    """
    subject = np.broadcast_to(subject, p.shape[:-2] + (3,))
    R_look = G.look_at(p, subject[..., None, :])
    R_off = np.swapaxes(R_look, -1, -2) @ R
    w = G.so3_log(R_off)  # (N,L,3) camera-local rotation vector
    ang = np.linalg.norm(w, axis=-1)
    k = np.argmax(ang, axis=-1)
    idx = np.arange(len(k))
    L = p.shape[-2]
    w_peak = w[idx, k]
    return {
        "max_excursion_deg": np.degrees(ang[idx, k]),
        "peak_time": k / (L - 1),
        "signed_pan_peak_deg": np.degrees(w_peak[:, 1]),
        "signed_roll_peak_deg": np.degrees(w_peak[:, 2]),
    }


def excursion_distribution_distance(feat_gen: dict, feat_ref: dict) -> dict:
    """1D Wasserstein per feature between two INDEPENDENT sample sets (unpaired)."""
    return {k: float(stats.wasserstein_distance(feat_gen[k], feat_ref[k])) for k in EXCURSION_FEATURES}


# ---------------------------------------------------------------------------
# metrics-v2: conditional calibration and E1b support
# ---------------------------------------------------------------------------
def conditional_dispersion(counts5: np.ndarray, target5: np.ndarray = TARGET5, n_sim: int = 4000,
                           seed: int = 0) -> dict:
    """Is p(mode | c) the SAME declared distribution for every condition?

    Pooled frequencies can match the target while individual conditions are skewed in
    opposite directions. Statistic: chi2 = sum_c sum_{k<4} (n_ck - K t_k)^2 / (K t_k),
    compared with its distribution under a perfectly calibrated sampler (multinomial
    draws at the same K and number of conditions). ``excess_ratio`` ~ 1 is calibrated.
    ``unassigned`` samples count as missing mass in n_ck (they inflate chi2).
    """
    counts5 = np.asarray(counts5, dtype=float)
    n_cond = counts5.shape[0]
    K = counts5.sum(1)
    if not np.all(K == K[0]):
        raise ValueError("conditional_dispersion expects equal K per condition")
    K = int(K[0])
    t = np.asarray(target5[:4], dtype=float)
    exp = K * t

    def chi2(n4):
        return float(np.sum((n4 - exp) ** 2 / exp))

    obs = chi2(counts5[:, :4])
    rng = np.random.default_rng(seed)
    sims = np.array([chi2(rng.multinomial(K, t, size=n_cond)) for _ in range(n_sim)])
    p_long = counts5[:, 2:4].sum(1) / K
    p_far = (counts5[:, 1] + counts5[:, 3]) / K
    t_long, t_far = t[2:].sum(), t[1] + t[3]
    return {
        "chi2": obs, "oracle_mean": float(sims.mean()), "oracle_q95": float(np.quantile(sims, 0.95)),
        "p_value": float((np.sum(sims >= obs) + 1) / (n_sim + 1)),
        "excess_ratio": float(obs / sims.mean()),
        "sd_p_long": float(p_long.std()), "binomial_sd_p_long": float(np.sqrt(t_long * (1 - t_long) / K)),
        "sd_p_far": float(p_far.std()), "binomial_sd_p_far": float(np.sqrt(t_far * (1 - t_far) / K)),
        "n_conditions": n_cond, "K": K,
    }


def dispersion_decomposition(counts5: np.ndarray, target5: np.ndarray = TARGET5, n_sim: int = 4000,
                             seed: int = 0) -> dict:
    """Exact split of the conditional chi2 into a global-bias part and a between-condition part.

    With p_ck = n_ck / K and pbar_k = mean_c p_ck (cross term vanishes):
        chi2 = K * C * sum_k (pbar_k - t_k)^2 / t_k        (bias:    pooled frequencies off target)
             + K * sum_c sum_k (p_ck - pbar_k)^2 / t_k     (between: conditions differ from each other)
    Each part is compared with its distribution under a calibrated multinomial sampler
    (same K and C). Bias alone can be large with between ~ oracle, and vice versa.
    """
    counts5 = np.asarray(counts5, dtype=float)
    C = counts5.shape[0]
    K = counts5.sum(1)
    if not np.all(K == K[0]):
        raise ValueError("equal K per condition required")
    K = float(K[0])
    t = np.asarray(target5[:4], dtype=float)

    def parts(n4):
        p = n4 / K
        pbar = p.mean(0)
        bias = K * C * np.sum((pbar - t) ** 2 / t)
        between = K * np.sum((p - pbar) ** 2 / t)
        return bias, between

    b_obs, w_obs = parts(counts5[:, :4])
    rng = np.random.default_rng(seed)
    sims = np.array([parts(rng.multinomial(int(K), t, size=C).astype(float)) for _ in range(n_sim)])
    out = {"n_conditions": C, "K": int(K)}
    for name, obs, col in (("bias", b_obs, 0), ("between", w_obs, 1)):
        s = sims[:, col]
        out[name] = {"chi2": float(obs), "oracle_mean": float(s.mean()), "oracle_q95": float(np.quantile(s, 0.95)),
                     "excess_ratio": float(obs / s.mean()),
                     "p_value": float((np.sum(s >= obs) + 1) / (n_sim + 1))}
    pbar = (counts5[:, :4] / K).mean(0)
    out["pooled_minus_target"] = (pbar - t).tolist()
    out["unassigned_mass"] = float(counts5[:, 4].sum() / counts5.sum())
    return out


def excursion_support_violation(feats: dict, cfg=FROZEN, tol_deg: float = 1.0) -> dict:
    """Fraction of E1b samples whose excursion features fall outside the DECLARED
    generator support (config ranges, not per-sample latents). Peak time gets a
    one-frame tolerance because features are frame-quantised."""
    lo, hi = cfg.latent.offset_amp_deg
    t0, t1 = cfg.latent.offset_peak_time
    frame = 1.0 / (cfg.split.L - 1)
    mx, pt = feats["max_excursion_deg"], feats["peak_time"]
    return {
        "amp_below_support": float(np.mean(mx < lo - tol_deg)),
        "amp_above_support": float(np.mean(mx > hi + tol_deg)),
        "amp_outside_support": float(np.mean((mx < lo - tol_deg) | (mx > hi + tol_deg))),
        "peak_time_outside_support": float(np.mean((pt < t0 - frame) | (pt > t1 + frame))),
    }


# ---------------------------------------------------------------------------
# Data audits
# ---------------------------------------------------------------------------
def mode_condition_independence(ds: dict, n_bins: int = 4) -> dict:
    """Chi-square test of hidden mode vs quantile bins of each public condition scalar."""
    out = {}
    cp = ds["condition_params"]
    cols = {"phi0": cp[:, 0], "r0": cp[:, 1], "r1": cp[:, 2], "h": cp[:, 3],
            "subject_x": ds["subject"][:, 0], "subject_y": ds["subject"][:, 1]}
    m = ds["latent_mode"]
    for name, x in cols.items():
        edges = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])
        b = np.digitize(x, edges)
        table = np.zeros((n_bins, 4), dtype=int)
        for bi, mi in zip(b, m):
            table[bi, mi] += 1
        chi2, pval, _, _ = stats.chi2_contingency(table)
        out[name] = {"chi2": float(chi2), "p_value": float(pval)}
    return out


def rotation_diagnostics(R: np.ndarray) -> dict:
    return G.rotation_validity(R)


__all__ = [n for n in dir() if not n.startswith("_")]
