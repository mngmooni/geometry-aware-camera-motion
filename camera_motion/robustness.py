"""Perturbation-robustness probes for the E1 mode classifier (GT paths only).

Two kinds of check:
* **Invariances** (must be exact): rigid motion of the whole scene (rotation about
  world-up + translation), uniform scaling, and denser temporal sampling.
* **Degradation curves**: i.i.d. position jitter (xyz / xy-only / z-only) and smooth
  low-frequency deformation. For each level we report agreement, WRONG-label rate
  (assigned to a different valid mode) and unassigned rate. A safe classifier fails
  towards ``unassigned``, never towards a wrong mode.
"""
from __future__ import annotations

import numpy as np

from . import geometry as G
from .mode_classifier import classify_batch

JITTER_LEVELS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
LOWFREQ_LEVELS = (0.0, 0.1, 0.2, 0.4, 0.6)


def _rates(labels, true):
    return {
        "agree": float(np.mean(labels == true)),
        "wrong": float(np.mean((labels != true) & (labels != 4))),
        "unassigned": float(np.mean(labels == 4)),
    }


def rigid_transform(P, S, rng):
    th = rng.uniform(0, 2 * np.pi, size=len(P))
    Rz = G.so3_exp(np.stack([np.zeros_like(th), np.zeros_like(th), th], -1))
    t = rng.normal(size=(len(P), 3)) * 5
    P2 = np.einsum("nij,nlj->nli", Rz, P) + t[:, None]
    S2 = np.einsum("nij,nj->ni", Rz, S) + t
    return P2, S2


def jitter(P, sigma, rng, axes=(0, 1, 2)):
    P2 = P.copy()
    for a in axes:
        P2[..., a] += rng.normal(size=P.shape[:-1]) * sigma
    return P2


def lowfreq(P, amp, rng, n_modes=3):
    L = P.shape[1]
    u = np.linspace(0, 1, L)
    k = np.arange(1, n_modes + 1)
    basis = np.sin(np.pi * np.outer(k, u))  # zero at both ends: endpoints untouched
    coef = rng.normal(size=(len(P), n_modes, 3)) * amp / np.sqrt(n_modes)
    return P + np.einsum("nkd,kl->nld", coef, basis)


def run_probes(P, S, true, family, seed=0):
    rng = np.random.default_rng(seed)
    base = classify_batch(P, S, family)["label"]
    out = {"baseline": _rates(base, true), "invariance": {}, "curves": {}}

    P2, S2 = rigid_transform(P, S, rng)
    out["invariance"]["rigid_z_rotation_translation"] = bool(
        np.array_equal(classify_batch(P2, S2, family)["label"], base))
    sc = rng.uniform(0.5, 2.0, size=(len(P), 1, 1))
    out["invariance"]["uniform_scale_0.5_2"] = bool(
        np.array_equal(classify_batch(P * sc, S * sc[:, 0], family)["label"], base))

    curves = out["curves"]
    for name, axes in (("jitter_xyz", (0, 1, 2)), ("jitter_xy", (0, 1)), ("jitter_z", (2,))):
        curves[name] = {str(s): _rates(classify_batch(jitter(P, s, rng, axes), S, family)["label"], true)
                        for s in JITTER_LEVELS}
    curves["lowfreq_xyz"] = {str(a): _rates(classify_batch(lowfreq(P, a, rng), S, family)["label"], true)
                             for a in LOWFREQ_LEVELS}
    rev = classify_batch(P[:, ::-1], S, family)
    out["time_reversed"] = {"wrong_rate": float(np.mean((rev["label"] != 4))),
                            "note": "reversed paths have no valid designed sweep -> must be unassigned"}
    out["max_wrong_rate_over_all_curves"] = max(v["wrong"] for c in curves.values() for v in c.values())
    return out


def temporal_resampling_invariance(family="E1a", L_dense=64, n=300, seed=0):
    """Same conditions/latents generated at L=32 and L=64 must get the same label."""
    from dataclasses import replace

    from . import dataset_generator as D
    from .config import FROZEN
    cfg64 = replace(FROZEN, split=replace(FROZEN.split, L=L_dense))
    a = D.generate_e1_train(family, n=n)
    b = D.generate_e1_train(family, n=n, cfg=cfg64)
    la = classify_batch(a["p"], a["subject"], family)["label"]
    lb = classify_batch(b["p"], b["subject"], family)["label"]
    return bool(np.array_equal(la, lb) and np.array_equal(la, a["latent_mode"]))
