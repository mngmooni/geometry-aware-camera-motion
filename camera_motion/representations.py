"""Euclidean trajectory representations for the FM baselines (numpy, model-agnostic).

All three share the SAME tensor shape so one network architecture serves every
variant: ``x`` is (N, T = L-1, C = 9) = 3 translation + 6 rotation (6D code) channels.
The start pose is anchored identically for every variant (never generated).

==========  =================  ==================================================
name        frame              per-step content (i = 1..L-1, or increments 0..L-2)
==========  =================  ==================================================
abs_pose    world              p_i, 6D(R_i)
rel_pose    initial camera     R0^T (p_i - p0), 6D(R0^T R_i)
rel_inc     initial camera     body increments R_i^T (p_{i+1}-p_i), 6D(R_i^T R_{i+1})
==========  =================  ==================================================

Conditioning (all derived from PUBLIC fields only):
* abs_pose:  [p_start, 6D(R_start), p_end, 6D(R_end), subject]            (21 dims)
* rel_*:     [p_end_rel, 6D(R_end_rel), subject_rel, up_rel]              (15 dims)
  where ``up_rel = R0^T e_z`` keeps gravity/orbital-plane information that the
  relative frame would otherwise discard. The relative encoding is invariant to
  world translation and to rotation about world-up; that invariance IS the frame
  contrast being tested. Shot category is constant ("orbit") in E1 and is omitted.
"""
from __future__ import annotations

import numpy as np

from . import geometry as G

T_CHANNELS = 9
REPRESENTATIONS = ("abs_pose", "rel_pose", "rel_inc")


def _pose_feat(R, p):
    return np.concatenate([p, G.rot6d_encode(R)], axis=-1)


def _unpack(x):
    return G.rot6d_decode(x[..., 3:9]), x[..., :3]


def encode_condition(rep: str, start_pose, end_pose, subject) -> np.ndarray:
    """(N,4,4),(N,4,4),(N,3) -> (N, Dc) float64."""
    R0, p0 = G.split_pose(start_pose)
    R1, p1 = G.split_pose(end_pose)
    if rep == "abs_pose":
        return np.concatenate([_pose_feat(R0, p0), _pose_feat(R1, p1), subject], axis=-1)
    if rep in ("rel_pose", "rel_inc"):
        R0t = np.swapaxes(R0, -1, -2)
        R1r = R0t @ R1
        p1r = (R0t @ (p1 - p0)[..., None])[..., 0]
        sr = (R0t @ (subject - p0)[..., None])[..., 0]
        up = R0t @ G.WORLD_UP
        return np.concatenate([_pose_feat(R1r, p1r), sr, up], axis=-1)
    raise ValueError(rep)


def encode_trajectories(rep: str, R, p) -> np.ndarray:
    """World poses (N,L,3,3),(N,L,3) -> x (N, L-1, 9)."""
    if rep == "abs_pose":
        return _pose_feat(R[:, 1:], p[:, 1:])
    Rr, pr = G.to_relative(R, p, R[:, 0], p[:, 0])
    if rep == "rel_pose":
        return _pose_feat(Rr[:, 1:], pr[:, 1:])
    if rep == "rel_inc":
        dR, dp = G.poses_to_increments(Rr, pr, frame="body")
        return _pose_feat(dR, dp)
    raise ValueError(rep)


def decode_trajectories(rep: str, x, start_pose):
    """x (N, L-1, 9) + public start pose (N,4,4) -> world (R (N,L,3,3), p (N,L,3)).

    6D codes are Gram-Schmidt projected, so every decoded rotation is valid.
    """
    R0, p0 = G.split_pose(start_pose)
    N = x.shape[0]
    if rep == "abs_pose":
        Rs, ps = _unpack(x)
        return np.concatenate([R0[:, None], Rs], 1), np.concatenate([p0[:, None], ps], 1)
    if rep == "rel_pose":
        Rr, pr = _unpack(x)
        Rr = np.concatenate([np.broadcast_to(np.eye(3), (N, 1, 3, 3)), Rr], 1)
        pr = np.concatenate([np.zeros((N, 1, 3)), pr], 1)
        R, p = G.to_world(Rr, pr, R0, p0)
        R[:, 0], p[:, 0] = R0, p0  # bit-exact anchor
        return R, p
    if rep == "rel_inc":
        dR, dp = _unpack(x)
        Rr, pr = G.increments_to_poses(np.broadcast_to(np.eye(3), (N, 3, 3)), np.zeros((N, 3)), dR, dp, "body")
        R, p = G.to_world(Rr, pr, R0, p0)
        R[:, 0], p[:, 0] = R0, p0
        return R, p
    raise ValueError(rep)


class Normalizer:
    """Per-channel standardisation fitted on the training split only."""

    def __init__(self, x_mean, x_std, c_mean, c_std):
        self.x_mean, self.x_std = np.asarray(x_mean), np.asarray(x_std)
        self.c_mean, self.c_std = np.asarray(c_mean), np.asarray(c_std)

    @classmethod
    def fit(cls, x, c, eps=1e-6):
        xm, xs = x.reshape(-1, x.shape[-1]).mean(0), x.reshape(-1, x.shape[-1]).std(0)
        cm, cs = c.mean(0), c.std(0)
        return cls(xm, np.maximum(xs, eps), cm, np.maximum(cs, eps))

    def nx(self, x):
        return (x - self.x_mean) / self.x_std

    def ux(self, x):
        return x * self.x_std + self.x_mean

    def nc(self, c):
        return (c - self.c_mean) / self.c_std

    def state(self):
        return {k: getattr(self, k).tolist() for k in ("x_mean", "x_std", "c_mean", "c_std")}

    @classmethod
    def from_state(cls, s):
        return cls(**{k: np.array(v) for k, v in s.items()})
