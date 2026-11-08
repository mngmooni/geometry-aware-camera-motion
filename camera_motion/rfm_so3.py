"""`rfm_so3`: Riemannian flow matching on SO(3)^T x R^{3T} (relative-pose, pose-level).

Matched to `fm_rel_pose` in frame (initial camera), trajectory semantics (poses of
frames 1..L-1, start anchored), conditioning (same 15-d relative condition), network
family, data, optimiser and budget. It differs in the generative process:

* state: rotation matrices R_i in SO(3) and standardised positions x_i in R^3
* prior: Haar-uniform rotations (independent per frame) x N(0, I) positions
  (Euclidean FM uses N(0, I) on 6D codes: a disclosed method-level difference)
* path: rotations follow the geodesic R_t = R0 Exp(t xi), xi = Log(R0^T R1);
        positions follow x_t = (1 - t) x0 + t x1
* target: body angular velocity xi (constant along the geodesic) and x1 - x0
* network input: [x_t, 6D(R_t)] (9 ch), output: [v_x, omega] (6 ch)
* sampler: R <- R Exp(dt omega) (stays on SO(3) exactly), x <- x + dt v
Rotations are never normalised; positions use a Normalizer fitted on training data.
This is a product-manifold model, not a coupled SE(3) flow.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import geometry as G
from . import representations as Rp
from . import so3_torch as S

IN_CH, OUT_CH = 9, 6
PRIOR = "Haar-uniform on SO(3) per frame x isotropic Gaussian N(0, I) on standardised positions"
PATH = "rotation geodesic R0 Exp(t Log(R0^T R1)); position linear; uniform t"


def encode(R_world, p_world):
    """World poses -> relative rotations (N,T,3,3) and positions (N,T,3) for frames 1..L-1."""
    Rr, pr = G.to_relative(R_world, p_world, R_world[:, 0], p_world[:, 0])
    return Rr[:, 1:], pr[:, 1:]


def decode(R_rel, x_n, norm, start_pose):
    """Generated (R_rel, standardised positions) -> world (R (N,L,3,3), p (N,L,3))."""
    R0, p0 = G.split_pose(start_pose)
    N = R_rel.shape[0]
    pr = x_n * norm.x_std + norm.x_mean
    Rr = np.concatenate([np.broadcast_to(np.eye(3), (N, 1, 3, 3)), R_rel], 1)
    pr = np.concatenate([np.zeros((N, 1, 3)), pr], 1)
    R, p = G.to_world(Rr, pr, R0, p0)
    R[:, 0], p[:, 0] = R0, p0
    return R, p


def features(R, x):
    return torch.cat([x, S.rot6d(R)], -1)


def loss(model, R1, x1, c, generator=None):
    B = R1.shape[0]
    R0 = S.uniform(R1.shape[:-2], generator=generator, device=R1.device, dtype=R1.dtype)
    x0 = torch.randn(x1.shape, generator=generator, device=x1.device, dtype=x1.dtype)
    t = torch.rand(B, generator=generator, device=x1.device, dtype=x1.dtype)
    xi = S.log(R0.transpose(-1, -2) @ R1)
    Rt = R0 @ S.exp(t[:, None, None] * xi)
    xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
    out = model(features(Rt, xt), t, c)
    return F.mse_loss(out, torch.cat([x1 - x0, xi], -1))


@torch.no_grad()
def sample(model, c, T, nfe=50, method="euler", generator=None):
    K = c.shape[0]
    R = S.uniform((K, T), generator=generator, device=c.device)
    x = torch.randn(K, T, 3, generator=generator, device=c.device)
    if method == "euler":
        dt = 1.0 / nfe
        for i in range(nfe):
            v = model(features(R, x), torch.full((K,), i * dt, device=c.device), c)
            x = x + dt * v[..., :3]
            R = R @ S.exp(dt * v[..., 3:])
    elif method == "midpoint":
        steps = nfe // 2
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((K,), i * dt, device=c.device)
            v1 = model(features(R, x), t, c)
            Rm, xm = R @ S.exp(0.5 * dt * v1[..., 3:]), x + 0.5 * dt * v1[..., :3]
            v2 = model(features(Rm, xm), t + 0.5 * dt, c)
            x = x + dt * v2[..., :3]
            R = R @ S.exp(dt * v2[..., 3:])
    else:
        raise ValueError(method)
    return S.project(R), x


def build_training_tensors(ds):
    R1, p1 = encode(ds["R"], ds["p"])
    c = Rp.encode_condition("rel_pose", ds["start_pose"], ds["end_pose"], ds["subject"])
    norm = Rp.Normalizer.fit(p1, c)
    return norm, R1.astype(np.float32), norm.nx(p1).astype(np.float32), norm.nc(c).astype(np.float32)
