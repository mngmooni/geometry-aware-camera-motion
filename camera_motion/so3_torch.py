"""Batched SO(3) operations in torch (float32/float64), numerically safe everywhere.

Log is computed through a unit quaternion (Shepperd's method + atan2), which stays
accurate near angle 0 and near pi; at exactly pi the axis sign is inherently ambiguous
and one consistent choice is returned. Conventions match ``geometry.py``.
"""
from __future__ import annotations

import torch


def hat(w):
    z = torch.zeros_like(w[..., 0])
    return torch.stack([torch.stack([z, -w[..., 2], w[..., 1]], -1),
                        torch.stack([w[..., 2], z, -w[..., 0]], -1),
                        torch.stack([-w[..., 1], w[..., 0], z], -1)], -2)


def exp(w):
    """Rodrigues; (...,3) -> (...,3,3)."""
    th = torch.linalg.norm(w, dim=-1)[..., None, None]
    K = hat(w)
    small = th < 1e-4
    ths = torch.where(small, torch.ones_like(th), th)
    a = torch.where(small, 1 - th**2 / 6, torch.sin(ths) / ths)
    b = torch.where(small, 0.5 - th**2 / 24, (1 - torch.cos(ths)) / ths**2)
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand(K.shape)
    return I + a * K + b * (K @ K)


def matrix_to_quat(R):
    """(...,3,3) -> (...,4) as (w, x, y, z) with w >= 0. Shepperd: pick the largest pivot."""
    m = R
    tr = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    cand = torch.stack([tr, m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]], -1)
    piv = cand.argmax(-1)
    r = torch.sqrt(torch.clamp(1 + 2 * cand.max(-1).values - tr, min=1e-12))  # = 2 * |largest component|
    q0 = torch.stack([r / 2, (m[..., 2, 1] - m[..., 1, 2]) / (2 * r), (m[..., 0, 2] - m[..., 2, 0]) / (2 * r),
                      (m[..., 1, 0] - m[..., 0, 1]) / (2 * r)], -1)
    q1 = torch.stack([(m[..., 2, 1] - m[..., 1, 2]) / (2 * r), r / 2, (m[..., 0, 1] + m[..., 1, 0]) / (2 * r),
                      (m[..., 0, 2] + m[..., 2, 0]) / (2 * r)], -1)
    q2 = torch.stack([(m[..., 0, 2] - m[..., 2, 0]) / (2 * r), (m[..., 0, 1] + m[..., 1, 0]) / (2 * r), r / 2,
                      (m[..., 1, 2] + m[..., 2, 1]) / (2 * r)], -1)
    q3 = torch.stack([(m[..., 1, 0] - m[..., 0, 1]) / (2 * r), (m[..., 0, 2] + m[..., 2, 0]) / (2 * r),
                      (m[..., 1, 2] + m[..., 2, 1]) / (2 * r), r / 2], -1)
    q = torch.stack([q0, q1, q2, q3], -2)
    q = torch.gather(q, -2, piv[..., None, None].expand(*piv.shape, 1, 4)).squeeze(-2)
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    return torch.where(q[..., :1] < 0, -q, q)


def quat_to_matrix(q):
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def log(R):
    """Principal log; (...,3,3) -> (...,3), angle in [0, pi]."""
    q = matrix_to_quat(R)
    v = q[..., 1:]
    s = torch.linalg.norm(v, dim=-1, keepdim=True)
    ang = 2 * torch.atan2(s, q[..., :1])
    scale = torch.where(s < 1e-7, 2.0 / torch.clamp(q[..., :1], min=1e-7), ang / torch.clamp(s, min=1e-12))
    return v * scale


def uniform(shape, generator=None, device=None, dtype=torch.float32):
    """Haar-uniform rotations: normalised 4D Gaussian -> quaternion -> matrix."""
    q = torch.randn(*shape, 4, generator=generator, device=device, dtype=dtype)
    return quat_to_matrix(q)


def rot6d(R):
    return torch.cat([R[..., :, 0], R[..., :, 1]], -1)


def project(R):
    """Nearest rotation (SVD); used only to remove float drift after many compositions."""
    U, _, Vh = torch.linalg.svd(R)
    d = torch.det(U @ Vh)
    D = torch.ones(*R.shape[:-2], 3, dtype=R.dtype, device=R.device)
    D[..., 2] = d
    return U @ torch.diag_embed(D) @ Vh
