"""Euclidean conditional flow matching (linear / rectified path) + temporal 1D-conv net.

Shared by every Euclidean variant (abs_pose, rel_pose, rel_inc); only the input
representation and condition encoding change. The SO(3) Riemannian variant (rfm_so3)
is a separate, later module.

Path:    x_t = (1 - t) x0 + t x1,   x0 ~ N(0, I) in normalised coordinates
Target:  v*  = x1 - x0
Loss:    E || v_theta(x_t, t, c) - v* ||^2
Sampler: fixed-step Euler (or midpoint) from t = 0 to 1 with a declared NFE.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        a = t[:, None] * 1000.0 * freqs[None]
        return torch.cat([a.sin(), a.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, ch: int, emb: int, dilation: int, kernel: int = 5):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.n1 = nn.GroupNorm(8, ch)
        self.c1 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.n2 = nn.GroupNorm(8, ch)
        self.c2 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.film = nn.Linear(emb, 2 * ch)

    def forward(self, h, e):
        scale, shift = self.film(e)[..., None].chunk(2, dim=1)
        y = self.c1(F.silu(self.n1(h)))
        y = self.n2(y) * (1 + scale) + shift
        y = self.c2(F.silu(y))
        return h + y


class TemporalVelocityNet(nn.Module):
    """v_theta(x_t, t, c): x (B, T, C) -> (B, T, C). FiLM-conditioned dilated 1D ResNet."""

    def __init__(self, T: int, C: int, cond_dim: int, hidden: int = 128, emb: int = 256,
                 dilations=(1, 2, 4, 8, 1, 2), out_channels: int | None = None):
        super().__init__()
        out_channels = out_channels or C
        self.t_emb = nn.Sequential(SinusoidalEmbedding(emb), nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.c_emb = nn.Sequential(nn.Linear(cond_dim, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.pos = nn.Parameter(torch.zeros(1, hidden, T))
        self.inp = nn.Conv1d(C, hidden, 1)
        self.blocks = nn.ModuleList([ResBlock(hidden, emb, d) for d in dilations])
        self.out_norm = nn.GroupNorm(8, hidden)
        self.out = nn.Conv1d(hidden, out_channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x, t, c):
        e = F.silu(self.t_emb(t) + self.c_emb(c))
        h = self.inp(x.transpose(1, 2)) + self.pos
        for b in self.blocks:
            h = b(h, e)
        return self.out(F.silu(self.out_norm(h))).transpose(1, 2)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def fm_loss(model, x1, c, generator=None):
    """Linear-path conditional FM loss on a normalised batch."""
    B = x1.shape[0]
    x0 = torch.randn(x1.shape, device=x1.device, generator=generator)
    t = torch.rand(B, device=x1.device, generator=generator)
    xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
    return F.mse_loss(model(xt, t, c), x1 - x0)


@torch.no_grad()
def sample(model, c, shape, nfe: int = 50, method: str = "euler", generator=None):
    """Integrate dx/dt = v(x, t, c) from N(0, I) at t=0 to t=1. Returns normalised x1."""
    x = torch.randn(shape, device=c.device, generator=generator)
    dt = 1.0 / nfe if method == "euler" else 2.0 / nfe
    steps = nfe if method == "euler" else nfe // 2
    for i in range(steps):
        t = torch.full((shape[0],), i * dt, device=c.device)
        if method == "euler":
            x = x + dt * model(x, t, c)
        elif method == "midpoint":
            xm = x + 0.5 * dt * model(x, t, c)
            x = x + dt * model(xm, t + 0.5 * dt, c)
        else:
            raise ValueError(method)
    return x


class EMA:
    """EMA with warmup: effective decay = min(decay, (1 + n) / (10 + n))."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.n = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow)
