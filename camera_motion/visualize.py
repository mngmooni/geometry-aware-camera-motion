"""Diagnostic figures (matplotlib, headless-safe).

Mode colours use a validated categorical palette (fixed order, light surface) and
are always paired with a legend and a distinct line style, so identity never relies
on colour alone. ``unassigned`` is neutral grey.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import geometry as G  # noqa: E402
from .config import CATEGORIES, MODE_NAMES  # noqa: E402

MODE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8a85")  # 4 modes + unassigned
MODE_STYLES = ("-", "--", "-.", ":", "-")
INK = "#3a3a38"
MUTED = "#8a8a85"
GRID = "#e6e5e0"
SURFACE = "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)
    ax.grid(True, color=GRID, linewidth=0.6)


def _legend(ax, labels_present):
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=MODE_COLORS[k], ls=MODE_STYLES[k], lw=2) for k in labels_present]
    ax.legend(handles, [CATEGORIES[k] for k in labels_present], fontsize=7, frameon=False, labelcolor=INK,
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)


def plot_condition_topdown(ds, condition_id, labels=None, ax=None, max_paths=60, title=None):
    """Top view (world xy) of all paths under one public condition, coloured by mode."""
    sel = np.nonzero(ds["condition_id"] == condition_id)[0][:max_paths]
    labels = ds["latent_mode"] if labels is None else labels
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4.2))
    s = ds["subject"][sel[0]]
    present = sorted(set(int(labels[i]) for i in sel))
    for i in sel:
        k = int(labels[i])
        P = ds["p"][i]
        ax.plot(P[:, 0], P[:, 1], color=MODE_COLORS[k], ls=MODE_STYLES[k], lw=1.4, alpha=0.75)
    P0 = ds["p"][sel[0]]
    ax.plot(*P0[0, :2], "o", color=INK, ms=7, mfc=SURFACE, mew=2, zorder=5)
    ax.plot(*P0[-1, :2], "s", color=INK, ms=7, mfc=SURFACE, mew=2, zorder=5)
    ax.plot(*s[:2], "*", color=INK, ms=11, zorder=5)
    ax.annotate("start", P0[0, :2], textcoords="offset points", xytext=(6, 6), fontsize=7, color=INK)
    ax.annotate("end", P0[-1, :2], textcoords="offset points", xytext=(6, 6), fontsize=7, color=INK)
    ax.annotate("subject", s[:2], textcoords="offset points", xytext=(6, -10), fontsize=7, color=INK)
    ax.set_aspect("equal")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title(title or condition_id, fontsize=9)
    _style(ax)
    _legend(ax, present)
    return ax


def plot_camera_axes_3d(ds, idx, ax=None, every=4, scale=0.5):
    """3D path with camera forward (z) and down (y) axes drawn every few frames."""
    if ax is None:
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, projection="3d")
    P, R, s = ds["p"][idx], ds["R"][idx], ds["subject"][idx]
    ax.plot(P[:, 0], P[:, 1], P[:, 2], color=MODE_COLORS[0], lw=1.6)
    for i in range(0, len(P), every):
        f, d = R[i][:, 2] * scale, R[i][:, 1] * scale * 0.6
        ax.quiver(*P[i], *f, color=INK, lw=0.9)
        ax.quiver(*P[i], *d, color=MUTED, lw=0.7)
    ax.scatter(*s, marker="*", s=80, color=INK)
    pts = np.vstack([P, s[None]])
    c, half = pts.mean(0), (pts.max(0) - pts.min(0)).max() / 2 + scale
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_zlim(c[2] - half, c[2] + half)
    ax.set_box_aspect((1, 1, 1))  # equal scale so look-at directions read correctly
    ax.set_title(f"{ds['condition_id'][idx]}  (dark = forward, grey = down)", fontsize=8, color=INK)
    ax.tick_params(labelsize=6, colors=MUTED)
    return ax


def plot_profiles(ds, condition_id, labels=None, axes=None, max_paths=60):
    """Unwrapped azimuth (deg) and radius/r_lin ratio versus frame for one condition."""
    from .mode_classifier import unwrapped_azimuth
    sel = np.nonzero(ds["condition_id"] == condition_id)[0][:max_paths]
    labels = ds["latent_mode"] if labels is None else labels
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(8, 3))
    for i in sel:
        k = int(labels[i])
        P, s = ds["p"][i], ds["subject"][i]
        steps, _ = unwrapped_azimuth(P, s)
        az = np.degrees(np.concatenate([[0], np.cumsum(steps)]))
        r = np.linalg.norm(P[:, :2] - s[None, :2], axis=-1)
        r_lin = r[0] + (r[-1] - r[0]) * np.linspace(0, 1, len(r))
        axes[0].plot(az, color=MODE_COLORS[k], ls=MODE_STYLES[k], lw=1.2, alpha=0.7)
        axes[1].plot(r / r_lin - 1, color=MODE_COLORS[k], ls=MODE_STYLES[k], lw=1.2, alpha=0.7)
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("unwrapped azimuth (deg)")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("r / r_lin − 1")
    for a in axes:
        _style(a)
    _legend(axes[0], sorted(set(int(labels[i]) for i in sel)))
    return axes


def plot_confusion(M, ax=None, title="GT audit confusion"):
    """rows: true mode; cols: predicted (incl. unassigned). Sequential single hue."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.imshow(M / np.maximum(M.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                    color="white" if v > 0.5 * M[i].sum() else INK)
    ax.set_xticks(range(5), [c.replace("+", "\n") for c in CATEGORIES], fontsize=7, color=INK)
    ax.set_yticks(range(4), [c.replace("+", " ") for c in MODE_NAMES], fontsize=7, color=INK)
    ax.set_xlabel("classifier output", color=INK)
    ax.set_ylabel("generator mode (hidden)", color=INK)
    ax.set_title(title, fontsize=9, color=INK)
    return ax


def plot_excursion_traces(ds, condition_id, axes=None, max_paths=40):
    """E1b: angle of R relative to look-at, per frame, split into pan / roll components."""
    sel = np.nonzero(ds["condition_id"] == condition_id)[0][:max_paths]
    R, p, s = ds["R"][sel], ds["p"][sel], ds["subject"][sel]
    R_look = G.look_at(p, s[:, None, :])
    w = np.degrees(G.so3_log(np.swapaxes(R_look, -1, -2) @ R))
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(8, 3))
    for n in range(len(sel)):
        axes[0].plot(w[n, :, 1], color=MODE_COLORS[0], lw=1, alpha=0.6)
        axes[1].plot(w[n, :, 2], color=MODE_COLORS[1], lw=1, alpha=0.6, ls="--")
    axes[0].set_title("pan offset (camera y)", fontsize=9)
    axes[1].set_title("roll offset (camera z)", fontsize=9)
    for a in axes:
        a.set_xlabel("frame")
        a.set_ylabel("deg")
        _style(a)
    return axes


def plot_feature_hists(feats_a: dict, feats_b: dict | None = None, labels=("GT", "model"), bins=20):
    """Overlaid histograms of E1b excursion features (independent sets, unpaired)."""
    from .metrics import EXCURSION_FEATURES
    fig, axes = plt.subplots(1, len(EXCURSION_FEATURES), figsize=(13, 2.8))
    for ax, k in zip(axes, EXCURSION_FEATURES):
        lo = min(np.min(feats_a[k]), np.min(feats_b[k]) if feats_b is not None else np.inf)
        hi = max(np.max(feats_a[k]), np.max(feats_b[k]) if feats_b is not None else -np.inf)
        edges = np.linspace(lo, hi, bins + 1)
        ax.hist(feats_a[k], bins=edges, density=True, color=MODE_COLORS[0], alpha=0.75,
                label=f"{labels[0]} (n={len(feats_a[k])})", edgecolor=SURFACE)
        if feats_b is not None:
            ax.hist(feats_b[k], bins=edges, density=True, histtype="step", color=MODE_COLORS[1], lw=2,
                    label=f"{labels[1]} (n={len(feats_b[k])})")
        ax.set_title(k + "  (density)", fontsize=8)
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def save(fig, path, dpi=140):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
