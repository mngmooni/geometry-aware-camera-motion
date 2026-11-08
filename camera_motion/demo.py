"""Compose camera-motion demo videos: a 2x2 grid of camera views + a top-down map.

Each tile is (label, colour, R (L,3,3), p (L,3)) in WORLD coordinates, i.e. exactly what
the generators / FM decoders output. Keyframes are upsampled for smooth playback only.
"""
from __future__ import annotations

import imageio.v2 as imageio
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from . import render as Rd

BG = (250, 250, 248)
INK = (40, 40, 38)
MUTED = (130, 130, 125)
GRIDC = (225, 224, 218)


def _font(size, bold=False):
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


def _hex(c):
    if isinstance(c, str):
        c = c.lstrip("#")
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
    return tuple(c)


class TopDownMap:
    def __init__(self, paths, subject, size=430, margin=34):
        pts = np.concatenate([p[:, :2] for p in paths] + [np.asarray(subject)[None, :2]])
        lo, hi = pts.min(0), pts.max(0)
        c, half = (lo + hi) / 2, (hi - lo).max() / 2 + 0.6
        self.size, self.c, self.scale = size, c, (size / 2 - margin) / half
        self.subject = np.asarray(subject)

    def px(self, xy):
        d = (np.asarray(xy)[..., :2] - self.c) * self.scale
        return np.stack([self.size / 2 + d[..., 0], self.size / 2 - d[..., 1]], -1)  # y up on screen

    def base(self, tiles):
        img = Image.new("RGB", (self.size, self.size), BG)
        d = ImageDraw.Draw(img)
        step = 1.0
        g0 = np.floor((self.c - self.size / 2 / self.scale) / step) * step
        for k in range(int(self.size / self.scale / step) + 2):
            x = self.px([g0[0] + k * step, self.c[1]])[0]
            y = self.px([self.c[0], g0[1] + k * step])[1]
            d.line([(x, 0), (x, self.size)], fill=GRIDC)
            d.line([(0, y), (self.size, y)], fill=GRIDC)
        for name, ang in (("E", 0.0), ("N", np.pi / 2), ("W", np.pi), ("S", -np.pi / 2)):
            q = self.px(self.subject[:2] + 7.5 * np.array([np.cos(ang), np.sin(ang)]))
            if 0 <= q[0] <= self.size and 0 <= q[1] <= self.size:
                d.rectangle([q[0] - 5, q[1] - 5, q[0] + 5, q[1] + 5], fill=Rd.PILLAR_COLORS[name])
        for _, col, _, p in tiles:
            d.line([tuple(x) for x in self.px(p)], fill=_hex(col) + (255,), width=2)
        s = self.px(self.subject)
        d.regular_polygon((s[0], s[1], 8), 5, fill=INK)
        p0, p1 = self.px(tiles[0][3][0]), self.px(tiles[0][3][-1])
        d.ellipse([p0[0] - 6, p0[1] - 6, p0[0] + 6, p0[1] + 6], outline=INK, width=2)
        d.rectangle([p1[0] - 6, p1[1] - 6, p1[0] + 6, p1[1] + 6], outline=INK, width=2)
        f = _font(12)
        d.text((p0[0] + 8, p0[1] - 16), "start", fill=INK, font=f)
        d.text((p1[0] + 8, p1[1] - 16), "end", fill=INK, font=f)
        d.text((s[0] + 10, s[1] + 2), "subject", fill=INK, font=f)
        d.text((8, self.size - 20), "top view · 1 m grid", fill=MUTED, font=f)
        return img

    def overlay(self, base, tiles, i):
        img = base.copy()
        d = ImageDraw.Draw(img)
        for _, col, R, p in tiles:
            q = self.px(p[i])
            fwd = R[i][:, 2]  # camera optical axis in world
            h = fwd[:2] / (np.linalg.norm(fwd[:2]) + 1e-9)
            tip = q + np.array([h[0], -h[1]]) * 22
            d.line([tuple(q), tuple(tip)], fill=_hex(col), width=3)
            d.ellipse([q[0] - 6, q[1] - 6, q[0] + 6, q[1] + 6], fill=_hex(col), outline=INK)
        return img


def compose_frames(tiles, subject, title, subtitle, view=(400, 225), factor=3):
    """tiles: list of 4 (label, colour, R, p). Returns list of HxWx3 uint8 frames."""
    assert len(tiles) == 4
    faces = Rd.build_scene(subject)
    up = [(lab, col, *Rd.upsample_poses(R, p, factor)) for lab, col, R, p in tiles]
    n = len(up[0][3])
    vw, vh = view
    pad, head = 10, 58
    mapw = 2 * vh + pad
    W = pad + 2 * vw + pad + pad + mapw + pad
    H = head + 2 * vh + 3 * pad + 18
    W, H = W + W % 2, H + H % 2
    tmap = TopDownMap([t[3] for t in tiles], subject, size=mapw)
    base = tmap.base(tiles)
    ft, fs, fl = _font(18, True), _font(13), _font(13, True)
    frames = []
    for i in range(n):
        canvas = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(canvas)
        d.text((pad, 8), title, fill=INK, font=ft)
        d.text((pad, 33), subtitle, fill=MUTED, font=fs)
        for k, (lab, col, R, p) in enumerate(up):
            x0 = pad + (k % 2) * (vw + pad)
            y0 = head + (k // 2) * (vh + pad)
            canvas.paste(Rd.render_view(faces, R[i], p[i], size=view), (x0, y0))
            d.rectangle([x0, y0, x0 + vw - 1, y0 + vh - 1], outline=_hex(col), width=3)
            tw = d.textlength(lab, font=fl)
            d.rectangle([x0 + 6, y0 + 6, x0 + 18 + tw, y0 + 26], fill=_hex(col))
            d.text((x0 + 12, y0 + 8), lab, fill=(255, 255, 255), font=fl)
        canvas.paste(tmap.overlay(base, up, i), (pad + 2 * vw + 2 * pad, head))
        frac = i / (n - 1)
        by = H - 14
        d.rectangle([pad, by, W - pad, by + 4], fill=GRIDC)
        d.rectangle([pad, by, pad + int((W - 2 * pad) * frac), by + 4], fill=MUTED)
        frames.append(np.asarray(canvas))
    return frames


def write_video(frames, mp4_path=None, gif_path=None, fps=24, gif_scale=0.5, gif_stride=2):
    if mp4_path:
        with imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=8, macro_block_size=2,
                                pixelformat="yuv420p", ffmpeg_params=["-preset", "veryfast"]) as w:
            for fr in frames:
                w.append_data(fr)
    if gif_path:
        small = [np.asarray(Image.fromarray(fr).resize((int(fr.shape[1] * gif_scale) // 2 * 2,
                                                         int(fr.shape[0] * gif_scale) // 2 * 2), Image.LANCZOS))
                 for fr in frames[::gif_stride]]
        imageio.mimsave(gif_path, small, duration=gif_stride / fps, loop=0)
