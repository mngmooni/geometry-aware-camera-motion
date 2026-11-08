"""Minimal software renderer for camera-motion demos (numpy + PIL, no Blender needed).

Uses the project's camera convention directly: poses are camera-to-world with OpenCV
axes (x right, y down, z forward), so a world point X maps to camera coordinates
x_c = R^T (X - p) and to pixels u = f x/z + cx, v = f y/z + cy.

Scene (built around the public subject position s):
* checkerboard ground plane under the subject
* the subject: three stacked, differently coloured blocks centred on s
* four coloured pillars at the compass points (N/E/S/W) so view direction is readable
Faces are clipped against the near plane, Lambert-shaded and drawn back-to-front.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

SKY_TOP = np.array([156, 190, 228])
SKY_BOTTOM = np.array([226, 236, 245])
LIGHT = np.array([0.35, -0.45, 0.82]) / np.linalg.norm([0.35, -0.45, 0.82])
PILLAR_COLORS = {"N": (222, 74, 74), "E": (236, 170, 40), "S": (60, 150, 220), "W": (70, 175, 110)}
SUBJECT_COLORS = ((235, 235, 232), (240, 120, 60), (50, 60, 75))  # bottom -> top


@dataclass
class Face:
    verts: np.ndarray  # (n, 3) world
    color: tuple
    normal: np.ndarray


def _box(center, size, color):
    c, (sx, sy, sz) = np.asarray(center, float), size
    d = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float) * [sx / 2, sy / 2, sz / 2] + c
    quads = [((0, 3, 2, 1), (0, 0, -1)), ((4, 5, 6, 7), (0, 0, 1)), ((0, 1, 5, 4), (0, -1, 0)),
             ((2, 3, 7, 6), (0, 1, 0)), ((1, 2, 6, 5), (1, 0, 0)), ((3, 0, 4, 7), (-1, 0, 0))]
    return [Face(d[list(q)], color, np.array(n, float)) for q, n in quads]


def build_scene(subject, ground_half=12.0, tile=1.0):
    s = np.asarray(subject, float)
    z0 = s[2] - 0.45  # ground under the subject
    faces = []
    n = int(2 * ground_half / tile)
    for i in range(n):
        for j in range(n):
            x0, y0 = s[0] - ground_half + i * tile, s[1] - ground_half + j * tile
            col = (206, 204, 196) if (i + j) % 2 == 0 else (178, 176, 168)
            v = np.array([[x0, y0, z0], [x0 + tile, y0, z0], [x0 + tile, y0 + tile, z0], [x0, y0 + tile, z0]])
            faces.append(Face(v, col, np.array([0, 0, 1.0])))
    for k, col in enumerate(SUBJECT_COLORS):  # subject totem, total height 0.9 centred on s
        faces += _box(s + [0, 0, -0.30 + 0.30 * k], (0.5 - 0.08 * k, 0.5 - 0.08 * k, 0.30), col)
    for name, ang in (("E", 0.0), ("N", np.pi / 2), ("W", np.pi), ("S", -np.pi / 2)):
        c = s + [7.5 * np.cos(ang), 7.5 * np.sin(ang), 0]
        faces += _box([c[0], c[1], z0 + 1.6], (0.7, 0.7, 3.2), PILLAR_COLORS[name])
    return faces


def _clip_near(pc, near):
    """Sutherland-Hodgman clip of a camera-space polygon against z >= near."""
    out = []
    for i in range(len(pc)):
        a, b = pc[i], pc[(i + 1) % len(pc)]
        ina, inb = a[2] >= near, b[2] >= near
        if ina:
            out.append(a)
        if ina != inb:
            t = (near - a[2]) / (b[2] - a[2])
            out.append(a + t * (b - a))
    return np.array(out)


def project(points_w, R, p, f, cx, cy):
    """World points (n,3) -> pixel (n,2) and camera depth (n,), OpenCV convention."""
    pc = (points_w - p) @ R  # = (R^T (X - p))^T
    return np.stack([f * pc[:, 0] / pc[:, 2] + cx, f * pc[:, 1] / pc[:, 2] + cy], -1), pc[:, 2]


def render_view(faces, R, p, size=(480, 270), fov_deg=60.0, near=0.05):
    W, H = size
    f = 0.5 * W / np.tan(np.radians(fov_deg) / 2)
    cx, cy = W / 2, H / 2
    grad = np.linspace(0, 1, H)[:, None, None]
    img = Image.fromarray((SKY_TOP * (1 - grad) + SKY_BOTTOM * grad).repeat(W, 1).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    polys = []
    for fc in faces:
        pc = (fc.verts - p) @ R
        if np.all(pc[:, 2] < near):
            continue
        if np.dot(fc.normal, p - fc.verts.mean(0)) <= 0:  # back-face cull
            continue
        pc = _clip_near(pc, near) if np.any(pc[:, 2] < near) else pc
        if len(pc) < 3:
            continue
        uv = np.stack([f * pc[:, 0] / pc[:, 2] + cx, f * pc[:, 1] / pc[:, 2] + cy], -1)
        if uv[:, 0].max() < -W or uv[:, 0].min() > 2 * W or uv[:, 1].max() < -H or uv[:, 1].min() > 2 * H:
            continue
        shade = 0.55 + 0.45 * max(0.0, float(np.dot(fc.normal, LIGHT)))
        depth = float(np.linalg.norm(pc.mean(0)))
        fog = min(1.0, depth / 30.0) * 0.35
        col = tuple(int((1 - fog) * c * shade + fog * 235) for c in fc.color)
        polys.append((depth, [tuple(x) for x in uv], col))
    for _, uv, col in sorted(polys, key=lambda t: -t[0]):
        draw.polygon(uv, fill=col)
    return img


def upsample_poses(R, p, factor=3):
    """Playback interpolation between generated keyframes: cubic spline for positions,
    SLERP for rotations. Keyframes are reproduced exactly; no new motion is invented."""
    L = len(p)
    t = np.arange(L, dtype=float)
    tt = np.linspace(0, L - 1, (L - 1) * factor + 1)
    p_up = CubicSpline(t, p, axis=0)(tt)
    R_up = Slerp(t, Rotation.from_matrix(R))(tt).as_matrix()
    return R_up, p_up
