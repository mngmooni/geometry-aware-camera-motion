"""SO(3) / pose geometry with one explicit, tested convention.

Conventions (frozen for the whole project)
------------------------------------------
* World frame: right-handed, **z-up**. The orbital plane of E1 is the world xy-plane.
* Camera frame: OpenCV style, **x right, y down, z forward** (optical axis).
* A pose ``T = (R, p)`` is **camera-to-world**: ``x_world = R @ x_cam + p``.
  The columns of ``R`` are the camera x/y/z axes expressed in world coordinates and
  ``p`` is the camera centre in world coordinates.
* Composition is matrix multiplication of homogeneous transforms:
  ``T_a @ T_b`` applies ``T_b`` first.
* Camera-local (body) rotation offsets multiply on the **right**: ``R = R_base @ R_off``.
* Rotation vectors / Log return the principal value, angle in ``[0, pi]``.
* 6D rotation code = first two columns of ``R`` concatenated ``[R[:,0], R[:,1]]``;
  decoding uses Gram-Schmidt (Zhou et al. 2019 continuous representation).
* Sequence time: frame index ``i = 0..L-1`` with fixed step ``dt = 1/(L-1)``
  (normalised shot duration). This is *camera time*, unrelated to the flow-matching
  interpolation time ``t``.

Increment convention
--------------------
``poses_to_increments`` uses **body-frame** increments:
``dR_i = R_i^T R_{i+1}``, ``dp_i = R_i^T (p_{i+1} - p_i)``.
Body-frame increments are invariant to a global left transform, so they are the same
whether computed from world or initial-camera-relative poses; only the anchor (start)
pose differs. ``frame="spatial"`` (``dR_i = R_{i+1} R_i^T``, ``dp_i = p_{i+1}-p_i``) is
also provided for ablations.

All functions accept arbitrary leading batch dimensions (``(..., 3, 3)``, ``(..., 3)``).
"""
from __future__ import annotations

import numpy as np

WORLD_UP = np.array([0.0, 0.0, 1.0])
_SMALL_ANGLE = 1e-6
_NEAR_PI = 1e-4


# ---------------------------------------------------------------------------
# so(3) basics
# ---------------------------------------------------------------------------
def hat(w: np.ndarray) -> np.ndarray:
    """(...,3) -> (...,3,3) skew-symmetric matrix."""
    w = np.asarray(w, dtype=float)
    out = np.zeros(w.shape[:-1] + (3, 3))
    out[..., 0, 1] = -w[..., 2]
    out[..., 0, 2] = w[..., 1]
    out[..., 1, 0] = w[..., 2]
    out[..., 1, 2] = -w[..., 0]
    out[..., 2, 0] = -w[..., 1]
    out[..., 2, 1] = w[..., 0]
    return out


def vee(W: np.ndarray) -> np.ndarray:
    """(...,3,3) skew matrix -> (...,3). Uses the antisymmetric part."""
    W = np.asarray(W, dtype=float)
    return 0.5 * np.stack(
        [W[..., 2, 1] - W[..., 1, 2], W[..., 0, 2] - W[..., 2, 0], W[..., 1, 0] - W[..., 0, 1]],
        axis=-1,
    )


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rodrigues formula. (...,3) rotation vector -> (...,3,3)."""
    w = np.asarray(w, dtype=float)
    theta = np.linalg.norm(w, axis=-1)[..., None, None]
    K = hat(w)
    K2 = K @ K
    small = theta < _SMALL_ANGLE
    safe = np.where(small, 1.0, theta)
    a = np.where(small, 1.0 - theta**2 / 6.0, np.sin(safe) / safe)
    b = np.where(small, 0.5 - theta**2 / 24.0, (1.0 - np.cos(safe)) / safe**2)
    return np.eye(3) + a * K + b * K2


def so3_log(R: np.ndarray) -> np.ndarray:
    """Principal logarithm. (...,3,3) -> (...,3) with angle in [0, pi].

    Handles the small-angle limit and the angle ~ pi branch (where the
    antisymmetric part vanishes) explicitly. At exactly pi the axis sign is
    inherently ambiguous; we return one consistent choice.
    """
    R = np.asarray(R, dtype=float)
    batch = R.shape[:-2]
    Rf = R.reshape(-1, 3, 3)
    out = np.zeros((Rf.shape[0], 3))
    cos = np.clip((np.trace(Rf, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos)
    v = vee(Rf)  # = sin(theta) * axis

    small = theta < _SMALL_ANGLE
    near_pi = (np.pi - theta) < _NEAR_PI
    gen = ~(small | near_pi)

    out[small] = v[small]  # first-order: R ~ I + hat(w)
    if np.any(gen):
        s = np.sin(theta[gen])
        out[gen] = (theta[gen] / s)[:, None] * v[gen]
    if np.any(near_pi):
        for k in np.nonzero(near_pi)[0]:
            # Exact: sym(R) = cos(t) I + (1 - cos(t)) a a^T  =>  a a^T recovered for any t.
            S = 0.5 * (Rf[k] + Rf[k].T)
            B = (S - cos[k] * np.eye(3)) / (1.0 - cos[k])
            j = int(np.argmax(np.diag(B)))
            a = B[:, j] / np.sqrt(max(B[j, j], 1e-12))
            a = a / np.linalg.norm(a)
            if np.dot(v[k], a) < 0:
                a = -a
            out[k] = theta[k] * a
    return out.reshape(batch + (3,))


def geodesic_angle(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray:
    """Angle (rad) of R_a^T R_b, in [0, pi]."""
    # atan2(sin, cos) stays accurate near 0 where arccos loses ~sqrt(eps).
    R_rel = np.swapaxes(R_a, -1, -2) @ R_b
    cos = (np.trace(R_rel, axis1=-2, axis2=-1) - 1.0) / 2.0
    sin = np.linalg.norm(vee(R_rel), axis=-1)
    return np.arctan2(sin, cos)


# ---------------------------------------------------------------------------
# 6D representation
# ---------------------------------------------------------------------------
def rot6d_encode(R: np.ndarray) -> np.ndarray:
    """(...,3,3) -> (...,6): first two columns."""
    R = np.asarray(R, dtype=float)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def rot6d_decode(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """(...,6) -> (...,3,3) via Gram-Schmidt. Always returns a valid rotation."""
    x = np.asarray(x, dtype=float)
    a1, a2 = x[..., :3], x[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), eps)
    a2p = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2p / np.maximum(np.linalg.norm(a2p, axis=-1, keepdims=True), eps)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


# ---------------------------------------------------------------------------
# Rotation validity diagnostics (not primary metrics)
# ---------------------------------------------------------------------------
def rotation_validity(R: np.ndarray) -> dict:
    R = np.asarray(R, dtype=float)
    I = np.eye(3)
    orth = np.linalg.norm(np.swapaxes(R, -1, -2) @ R - I, axis=(-2, -1))
    det = np.linalg.det(R)
    return {"max_orth_err": float(np.max(orth)), "max_det_err": float(np.max(np.abs(det - 1.0)))}


def is_valid_rotation(R: np.ndarray, tol: float = 1e-6) -> bool:
    d = rotation_validity(R)
    return d["max_orth_err"] < tol and d["max_det_err"] < tol


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------
def look_at(p: np.ndarray, target: np.ndarray, up: np.ndarray = WORLD_UP) -> np.ndarray:
    """Camera-to-world rotation (OpenCV axes) looking from p towards target.

    columns = [right, down, forward]; right = forward x up. Undefined when the
    view direction is parallel to ``up`` (raises).
    """
    p = np.asarray(p, dtype=float)
    target = np.asarray(target, dtype=float)
    f = target - p
    f = f / np.linalg.norm(f, axis=-1, keepdims=True)
    r = np.cross(f, np.broadcast_to(up, f.shape))
    rn = np.linalg.norm(r, axis=-1, keepdims=True)
    if np.any(rn < 1e-8):
        raise ValueError("look_at: view direction parallel to world up")
    r = r / rn
    d = np.cross(f, r)
    return np.stack([r, d, f], axis=-1)


# ---------------------------------------------------------------------------
# Homogeneous poses
# ---------------------------------------------------------------------------
def make_pose(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    p = np.asarray(p, dtype=float)
    T = np.zeros(R.shape[:-2] + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = p
    T[..., 3, 3] = 1.0
    return T


def split_pose(T: np.ndarray):
    T = np.asarray(T, dtype=float)
    return T[..., :3, :3], T[..., :3, 3]


def pose_inverse(T: np.ndarray) -> np.ndarray:
    R, p = split_pose(T)
    Rt = np.swapaxes(R, -1, -2)
    return make_pose(Rt, -(Rt @ p[..., None])[..., 0])


# ---------------------------------------------------------------------------
# Frames: world <-> initial-camera-relative
# ---------------------------------------------------------------------------
def to_relative(R: np.ndarray, p: np.ndarray, R0: np.ndarray, p0: np.ndarray):
    """Express world poses (…,L,3,3)/(…,L,3) in the frame of the anchor pose (R0,p0).

    R_rel = R0^T R,  p_rel = R0^T (p - p0). The anchor maps to (I, 0).
    """
    R0t = np.swapaxes(R0, -1, -2)[..., None, :, :]
    R_rel = R0t @ R
    p_rel = (R0t @ (p - p0[..., None, :])[..., None])[..., 0]
    return R_rel, p_rel


def to_world(R_rel: np.ndarray, p_rel: np.ndarray, R0: np.ndarray, p0: np.ndarray):
    """Inverse of :func:`to_relative`."""
    R0e = R0[..., None, :, :]
    R = R0e @ R_rel
    p = (R0e @ p_rel[..., None])[..., 0] + p0[..., None, :]
    return R, p


def point_to_relative(x: np.ndarray, R0: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """A world point (…,3) expressed in the anchor frame."""
    return (np.swapaxes(R0, -1, -2) @ (x - p0)[..., None])[..., 0]


# ---------------------------------------------------------------------------
# Increments
# ---------------------------------------------------------------------------
def poses_to_increments(R: np.ndarray, p: np.ndarray, frame: str = "body"):
    """(…,L,3,3),(…,L,3) -> increments (…,L-1,3,3),(…,L-1,3)."""
    Ra, Rb = R[..., :-1, :, :], R[..., 1:, :, :]
    dp_w = p[..., 1:, :] - p[..., :-1, :]
    if frame == "body":
        Rat = np.swapaxes(Ra, -1, -2)
        return Rat @ Rb, (Rat @ dp_w[..., None])[..., 0]
    if frame == "spatial":
        return Rb @ np.swapaxes(Ra, -1, -2), dp_w
    raise ValueError(frame)


def increments_to_poses(R_start, p_start, dR, dp, frame: str = "body"):
    """Integrate increments from a start pose. Returns (…,L,3,3),(…,L,3).

    Exact inverse of :func:`poses_to_increments` (up to float round-off).
    For decoded network outputs, accumulated error = drift; measure it, don't hide it.
    """
    n = dR.shape[-3]
    Rs = [np.asarray(R_start, dtype=float)]
    ps = [np.asarray(p_start, dtype=float)]
    for i in range(n):
        Ri, pi = Rs[-1], ps[-1]
        if frame == "body":
            Rs.append(Ri @ dR[..., i, :, :])
            ps.append(pi + (Ri @ dp[..., i, :, None])[..., 0])
        elif frame == "spatial":
            Rs.append(dR[..., i, :, :] @ Ri)
            ps.append(pi + dp[..., i, :])
        else:
            raise ValueError(frame)
    return np.stack(Rs, axis=-3), np.stack(ps, axis=-2)


# ---------------------------------------------------------------------------
# Angular helpers
# ---------------------------------------------------------------------------
def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap to (-pi, pi]."""
    return np.angle(np.exp(1j * np.asarray(a, dtype=float)))


def angular_velocity_body(R: np.ndarray, dt: float) -> np.ndarray:
    """Body angular velocity (…,L-1,3) from consecutive rotations: Log(R_i^T R_{i+1})/dt."""
    dR = np.swapaxes(R[..., :-1, :, :], -1, -2) @ R[..., 1:, :, :]
    return so3_log(dR) / dt
