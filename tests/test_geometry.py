import numpy as np
import pytest

from camera_motion import geometry as G

rng = np.random.default_rng(0)


def random_rotvecs(n, max_angle=np.pi):
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    return axis * rng.uniform(0, max_angle, size=(n, 1))


def test_exp_log_roundtrip_general():
    w = random_rotvecs(2000, np.pi - 1e-3)
    R = G.so3_exp(w)
    assert G.is_valid_rotation(R, 1e-10)
    np.testing.assert_allclose(G.so3_log(R), w, atol=1e-8)


def test_exp_log_small_angle():
    w = random_rotvecs(200, 1e-7)
    np.testing.assert_allclose(G.so3_log(G.so3_exp(w)), w, atol=1e-12)


@pytest.mark.parametrize("eps", [0.0, 1e-9, 1e-6, 1e-5, 5e-5])
def test_log_near_pi_reconstructs_rotation(eps):
    w = random_rotvecs(200, 1.0)
    w = w / np.linalg.norm(w, axis=-1, keepdims=True) * (np.pi - eps)
    R = G.so3_exp(w)
    w2 = G.so3_log(R)
    assert np.all(np.linalg.norm(w2, axis=-1) <= np.pi + 1e-9)
    np.testing.assert_allclose(G.so3_exp(w2), R, atol=1e-6)
    if eps > 0:  # away from exactly pi the sign is determined
        np.testing.assert_allclose(w2, w, atol=1e-4)


def test_geodesic_angle_matches_log_norm():
    R1 = G.so3_exp(random_rotvecs(300))
    R2 = G.so3_exp(random_rotvecs(300))
    ang = G.geodesic_angle(R1, R2)
    ref = np.linalg.norm(G.so3_log(np.swapaxes(R1, -1, -2) @ R2), axis=-1)
    np.testing.assert_allclose(ang, ref, atol=1e-6)


def test_rot6d_roundtrip_and_validity():
    R = G.so3_exp(random_rotvecs(500))
    np.testing.assert_allclose(G.rot6d_decode(G.rot6d_encode(R)), R, atol=1e-12)
    junk = rng.normal(size=(500, 6)) * 3
    assert G.is_valid_rotation(G.rot6d_decode(junk), 1e-10)


def test_look_at_convention():
    p = np.array([3.0, 0.0, 1.0])
    s = np.array([0.0, 0.0, 0.0])
    R = G.look_at(p, s)
    assert G.is_valid_rotation(R, 1e-12)
    f = R[:, 2]
    np.testing.assert_allclose(f, (s - p) / np.linalg.norm(s - p))  # z forward
    assert abs(R[2, 0]) < 1e-12  # camera x (right) is horizontal
    assert R[2, 1] < 0  # camera y (down) points downward in world
    # camera at +x looking at origin: right should be world -y... check via cross product
    np.testing.assert_allclose(np.cross(R[:, 0], R[:, 1]), R[:, 2], atol=1e-12)
    with pytest.raises(ValueError):
        G.look_at(np.array([0.0, 0, 5]), s)


def test_relative_world_roundtrip_and_anchor_identity():
    L = 16
    R = G.so3_exp(random_rotvecs(4 * L).reshape(4, L, 3))
    p = rng.normal(size=(4, L, 3))
    Rr, pr = G.to_relative(R, p, R[:, 0], p[:, 0])
    np.testing.assert_allclose(Rr[:, 0], np.broadcast_to(np.eye(3), (4, 3, 3)), atol=1e-12)
    np.testing.assert_allclose(pr[:, 0], 0, atol=1e-12)
    Rw, pw = G.to_world(Rr, pr, R[:, 0], p[:, 0])
    np.testing.assert_allclose(Rw, R, atol=1e-12)
    np.testing.assert_allclose(pw, p, atol=1e-12)


def test_relative_frame_invariant_to_global_transform():
    L = 10
    R = G.so3_exp(random_rotvecs(L))
    p = rng.normal(size=(L, 3))
    Rg = G.so3_exp(random_rotvecs(1))[0]
    tg = rng.normal(size=3)
    R2, p2 = Rg @ R, (Rg @ p.T).T + tg
    a = G.to_relative(R, p, R[0], p[0])
    b = G.to_relative(R2, p2, R2[0], p2[0])
    np.testing.assert_allclose(a[0], b[0], atol=1e-12)
    np.testing.assert_allclose(a[1], b[1], atol=1e-12)


@pytest.mark.parametrize("frame", ["body", "spatial"])
def test_increment_roundtrip(frame):
    L = 32
    R = G.so3_exp(random_rotvecs(3 * L).reshape(3, L, 3))
    p = rng.normal(size=(3, L, 3))
    dR, dp = G.poses_to_increments(R, p, frame)
    R2, p2 = G.increments_to_poses(R[:, 0], p[:, 0], dR, dp, frame)
    np.testing.assert_allclose(R2, R, atol=1e-10)
    np.testing.assert_allclose(p2, p, atol=1e-10)


def test_body_increments_invariant_to_global_transform():
    L = 12
    R = G.so3_exp(random_rotvecs(L))
    p = rng.normal(size=(L, 3))
    Rg = G.so3_exp(random_rotvecs(1))[0]
    dR1, dp1 = G.poses_to_increments(R, p)
    dR2, dp2 = G.poses_to_increments(Rg @ R, (Rg @ p.T).T + 5.0)
    np.testing.assert_allclose(dR1, dR2, atol=1e-12)
    np.testing.assert_allclose(dp1, dp2, atol=1e-12)


def test_pose_inverse():
    T = G.make_pose(G.so3_exp(random_rotvecs(5)), rng.normal(size=(5, 3)))
    np.testing.assert_allclose(T @ G.pose_inverse(T), np.broadcast_to(np.eye(4), (5, 4, 4)), atol=1e-12)
