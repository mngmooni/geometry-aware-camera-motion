import numpy as np
import pytest

from camera_motion import dataset_generator as D
from camera_motion import geometry as G
from camera_motion import representations as Rp


@pytest.fixture(scope="module")
def ds():
    return D.generate_e1_audit("E1b", n_cond=2, group=30)


@pytest.mark.parametrize("rep", Rp.REPRESENTATIONS)
def test_roundtrip_exact(ds, rep):
    x = Rp.encode_trajectories(rep, ds["R"], ds["p"])
    assert x.shape == (len(ds["p"]), ds["p"].shape[1] - 1, Rp.T_CHANNELS)
    R, p = Rp.decode_trajectories(rep, x, ds["start_pose"])
    np.testing.assert_allclose(R, ds["R"], atol=1e-9)
    np.testing.assert_allclose(p, ds["p"], atol=1e-9)
    assert np.all(R[:, 0] == ds["R"][:, 0]) and np.all(p[:, 0] == ds["p"][:, 0])  # exact anchor


@pytest.mark.parametrize("rep", Rp.REPRESENTATIONS)
def test_decode_always_valid_rotation(ds, rep):
    x = np.random.default_rng(0).normal(size=(10, 31, 9)) * 3
    R, _ = Rp.decode_trajectories(rep, x, ds["start_pose"][:10])
    assert G.is_valid_rotation(R, 1e-8)


def test_condition_uses_public_fields_only(ds):
    for rep in Rp.REPRESENTATIONS:
        c = Rp.encode_condition(rep, ds["start_pose"], ds["end_pose"], ds["subject"])
        assert c.shape[1] == (21 if rep == "abs_pose" else 15)
        sel = ds["condition_id"] == ds["condition_id"][0]
        assert np.all(c[sel] == c[sel][0])  # same public condition -> same vector


@pytest.mark.parametrize("rep", ["rel_pose", "rel_inc"])
def test_relative_encoding_invariant_to_world_yaw_and_translation(ds, rep):
    Rz = G.so3_exp(np.array([0, 0, 1.234]))
    t = np.array([7.0, -3.0, 2.0])
    Rw, pw = Rz @ ds["R"], np.einsum("ij,nlj->nli", Rz, ds["p"]) + t
    T0 = G.make_pose(Rz @ ds["start_pose"][:, :3, :3], ds["start_pose"][:, :3, 3] @ Rz.T + t)
    T1 = G.make_pose(Rz @ ds["end_pose"][:, :3, :3], ds["end_pose"][:, :3, 3] @ Rz.T + t)
    s = ds["subject"] @ Rz.T + t
    np.testing.assert_allclose(Rp.encode_trajectories(rep, Rw, pw), Rp.encode_trajectories(rep, ds["R"], ds["p"]), atol=1e-9)
    np.testing.assert_allclose(Rp.encode_condition(rep, T0, T1, s),
                               Rp.encode_condition(rep, ds["start_pose"], ds["end_pose"], ds["subject"]), atol=1e-9)


def test_abs_encoding_is_not_invariant(ds):
    t = np.array([7.0, 0, 0])
    a = Rp.encode_trajectories("abs_pose", ds["R"], ds["p"])
    b = Rp.encode_trajectories("abs_pose", ds["R"], ds["p"] + t)
    assert not np.allclose(a, b)


def test_normalizer_roundtrip(ds):
    x = Rp.encode_trajectories("abs_pose", ds["R"], ds["p"])
    c = Rp.encode_condition("abs_pose", ds["start_pose"], ds["end_pose"], ds["subject"])
    n = Rp.Normalizer.fit(x, c)
    np.testing.assert_allclose(n.ux(n.nx(x)), x, atol=1e-12)
    n2 = Rp.Normalizer.from_state(n.state())
    np.testing.assert_allclose(n2.nx(x), n.nx(x))


def _long_sweep_path():
    from camera_motion.config import FROZEN
    rng = np.random.default_rng(11)
    cond = D.sample_conditions(rng, 1)
    s = cond.subject[0]
    p = D.e1_positions(s, cond.phi0[0], cond.r0[0], cond.r1[0], cond.h[0], 3, 0.4, 0.4, FROZEN.split.L)  # long_cw+far
    R = G.look_at(p, s) @ D.offset_rotations(FROZEN.split.L, 1, np.deg2rad(55), 0.5)
    return R[None], p[None]


def test_rel_inc_long_rotation_roundtrip_and_anchor():
    """-270 deg sweep with a 55 deg roll excursion: total rotation history far beyond pi."""
    R, p = _long_sweep_path()
    x = Rp.encode_trajectories("rel_inc", R, p)
    R2, p2 = Rp.decode_trajectories("rel_inc", x, G.make_pose(R[:, 0], p[:, 0]))
    np.testing.assert_allclose(R2, R, atol=1e-9)
    np.testing.assert_allclose(p2, p, atol=1e-9)
    assert np.all(R2[:, 0] == R[:, 0]) and np.all(p2[:, 0] == p[:, 0])
    # every increment is small even though the cumulative yaw is 270 deg (no principal-log wrap)
    dR, _ = G.poses_to_increments(R, p)
    assert np.degrees(np.linalg.norm(G.so3_log(dR), axis=-1)).max() < 30


def test_rel_inc_composition_order_matters():
    """Body increments must be right-multiplied: R_{i+1} = R_i dR_i. The wrong order fails."""
    R, p = _long_sweep_path()
    dR, dp = G.poses_to_increments(R, p, "body")
    Rw, _ = G.increments_to_poses(R[:, 0], p[:, 0], dR, dp, "spatial")  # deliberately wrong order
    assert np.degrees(G.geodesic_angle(Rw[:, -1], R[:, -1])).max() > 1.0


def test_rel_inc_endpoint_drift_accumulates():
    """Equal per-step noise in increments vs poses: increments drift at the end, poses do not."""
    R, p = _long_sweep_path()
    rng = np.random.default_rng(0)
    T0 = G.make_pose(R[:, 0], p[:, 0])
    eps = rng.normal(size=(200, 31, 9)) * 0.01
    xi = Rp.encode_trajectories("rel_inc", R, p) + eps
    xp = Rp.encode_trajectories("rel_pose", R, p) + eps
    _, pi = Rp.decode_trajectories("rel_inc", xi, np.repeat(T0, 200, 0))
    _, pp = Rp.decode_trajectories("rel_pose", xp, np.repeat(T0, 200, 0))
    e_inc = np.linalg.norm(pi[:, -1] - p[0, -1], axis=-1).mean()
    e_pose = np.linalg.norm(pp[:, -1] - p[0, -1], axis=-1).mean()
    assert e_inc > 2 * e_pose
