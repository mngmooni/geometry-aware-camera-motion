import os

import numpy as np
import pytest

from camera_motion import dataset_generator as D
from camera_motion import geometry as G
from camera_motion.config import FROZEN
from camera_motion.metrics import mode_condition_independence

TARGET = np.array(FROZEN.modes.joint())


@pytest.fixture(scope="module")
def train_a():
    return D.generate_e1_train("E1a")


@pytest.fixture(scope="module")
def audit_a():
    return D.generate_e1_audit("E1a")


@pytest.fixture(scope="module")
def audit_b():
    return D.generate_e1_audit("E1b")


def test_declared_probabilities():
    np.testing.assert_allclose(TARGET, [0.42, 0.28, 0.18, 0.12])
    assert abs(TARGET.sum() - 1) < 1e-12


def test_stratified_counts_exact():
    assert D.stratified_counts(100, TARGET).tolist() == [42, 28, 18, 12]
    for n in (1, 7, 33, 999, 1000):
        c = D.stratified_counts(n, TARGET)
        assert c.sum() == n and np.all(np.abs(c - TARGET * n) < 1)


def test_determinism():
    a = D.generate_e1_train("E1a", n=50)
    b = D.generate_e1_train("E1a", n=50)
    np.testing.assert_array_equal(a["p"], b["p"])
    np.testing.assert_array_equal(a["R"], b["R"])
    np.testing.assert_array_equal(a["latent_mode"], b["latent_mode"])


def test_train_many_unique_conditions_and_split_level_stratification(train_a):
    n = len(train_a["p"])
    assert len(np.unique(train_a["condition_id"])) == n  # one path per condition
    counts = np.bincount(train_a["latent_mode"], minlength=4)
    assert counts.tolist() == D.stratified_counts(n, TARGET).tolist()
    # distinct continuous condition values (not repetitions)
    assert len(np.unique(np.round(train_a["condition_params"][:, 0], 9))) == n


def test_train_mode_independent_of_condition(train_a):
    res = mode_condition_independence(train_a)
    for name, r in res.items():
        assert r["p_value"] > 1e-3, (name, r)


def test_audit_exact_counts_per_condition(audit_a, audit_b):
    for ds in (audit_a, audit_b):
        for cid in np.unique(ds["condition_id"]):
            m = ds["latent_mode"][ds["condition_id"] == cid]
            assert np.bincount(m, minlength=4).tolist() == [42, 28, 18, 12]


@pytest.mark.parametrize("fixture", ["audit_a", "audit_b"])
def test_shared_public_condition_is_bit_exact(fixture, request):
    ds = request.getfixturevalue(fixture)
    for cid in np.unique(ds["condition_id"]):
        sel = ds["condition_id"] == cid
        for key in ("start_pose", "end_pose", "subject"):
            v = ds[key][sel]
            assert np.all(v == v[0]), key
        # and the poses inside the paths agree with the public condition
        assert np.all(ds["p"][sel][:, 0] == ds["start_pose"][sel][:, :3, 3])
        assert np.all(ds["R"][sel][:, -1] == ds["end_pose"][sel][:, :3, :3])


def test_all_rotations_valid(train_a, audit_b):
    for ds in (train_a, audit_b):
        assert G.is_valid_rotation(ds["R"], 1e-9)


def test_e1a_rotation_is_lookat(audit_a):
    R_look = G.look_at(audit_a["p"], audit_a["subject"][:, None, :])
    np.testing.assert_allclose(R_look, audit_a["R"], atol=1e-12)


def test_e1b_same_position_different_rotation_identical_endpoints():
    cfg = FROZEN
    rng = np.random.default_rng(3)
    cond = D.sample_conditions(rng, 1)
    s = cond.subject[0]
    p = D.e1_positions(s, cond.phi0[0], cond.r0[0], cond.r1[0], cond.h[0], 0, 0.1, 0.35, cfg.split.L)
    R0 = G.look_at(p, s)
    Ra = R0 @ D.offset_rotations(cfg.split.L, 0, np.deg2rad(40), 0.4)
    Rb = R0 @ D.offset_rotations(cfg.split.L, 1, -np.deg2rad(55), 0.6)
    assert np.all(Ra[0] == Rb[0]) and np.all(Ra[-1] == Rb[-1])
    assert np.all(Ra[0] == R0[0]) and np.all(Ra[-1] == R0[-1])
    mid = np.degrees(G.geodesic_angle(Ra[1:-1], Rb[1:-1]))
    assert mid.max() > 40  # genuinely different intermediate orientation


def test_e1b_offset_is_nontrivial_and_alias_free(audit_b):
    R_look = G.look_at(audit_b["p"], audit_b["subject"][:, None, :])
    ang = np.degrees(G.geodesic_angle(R_look, audit_b["R"]))
    assert np.all(ang[:, 0] == 0) and np.all(ang[:, -1] == 0)
    assert np.median(ang.max(axis=1)) > 25
    step = np.degrees(G.geodesic_angle(audit_b["R"][:, :-1], audit_b["R"][:, 1:]))
    assert step.max() < 45


def test_e1b_offsets_independent_of_mode(audit_b):
    off = audit_b["orientation_offset_params"]
    m = audit_b["latent_mode"]
    # pan/roll split should not be extreme within any mode
    for k in range(4):
        frac = np.mean(off[m == k, 0] == 0)
        assert 0.2 < frac < 0.8


def test_split_condition_ids_and_values_disjoint():
    names = {
        "train": D.generate_e1_train("E1a", n=300),
        "audit": D.generate_e1_audit("E1a"),
        "eval_id": D.generate_e1_eval("E1a", "eval_id", n_cond=30, n_ref=5),
        "eval_ood": D.generate_e1_eval("E1a", "eval_ood_position", n_cond=10, n_ref=5),
    }
    ids = {k: set(v["condition_id"]) for k, v in names.items()}
    vals = {k: set(map(tuple, np.round(v["condition_params"], 9))) for k, v in names.items()}
    keys = list(names)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not (ids[keys[i]] & ids[keys[j]])
            assert not (vals[keys[i]] & vals[keys[j]])


def test_ood_position_outside_train_support():
    ood = D.generate_e1_eval("E1a", "eval_ood_position", n_cond=20, n_ref=2)
    rad = np.linalg.norm(ood["subject"][:, :2], axis=-1)
    lo, hi = FROZEN.cond.subject_xy_train
    assert np.all(rad >= FROZEN.cond.subject_xy_ood_annulus[0])
    assert np.all(np.max(np.abs(ood["subject"][:, :2]), axis=-1) > hi) or np.all(rad > np.sqrt(2) * hi)


def test_public_view_has_no_hidden_fields(audit_a):
    pub = D.public_conditions(audit_a)
    assert set(pub) == set(D.PUBLIC_KEYS)
    assert len(pub["condition_id"]) == len(np.unique(audit_a["condition_id"]))


def test_conditions_from_public_roundtrip(audit_a):
    pub = D.public_conditions(audit_a)
    i0 = np.nonzero(audit_a["condition_id"] == pub["condition_id"][0])[0][0]
    c = D.conditions_from_public(pub["start_pose"][0], pub["end_pose"][0], pub["subject"][0])
    phi0, r0, r1, h = audit_a["condition_params"][i0]
    assert abs(G.wrap_angle(c.phi0[0] - phi0)) < 1e-12
    np.testing.assert_allclose([c.r0[0], c.r1[0], c.h[0]], [r0, r1, h], atol=1e-12)


def test_save_load_roundtrip(tmp_path, audit_a):
    path = os.path.join(tmp_path, "a.npz")
    D.save_dataset(audit_a, path)
    b = D.load_dataset(path)
    np.testing.assert_array_equal(b["p"], audit_a["p"])
    np.testing.assert_array_equal(b["condition_id"], audit_a["condition_id"])
    assert b["meta"]["declared_target_mode_probability"] == list(TARGET)


def test_e0_sanity():
    e0 = D.generate_e0(n_per_kind=5)
    assert G.is_valid_rotation(e0["R"], 1e-9)
    tr = e0["shot"] == "tracking"
    assert np.all(e0["R"][tr] == e0["R"][tr][:, :1])  # fixed orientation
    do = e0["shot"] == "dolly"
    d = np.linalg.norm(e0["p"][do] - e0["subject"][do][:, None], axis=-1)
    assert np.all(np.diff(d, axis=1) < 0)  # moving in
