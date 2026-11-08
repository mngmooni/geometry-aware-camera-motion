import numpy as np
import pytest

from camera_motion import dataset_generator as D
from camera_motion import geometry as G
from camera_motion import metrics as M
from camera_motion.config import UNASSIGNED_ID


def test_frequency5_counts_unassigned_without_renormalising():
    labels = np.array([0] * 40 + [1] * 30 + [UNASSIGNED_ID] * 30)
    f = M.frequency5(labels)
    np.testing.assert_allclose(f, [0.4, 0.3, 0, 0, 0.3])
    rep = M.mode_report(labels)
    assert rep["coverage4"] == 2
    assert rep["unassigned_rate"] == pytest.approx(0.3)
    # TV over five categories
    expected = 0.5 * (abs(0.4 - 0.42) + abs(0.3 - 0.28) + 0.18 + 0.12 + 0.3)
    assert rep["tv5"] == pytest.approx(expected)
    # secondary assigned-only frequencies are labelled separately
    np.testing.assert_allclose(rep["secondary_freq_given_assigned"], [4 / 7, 3 / 7, 0, 0])


def test_perfect_frequencies_give_zero_tv():
    labels = np.repeat(np.arange(4), [42, 28, 18, 12])
    rep = M.mode_report(labels)
    assert rep["tv5"] == pytest.approx(0.0) and rep["coverage4"] == 4
    assert rep["minority_mode"] == "long_cw+far"


@pytest.fixture(scope="module")
def eval_panel():
    ds = D.generate_e1_eval("E1a", "eval_id", n_cond=8, n_ref=4)
    return ds, D.public_conditions(ds)


def test_contract_oracle_sampler_is_calibrated(eval_panel):
    _, pub = eval_panel
    out = M.evaluate_mode_contract(D.oracle_sampler("E1a"), pub, "E1a", K=300, seed=1)
    P = out["pooled"]
    assert P["unassigned_rate_mean"] == 0
    assert P["full_coverage_rate"] == 1.0
    assert P["tv5_mean"] < 0.08  # finite-K noise only
    assert out["endpoint"]["e_p_mean"] < 1e-9 and out["endpoint"]["e_R_deg_mean"] < 1e-6


def test_contract_collapsed_sampler(eval_panel):
    _, pub = eval_panel
    oracle = D.oracle_sampler("E1a")

    def collapsed(row, K, rng):
        cond = D.conditions_from_public(row["start_pose"], row["end_pose"], row["subject"])
        out = [D._e1_one(cond, 0, 0, rng, "E1a", D.FROZEN)[:2] for _ in range(K)]
        return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])

    out = M.evaluate_mode_contract(collapsed, pub, "E1a", K=60)
    assert out["pooled"]["coverage4_mean"] == 1
    assert out["pooled"]["tv5_mean"] == pytest.approx(0.58)
    assert out["pooled"]["minority_error_mean"] == pytest.approx(-0.12)
    del oracle


def test_contract_garbage_sampler_is_all_unassigned(eval_panel):
    _, pub = eval_panel

    def garbage(row, K, rng):
        L = 32
        return np.broadcast_to(np.eye(3), (K, L, 3, 3)).copy(), rng.normal(size=(K, L, 3)) * 10

    out = M.evaluate_mode_contract(garbage, pub, "E1a", K=50)
    assert out["pooled"]["unassigned_rate_mean"] == 1.0
    assert out["pooled"]["tv5_mean"] == pytest.approx(1.0)


def test_contract_sampler_sees_only_public_fields(eval_panel):
    _, pub = eval_panel
    seen = []

    def spy(row, K, rng):
        seen.append(set(row))
        return D.oracle_sampler("E1a")(row, K, rng)

    M.evaluate_mode_contract(spy, pub, "E1a", K=5, min_k=1)
    assert all(s == set(D.PUBLIC_KEYS) for s in seen)


def test_min_k_flag(eval_panel):
    _, pub = eval_panel
    out = M.evaluate_mode_contract(D.oracle_sampler("E1a"), pub, "E1a", K=10, min_k=50)
    assert out["pooled"]["any_condition_under_min_k"]


def test_endpoint_and_start_errors_zero_on_gt():
    ds = D.generate_e1_audit("E1b", n_cond=1, group=20)
    e = M.endpoint_errors(ds["R"], ds["p"], ds["end_pose"][0])
    s = M.start_errors(ds["R"], ds["p"], ds["start_pose"][0])
    assert e["e_p"].max() == 0 and e["e_R_deg"].max() < 1e-6
    assert s["e_p"].max() == 0 and s["e_R_deg"].max() < 1e-6


def test_endpoint_error_detects_rotation_offset():
    ds = D.generate_e1_audit("E1a", n_cond=1, group=5)
    R = ds["R"].copy()
    R[:, -1] = R[:, -1] @ G.so3_exp(np.array([0, 0, np.deg2rad(10)]))
    e = M.endpoint_errors(R, ds["p"], ds["end_pose"][0])
    np.testing.assert_allclose(e["e_R_deg"], 10, atol=1e-8)


def test_excursion_features_recover_latents():
    ds = D.generate_e1_audit("E1b", n_cond=1, group=100)
    f = M.orientation_excursion_features(ds["R"], ds["p"], ds["subject"])
    off = ds["orientation_offset_params"]
    amp = np.degrees(np.abs(off[:, 1]))
    assert np.max(np.abs(f["max_excursion_deg"] - amp)) < 1.0  # grid-sampled peak
    assert np.max(np.abs(f["peak_time"] - off[:, 2])) <= 1 / 31 + 1e-9
    pan = off[:, 0] == 0
    np.testing.assert_array_equal(np.sign(f["signed_pan_peak_deg"][pan]), np.sign(off[pan, 1]))
    np.testing.assert_array_equal(np.sign(f["signed_roll_peak_deg"][~pan]), np.sign(off[~pan, 1]))


def test_excursion_distribution_distance_is_unpaired_and_discriminative():
    a = D.generate_e1_eval("E1b", "eval_id", n_cond=1, n_ref=200)
    cid = a["condition_id"][0]
    pub = D.public_conditions(a)
    rng = np.random.default_rng(7)
    R2, p2 = D.oracle_sampler("E1b")({k: pub[k][0] for k in pub}, 200, rng)
    R0, p0 = D.oracle_sampler("E1a")({k: pub[k][0] for k in pub}, 200, rng)  # no excursion
    fa = M.orientation_excursion_features(a["R"], a["p"], a["subject"])
    fb = M.orientation_excursion_features(R2, p2, a["subject"][0])
    fc = M.orientation_excursion_features(R0, p0, a["subject"][0])
    same = M.excursion_distribution_distance(fb, fa)
    diff = M.excursion_distribution_distance(fc, fa)
    assert same["max_excursion_deg"] < 5 < diff["max_excursion_deg"]
    # unpaired: permuting one set does not change the distance
    perm = {k: v[np.random.default_rng(1).permutation(len(v))] for k, v in fb.items()}
    assert M.excursion_distribution_distance(perm, fa) == same
    del cid


def test_smoothness_ratio_gt_vs_gt():
    ds = D.generate_e1_audit("E1a", n_cond=2, group=100)
    a = M.smoothness(ds["R"][:100], ds["p"][:100])
    b = M.smoothness(ds["R"][100:], ds["p"][100:])
    r = M.smoothness_ratio(a, b)
    assert all(0.5 < v < 2.0 for v in r.values())


def test_aggregate_reports_bootstrap():
    reps = {f"c{i}": M.mode_report(np.repeat(np.arange(4), [42, 28, 18, 12])) for i in range(5)}
    agg = M.aggregate_condition_reports(reps)
    assert agg["n_conditions"] == 5 and agg["tv5_mean"] == pytest.approx(0.0)
