import numpy as np
import pytest

from camera_motion import dataset_generator as D
from camera_motion import metrics as M
from camera_motion.config import FROZEN


def test_dispersion_calibrated_sampler_is_near_one():
    rng = np.random.default_rng(0)
    counts = np.concatenate([rng.multinomial(200, M.TARGET5[:4], size=50), np.zeros((50, 1))], 1)
    d = M.conditional_dispersion(counts, seed=1)
    assert 0.75 < d["excess_ratio"] < 1.3 and d["p_value"] > 0.01
    assert d["sd_p_long"] == pytest.approx(d["binomial_sd_p_long"], rel=0.35)


def test_dispersion_detects_offsetting_skews():
    # half the conditions over-produce long_cw, half under-produce: pooled is exact
    K = 200
    hi = np.array([0.30, 0.20, 0.30, 0.20]) * K
    lo = np.array([0.54, 0.36, 0.06, 0.04]) * K
    counts = np.vstack([np.tile(hi, (25, 1)), np.tile(lo, (25, 1))])
    counts = np.concatenate([counts, np.zeros((50, 1))], 1)
    pooled = counts[:, :4].sum(0) / counts.sum()
    np.testing.assert_allclose(pooled, M.TARGET5[:4], atol=1e-12)
    d = M.conditional_dispersion(counts)
    assert d["excess_ratio"] > 5 and d["p_value"] < 1e-3


def test_dispersion_requires_equal_k():
    with pytest.raises(ValueError):
        M.conditional_dispersion(np.array([[10, 0, 0, 0, 0], [5, 0, 0, 0, 0]]))


def test_support_violation_zero_on_gt_and_positive_when_shrunk():
    ds = D.generate_e1_eval("E1b", "eval_id", n_cond=2, n_ref=200)
    f = M.orientation_excursion_features(ds["R"], ds["p"], ds["subject"])
    v = M.excursion_support_violation(f)
    assert v["amp_outside_support"] == 0 and v["peak_time_outside_support"] == 0
    f2 = dict(f, max_excursion_deg=f["max_excursion_deg"] * 0.5)
    assert 0.35 < M.excursion_support_violation(f2)["amp_below_support"] < 0.55  # orig amp < 38 deg


def test_e1b_eval_uses_200_refs_and_same_conditions():
    a = D.generate_e1_eval("E1b", "eval_id", n_cond=3)
    b = D.generate_e1_eval("E1b", "eval_id", n_cond=3, n_ref=20)
    assert np.sum(a["condition_id"] == a["condition_id"][0]) == FROZEN.split.eval_gt_ref_per_condition_e1b
    pa, pb = D.public_conditions(a), D.public_conditions(b)
    np.testing.assert_array_equal(pa["start_pose"], pb["start_pose"])  # ref size does not move conditions
    np.testing.assert_array_equal(pa["subject"], pb["subject"])


def test_dispersion_decomposition_is_exact_and_separates_parts():
    rng = np.random.default_rng(3)
    counts = np.concatenate([rng.multinomial(200, M.TARGET5[:4], size=40), np.zeros((40, 1))], 1)
    d = M.dispersion_decomposition(counts, seed=1)
    full = M.conditional_dispersion(counts, seed=1)
    assert d["bias"]["chi2"] + d["between"]["chi2"] == pytest.approx(full["chi2"], rel=1e-10)
    assert d["bias"]["p_value"] > 0.01 and d["between"]["p_value"] > 0.01
    # pure global bias: every condition identical but shifted -> between ~ 0, bias huge
    skew = np.tile(np.array([0.52, 0.28, 0.12, 0.08, 0.0]) * 200, (40, 1))
    d2 = M.dispersion_decomposition(skew)
    assert d2["between"]["chi2"] == pytest.approx(0.0, abs=1e-9) and d2["bias"]["p_value"] < 1e-3
