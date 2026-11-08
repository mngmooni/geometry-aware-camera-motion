import numpy as np
import pytest

from camera_motion import dataset_generator as D
from camera_motion import robustness as Rb


@pytest.fixture(scope="module", params=["E1a", "E1b"])
def probes(request):
    ds = D.generate_e1_train(request.param, n=600)
    return Rb.run_probes(ds["p"], ds["subject"], ds["latent_mode"], request.param, seed=1)


def test_exact_invariances(probes):
    assert probes["baseline"]["agree"] == 1.0
    assert all(probes["invariance"].values()), probes["invariance"]


def test_wrong_mode_assignment_is_rare(probes):
    # Observed: 0 everywhere except xy-jitter sigma=0.5 m (~0.1%, radial flips on a
    # 3-5 m orbit that is already badly corrupted). Fail-safe = unassigned, not wrong.
    assert probes["max_wrong_rate_over_all_curves"] <= 0.005
    for name, curve in probes["curves"].items():
        for level, r in curve.items():
            if float(level) <= 0.3:
                assert r["wrong"] == 0.0, (name, level, r)


def test_moderate_noise_keeps_agreement(probes):
    c = probes["curves"]
    assert c["jitter_xyz"]["0.1"]["agree"] >= 0.99
    assert c["jitter_xyz"]["0.2"]["agree"] >= 0.95  # was 0.64 with the v1.0 max-planarity rule
    assert c["jitter_xy"]["0.3"]["agree"] >= 0.99
    assert c["lowfreq_xyz"]["0.2"]["agree"] >= 0.99


def test_degrades_towards_unassigned(probes):
    c = probes["curves"]["jitter_z"]
    assert c["0.5"]["unassigned"] > c["0.1"]["unassigned"]


def test_time_reversal_is_unassigned(probes):
    assert probes["time_reversed"]["wrong_rate"] == 0.0


def test_temporal_resampling_invariance():
    assert Rb.temporal_resampling_invariance("E1a", L_dense=64, n=200)
