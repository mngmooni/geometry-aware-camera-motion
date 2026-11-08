import inspect

import numpy as np
import pytest

from camera_motion import dataset_generator as D
from camera_motion import mode_classifier as C
from camera_motion.config import FROZEN, UNASSIGNED_ID
from camera_motion.metrics import confusion_matrix


@pytest.fixture(scope="module", params=["E1a", "E1b"])
def audit(request):
    return D.generate_e1_audit(request.param)


def circle(sweep, L=32, r=4.0, bump=0.0, s=np.zeros(3)):
    u = np.linspace(0, 1, L)
    phi = sweep * u
    rr = r * (1 + bump * np.sin(np.pi * u))
    return np.stack([s[0] + rr * np.cos(phi), s[1] + rr * np.sin(phi), np.full(L, s[2] + 1.0)], -1)


def test_gt_audit_confusion_is_diagonal(audit):
    res = C.classify_dataset(audit)
    M = confusion_matrix(audit["latent_mode"], res["label"])
    assert M[:, 4].sum() == 0
    assert np.all(M[:, :4] == np.diag(np.diag(M[:, :4])))


def test_gt_margins_clear_thresholds(audit):
    res = C.classify_dataset(audit)
    spec = FROZEN.clf
    assert np.min(np.abs(res["rho"])) > 2 * spec.tau_far
    d = res["delta_phi"]
    short = audit["latent_mode"] < 2
    np.testing.assert_allclose(d[short], np.pi / 2, atol=1e-9)
    np.testing.assert_allclose(d[~short], -1.5 * np.pi, atol=1e-9)


def test_long_cw_not_confused_with_short_ccw():
    # same endpoints (bearing 0 -> 90 deg) but opposite windings
    short = C.classify_path(circle(np.pi / 2, bump=-0.3), np.zeros(3), "E1a")
    long = C.classify_path(circle(-1.5 * np.pi, bump=-0.3), np.zeros(3), "E1a")
    assert short.name == "short_ccw+near" and long.name == "long_cw+near"
    np.testing.assert_allclose(short.delta_phi, np.pi / 2)
    np.testing.assert_allclose(long.delta_phi, -1.5 * np.pi)


def test_far_branch():
    assert C.classify_path(circle(np.pi / 2, bump=0.3), np.zeros(3), "E1a").name == "short_ccw+far"


def test_aliasing_guard():
    P = circle(-1.5 * np.pi, L=3, bump=-0.3)  # 135 deg per step
    r = C.classify_path(P, np.zeros(3), "E1a")
    assert r.label == UNASSIGNED_ID and r.reason == "angular_ambiguous"


def test_wrong_sweep_unassigned():
    r = C.classify_path(circle(np.pi, bump=-0.3), np.zeros(3), "E1a")  # 180 deg: neither branch
    assert r.reason == "angular_ambiguous"


def test_radial_gray_zone():
    r = C.classify_path(circle(np.pi / 2, bump=0.0), np.zeros(3), "E1a")
    assert r.reason == "radial_ambiguous"


@pytest.mark.parametrize("bad", ["nan", "subject", "nonplanar"])
def test_invalid_geometry(bad):
    P = circle(np.pi / 2, bump=-0.3)
    if bad == "nan":
        P[5, 0] = np.nan
    elif bad == "subject":
        P[10, :2] = 0.1
    else:
        P[:, 2] += np.linspace(0, 2, len(P))
    r = C.classify_path(P, np.zeros(3), "E1a")
    assert r.reason == "invalid_geometry" and r.label == UNASSIGNED_ID


def test_scope_guard_rejects_e2_and_e0():
    e2 = D.generate_e2_composition_stub(n=5)
    with pytest.raises(C.ClassifierScopeError):
        C.classify_dataset(e2)
    with pytest.raises(C.ClassifierScopeError):
        C.classify_path(e2["p"][0], e2["subject"][0], "E0")


def test_classifier_is_label_blind(audit):
    # signature carries no label/seed/model inputs
    params = set(inspect.signature(C.classify_path).parameters)
    assert params == {"positions", "subject", "family", "spec", "sweeps"}
    # shuffling hidden labels in the dict cannot change the output
    shuffled = dict(audit)
    shuffled["latent_mode"] = np.random.default_rng(0).permutation(audit["latent_mode"])
    shuffled["orientation_offset_params"] = np.zeros_like(audit["orientation_offset_params"])
    a = C.classify_dataset(audit)["label"]
    b = C.classify_dataset(shuffled)["label"]
    np.testing.assert_array_equal(a, b)
