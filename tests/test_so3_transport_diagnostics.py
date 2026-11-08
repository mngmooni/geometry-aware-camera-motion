import importlib.util
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from camera_motion import so3_torch as S  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("dx", os.path.join(ROOT, "scripts", "diagnose_so3_transport.py"))
dx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dx)


def _smooth_targets(n=50, T=31):
    t = torch.linspace(0, 1, T, dtype=torch.float64)
    w = torch.zeros(n, T, 3, dtype=torch.float64)
    w[..., 2] = 1.5 * t  # constant-rate yaw: omega constant across frames
    return S.exp(w)


def test_step_metrics_on_constant_rate_rotation():
    d, a = dx.step_metrics(_smooth_targets())
    np.testing.assert_allclose(d.numpy(), 1.5 / 30, atol=1e-12)
    np.testing.assert_allclose(a.numpy(), 0.0, atol=1e-12)


def test_geodesic_path_endpoints():
    R1 = _smooth_targets(20)
    R0 = dx.independent_haar(R1.shape[:2], 3)
    np.testing.assert_allclose(dx.geodesic_path(R0, R1, 0.0)[0].numpy(), R0.numpy(), atol=1e-14)
    np.testing.assert_allclose(dx.geodesic_path(R0, R1, 1.0)[0].numpy(), R1.numpy(), atol=1e-12)


def test_shared_prior_is_target_independent_and_flat_in_time():
    Qa = dx.shared_haar(10, 31, 5)
    Qb = dx.shared_haar(10, 31, 5)
    assert torch.equal(Qa, Qb)  # depends only on (n, T, seed), never on targets
    d, a = dx.step_metrics(Qa)
    assert d.max() < 1e-7 and a.max() < 1e-7
    # same per-frame marginal as Haar: mean angle ~ pi/2 + 2/pi
    ang = torch.linalg.norm(S.log(dx.shared_haar(20000, 1, 9)[:, 0]), dim=-1).mean().item()
    assert abs(ang - (np.pi / 2 + 2 / np.pi)) < 0.03


def test_independent_prior_matches_haar_step_statistics():
    d, _ = dx.step_metrics(dx.independent_haar((4000, 31), 1))
    assert abs(d.mean().item() - (np.pi / 2 + 2 / np.pi)) < 0.02  # adjacent independent Haar frames


def test_near_pi_share_matches_haar_theory():
    R1 = _smooth_targets(2000)
    R0 = dx.independent_haar(R1.shape[:2], 2)
    _, xi = dx.geodesic_path(R0, R1, 1.0)
    frac = (xi.norm(dim=-1) > np.pi - np.radians(10)).double().mean().item()
    delta = np.radians(10)
    theory = (delta + np.sin(delta)) / np.pi  # integral of (1 - cos th)/pi over [pi - delta, pi]
    assert abs(frac - theory) < 0.01


def test_runs_end_to_end_small(tmp_path):
    a = dx.parse(["--out", str(tmp_path), "--n", "40", "--n-train", "200"])
    s = dx.main(a)
    for fam in ("E1a", "E1b"):
        rows = s["roughness"][fam]
        gt_t1 = [r for r in rows if r["t"] == 1.0]
        assert all(abs(r["delta_median"] - gt_t1[0]["delta_median"]) < 1e-6 for r in gt_t1)  # all paths end at data
        assert s["transport_targets"][fam]["independent_haar"]["checks"]["all_finite"]
    assert os.path.exists(os.path.join(tmp_path, "temporal_roughness_vs_flow_time.png"))
