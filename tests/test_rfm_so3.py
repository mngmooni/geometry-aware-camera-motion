import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from camera_motion import dataset_generator as D  # noqa: E402
from camera_motion import fm, rfm_so3  # noqa: E402
from camera_motion import geometry as G  # noqa: E402
from camera_motion import so3_torch as S  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(0)


def rotvecs(n, lo=0.0, hi=np.pi):
    ax = rng.normal(size=(n, 3))
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    return ax * rng.uniform(lo, hi, size=(n, 1))


def test_exp_log_match_numpy_including_near_pi():
    for lo, hi in ((0, 1e-6), (0, np.pi - 1e-3), (np.pi - 1e-4, np.pi)):
        w = rotvecs(2000, lo, hi)
        R = S.exp(torch.tensor(w))
        np.testing.assert_allclose(R.numpy(), G.so3_exp(w), atol=1e-12)
        np.testing.assert_allclose(S.exp(S.log(R)).numpy(), R.numpy(), atol=1e-9)
        assert torch.linalg.norm(S.log(R), dim=-1).max() <= np.pi + 1e-9


def test_haar_uniform_statistics():
    U = S.uniform((100000,), generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    assert G.is_valid_rotation(U.numpy(), 1e-9)
    ang = torch.linalg.norm(S.log(U), dim=-1).numpy()
    assert abs(ang.mean() - (np.pi / 2 + 2 / np.pi)) < 0.01  # Haar angle density (1 - cos)/pi
    assert abs(U.diagonal(dim1=-2, dim2=-1).sum(-1).mean().item()) < 0.02  # E[tr R] = 0


def test_geodesic_endpoints_and_constant_body_velocity():
    R0 = S.exp(torch.tensor(rotvecs(500)))
    R1 = S.exp(torch.tensor(rotvecs(500, 0, np.pi - 1e-2)))
    xi = S.log(R0.transpose(-1, -2) @ R1)
    np.testing.assert_allclose((R0 @ S.exp(1.0 * xi)).numpy(), R1.numpy(), atol=1e-9)
    # integrating the constant body velocity with the sampler's update rule reaches R1
    R = R0.clone()
    for _ in range(50):
        R = R @ S.exp(xi / 50)
    np.testing.assert_allclose(R.numpy(), R1.numpy(), atol=1e-9)
    t = 0.37
    Rt = R0 @ S.exp(t * xi)
    np.testing.assert_allclose(S.log(Rt.transpose(-1, -2) @ R1).numpy(), ((1 - t) * xi).numpy(), atol=1e-8)


def test_encode_decode_roundtrip_and_anchor():
    ds = D.generate_e1_audit("E1b", n_cond=1, group=6)
    norm, R1, X, _ = rfm_so3.build_training_tensors(ds)
    R, p = rfm_so3.decode(R1.astype(np.float64), X.astype(np.float64), norm, ds["start_pose"])
    np.testing.assert_allclose(R, ds["R"], atol=1e-5)
    np.testing.assert_allclose(p, ds["p"], atol=1e-5)
    assert np.all(R[:, 0] == ds["R"][:, 0]) and np.all(p[:, 0] == ds["p"][:, 0])


def test_sampler_outputs_valid_rotations():
    net = fm.TemporalVelocityNet(31, rfm_so3.IN_CH, 15, hidden=32, out_channels=rfm_so3.OUT_CH)
    with torch.no_grad():
        net.out.weight.normal_(0, 0.5)
    for m in ("euler", "midpoint"):
        R, x = rfm_so3.sample(net, torch.randn(4, 15), 31, nfe=10, method=m)
        assert R.shape == (4, 31, 3, 3) and x.shape == (4, 31, 3)
        assert G.is_valid_rotation(R.double().numpy(), 1e-5) and torch.isfinite(x).all()


def test_learns_bimodal_rotation_toy():
    """Each frame's target is Rz(+100 deg) w.p. 0.7 or Rz(-100 deg) w.p. 0.3 (shared per sample)."""
    torch.manual_seed(0)
    T, n = 4, 4000
    mode = (torch.rand(n) < 0.7)
    ang = torch.where(mode, torch.tensor(np.deg2rad(100.0)), torch.tensor(np.deg2rad(-100.0))).float()
    w = torch.zeros(n, 3)
    w[:, 2] = ang
    R1 = S.exp(w)[:, None].expand(n, T, 3, 3).contiguous()
    x1 = torch.zeros(n, T, 3)
    c = torch.zeros(n, 1)
    net = fm.TemporalVelocityNet(T, rfm_so3.IN_CH, 1, hidden=32, emb=64, dilations=(1,), out_channels=rfm_so3.OUT_CH)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(1500):
        i = torch.randint(0, n, (256,))
        loss = rfm_so3.loss(net, R1[i], x1[i], c[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    R, _ = rfm_so3.sample(net, torch.zeros(1000, 1), T, nfe=50)
    yaw = torch.atan2(R[:, 0, 1, 0], R[:, 0, 0, 0]).numpy()
    frac_pos = np.mean(yaw > 0)
    assert 0.6 < frac_pos < 0.8, frac_pos
    near = np.minimum(np.abs(yaw - np.deg2rad(100)), np.abs(yaw + np.deg2rad(100)))
    assert np.median(np.degrees(near)) < 10


def test_train_and_eval_scripts_run(tmp_path):
    out = str(tmp_path / "rfm")
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "train_fm.py"), "--rep", "rfm_so3", "--family", "E1b",
                    "--n-train", "200", "--steps", "6", "--batch", "16", "--hidden", "32", "--log-every", "3",
                    "--ckpt-every", "3", "--out", out, "--cpu"], check=True, capture_output=True)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "eval_fm.py"), "--run", out, "--K", "20",
                        "--nfe", "4", "--max-conditions", "2", "--cpu"], check=True, capture_output=True, text=True)
    assert "rfm_so3" in r.stdout and "TV5" in r.stdout
