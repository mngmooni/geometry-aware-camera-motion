import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from camera_motion import fm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_net_shapes_and_zero_init():
    net = fm.TemporalVelocityNet(31, 9, 21)
    x, t, c = torch.randn(5, 31, 9), torch.rand(5), torch.randn(5, 21)
    out = net(x, t, c)
    assert out.shape == x.shape
    assert torch.all(out == 0)  # zero-initialised output head


def test_fm_learns_bimodal_toy():
    """1-frame, 1-channel toy: target is +-2 with p = 0.7 / 0.3; samples must recover both modes."""
    torch.manual_seed(0)
    net = fm.TemporalVelocityNet(T=4, C=1, cond_dim=1, hidden=32, emb=64, dilations=(1,))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    n = 4000
    mode = (torch.rand(n) < 0.7).float()
    x1 = (mode * 2 - (1 - mode) * 2)[:, None, None].expand(n, 4, 1).contiguous()
    c = torch.zeros(n, 1)
    for _ in range(1500):
        i = torch.randint(0, n, (256,))
        loss = fm.fm_loss(net, x1[i], c[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    s = fm.sample(net, torch.zeros(2000, 1), (2000, 4, 1), nfe=50)[:, 0, 0].numpy()
    frac_pos = np.mean(s > 0)
    assert 0.6 < frac_pos < 0.8, frac_pos
    assert np.median(np.abs(np.abs(s) - 2)) < 0.3


def test_sampler_methods_agree_in_shape():
    net = fm.TemporalVelocityNet(31, 9, 21)
    c = torch.randn(3, 21)
    for m in ("euler", "midpoint"):
        assert fm.sample(net, c, (3, 31, 9), nfe=10, method=m).shape == (3, 31, 9)


def test_ema_warmup_tracks_early_weights():
    net = torch.nn.Linear(2, 2)
    ema = fm.EMA(net, 0.999)
    with torch.no_grad():
        net.weight.add_(1.0)
    ema.update(net)
    # warmup decay at n=1 is 2/11, so EMA moves most of the way to the new weights
    np.testing.assert_allclose(ema.shadow["weight"].numpy(), net.weight.detach().numpy() - 2 / 11, atol=1e-6)


def test_train_script_runs_and_resumes(tmp_path):
    out = str(tmp_path / "run")
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "train_fm.py"), "--rep", "rel_inc",
           "--n-train", "200", "--steps", "6", "--batch", "16", "--hidden", "32",
           "--log-every", "3", "--ckpt-every", "3", "--out", out, "--cpu"]
    subprocess.run(cmd, check=True, capture_output=True)
    cmd[cmd.index("--steps") + 1] = "9"
    r = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert "resumed from step 6" in r.stdout
    ck = torch.load(os.path.join(out, "ckpt_last.pt"), weights_only=False)
    assert ck["step"] == 9
