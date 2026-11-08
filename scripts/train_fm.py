"""Train a Euclidean FM baseline.

Phase 2 runs ONLY  --rep abs_pose  (experiment key fm_abs_pose). rel_pose / rel_inc use
the identical script and architecture and are run at later gates.

    python scripts/train_fm.py --rep abs_pose --family E1a --steps 20000 --out runs/fm_abs_pose_E1a_s0
    (re-running the same command resumes from runs/.../ckpt_last.pt)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_motion import dataset_generator as D  # noqa: E402
from camera_motion import fm  # noqa: E402
from camera_motion import representations as Rp  # noqa: E402
from camera_motion import rfm_so3  # noqa: E402
from camera_motion.config import FROZEN  # noqa: E402


def build_training_tensors(rep: str, family: str, n_train: int):
    ds = D.generate_e1_train(family, n=n_train)
    x = Rp.encode_trajectories(rep, ds["R"], ds["p"])
    c = Rp.encode_condition(rep, ds["start_pose"], ds["end_pose"], ds["subject"])
    norm = Rp.Normalizer.fit(x, c)
    return ds, norm, norm.nx(x).astype(np.float32), norm.nc(c).astype(np.float32)


def lr_at(step, total, base, warmup=500):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + np.cos(np.pi * min(p, 1.0)))


def main(a):
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    t_data = time.time()
    riem = a.rep == "rfm_so3"
    if riem:
        ds = D.generate_e1_train(a.family, n=a.n_train)
        norm, Rtr, X, Cn = rfm_so3.build_training_tensors(ds)
        R_t = torch.from_numpy(Rtr).to(dev)
        Ch, out_ch = rfm_so3.IN_CH, rfm_so3.OUT_CH
    else:
        ds, norm, X, Cn = build_training_tensors(a.rep, a.family, a.n_train)
        Ch = X.shape[-1]
        out_ch = Ch
    t_data = time.time() - t_data
    X_t, C_t = torch.from_numpy(X).to(dev), torch.from_numpy(Cn).to(dev)
    N, T = X.shape[:2]

    model = fm.TemporalVelocityNet(T, Ch, C_t.shape[1], hidden=a.hidden, out_channels=out_ch).to(dev)
    ema = fm.EMA(model, a.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    step = 0
    ck = os.path.join(a.out, "ckpt_last.pt")
    if os.path.exists(ck) and not a.no_resume:
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"])
        ema.shadow = {k: v.to(dev) for k, v in s["ema"].items()}
        opt.load_state_dict(s["opt"])
        step = s["step"]
        ema.n = step
        torch.set_rng_state(s["torch_rng"])
        print(f"resumed from step {step}")

    run_meta = {
        "experiment_key": "rfm_so3" if riem else f"fm_{a.rep}", "rep": a.rep, "out_channels": out_ch, "family": a.family, "n_train": int(N),
        "n_train_conditions": int(len(np.unique(ds["condition_id"]))),
        "train_mode_counts": np.bincount(ds["latent_mode"], minlength=4).tolist(),
        "T": T, "channels": Ch, "cond_dim": int(C_t.shape[1]), "hidden": a.hidden,
        "params": fm.count_params(model), "steps": a.steps, "batch": a.batch, "lr": a.lr, "ema": a.ema,
        "seed": a.seed,
        "prior": rfm_so3.PRIOR if riem else "isotropic Gaussian N(0, I) in per-channel standardised coordinates",
        "path": rfm_so3.PATH if riem else "linear (rectified) x_t=(1-t)x0+t x1, uniform t", "device": str(dev),
        "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
        "torch": torch.__version__, "python": platform.python_version(),
        "classifier_version": FROZEN.clf.version, "data_gen_sec": round(t_data, 2),
    }
    with open(os.path.join(a.out, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)
    print(json.dumps({k: run_meta[k] for k in ("experiment_key", "n_train", "params", "device", "gpu")}))

    log = open(os.path.join(a.out, "train_log.jsonl"), "a")
    gen = torch.Generator(device=dev).manual_seed(a.seed * 1000 + step)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0, ema_loss, start_step = time.time(), None, step
    model.train()
    while step < a.steps:
        idx = torch.randint(0, N, (a.batch,), device=dev, generator=gen)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr)
        if riem:
            loss = rfm_so3.loss(model, R_t[idx], X_t[idx], C_t[idx], generator=gen)
        else:
            loss = fm.fm_loss(model, X_t[idx], C_t[idx], generator=gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)
        step += 1
        lv = loss.item()
        ema_loss = lv if ema_loss is None else 0.98 * ema_loss + 0.02 * lv
        if step % a.log_every == 0 or step == a.steps:
            el = time.time() - t0
            rec = {"step": step, "loss": lv, "loss_smooth": ema_loss, "grad_norm": float(gn),
                   "lr": opt.param_groups[0]["lr"], "elapsed_sec": round(el, 1)}
            if dev.type == "cuda":
                rec["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"step {step:6d}  loss {ema_loss:.4f}  lr {rec['lr']:.2e}  {el:.0f}s")
        if step % a.ckpt_every == 0 or step == a.steps:
            torch.save({"model": model.state_dict(), "ema": ema.shadow, "opt": opt.state_dict(),
                        "step": step, "torch_rng": torch.get_rng_state(), "norm": norm.state(),
                        "meta": run_meta}, ck)
    log.close()
    el = time.time() - t0
    summary = {"final_step": step, "train_wall_sec_this_session": round(el, 1),
               "steps_per_sec": round((step - start_step) / max(el, 1e-9), 2)}
    if dev.type == "cuda":
        summary["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    with open(os.path.join(a.out, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("done", summary)


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="abs_pose", choices=Rp.REPRESENTATIONS + ("rfm_so3",))
    ap.add_argument("--family", default="E1a", choices=["E1a", "E1b"])
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--out", default="runs/fm_abs_pose_E1a_s0")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    main(parse())
