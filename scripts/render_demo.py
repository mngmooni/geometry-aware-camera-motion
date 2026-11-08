"""Render camera-motion demo videos from saved model samples (no GPU needed).

    # four sampled modes from ONE public condition (E1a)
    python scripts/render_demo.py modes --eval-dir <run>/eval_m2_K200_nfe50_euler --family E1a --out demo/
    # E1b: two GT paths vs two generated paths under the same condition (orientation excursions)
    python scripts/render_demo.py gt_vs_gen --eval-dir <run>/eval_m2_K200_nfe50_euler --family E1b --out demo/

Samples come from ``<panel>_per_sample.npz`` (kept_R / kept_p written by eval_fm.py).
Representative samples are chosen by the frozen label-blind classifier: for each mode,
the sample closest to that mode's median (unwrapped sweep, interior radial offset).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_motion import dataset_generator as D  # noqa: E402
from camera_motion import demo as Dm  # noqa: E402
from camera_motion import metrics as M  # noqa: E402
from camera_motion import mode_classifier as C  # noqa: E402
from camera_motion.config import MODE_NAMES  # noqa: E402
from camera_motion.visualize import MODE_COLORS  # noqa: E402


def load_kept(eval_dir, panel):
    z = np.load(os.path.join(eval_dir, f"{panel}_per_sample.npz"))
    if z["kept_p"].size == 0:
        raise SystemExit("no kept samples in this eval (run eval_fm.py with --keep >= 1)")
    rep = json.load(open(os.path.join(eval_dir, "eval_report.json")))
    return z["kept_R"].astype(float), z["kept_p"].astype(float), z["kept_condition_ids"], rep


def representative(p, subject, family, mode):
    cls = C.classify_batch(p, np.broadcast_to(subject, (len(p), 3)), family)
    idx = np.nonzero(cls["label"] == mode)[0]
    if len(idx) == 0:
        return None
    feat = np.stack([cls["delta_phi"][idx], cls["rho"][idx]], -1)
    med = np.median(feat, 0)
    spread = feat.std(0) + 1e-9
    return int(idx[np.argmin((((feat - med) / spread) ** 2).sum(1))])


def modes_demo(a):
    R, p, cids, rep = load_kept(a.eval_dir, a.panel)
    pub = D.public_conditions(D.generate_e1_eval(a.family, a.panel, n_ref=1))
    ci = a.condition
    if ci < 0:  # auto: first kept condition in which all four modes were generated
        for j in range(len(cids)):
            sj = pub["subject"][list(pub["condition_id"]).index(str(cids[j]))]
            if all(representative(p[j], sj, a.family, m) is not None for m in range(4)):
                ci = j
                break
        else:
            raise SystemExit("no kept condition contains all four modes")
    cid = str(cids[ci])
    s = pub["subject"][list(pub["condition_id"]).index(cid)]
    tiles = []
    for m in range(4):
        k = representative(p[ci], s, a.family, m)
        if k is None:
            raise SystemExit(f"mode {MODE_NAMES[m]} not generated for {cid}; try another --condition")
        tiles.append((MODE_NAMES[m].replace("+", " · "), MODE_COLORS[m], R[ci, k], p[ci, k]))
    key = rep["experiment_key"]
    title = f"{key}: four camera moves sampled for ONE shot request"
    sub = (f"same start pose, end pose and subject ({cid}); the model chose different routes "
           f"· {rep['solver']} NFE {rep['nfe']} · 32 keyframes, interpolated for playback")
    frames = Dm.compose_frames(tiles, s, title, sub)
    base = os.path.join(a.out, f"{key}_{a.family}_{a.panel}_c{ci}_modes")
    Dm.write_video(frames, base + ".mp4", base + ".gif")
    return base


def gt_vs_gen_demo(a):
    R, p, cids, rep = load_kept(a.eval_dir, a.panel)
    ci = max(a.condition, 0)
    cid = str(cids[ci])
    ds = D.generate_e1_eval(a.family, a.panel)
    sel = ds["condition_id"] == cid
    s = ds["subject"][sel][0]
    gR, gp = ds["R"][sel], ds["p"][sel]
    fg = M.orientation_excursion_features(gR, gp, s)
    fm = M.orientation_excursion_features(R[ci], p[ci], s)
    # one pan-dominant and one roll-dominant example from each set, both with large excursions

    def pick(feats, axis):
        key = "signed_pan_peak_deg" if axis == "pan" else "signed_roll_peak_deg"
        other = "signed_roll_peak_deg" if axis == "pan" else "signed_pan_peak_deg"
        score = np.abs(feats[key]) - np.abs(feats[other])
        return int(np.argmax(score))

    # restrict all four examples to the same positional mode so only orientation differs
    lg = C.classify_batch(gp, np.broadcast_to(s, (len(gp), 3)), a.family)["label"]
    lm = C.classify_batch(p[ci], np.broadcast_to(s, (len(p[ci]), 3)), a.family)["label"]
    mode = a.mode
    ig, im = np.nonzero(lg == mode)[0], np.nonzero(lm == mode)[0]
    if len(ig) < 2 or len(im) < 2:
        raise SystemExit(f"not enough samples of mode {MODE_NAMES[mode]} for {cid}")
    sub = lambda f, idx: {k: v[idx] for k, v in f.items()}  # noqa: E731
    g1, g2 = ig[pick(sub(fg, ig), "pan")], ig[pick(sub(fg, ig), "roll")]
    m1, m2 = im[pick(sub(fm, im), "pan")], im[pick(sub(fm, im), "roll")]
    tiles = [("GT · pan excursion", "#8a8a85", gR[g1], gp[g1]), ("GT · roll excursion", "#5f5f5a", gR[g2], gp[g2]),
             ("generated · pan", MODE_COLORS[0], R[ci, m1], p[ci, m1]), ("generated · roll", MODE_COLORS[1], R[ci, m2], p[ci, m2])]
    key = rep["experiment_key"]
    title = f"{key} on E1b: camera orientation that is NOT determined by position"
    sub = (f"top: ground truth · bottom: generated · same condition {cid} and route ({MODE_NAMES[mode]}); "
           f"independent samples, not paired · start/end orientation identical")
    frames = Dm.compose_frames(tiles, s, title, sub)
    base = os.path.join(a.out, f"{key}_{a.family}_{a.panel}_c{ci}_gt_vs_gen")
    Dm.write_video(frames, base + ".mp4", base + ".gif")
    return base


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["modes", "gt_vs_gen"])
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--family", default="E1a")
    ap.add_argument("--panel", default="eval_id")
    ap.add_argument("--condition", type=int, default=-1, help="index into the kept conditions (-1: auto)")
    ap.add_argument("--mode", type=int, default=0, help="gt_vs_gen: positional mode shared by all tiles")
    ap.add_argument("--out", default="demo")
    return ap.parse_args(argv)


if __name__ == "__main__":
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    base = modes_demo(a) if a.kind == "modes" else gt_vs_gen_demo(a)
    print("wrote", base + ".mp4", "and .gif")
