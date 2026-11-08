"""Evaluate a trained Euclidean FM run under the frozen contract (metrics version m2).

    python scripts/eval_fm.py --run runs/fm_abs_pose_E1a_s0 --K 200 --nfe 50

Every panel is scored twice with the SAME procedure, conditions, K and seeds:
  * ``model``  : K samples from the EMA model per held-out public condition
  * ``oracle`` : K fresh GT re-draws per condition (a perfectly calibrated sampler)
The oracle row is the matched noise floor for every metric (TV5, conditional
dispersion, E1b W1 and support violation, smoothness ratio).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_motion import dataset_generator as D  # noqa: E402
from camera_motion import fm  # noqa: E402
from camera_motion import metrics as M  # noqa: E402
from camera_motion import mode_classifier as C  # noqa: E402
from camera_motion import representations as Rp  # noqa: E402
from camera_motion import rfm_so3  # noqa: E402
from camera_motion import visualize as V  # noqa: E402
from camera_motion.config import CATEGORIES, FROZEN, METRICS_VERSION  # noqa: E402


def load_run(run_dir, dev):
    s = torch.load(os.path.join(run_dir, "ckpt_last.pt"), map_location=dev, weights_only=False)
    meta = s["meta"]
    model = fm.TemporalVelocityNet(meta["T"], meta["channels"], meta["cond_dim"], hidden=meta["hidden"],
                                   out_channels=meta.get("out_channels")).to(dev)
    model.load_state_dict(s["ema"])
    model.eval()
    return model, Rp.Normalizer.from_state(s["norm"]), meta, s["step"]


def make_sampler(model, norm, rep, dev, nfe, method):
    if rep == "rfm_so3":
        def sample_rfm(row, K, rng):
            c = Rp.encode_condition("rel_pose", row["start_pose"][None], row["end_pose"][None], row["subject"][None])
            c = torch.from_numpy(norm.nc(c).astype(np.float32)).to(dev).expand(K, -1)
            g = torch.Generator(device=dev).manual_seed(int(rng.integers(2**31)))
            R, x = rfm_so3.sample(model, c, model.pos.shape[-1], nfe=nfe, method=method, generator=g)
            return rfm_so3.decode(R.cpu().numpy().astype(np.float64), x.cpu().numpy().astype(np.float64), norm,
                                  np.broadcast_to(row["start_pose"], (K, 4, 4)))
        return sample_rfm

    def sample_fn(row, K, rng):
        c = Rp.encode_condition(rep, row["start_pose"][None], row["end_pose"][None], row["subject"][None])
        c = torch.from_numpy(norm.nc(c).astype(np.float32)).to(dev).expand(K, -1)
        g = torch.Generator(device=dev).manual_seed(int(rng.integers(2**31)))
        x = fm.sample(model, c, (K, model.pos.shape[-1], Rp.T_CHANNELS), nfe=nfe, method=method, generator=g)
        x = norm.ux(x.cpu().numpy().astype(np.float64))
        return Rp.decode_trajectories(rep, x, np.broadcast_to(row["start_pose"], (K, 4, 4)))
    return sample_fn


def _cat(dicts, k):
    return np.concatenate([d[k] for d in dicts])


def eval_panel(sample_fn, ds_panel, family, K, seed, keep_samples=0):
    """Score one sampler on one panel. Returns (summary, per_condition, per_sample_rows, kept, extras)."""
    pub = D.public_conditions(ds_panel)
    rng = np.random.default_rng(seed)
    per_cond, rows, kept_R, kept_p, t_sample = {}, [], [], [], 0.0
    sm_gen, sm_ref, w1, fg_all, fr_all, counts, jp_gen, jp_ref = [], [], [], [], [], [], [], []
    for i, cid in enumerate(pub["condition_id"]):
        row = {k: pub[k][i] for k in pub}
        t0 = time.time()
        R, p = sample_fn(row, K, rng)
        t_sample += time.time() - t0
        cls = C.classify_batch(p, np.broadcast_to(row["subject"], (K, 3)), family)
        rep = M.mode_report(cls["label"])
        per_cond[str(cid)] = rep
        counts.append(rep["counts5"])
        ep = M.endpoint_errors(R, p, row["end_pose"])
        st = M.start_errors(R, p, row["start_pose"])
        for k in range(K):
            rows.append((str(cid), k, int(cls["label"][k]), str(cls["reason"][k]),
                         float(ep["e_p"][k]), float(ep["e_R_deg"][k]), float(st["e_p"][k])))
        ref = ds_panel["condition_id"] == cid
        sm_gen.append(M.smoothness(R, p))
        sm_ref.append(M.smoothness(ds_panel["R"][ref], ds_panel["p"][ref]))
        jp_gen.append(M.jerk_profile(p))
        jp_ref.append(M.jerk_profile(ds_panel["p"][ref]))
        if family == "E1b":
            fg = M.orientation_excursion_features(R, p, row["subject"])
            fr = M.orientation_excursion_features(ds_panel["R"][ref], ds_panel["p"][ref], row["subject"])
            w1.append(M.excursion_distribution_distance(fg, fr))
            fg_all.append(fg)
            fr_all.append(fr)
        if i < keep_samples:
            kept_R.append(R.astype(np.float32))
            kept_p.append(p.astype(np.float32))
    labels = np.array([r[2] for r in rows])
    e_p = np.array([r[4] for r in rows])
    e_R = np.array([r[5] for r in rows])
    sm_g = {k: _cat(sm_gen, k) for k in sm_gen[0]}
    sm_r = {k: _cat(sm_ref, k) for k in sm_ref[0]}
    out = {
        "n_conditions": len(pub["condition_id"]), "K": K,
        "mode": M.aggregate_condition_reports(per_cond),
        "conditional_dispersion": M.conditional_dispersion(np.array(counts), seed=seed),
        "endpoint": {"e_p_mean": float(e_p.mean()), "e_p_median": float(np.median(e_p)),
                     "e_p_p90": float(np.quantile(e_p, 0.9)), "e_R_deg_mean": float(e_R.mean()),
                     "e_R_deg_median": float(np.median(e_R)), "e_R_deg_p90": float(np.quantile(e_R, 0.9))},
        "endpoint_by_assigned_mode": {CATEGORIES[m]: float(np.median(e_p[labels == m]))
                                      for m in range(5) if np.any(labels == m)},
        "start_anchor_max_err": float(max(r[6] for r in rows)),
        "unassigned_reasons": {r: int(sum(1 for x in rows if x[3] == r)) for r in C.REASONS},
        "smoothness_ratio_gen_over_gt": M.smoothness_ratio(sm_g, sm_r),
        "smoothness_abs_median": {"gen": {k: float(np.median(v)) for k, v in sm_g.items()},
                                  "gt": {k: float(np.median(v)) for k, v in sm_r.items()},
                                  "units": "m/s^2, m/s^3, rad/s, rad/s^2 with dt = 1/(L-1)"},
        "jerk_profile_median": {"gen": np.median(jp_gen, 0).tolist(), "gt": np.median(jp_ref, 0).tolist(),
                                "note": "index i uses frames i..i+3; index 0 touches the anchored start"},
        "dispersion_decomposition": M.dispersion_decomposition(np.array(counts), seed=seed),
        "sampling_sec_per_condition": round(t_sample / len(pub["condition_id"]), 4),
    }
    extras = {"pub": pub}
    if family == "E1b":
        fg = {k: _cat(fg_all, k) for k in fg_all[0]}
        fr = {k: _cat(fr_all, k) for k in fr_all[0]}
        out["excursion_w1_condition_mean"] = {k: float(np.mean([w[k] for w in w1])) for k in w1[0]}
        out["excursion_w1_pooled"] = M.excursion_distribution_distance(fg, fr)
        out["excursion_support_violation"] = M.excursion_support_violation(fg)
        out["excursion_support_violation_gt_ref"] = M.excursion_support_violation(fr)
        per_c = [M.excursion_support_violation(f)["amp_outside_support"] for f in fg_all]
        bad = (fg["max_excursion_deg"] < FROZEN.latent.offset_amp_deg[0] - 1) | \
              (fg["max_excursion_deg"] > FROZEN.latent.offset_amp_deg[1] + 1)
        out["excursion_support_violation_by_condition"] = {
            "median": float(np.median(per_c)), "max": float(np.max(per_c)),
            "frac_conditions_over_10pct": float(np.mean(np.array(per_c) > 0.10)),
            "violator_peak_time_quartiles": np.quantile(fg["peak_time"][bad], [0.25, 0.5, 0.75]).tolist() if bad.any() else None,
            "violator_signed_pan_or_roll_abs_median_deg": float(np.median(np.maximum(
                np.abs(fg["signed_pan_peak_deg"][bad]), np.abs(fg["signed_roll_peak_deg"][bad])))) if bad.any() else None}
        extras["feats"] = (fg, fr)
    return out, per_cond, rows, (kept_R, kept_p), extras


def figures(panel_name, model_res, oracle_res, extras, kept, fam, fig_dir):
    pub = extras["pub"]
    gen_R, gen_p = kept
    n = min(3, len(gen_p))
    fig, axes = V.plt.subplots(1, n + 1, figsize=(4.3 * (n + 1), 4.6))
    for j in range(n):
        cid = pub["condition_id"][j]
        fake = {"condition_id": np.repeat(cid, len(gen_p[j])), "p": gen_p[j],
                "subject": np.broadcast_to(pub["subject"][j], (len(gen_p[j]), 3))}
        lab = C.classify_batch(gen_p[j], fake["subject"], fam)["label"]
        V.plot_condition_topdown(fake, cid, labels=lab, ax=axes[j], max_paths=80, title=f"generated · {cid}")
    xs = np.arange(5)
    ax = axes[-1]
    ax.bar(xs - 0.27, FROZEN.modes.target5(), 0.26, color=V.MUTED, label="declared p(mode|c)")
    ax.bar(xs, oracle_res["mode"]["pooled_freq5"], 0.26, color=V.GRID, edgecolor=V.MUTED, label="oracle (GT re-draw)")
    ax.bar(xs + 0.27, model_res["mode"]["pooled_freq5"], 0.26, color=[V.MODE_COLORS[k] for k in range(5)],
           label="model")
    ax.set_xticks(xs, [c.replace("+", "\n") for c in CATEGORIES], fontsize=7)
    d, do = model_res["conditional_dispersion"], oracle_res["conditional_dispersion"]
    ax.set_title(f"{panel_name}: TV5 {model_res['mode']['tv5_mean']:.3f} (oracle {oracle_res['mode']['tv5_mean']:.3f})\n"
                 f"dispersion ×{d['excess_ratio']:.2f} (oracle ×{do['excess_ratio']:.2f})", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    V._style(ax)
    fig.tight_layout()
    V.save(fig, os.path.join(fig_dir, f"{panel_name}_samples_and_modes.png"))
    if "feats" in extras:
        fig = V.plot_feature_hists(extras["feats"][1], extras["feats"][0], labels=("GT ref", "model"))
        V.save(fig, os.path.join(fig_dir, f"{panel_name}_excursion_hists.png"))


def _line(name, r):
    m, d = r["mode"], r["conditional_dispersion"]
    s = (f"  {name:7s} TV5 {m['tv5_mean']:.3f} CI{np.round(m['tv5_ci95'], 3).tolist()} | dispersion x{d['excess_ratio']:.2f} "
         f"(p={d['p_value']:.3g}, sd P_long {d['sd_p_long']:.3f} vs {d['binomial_sd_p_long']:.3f}) | cov {m['coverage4_mean']:.2f} "
         f"| unasg {m['unassigned_rate_mean']:.3f} | minErr {m['minority_error_mean']:+.3f} | e_p med {r['endpoint']['e_p_median']:.3f} m "
         f"| e_R med {r['endpoint']['e_R_deg_median']:.2f} deg")
    if "excursion_w1_pooled" in r:
        s += ("\n          W1 pooled " + str({k: round(v, 3) for k, v in r["excursion_w1_pooled"].items()})
              + "\n          support violation " + str({k: round(v, 3) for k, v in r["excursion_support_violation"].items()}))
    return s


def main(a):
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    model, norm, meta, step = load_run(a.run, dev)
    fam, rep = meta["family"], meta["rep"]
    sampler = make_sampler(model, norm, rep, dev, a.nfe, a.method)
    panels = [("eval_id", D.generate_e1_eval(fam, "eval_id"))]
    if fam == "E1a":
        panels.append(("eval_ood_position", D.generate_e1_eval(fam, "eval_ood_position")))
    if a.max_conditions:
        panels = [(n, _subset(ds, a.max_conditions)) for n, ds in panels]

    out_dir = os.path.join(a.run, f"eval_{a.tag}_K{a.K}_nfe{a.nfe}_{a.method}")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    report = {"experiment_key": meta["experiment_key"], "family": fam, "rep": rep, "train_step": step,
              "solver": a.method, "nfe": a.nfe, "K": a.K, "seed": a.seed, "device": str(dev),
              "classifier_version": FROZEN.clf.version, "metrics_version": METRICS_VERSION,
              "params": meta["params"], "panels": {}}
    for name, ds_panel in panels:
        res, per_cond, rows, kept, extras = eval_panel(sampler, ds_panel, fam, a.K, a.seed, keep_samples=a.keep)
        o_seed = a.seed + 1 + zlib.crc32(name.encode()) % 10_000  # independent oracle draw per panel
        ores, *_ = eval_panel(D.oracle_sampler(fam), ds_panel, fam, a.K, o_seed)
        figures(name, res, ores, extras, kept, fam, fig_dir)
        report["panels"][name] = {"model": res, "oracle": ores}
        np.savez_compressed(
            os.path.join(out_dir, f"{name}_per_sample.npz"),
            condition_id=np.array([r[0] for r in rows]), sample_idx=np.array([r[1] for r in rows]),
            label=np.array([r[2] for r in rows]), reason=np.array([r[3] for r in rows]),
            e_p=np.array([r[4] for r in rows]), e_R_deg=np.array([r[5] for r in rows]),
            kept_R=np.array(kept[0]) if kept[0] else np.zeros(0),
            kept_p=np.array(kept[1]) if kept[1] else np.zeros(0),
            kept_condition_ids=extras["pub"]["condition_id"][:len(kept[1])])
        with open(os.path.join(out_dir, f"{name}_per_condition.json"), "w") as f:
            json.dump(per_cond, f, indent=1)
        print(f"[{meta['experiment_key']} {fam} {name}] {a.method} NFE {a.nfe}, K {a.K}, "
              f"{res['n_conditions']} conditions")
        print(_line("model", res))
        print(_line("oracle", ores))
        print("  smoothness gen/GT:", {k: round(v, 2) for k, v in res["smoothness_ratio_gen_over_gt"].items()},
              "| oracle:", {k: round(v, 2) for k, v in ores["smoothness_ratio_gen_over_gt"].items()})
    with open(os.path.join(out_dir, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", out_dir)
    return report


def _subset(ds, n_cond):
    keep = np.isin(ds["condition_id"], np.unique(ds["condition_id"])[:n_cond])
    return {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in ds.items()}


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--nfe", type=int, default=50)
    ap.add_argument("--method", default="euler", choices=["euler", "midpoint"])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--keep", type=int, default=5, help="conditions whose raw samples are saved")
    ap.add_argument("--max-conditions", type=int, default=0, help="debug: limit conditions per panel")
    ap.add_argument("--tag", default=METRICS_VERSION)
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    main(parse())
