"""Solver / NFE sensitivity on ONE checkpoint (no retraining).

    python scripts/diagnose_solver.py --run runs/fm_rel_pose_E1a_s0 --grid euler:20 euler:50 euler:100 midpoint:50 midpoint:100

Same held-out conditions, K and evaluation seed for every setting, so differences are
due to the ODE integration only. Reports TV5, dispersion decomposition, endpoint,
absolute and GT-relative jerk, and the per-frame jerk profile. If jerk falls sharply
with NFE, the excess is integration error rather than a learned property.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_motion import dataset_generator as D  # noqa: E402
from camera_motion import visualize as V  # noqa: E402
from scripts.eval_fm import _subset, eval_panel, load_run, make_sampler  # noqa: E402


def main(a):
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    model, norm, meta, step = load_run(a.run, dev)
    ds = D.generate_e1_eval(meta["family"], a.panel)
    if a.max_conditions:
        ds = _subset(ds, a.max_conditions)
    out = {"run": a.run, "experiment_key": meta["experiment_key"], "family": meta["family"], "panel": a.panel,
           "K": a.K, "seed": a.seed, "train_step": step, "settings": {}}
    profiles = {}
    for spec in a.grid:
        method, nfe = spec.split(":")
        nfe = int(nfe)
        res, *_ = eval_panel(make_sampler(model, norm, meta["rep"], dev, nfe, method), ds, meta["family"], a.K, a.seed)
        dd = res["dispersion_decomposition"]
        out["settings"][spec] = {
            "tv5": res["mode"]["tv5_mean"], "unassigned": res["mode"]["unassigned_rate_mean"],
            "bias_excess": dd["bias"]["excess_ratio"], "between_excess": dd["between"]["excess_ratio"],
            "pooled_minus_target": dd["pooled_minus_target"],
            "e_p_median": res["endpoint"]["e_p_median"], "e_R_deg_median": res["endpoint"]["e_R_deg_median"],
            "jerk_ratio": res["smoothness_ratio_gen_over_gt"]["trans_jerk"],
            "jerk_abs_median_gen": res["smoothness_abs_median"]["gen"]["trans_jerk"],
            "jerk_abs_median_gt": res["smoothness_abs_median"]["gt"]["trans_jerk"],
            "ang_acc_ratio": res["smoothness_ratio_gen_over_gt"]["ang_acc"],
            "sampling_sec_per_condition": res["sampling_sec_per_condition"],
        }
        if "excursion_w1_pooled" in res:
            out["settings"][spec]["e1b_w1_max_excursion_deg"] = res["excursion_w1_pooled"]["max_excursion_deg"]
            out["settings"][spec]["e1b_amp_outside_support"] = res["excursion_support_violation"]["amp_outside_support"]
        profiles[spec] = (res["jerk_profile_median"]["gen"], res["jerk_profile_median"]["gt"])
        s = out["settings"][spec]
        print(f"{spec:13s} TV5 {s['tv5']:.3f} | bias x{s['bias_excess']:.1f} between x{s['between_excess']:.2f} "
              f"| e_p {s['e_p_median'] * 100:.2f} cm | jerk/GT {s['jerk_ratio']:.2f} "
              f"(abs {s['jerk_abs_median_gen']:.0f} vs GT {s['jerk_abs_median_gt']:.0f}) | {s['sampling_sec_per_condition']:.3f} s/cond"
              + (f" | E1b W1 amp {s['e1b_w1_max_excursion_deg']:.2f} deg, out-of-support {100 * s['e1b_amp_outside_support']:.1f}%"
                 if "e1b_w1_max_excursion_deg" in s else ""))
    out_dir = os.path.join(a.run, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"solver_sweep_{a.panel}.json"), "w") as f:
        json.dump(out, f, indent=2)
    fig, ax = V.plt.subplots(figsize=(7, 3.4))
    for i, (spec, (g, gt)) in enumerate(profiles.items()):
        ax.plot(g, color=V.MODE_COLORS[i % 5], ls=V.MODE_STYLES[i % 5], lw=1.8, label=spec)
    ax.plot(gt, color=V.INK, lw=2.4, label="GT")
    ax.set_yscale("log")
    ax.set_xlabel("jerk index i (frames i..i+3; 0 touches the anchored start)")
    ax.set_ylabel("median |jerk| (m/s³)")
    ax.set_title(f"{meta['experiment_key']} {meta['family']} {a.panel}: jerk profile by solver", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    V._style(ax)
    fig.tight_layout()
    V.save(fig, os.path.join(out_dir, f"jerk_profile_{a.panel}.png"))
    print("wrote", out_dir)
    return out


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--panel", default="eval_id")
    ap.add_argument("--grid", nargs="+", default=["euler:20", "euler:50", "euler:100", "midpoint:50", "midpoint:100"])
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-conditions", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    main(parse())
