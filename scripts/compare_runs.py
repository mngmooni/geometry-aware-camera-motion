"""Paired comparison of two evaluated runs on the SAME held-out conditions.

    python scripts/compare_runs.py --a runs/fm_abs_pose_E1a_s0/eval_m2_K200_nfe50_euler \
                                   --b runs/fm_rel_pose_E1a_s0/eval_m2_K200_nfe50_euler

Per-condition statistics are differenced (b - a) and a paired bootstrap over
conditions gives a 95% CI. Negative = b better for every error-type metric.
Panel-level statistics (conditional dispersion, smoothness, E1b W1/support) are shown
side by side with their matched oracle values; they have no paired CI here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera_motion import metrics as M  # noqa: E402
from camera_motion.config import FROZEN  # noqa: E402

T = np.array(FROZEN.modes.joint())
T_LONG, T_FAR = T[2:].sum(), T[1] + T[3]


def per_condition_table(eval_dir, panel):
    pc = json.load(open(os.path.join(eval_dir, f"{panel}_per_condition.json")))
    z = np.load(os.path.join(eval_dir, f"{panel}_per_sample.npz"))
    out = {}
    for cid, r in pc.items():
        c = np.array(r["counts5"], dtype=float)
        K = c.sum()
        sel = z["condition_id"] == cid
        out[cid] = {
            "tv5": r["tv5"],
            "abs_err_p_long": abs(c[2:4].sum() / K - T_LONG),
            "abs_err_p_far": abs((c[1] + c[3]) / K - T_FAR),
            "unassigned": r["unassigned_rate"],
            "e_p_median": float(np.median(z["e_p"][sel])),
            "e_R_deg_median": float(np.median(z["e_R_deg"][sel])),
        }
    return out


def _model(p):
    """m2 reports nest {model, oracle}; m1 (gate-2) reports are flat model-only."""
    return p["model"] if "model" in p else p


def _oracle(p):
    if "oracle" in p:
        return p["oracle"]
    return {"mode": {"tv5_mean": p.get("oracle_noise_floor", {}).get("tv5_mean", float("nan"))}}


def dispersion_from_counts(eval_dir, panel, seed=0):
    """Recompute conditional dispersion from saved counts (works for m1 and m2 evals)."""
    pc = json.load(open(os.path.join(eval_dir, f"{panel}_per_condition.json")))
    return M.conditional_dispersion(np.array([pc[c]["counts5"] for c in sorted(pc)]), seed=seed)


def paired(a, b, n_boot=5000, seed=0):
    ids = sorted(set(a) & set(b))
    if len(ids) != len(a) or len(ids) != len(b):
        raise ValueError("runs were not evaluated on the same conditions")
    rng = np.random.default_rng(seed)
    res = {}
    for k in a[ids[0]]:
        va = np.array([a[i][k] for i in ids])
        vb = np.array([b[i][k] for i in ids])
        d = vb - va
        bs = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(1)
        res[k] = {"a_mean": float(va.mean()), "b_mean": float(vb.mean()), "diff_mean": float(d.mean()),
                  "diff_ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                  "frac_conditions_b_better": float(np.mean(d < 0))}
    return res, len(ids)


def main(a_dir, b_dir, out=None):
    ra = json.load(open(os.path.join(a_dir, "eval_report.json")))
    rb = json.load(open(os.path.join(b_dir, "eval_report.json")))
    for key in ("family", "K", "seed", "nfe", "solver", "metrics_version", "classifier_version"):
        if key == "seed" and key not in ra:  # m1 reports did not record the eval seed (it was 123)
            continue
        if ra.get(key) != rb.get(key):
            print(f"WARNING: {key} differs: {ra.get(key)} vs {rb.get(key)}")
    result = {"a": ra["experiment_key"], "b": rb["experiment_key"], "family": ra["family"], "panels": {}}
    print(f"\n=== {rb['experiment_key']} (b) minus {ra['experiment_key']} (a) | {ra['family']} | "
          f"K {ra['K']} | {ra['solver']} NFE {ra['nfe']} ===")
    for panel in ra["panels"]:
        if panel not in rb["panels"]:
            continue
        stats, n = paired(per_condition_table(a_dir, panel), per_condition_table(b_dir, panel))
        pa, pb = ra["panels"][panel], rb["panels"][panel]
        ma, mb = _model(pa), _model(pb)
        oa, ob = _oracle(pa), _oracle(pb)
        o = ob if "conditional_dispersion" in ob else oa  # prefer the m2 oracle row
        nan = float("nan")
        da, db = dispersion_from_counts(a_dir, panel), dispersion_from_counts(b_dir, panel)
        side = {
            "dispersion_excess": [da["excess_ratio"], db["excess_ratio"],
                                  o.get("conditional_dispersion", {}).get("excess_ratio", nan)],
            "sd_p_long (binomial)": [da["sd_p_long"], db["sd_p_long"], da["binomial_sd_p_long"]],
            "sd_p_far (binomial)": [da["sd_p_far"], db["sd_p_far"], da["binomial_sd_p_far"]],
            "tv5_pooled_mean": [ma["mode"]["tv5_mean"], mb["mode"]["tv5_mean"], o["mode"]["tv5_mean"]],
        }
        for k in ma["smoothness_ratio_gen_over_gt"]:
            side[f"smooth_{k}"] = [ma["smoothness_ratio_gen_over_gt"][k], mb["smoothness_ratio_gen_over_gt"][k],
                                   o.get("smoothness_ratio_gen_over_gt", {}).get(k, nan)]
        if "excursion_w1_pooled" in ma and "excursion_w1_pooled" in mb:
            for k, v in ma["excursion_w1_pooled"].items():
                side[f"w1_{k}"] = [v, mb["excursion_w1_pooled"][k],
                                   o.get("excursion_w1_pooled", {}).get(k, nan)]
        if "excursion_support_violation" in ma and "excursion_support_violation" in mb:
            side["support_amp_outside"] = [ma["excursion_support_violation"]["amp_outside_support"],
                                           mb["excursion_support_violation"]["amp_outside_support"],
                                           o["excursion_support_violation"]["amp_outside_support"]]
        result["panels"][panel] = {"paired": stats, "n_conditions": n, "side_by_side_a_b_oracle": side}
        print(f"\n[{panel}] {n} paired conditions  (diff = b - a, 95% paired-bootstrap CI)")
        for k, s in stats.items():
            sig = "*" if (s["diff_ci95"][0] > 0 or s["diff_ci95"][1] < 0) else " "
            print(f"  {k:16s} a {s['a_mean']:.4f}  b {s['b_mean']:.4f}  diff {s['diff_mean']:+.4f} "
                  f"[{s['diff_ci95'][0]:+.4f}, {s['diff_ci95'][1]:+.4f}]{sig}  b better in {s['frac_conditions_b_better']:.0%}")
        print("  panel-level (a | b | oracle):")
        for k, v in side.items():
            print(f"    {k:28s} {v[0]:8.3f} | {v[1]:8.3f} | {v[2]:8.3f}")
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default=None)
    x = ap.parse_args()
    main(x.a, x.b, x.out)
