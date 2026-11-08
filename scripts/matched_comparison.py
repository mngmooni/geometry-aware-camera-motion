"""Matched comparison table (csv + md) + condition-level residual plots for two runs.

    python scripts/matched_comparison.py --family E1a --a <abs eval dir> --b <rel eval dir> --out <dir>

Every row states whether it is MATCHED (same held-out conditions, K, evaluation seed /
base noise, solver, NFE, classifier version and computation) and, if not, why.
Uncertainty: paired bootstrap over CONDITIONS (never over samples within a condition).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_motion import metrics as M  # noqa: E402
from camera_motion import visualize as V  # noqa: E402
from camera_motion.config import FROZEN  # noqa: E402
from scripts.compare_runs import _model, _oracle, paired, per_condition_table  # noqa: E402

T = np.array(FROZEN.modes.joint())


def counts(eval_dir, panel):
    pc = json.load(open(os.path.join(eval_dir, f"{panel}_per_condition.json")))
    ids = sorted(pc)
    return ids, np.array([pc[c]["counts5"] for c in ids], dtype=float)


def _version(rep):
    return rep.get("metrics_version") or "m1"


def residual_plot(ids, ca, cb, la, lb, panel, path):
    K = ca[0].sum()
    fig, axes = V.plt.subplots(1, 2, figsize=(11, 3.4), sharex=True)
    x = np.arange(len(ids))
    for ax, (name, sel, t) in zip(axes, (("P(long_cw)", [2, 3], T[2:].sum()), ("P(far)", [1, 3], T[1] + T[3]))):
        sd = np.sqrt(t * (1 - t) / K)
        ax.axhspan(-1.96 * sd, 1.96 * sd, color=V.GRID, label="calibrated 95% band (binomial)")
        ax.axhline(0, color=V.MUTED, lw=1)
        ax.plot(x, ca[:, sel].sum(1) / K - t, "o", color=V.MODE_COLORS[1], ms=5, label=la)
        ax.plot(x, cb[:, sel].sum(1) / K - t, "s", color=V.MODE_COLORS[0], ms=5, mfc="none", mew=1.5, label=lb)
        ax.set_title(f"{panel}: {name} − target, per condition", fontsize=9)
        ax.set_xlabel("held-out condition (sorted id)")
        V._style(ax)
    axes[0].set_ylabel("frequency residual")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    V.save(fig, path)


def main(family, a_dir, b_dir, out_dir, label_a=None, label_b=None):
    os.makedirs(out_dir, exist_ok=True)
    ra = json.load(open(os.path.join(a_dir, "eval_report.json")))
    rb = json.load(open(os.path.join(b_dir, "eval_report.json")))
    la, lb = label_a or ra["experiment_key"], label_b or rb["experiment_key"]
    va, vb = _version(ra), _version(rb)
    same_eval = all(ra.get(k) == rb.get(k) for k in ("K", "nfe", "solver", "classifier_version", "family"))
    rows = []

    def add(panel, metric, a, b, oracle, diff=None, ci=None, matched=True, note=""):
        rows.append({"family": family, "panel": panel, "metric": metric, la: a, lb: b,
                     "diff_b_minus_a": diff, "ci95_low": ci[0] if ci else None, "ci95_high": ci[1] if ci else None,
                     "oracle": oracle, "matched": "yes" if matched else "no", "note": note})

    for panel in ra["panels"]:
        if panel not in rb["panels"]:
            continue
        ma, mb, ob = _model(ra["panels"][panel]), _model(rb["panels"][panel]), _oracle(rb["panels"][panel])
        stats, n = paired(per_condition_table(a_dir, panel), per_condition_table(b_dir, panel))
        for k, s in stats.items():
            add(panel, f"per-condition {k} (mean over {n})", s["a_mean"], s["b_mean"], None, s["diff_mean"],
                s["diff_ci95"], same_eval, "paired by condition; identical eval seed -> identical base-noise seeds")
        ids, ca = counts(a_dir, panel)
        _, cb = counts(b_dir, panel)
        da, db = M.dispersion_decomposition(ca), M.dispersion_decomposition(cb)
        for part in ("bias", "between"):
            add(panel, f"dispersion {part} chi2 (excess ratio)", da[part]["excess_ratio"], db[part]["excess_ratio"],
                1.0, note=f"p_a={da[part]['p_value']:.3g}, p_b={db[part]['p_value']:.3g}; recomputed from saved counts")
        add(panel, "TV5 mean", ma["mode"]["tv5_mean"], mb["mode"]["tv5_mean"], ob["mode"]["tv5_mean"])
        for k in ("e_p_mean", "e_p_p90", "e_R_deg_mean", "e_R_deg_p90"):
            add(panel, f"endpoint {k}", ma["endpoint"][k], mb["endpoint"][k], 0.0)
        smooth_matched = family == "E1a"
        for k in ("trans_acc", "trans_jerk", "ang_acc"):
            add(panel, f"smoothness {k} / GT", ma["smoothness_ratio_gen_over_gt"][k], mb["smoothness_ratio_gen_over_gt"][k],
                ob.get("smoothness_ratio_gen_over_gt", {}).get(k), matched=smooth_matched,
                note="" if smooth_matched else "GT reference set differs (m1: 20 refs/cond, m2: 200)")
        if "excursion_w1_pooled" in mb:
            e_ok = "excursion_support_violation" in ma and va == vb
            for k in ("max_excursion_deg", "peak_time"):
                add(panel, f"E1b W1 pooled {k}", ma.get("excursion_w1_pooled", {}).get(k), mb["excursion_w1_pooled"][k],
                    ob["excursion_w1_pooled"][k], matched=e_ok,
                    note="" if e_ok else f"a evaluated with {va} (20 GT refs/cond): pending matched m2 re-evaluation")
            add(panel, "E1b amplitude outside support", ma.get("excursion_support_violation", {}).get("amp_outside_support"),
                mb["excursion_support_violation"]["amp_outside_support"], ob["excursion_support_violation"]["amp_outside_support"],
                matched=e_ok, note="" if e_ok else "not computed for a")
        residual_plot(ids, ca, cb, la, lb, panel, os.path.join(out_dir, f"{family}_{panel}_residuals.png"))

    with open(os.path.join(out_dir, f"{family}_matched_comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def fmt(v):
        return "—" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))

    lines = [f"# {family}: {lb} vs {la}", "",
             f"Evaluators: a = {va}, b = {vb}; K = {ra['K']}, {ra['solver']} NFE {ra['nfe']}, classifier "
             f"{ra.get('classifier_version')}. **One training seed per model.** Paired 95% CIs are bootstrap over "
             "held-out conditions. `matched = no` rows must not be used for claims.", "",
             f"| panel | metric | {la} | {lb} | b − a [95% CI] | oracle | matched | note |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d = "—" if r["diff_b_minus_a"] is None else f"{r['diff_b_minus_a']:+.4g} [{r['ci95_low']:+.4g}, {r['ci95_high']:+.4g}]"
        lines.append(f"| {r['panel']} | {r['metric']} | {fmt(r[la])} | {fmt(r[lb])} | {d} | {fmt(r['oracle'])} | "
                     f"{r['matched']} | {r['note']} |")
    with open(os.path.join(out_dir, f"{family}_matched_comparison.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    x = ap.parse_args()
    main(x.family, x.a, x.b, x.out, x.label_a, x.label_b)
