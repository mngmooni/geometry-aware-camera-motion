"""Deliverable 1: compact benchmark table + GT-vs-generated figures from finished evaluations.

    python scripts/make_benchmark.py \
        --run rel_pose=runs/fm_rel_pose_E1a_s0/eval_m2_K200_nfe50_euler \
        --run rel_pose=runs/fm_rel_pose_E1b_s0/eval_m2_K200_nfe50_euler \
        --run rel_inc=runs/fm_rel_inc_E1a_s0/eval_m2_K200_nfe50_euler ... \
        --out results/benchmark

Rows evaluated with an older evaluator (m1) are marked *exploratory*; they share the
held-out conditions, K, seed and solver but not every metric. Figures use the samples
kept by eval_fm.py (first few held-out conditions) and fresh GT draws for the same condition.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_motion import dataset_generator as D  # noqa: E402
from camera_motion import geometry as G  # noqa: E402
from camera_motion import mode_classifier as C  # noqa: E402
from camera_motion import visualize as V  # noqa: E402

PANEL_NAMES = {"eval_id": "held-out (ID)", "eval_ood_position": "OOD position"}


def _m(p):
    return p["model"] if "model" in p else p


def _o(p):
    return p.get("oracle")


def table(runs, out):
    rows = []
    for label, d in runs:
        rep = json.load(open(os.path.join(d, "eval_report.json")))
        ver = rep.get("metrics_version") or "m1"
        for panel, pr in rep["panels"].items():
            m, o = _m(pr), _o(pr)
            otv = o["mode"]["tv5_mean"] if o else pr.get("oracle_noise_floor", {}).get("tv5_mean")
            r = {"model": label, "data": rep["family"], "panel": PANEL_NAMES.get(panel, panel),
                 "TV5": m["mode"]["tv5_mean"], "TV5_oracle": otv,
                 "coverage_of_4": m["mode"]["coverage4_mean"], "unassigned_%": 100 * m["mode"]["unassigned_rate_mean"],
                 "endpoint_cm_median": 100 * m["endpoint"]["e_p_median"], "endpoint_deg_median": m["endpoint"]["e_R_deg_median"],
                 "jerk_vs_GT": m["smoothness_ratio_gen_over_gt"]["trans_jerk"],
                 "E1b_W1_amp_deg": m.get("excursion_w1_pooled", {}).get("max_excursion_deg") if ver != "m1" else None,
                 "E1b_W1_amp_oracle": (o or {}).get("excursion_w1_pooled", {}).get("max_excursion_deg") if o else None,
                 "E1b_amp_outside_%": 100 * m["excursion_support_violation"]["amp_outside_support"]
                 if "excursion_support_violation" in m else None,
                 "evaluator": ver, "status": "exploratory (older evaluator)" if ver == "m1" else "main"}
            rows.append(r)
    with open(os.path.join(out, "benchmark_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def f(v, nd=3):
        return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    lines = ["| model | data | panel | TV5 (oracle) ↓ | modes found | unassigned % | endpoint cm | endpoint ° | jerk / GT | E1b W1 amp ° (oracle) | E1b amp out % | status |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        w1 = "—" if r["E1b_W1_amp_deg"] is None else f"{r['E1b_W1_amp_deg']:.2f} ({f(r['E1b_W1_amp_oracle'], 2)})"
        lines.append(f"| {r['model']} | {r['data']} | {r['panel']} | {f(r['TV5'])} ({f(r['TV5_oracle'])}) | "
                     f"{f(r['coverage_of_4'], 1)}/4 | {f(r['unassigned_%'], 2)} | {f(r['endpoint_cm_median'], 2)} | "
                     f"{f(r['endpoint_deg_median'], 2)} | {f(r['jerk_vs_GT'], 2)} | {w1} | {f(r['E1b_amp_outside_%'], 1)} | {r['status']} |")
    notes = ["", "TV5: total-variation distance between the generated 5-category mode frequencies (4 routes + "
             "unassigned) and the declared p(route|shot); the oracle column is a perfect sampler (fresh ground-truth "
             "draws) at the same K, i.e. the finite-sample floor. Endpoint: median error of the generated last pose "
             "(soft-conditioned, never projected). Jerk / GT: median translational jerk relative to ground truth. "
             "One training seed per model; differences are exploratory unless paired CIs are reported."]
    with open(os.path.join(out, "benchmark_table.md"), "w") as fh:
        fh.write("\n".join(lines + notes) + "\n")
    return rows


def _kept(d, panel):
    z = np.load(os.path.join(d, f"{panel}_per_sample.npz"))
    return z["kept_R"].astype(float), z["kept_p"].astype(float), z["kept_condition_ids"]


def trajectory_figure(runs, family, out, cond_index=0, n_show=60):
    runs = [(lab, d) for lab, d in runs if json.load(open(os.path.join(d, "eval_report.json")))["family"] == family]
    if not runs:
        return None
    _, _, cids = _kept(runs[0][1], "eval_id")
    cid = str(cids[cond_index])
    ds = D.generate_e1_eval(family, "eval_id")
    sel = ds["condition_id"] == cid
    gt = {"condition_id": ds["condition_id"][sel][:n_show], "p": ds["p"][sel][:n_show], "R": ds["R"][sel][:n_show],
          "subject": ds["subject"][sel][:n_show], "latent_mode": ds["latent_mode"][sel][:n_show]}
    s = gt["subject"][0]
    ncol = 1 + len(runs)
    rows = 2 if family == "E1b" else 1
    fig, axes = V.plt.subplots(rows, ncol, figsize=(4.4 * ncol, 4.9 * rows), squeeze=False)
    gt_lab = C.classify_batch(gt["p"], gt["subject"], family)["label"]
    V.plot_condition_topdown(gt, cid, labels=gt_lab, ax=axes[0, 0], max_paths=n_show, title="ground truth")
    sets = [("ground truth", gt["R"], gt["p"])]
    for j, (lab, d) in enumerate(runs, 1):
        R, p, kc = _kept(d, "eval_id")
        k = list(kc).index(cid)
        fake = {"condition_id": np.repeat(cid, n_show), "p": p[k][:n_show], "subject": np.broadcast_to(s, (n_show, 3))}
        V.plot_condition_topdown(fake, cid, labels=C.classify_batch(fake["p"], fake["subject"], family)["label"],
                                 ax=axes[0, j], max_paths=n_show, title=f"generated · {lab}")
        sets.append((lab, R[k][:n_show], p[k][:n_show]))
    if family == "E1b":
        for j, (lab, R, p) in enumerate(sets):
            look = G.look_at(p, np.broadcast_to(s, p.shape))
            w = np.degrees(np.linalg.norm(G.so3_log(np.swapaxes(look, -1, -2) @ R), axis=-1))
            ax = axes[1, j]
            ax.plot(w.T, color=V.MODE_COLORS[0] if j else V.MUTED, lw=0.8, alpha=0.5)
            ax.axhspan(20, 60, color=V.GRID, zorder=0, label="declared peak range")
            ax.set_xlabel("frame")
            ax.set_ylabel("angle from look-at (deg)")
            ax.set_title(f"orientation excursion · {lab}", fontsize=9)
            ax.set_ylim(0, 75)
            V._style(ax)
        axes[1, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{family} · one held-out shot request ({cid}): {n_show} ground-truth vs {n_show} generated paths",
                 fontsize=11, color=V.INK)
    fig.tight_layout()
    path = os.path.join(out, f"{family}_gt_vs_generated.png")
    V.save(fig, path, dpi=130)
    return path


def main(runs, out, cond_index=0):
    os.makedirs(out, exist_ok=True)
    table(runs, out)
    figs = [trajectory_figure(runs, fam, out, cond_index) for fam in ("E1a", "E1b")]
    print("wrote", os.path.join(out, "benchmark_table.md"), [f for f in figs if f])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=eval_dir (repeatable)")
    ap.add_argument("--out", default="results/benchmark")
    ap.add_argument("--condition", type=int, default=0)
    a = ap.parse_args()
    main([tuple(r.split("=", 1)) for r in a.run], a.out, a.condition)
