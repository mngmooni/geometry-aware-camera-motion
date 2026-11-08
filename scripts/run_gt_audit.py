"""Phase-1 gate: build all splits, audit generator + classifier, save report and figures.

Usage:  python scripts/run_gt_audit.py --out outputs/phase1
No model is trained or sampled here. The oracle sampler is a GT re-draw used only to
show what a perfectly calibrated model would score under the frozen evaluation contract.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_motion import dataset_generator as D  # noqa: E402
from camera_motion import geometry as G  # noqa: E402
from camera_motion import metrics as M  # noqa: E402
from camera_motion import mode_classifier as C  # noqa: E402
from camera_motion import visualize as V  # noqa: E402
from camera_motion.config import CATEGORIES, FROZEN  # noqa: E402


def audit_family(ds_audit):
    res = C.classify_dataset(ds_audit)
    cm = M.confusion_matrix(ds_audit["latent_mode"], res["label"])
    per_cond = {}
    for cid in np.unique(ds_audit["condition_id"]):
        sel = ds_audit["condition_id"] == cid
        per_cond[str(cid)] = {
            "generator_counts": np.bincount(ds_audit["latent_mode"][sel], minlength=4).tolist(),
            "classifier_report": M.mode_report(res["label"][sel]),
        }
    rho, dphi = res["rho"], np.degrees(res["delta_phi"])
    return res, cm, {
        "n_paths": int(len(res["label"])),
        "agreement": float(np.mean(res["label"] == ds_audit["latent_mode"])),
        "unassigned_rate": float(np.mean(res["label"] == 4)),
        "reasons": {r: int(np.sum(res["reason"] == r)) for r in C.REASONS},
        "confusion_rows_true_cols_pred": cm.tolist(),
        "rho_margin_min_abs": float(np.nanmin(np.abs(rho))),
        "rho_thresholds": [FROZEN.clf.tau_near, FROZEN.clf.tau_far],
        "delta_phi_deg_range": [float(np.nanmin(dphi)), float(np.nanmax(dphi))],
        "max_azimuth_step_deg": float(np.degrees(np.nanmax(res["max_step"]))),
        "per_condition": per_cond,
    }


def main(out: str, k_oracle: int = 200):
    t0 = time.time()
    fig_dir = os.path.join(out, "figures")
    data_dir = os.path.join(out, "data")
    os.makedirs(fig_dir, exist_ok=True)
    built = D.build_all(data_dir)
    report = {"config": FROZEN.to_dict(), "splits": {}}

    for name, ds in built.items():
        report["splits"][name] = {
            "n_paths": int(len(ds["p"])),
            "n_conditions": int(len(np.unique(ds["condition_id"]))),
            "rotation_validity": G.rotation_validity(ds["R"]),
        }
        if ds["meta"]["family"] in ("E1a", "E1b"):
            report["splits"][name]["mode_counts"] = np.bincount(ds["latent_mode"], minlength=4).tolist()

    # --- training split: diversity + independence
    for fam in ("E1a", "E1b"):
        tr = built[f"{fam}_train"]
        report[f"{fam}_train_independence_chi2"] = M.mode_condition_independence(tr)

    # --- condition-id disjointness across splits
    ids = {n: set(ds["condition_id"]) for n, ds in built.items()}
    names = list(ids)
    report["condition_ids_disjoint"] = all(
        not (ids[a] & ids[b]) for i, a in enumerate(names) for b in names[i + 1:])

    # --- GT audit
    for fam in ("E1a", "E1b"):
        ds = built[f"{fam}_gt_audit"]
        res, cm, rep = audit_family(ds)
        report[f"{fam}_gt_audit"] = rep
        fig, ax = V.plt.subplots(figsize=(5.4, 3.2))
        V.plot_confusion(cm, ax, title=f"{fam} GT audit: classifier vs hidden generator mode")
        V.save(fig, os.path.join(fig_dir, f"{fam}_audit_confusion.png"))
        cid = ds["condition_id"][0]
        fig, axes = V.plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 1.2, 1.2]})
        V.plot_condition_topdown(ds, cid, ax=axes[0], max_paths=100,
                                 title=f"{fam}: 100 GT paths, one public condition")
        V.plot_profiles(ds, cid, axes=axes[1:], max_paths=100)
        fig.tight_layout()
        V.save(fig, os.path.join(fig_dir, f"{fam}_audit_condition.png"))

    # --- E1b: excursion features (GT generator audit; latents used ONLY here)
    b = built["E1b_gt_audit"]
    fb = M.orientation_excursion_features(b["R"], b["p"], b["subject"])
    off = b["orientation_offset_params"]
    report["E1b_excursion_generator_audit"] = {
        "max_abs_err_vs_latent_amp_deg": float(np.max(np.abs(fb["max_excursion_deg"] - np.degrees(np.abs(off[:, 1]))))),
        "max_abs_err_vs_latent_peak_time": float(np.max(np.abs(fb["peak_time"] - off[:, 2]))),
        "median_max_excursion_deg": float(np.median(fb["max_excursion_deg"])),
        "pan_fraction": float(np.mean(off[:, 0] == 0)),
        "endpoint_offset_deg_max": float(np.degrees(np.max(G.geodesic_angle(
            G.look_at(b["p"][:, [0, -1]], b["subject"][:, None, :]), b["R"][:, [0, -1]])))),
    }
    fig, axes = V.plt.subplots(1, 2, figsize=(9, 3))
    V.plot_excursion_traces(b, b["condition_id"][0], axes=axes)
    fig.tight_layout()
    V.save(fig, os.path.join(fig_dir, "E1b_excursion_traces.png"))
    # two INDEPENDENT GT sets under the same condition -> sampling-noise floor for W1
    ev = built["E1b_eval_id"]
    pub = D.public_conditions(ev)
    row = {k: pub[k][0] for k in pub}
    rng = np.random.default_rng(11)
    R1, p1 = D.oracle_sampler("E1b")(row, 200, rng)
    R2, p2 = D.oracle_sampler("E1b")(row, 200, rng)
    f1 = M.orientation_excursion_features(R1, p1, row["subject"])
    f2 = M.orientation_excursion_features(R2, p2, row["subject"])
    report["E1b_w1_noise_floor_K200"] = M.excursion_distribution_distance(f1, f2)
    fig = V.plot_feature_hists(f1, f2, labels=("GT set A", "GT set B (independent)"))
    V.save(fig, os.path.join(fig_dir, "E1b_feature_hists_noise_floor.png"))

    # --- Oracle under the frozen evaluation contract (ceiling, not a model result)
    for fam, panel in (("E1a", "eval_id"), ("E1a", "eval_ood_position"), ("E1b", "eval_id")):
        ds = built[f"{fam}_{panel}"]
        out_c = M.evaluate_mode_contract(D.oracle_sampler(fam), D.public_conditions(ds), fam, K=k_oracle, seed=3)
        report[f"oracle_contract_{fam}_{panel}"] = {"pooled": out_c["pooled"], "endpoint": out_c["endpoint"]}

    # --- GT smoothness reference (for later ratio reporting)
    for fam in ("E1a", "E1b"):
        tr = built[f"{fam}_train"]
        sm = M.smoothness(tr["R"], tr["p"])
        report[f"{fam}_train_smoothness_median"] = {k: float(np.median(v)) for k, v in sm.items()}

    # --- E0 + E2 scope
    e0 = built["E0_sanity"]
    fig = V.plt.figure(figsize=(12, 3.6))
    for j, kind in enumerate(("orbit", "dolly", "tracking")):
        ax = fig.add_subplot(1, 3, j + 1, projection="3d")
        V.plot_camera_axes_3d(e0, int(np.nonzero(e0["shot"] == kind)[0][0]), ax=ax)
    V.save(fig, os.path.join(fig_dir, "E0_camera_axes.png"))
    try:
        C.classify_dataset(built["E2_composition_stub"])
        report["E2_scope_guard"] = "FAILED: classifier accepted E2"
    except C.ClassifierScopeError:
        report["E2_scope_guard"] = "ok: E1 classifier refuses E2"

    # --- classifier perturbation robustness (GT train paths)
    from camera_motion import robustness as Rb
    rob = {}
    for fam in ("E1a", "E1b"):
        tr = built[f"{fam}_train"]
        rob[fam] = Rb.run_probes(tr["p"], tr["subject"], tr["latent_mode"], fam, seed=5)
    rob["temporal_resampling_L64_invariant"] = Rb.temporal_resampling_invariance("E1a", 64, 300)
    report["classifier_robustness"] = rob
    fig, axes = V.plt.subplots(1, 4, figsize=(14, 3), sharey=True)
    for ax, name in zip(axes, ("jitter_xyz", "jitter_xy", "jitter_z", "lowfreq_xyz")):
        c = rob["E1a"]["curves"][name]
        lv = [float(k) for k in c]
        ax.plot(lv, [c[k]["agree"] for k in c], "-o", color=V.MODE_COLORS[0], lw=2, ms=6, label="agree")
        ax.plot(lv, [c[k]["unassigned"] for k in c], "--s", color=V.MUTED, lw=2, ms=6, label="unassigned")
        ax.plot(lv, [c[k]["wrong"] for k in c], ":^", color=V.MODE_COLORS[1], lw=2, ms=6, label="wrong mode")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("sigma / amplitude (m)")
        V._style(ax)
    axes[0].set_ylabel("fraction of GT paths")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"E1a classifier robustness ({FROZEN.clf.version})", fontsize=10, color=V.INK)
    fig.tight_layout()
    V.save(fig, os.path.join(fig_dir, "E1a_classifier_robustness.png"))

    report["runtime_sec"] = round(time.time() - t0, 2)
    with open(os.path.join(out, "phase1_audit_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # --- console summary
    print(f"Phase-1 audit written to {out}  ({report['runtime_sec']} s)")
    for fam in ("E1a", "E1b"):
        r = report[f"{fam}_gt_audit"]
        print(f"[{fam} GT audit] n={r['n_paths']} agreement={r['agreement']:.4f} "
              f"unassigned={r['unassigned_rate']:.4f} min|rho|={r['rho_margin_min_abs']:.3f} "
              f"max step={r['max_azimuth_step_deg']:.1f} deg")
    for k in [k for k in report if k.startswith("oracle_contract")]:
        p = report[k]["pooled"]
        print(f"[{k}] TV5={p['tv5_mean']:.3f} CI{np.round(p['tv5_ci95'], 3).tolist()} "
              f"coverage4={p['coverage4_mean']:.2f} unassigned={p['unassigned_rate_mean']:.3f}")
    print("[E1b] W1 noise floor (K=200):", {k: round(v, 3) for k, v in report["E1b_w1_noise_floor_K200"].items()})
    print("[split ids disjoint]", report["condition_ids_disjoint"], "|", report["E2_scope_guard"])
    worst_p = min(v["p_value"] for f in ("E1a", "E1b") for v in report[f"{f}_train_independence_chi2"].values())
    print(f"[train mode-condition independence] min chi2 p = {worst_p:.3f}")
    rb = report["classifier_robustness"]
    print(f"[robustness {FROZEN.clf.version}] invariances E1a={rb['E1a']['invariance']} "
          f"L64={rb['temporal_resampling_L64_invariant']} | xyz jitter 0.2 m agree="
          f"{rb['E1a']['curves']['jitter_xyz']['0.2']['agree']:.3f} | max wrong-mode rate="
          f"{max(rb[f]['max_wrong_rate_over_all_curves'] for f in ('E1a', 'E1b')):.4f}")
    print("categories:", CATEGORIES)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/phase1")
    ap.add_argument("--k-oracle", type=int, default=200)
    a = ap.parse_args()
    main(a.out, a.k_oracle)
