"""Why did rfm_so3 underperform? Path-level diagnostics of the SO(3) transport (no training).

    python scripts/diagnose_so3_transport.py --out outputs/diagnostics/so3_transport \
        --sweep rfm_so3_E1b=<run>/diagnostics/solver_sweep_eval_id.json \
        --sweep fm_rel_pose_E1b=<run>/diagnostics/solver_sweep_eval_id.json

Targets: the first ``--n`` trajectories of the frozen E1a / E1b TRAINING split (the
split is generated with ``--n-train`` paths, exactly as in training, and is already
shuffled independently of the hidden modes), encoded with ``rfm_so3.encode``:
relative rotations of frames 1..L-1, the anchored frame 0 = I is not generated.

A  temporal roughness of the RFM interpolation R_t = R0 Exp(t Log(R0^T R1)) at fixed
   flow times, with the training prior (independent Haar per frame);
B  the same path from a temporally coherent counterfactual prior (one Haar rotation Q
   per trajectory, shared by all frames; drawn independently of the targets);
C  statistics of the transport targets xi = Log(R0^T R1): angle, near-pi share,
   frame-to-frame variation, abrupt flips, numerical round-trip checks;
D  GT rotation geometry, and the Euclidean fm_rel_pose path (linear in standardised
   9-channel coordinates, 6D decoded by Gram-Schmidt) measured with the same metrics;
E  summary of existing solver sweeps (optional ``--sweep`` inputs).

Units: degrees per camera frame (delta) and degrees per frame^2 (a). ``a`` is a discrete
body-frame variation proxy, not a covariant acceleration or a cinematic jerk measure.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_motion import dataset_generator as D  # noqa: E402
from camera_motion import geometry as G  # noqa: E402
from camera_motion import metrics as M  # noqa: E402
from camera_motion import representations as Rp  # noqa: E402
from camera_motion import rfm_so3  # noqa: E402
from camera_motion import so3_torch as S  # noqa: E402
from camera_motion.config import FROZEN  # noqa: E402

FLOW_T = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
DEG = 180.0 / np.pi
PATHS = ("rfm_independent_haar", "rfm_shared_haar", "euclid_6d_rel_pose")


# ---------------------------------------------------------------------------
# primitives (unit-tested)
# ---------------------------------------------------------------------------
def step_metrics(R):
    """R (N,T,3,3) torch -> delta (N,T-1) inter-frame angle [rad], a (N,T-2) |omega_{i+1}-omega_i| [rad]."""
    w = S.log(R[:, :-1].transpose(-1, -2) @ R[:, 1:])
    return w.norm(dim=-1), (w[:, 1:] - w[:, :-1]).norm(dim=-1)


def geodesic_path(R0, R1, t):
    xi = S.log(R0.transpose(-1, -2) @ R1)
    return R0 @ S.exp(t * xi), xi


def independent_haar(shape, seed):
    return S.uniform(shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def shared_haar(n, T, seed):
    """One Haar rotation per trajectory, repeated over frames. Uses only (n, T, seed): no target access."""
    Q = S.uniform((n,), generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return Q[:, None].expand(n, T, 3, 3).contiguous()


def angle(Ra, Rb):
    return torch.from_numpy(G.geodesic_angle(Ra.numpy(), Rb.numpy()))


def summarize(x_deg):
    x = np.asarray(x_deg, dtype=float).ravel()
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p95": float(np.quantile(x, 0.95)),
            "max": float(x.max())}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_family(family, n, n_train):
    ds = D.generate_e1_train(family, n=n_train)
    R1_all, _ = rfm_so3.encode(ds["R"], ds["p"])
    x_all = Rp.encode_trajectories("rel_pose", ds["R"], ds["p"])
    c_all = Rp.encode_condition("rel_pose", ds["start_pose"], ds["end_pose"], ds["subject"])
    norm = Rp.Normalizer.fit(x_all, c_all)  # identical to the fm_rel_pose training normaliser
    sel = np.arange(n)
    return {"ds": {k: v[sel] for k, v in ds.items() if isinstance(v, np.ndarray) and len(v) == len(ds["p"])},
            "R1": torch.from_numpy(R1_all[sel]), "x1": norm.nx(x_all[sel]), "norm": norm}


# ---------------------------------------------------------------------------
# A / B / D-euclid: roughness along flow time
# ---------------------------------------------------------------------------
def path_states(path, fam, t, seed):
    R1 = fam["R1"]
    n, T = R1.shape[:2]
    if path == "rfm_independent_haar":
        Rt, _ = geodesic_path(independent_haar((n, T), seed), R1, t)
    elif path == "rfm_shared_haar":
        Rt, _ = geodesic_path(shared_haar(n, T, seed), R1, t)
    elif path == "euclid_6d_rel_pose":
        x0 = np.random.default_rng(seed).normal(size=fam["x1"].shape)
        xt = (1 - t) * x0 + t * fam["x1"]
        Rt = torch.from_numpy(G.rot6d_decode(fam["norm"].ux(xt)[..., 3:9]))
    else:
        raise ValueError(path)
    return Rt


def roughness_table(fam, seed):
    rows = []
    R1 = fam["R1"]
    I = torch.eye(3, dtype=R1.dtype).expand(R1.shape[0], 3, 3)
    for path in PATHS:
        for t in FLOW_T:
            Rt = path_states(path, fam, t, seed)
            d, a = step_metrics(Rt)
            rows.append({"path": path, "t": t,
                         **{f"delta_{k}": v for k, v in summarize(d.numpy() * DEG).items()},
                         **{f"a_{k}": v for k, v in summarize(a.numpy() * DEG).items()},
                         "dist_to_target_median": float(np.median(angle(Rt, R1).numpy() * DEG)),
                         "anchor_jump_median": float(np.median(angle(I, Rt[:, 0]).numpy() * DEG))})
    return rows


# ---------------------------------------------------------------------------
# C: transport targets
# ---------------------------------------------------------------------------
def target_stats(fam, seed):
    R1 = fam["R1"]
    n, T = R1.shape[:2]
    out, hist = {}, {}
    for name, R0 in (("independent_haar", independent_haar((n, T), seed)), ("shared_haar", shared_haar(n, T, seed))):
        Rt1, xi = geodesic_path(R0, R1, 1.0)
        ang = xi.norm(dim=-1).numpy() * DEG
        dxi = (xi[:, 1:] - xi[:, :-1]).norm(dim=-1).numpy() * DEG
        rel = R0.transpose(-1, -2) @ R1
        out[name] = {
            "xi_angle_deg": summarize(ang),
            "frac_within_5deg_of_pi": float(np.mean(ang > 175)), "frac_within_10deg_of_pi": float(np.mean(ang > 170)),
            "adjacent_dxi_deg": summarize(dxi),
            "frac_adjacent_dxi_over_90deg": float(np.mean(dxi > 90)),
            "frac_trajectories_with_flip": float(np.mean((dxi > 90).any(1))),
            "checks": {
                "exp_log_roundtrip_max_err": float((S.exp(xi) - rel).abs().max()),
                "exp_log_roundtrip_max_err_near_pi": float((S.exp(xi) - rel).abs()[torch.from_numpy(ang > 170)].max())
                if np.any(ang > 170) else 0.0,
                "endpoint_t0_max_err": float((geodesic_path(R0, R1, 0.0)[0] - R0).abs().max()),
                "endpoint_t1_max_err": float((Rt1 - R1).abs().max()),
                "orthogonality_max_err_mid": G.rotation_validity(geodesic_path(R0, R1, 0.5)[0].numpy())["max_orth_err"],
                "det_max_err_mid": G.rotation_validity(geodesic_path(R0, R1, 0.5)[0].numpy())["max_det_err"],
                "all_finite": bool(torch.isfinite(xi).all() and torch.isfinite(Rt1).all()),
            },
        }
        hist[name] = (ang, dxi)
    # Euclidean analogue: angle between the decoded prior rotation and its target, per frame
    R0e = path_states("euclid_6d_rel_pose", fam, 0.0, seed)
    ang_e = angle(R0e, R1).numpy() * DEG
    out["euclid_6d_prior_to_target_angle_deg"] = summarize(ang_e)
    hist["euclid"] = (ang_e, None)
    return out, hist


# ---------------------------------------------------------------------------
# D: GT geometry
# ---------------------------------------------------------------------------
def gt_geometry(fam):
    ds = fam["ds"]
    R = torch.from_numpy(ds["R"])
    d, a = step_metrics(R)
    from_anchor = angle(R[:, :1].expand_as(R), R).numpy() * DEG  # (N,L) principal angle to frame 0
    look = G.look_at(ds["p"], ds["subject"][:, None, :])
    dev = G.geodesic_angle(look, ds["R"]) * DEG
    exc = M.orientation_excursion_features(ds["R"], ds["p"], ds["subject"])
    return {
        "step_angle_deg": summarize(d.numpy() * DEG),
        "omega_variation_deg": summarize(a.numpy() * DEG),
        "principal_angle_from_anchor_deg": summarize(from_anchor),
        "frac_frames_over_150deg_from_anchor": float(np.mean(from_anchor > 150)),
        "frac_trajectories_reaching_170deg_from_anchor": float(np.mean(from_anchor.max(1) > 170)),
        "deviation_from_lookat_deg": summarize(dev),
        "e1b_max_excursion_deg": summarize(exc["max_excursion_deg"]),
        "_per_frame_from_anchor_median": np.median(from_anchor, 0).tolist(),
        "_per_frame_from_anchor_p95": np.quantile(from_anchor, 0.95, 0).tolist(),
        "_steps": d.numpy().ravel() * DEG,
    }


# ---------------------------------------------------------------------------
# E: solver sweeps
# ---------------------------------------------------------------------------
def solver_summary(sweeps):
    out = {}
    for name, path in sweeps:
        s = json.load(open(path))["settings"]
        g = lambda k, m: s[k].get(m)  # noqa: E731
        keys = ("tv5", "bias_excess", "jerk_ratio", "ang_acc_ratio", "e1b_w1_max_excursion_deg", "e1b_amp_outside_support")
        out[name] = {"settings": {k: {m: s[k].get(m) for m in keys} for k in s}}
        if all(k in s for k in ("euler:20", "euler:100", "euler:50", "midpoint:50")):
            out[name]["nfe_effect_euler20_to_100"] = {m: [g("euler:20", m), g("euler:100", m)] for m in keys}
            out[name]["solver_effect_at_nfe50"] = {m: [g("euler:50", m), g("midpoint:50", m)] for m in keys}
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figures(out, rough, tstats, gt, solver):
    from camera_motion import visualize as V
    col = {"rfm_independent_haar": V.MODE_COLORS[1], "rfm_shared_haar": V.MODE_COLORS[0], "euclid_6d_rel_pose": V.MODE_COLORS[2]}
    lab = {"rfm_independent_haar": "RFM path, independent Haar prior (trained)",
           "rfm_shared_haar": "RFM path, shared Haar prior (counterfactual)",
           "euclid_6d_rel_pose": "Euclidean 6D path (fm_rel_pose), decoded"}
    sty = {"rfm_independent_haar": "-o", "rfm_shared_haar": "--s", "euclid_6d_rel_pose": ":^"}
    fams = list(rough)

    def panel(ax, rows, key, title, gt_val, ylog=True):
        for p in PATHS:
            r = [x for x in rows if x["path"] == p]
            ax.plot([x["t"] for x in r], [x[key] for x in r], sty[p], color=col[p], lw=2, ms=6, label=lab[p])
        ax.axhline(gt_val, color=V.INK, lw=1.5, ls="-", alpha=0.7, label="ground truth")
        if ylog:
            ax.set_yscale("log")
        ax.set_xlabel("flow time t (0 = prior, 1 = data)")
        ax.set_title(title, fontsize=9)
        V._style(ax)

    fig, axes = V.plt.subplots(len(fams), 3, figsize=(15, 3.8 * len(fams)), squeeze=False)
    for i, f in enumerate(fams):
        panel(axes[i, 0], rough[f], "delta_median", f"{f}: median inter-frame angle δ (deg/frame)", gt[f]["step_angle_deg"]["median"])
        panel(axes[i, 1], rough[f], "a_median", f"{f}: median angular-velocity change a (deg/frame²)", gt[f]["omega_variation_deg"]["median"])
        panel(axes[i, 2], rough[f], "dist_to_target_median", f"{f}: median distance to target R1 (deg)", 0.0, ylog=False)
        axes[i, 0].set_ylabel("degrees")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    V.save(fig, os.path.join(out, "temporal_roughness_vs_flow_time.png"))

    f0 = fams[-1]
    fig, axes = V.plt.subplots(1, 2, figsize=(11, 3.6))
    for p in ("rfm_independent_haar", "rfm_shared_haar"):
        r = [x for x in rough[f0] if x["path"] == p]
        axes[0].plot([x["t"] for x in r], [x["delta_p95"] for x in r], sty[p], color=col[p], lw=2, label=lab[p])
        axes[1].plot([x["t"] for x in r], [x["a_p95"] for x in r], sty[p], color=col[p], lw=2, label=lab[p])
    for ax, k, ttl in ((axes[0], "step_angle_deg", "95th pct inter-frame angle δ"), (axes[1], "omega_variation_deg", "95th pct angular-velocity change a")):
        ax.axhline(gt[f0][k]["p95"], color=V.INK, lw=1.5, label="ground truth")
        ax.set_yscale("log")
        ax.set_xlabel("flow time t")
        ax.set_ylabel("degrees")
        ax.set_title(f"{f0}: {ttl} — same targets, same seed", fontsize=9)
        V._style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    V.save(fig, os.path.join(out, "independent_vs_correlated_prior.png"))

    fig, axes = V.plt.subplots(1, 3, figsize=(15, 3.6))
    ang_i, dxi_i = tstats[f0][1]["independent_haar"]
    ang_s, dxi_s = tstats[f0][1]["shared_haar"]
    ang_e = tstats[f0][1]["euclid"][0]
    bins = np.linspace(0, 180, 37)
    axes[0].hist(ang_i.ravel(), bins=bins, density=True, color=col["rfm_independent_haar"], alpha=0.75, label="RFM ‖ξ‖ (Haar prior)")
    axes[0].hist(ang_e.ravel(), bins=bins, density=True, histtype="step", lw=2, color=col["euclid_6d_rel_pose"], label="Euclidean: angle(decoded prior, target)")
    axes[0].axvspan(170, 180, color=V.GRID, zorder=0, label="within 10° of π")
    axes[0].set_xlabel("per-frame transport angle (deg)")
    axes[0].set_title(f"{f0}: how far each frame must travel", fontsize=9)
    b2 = np.linspace(0, 360, 49)
    axes[1].hist(dxi_i.ravel(), bins=b2, density=True, color=col["rfm_independent_haar"], alpha=0.75, label="independent Haar")
    axes[1].hist(dxi_s.ravel(), bins=b2, density=True, histtype="step", lw=2, color=col["rfm_shared_haar"], label="shared Haar")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("‖ξ_{i+1} − ξ_i‖ (deg)")
    axes[1].set_title("frame-to-frame change of the velocity target", fontsize=9)
    st = tstats[f0][0]
    vals = [st["independent_haar"]["frac_within_10deg_of_pi"], st["shared_haar"]["frac_within_10deg_of_pi"],
            st["independent_haar"]["frac_adjacent_dxi_over_90deg"], st["shared_haar"]["frac_adjacent_dxi_over_90deg"]]
    names = ["‖ξ‖ within 10° of π\n(independent)", "‖ξ‖ within 10° of π\n(shared)", "adjacent Δξ > 90°\n(independent)", "adjacent Δξ > 90°\n(shared)"]
    axes[2].bar(range(4), np.array(vals) * 100, color=[col["rfm_independent_haar"], col["rfm_shared_haar"]] * 2)
    for k, v in enumerate(vals):
        axes[2].text(k, v * 100 + 0.5, f"{100 * v:.1f}%", ha="center", fontsize=8, color=V.INK)
    axes[2].set_xticks(range(4), names, fontsize=7)
    axes[2].set_ylabel("% of frames / frame pairs")
    axes[2].set_title("near-π targets and abrupt target flips", fontsize=9)
    for ax in axes:
        V._style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    V.save(fig, os.path.join(out, "transport_target_statistics.png"))

    fig, axes = V.plt.subplots(1, 3, figsize=(15, 3.6))
    for i, f in enumerate(fams):
        axes[0].hist(gt[f]["_steps"], bins=40, density=True, alpha=0.6 if i == 0 else 1, histtype="bar" if i == 0 else "step",
                     lw=2, color=V.MODE_COLORS[i], label=f)
        L = len(gt[f]["_per_frame_from_anchor_median"])
        axes[1].plot(range(L), gt[f]["_per_frame_from_anchor_median"], color=V.MODE_COLORS[i], lw=2, label=f"{f} median")
        axes[1].plot(range(L), gt[f]["_per_frame_from_anchor_p95"], color=V.MODE_COLORS[i], lw=1.5, ls="--", label=f"{f} 95th pct")
    axes[0].set_xlabel("GT inter-frame rotation (deg/frame)")
    axes[0].set_title("GT rotation steps are small", fontsize=9)
    axes[1].axhline(180, color=V.MUTED, lw=1)
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("principal angle from start (deg)")
    axes[1].set_title("GT relative orientation range (what rfm_so3 generates)", fontsize=9)
    if solver:
        ax = axes[2]
        for j, (name, s) in enumerate(solver.items()):
            ks = list(s["settings"])
            w1 = [s["settings"][k].get("e1b_w1_max_excursion_deg") for k in ks]
            if any(v is None for v in w1):
                continue
            ax.plot(range(len(ks)), w1, "-o", color=V.MODE_COLORS[1 if "rfm" in name else 2], lw=2, label=f"{name}: E1b W1 (deg)")
            ax.set_xticks(range(len(ks)), ks, fontsize=7, rotation=20)
        ax.set_title("existing solver sweep: E1b orientation W1", fontsize=9)
        ax.set_ylim(bottom=0)
    for ax in axes:
        V._style(ax)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    V.save(fig, os.path.join(out, "gt_rotation_geometry.png"))


def main(a):
    """Run a diagnostic without leaking its float64 default into other tests."""
    original_dtype = torch.get_default_dtype()
    try:
        return _main_impl(a)
    finally:
        torch.set_default_dtype(original_dtype)


def _main_impl(a):
    os.makedirs(a.out, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    rough, tstats, gt = {}, {}, {}
    for fam in ("E1a", "E1b"):
        F = load_family(fam, a.n, a.n_train)
        rough[fam] = roughness_table(F, a.seed)
        tstats[fam] = target_stats(F, a.seed)
        gt[fam] = gt_geometry(F)
    solver = solver_summary([tuple(s.split("=", 1)) for s in a.sweep]) if a.sweep else {}
    figures(a.out, rough, tstats, gt, solver)
    with open(os.path.join(a.out, "metrics.csv"), "w", newline="") as f:
        rows = [{"family": fam, **r} for fam in rough for r in rough[fam]]
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "setup": {"n_trajectories_per_family": a.n, "training_split_size": a.n_train, "seed": a.seed,
                  "subset": "first n rows of the frozen training split (generated with n_train paths)",
                  "flow_times": list(FLOW_T), "frames_generated": int(FROZEN.split.L - 1),
                  "interior_metrics": "frames 1..L-1 only; anchor jump reported separately",
                  "units": "deg/frame (delta), deg/frame^2 (a)", "classifier_version": FROZEN.clf.version},
        "roughness": rough,
        "transport_targets": {f: tstats[f][0] for f in tstats},
        "gt_geometry": {f: {k: v for k, v in gt[f].items() if not k.startswith("_")} for f in gt},
        "solver_sweeps": solver,
    }
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for fam in rough:
        print(f"\n[{fam}] GT: step median {gt[fam]['step_angle_deg']['median']:.2f} deg/frame, "
              f"a median {gt[fam]['omega_variation_deg']['median']:.3f}")
        for r in rough[fam]:
            if r["t"] in (0.0, 0.5, 0.9, 1.0):
                print(f"  {r['path']:22s} t={r['t']:.2f} delta med {r['delta_median']:8.2f} p95 {r['delta_p95']:8.2f} | "
                      f"a med {r['a_median']:8.2f} | dist-to-target {r['dist_to_target_median']:7.2f} | anchor {r['anchor_jump_median']:7.2f}")
        st = tstats[fam][0]
        for k in ("independent_haar", "shared_haar"):
            print(f"  xi[{k}] angle med {st[k]['xi_angle_deg']['median']:.1f} | near-pi(10) {100 * st[k]['frac_within_10deg_of_pi']:.1f}% "
                  f"| dxi>90 {100 * st[k]['frac_adjacent_dxi_over_90deg']:.2f}% | traj w/ flip {100 * st[k]['frac_trajectories_with_flip']:.1f}% | checks {st[k]['checks']}")
        print(f"  euclid prior->target angle {st['euclid_6d_prior_to_target_angle_deg']}")
    print("wrote", a.out)
    return summary


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/diagnostics/so3_transport")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="append", default=[], help="name=solver_sweep_eval_id.json (repeatable)")
    return ap.parse_args(argv)


if __name__ == "__main__":
    main(parse())
