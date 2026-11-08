"""Seeded synthetic camera-trajectory generator: E0 / E1a / E1b / E2-stub.

Three sampling protocols are kept strictly separate (handoff v1.5):

* ``train``      many distinct public conditions, 1 path each (default); the hidden
                 4-mode label is stratified at the **split level** (largest remainder)
                 and shuffled with its own RNG, independently of the conditions.
* ``gt_audit``   a small panel of fixed conditions, ``audit_group_size`` paths each,
                 with the exact 42/28/18/12 allocation. Used only to audit the
                 generator and the classifier.
* ``eval_id`` / ``eval_ood_position``   held-out conditions. Models are sampled K times
                 per condition; the small GT set stored here is a *distributional
                 reference only* (never paired with model samples by index).

E1 geometry (positions)
-----------------------
For condition (subject s, start bearing phi0, start/end radius r0/r1, height h) and
camera-time u in [0, 1]:

    tau(u)   = u + w sin(2 pi u) / (2 pi)                      hidden time warp
    phi(u)   = phi0 + dphi * tau,  dphi in {+pi/2 (short_ccw), -3pi/2 (long_cw)}
    r(u)     = lerp(r0, r1, tau) * (1 + sgn * A * sin(pi tau)),  sgn = -1 near / +1 far
    p(u)     = s + (r cos phi, r sin phi, h)

Both angular branches end on the same bearing phi0 + pi/2 (mod 2pi), and the bump
vanishes at both ends, so every mode shares the exact public start/end pose. The
first/last frames are written from the condition itself (not from the mode) so the
endpoint identity is bit-exact.

Orientation: E1a uses pure look-at to the subject. E1b right-multiplies a camera-local
pan (about camera y) or roll (about camera z) offset ``Exp(axis * a * f(u))`` with
f(0)=f(1)=0 and a single smooth peak at a hidden time, so start/end poses are unchanged.
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass

import numpy as np

from . import geometry as G
from .config import FROZEN, MODE_NAMES, Config

AXIS_IDS = {"none": -1, "pan": 0, "roll": 1}
AXIS_VEC = {0: np.array([0.0, 1.0, 0.0]), 1: np.array([0.0, 0.0, 1.0])}


# ---------------------------------------------------------------------------
# RNG plumbing
# ---------------------------------------------------------------------------
def _seed_seq(master: int, *names: str) -> np.random.SeedSequence:
    keys = [zlib.crc32(n.encode()) for n in names]
    return np.random.SeedSequence([master, *keys])


def make_rngs(master: int, family: str, split: str):
    """Independent streams for conditions, hidden modes, and other hidden latents."""
    ss = _seed_seq(master, family, split)
    c, m, l = ss.spawn(3)
    return np.random.default_rng(c), np.random.default_rng(m), np.random.default_rng(l)


def stratified_counts(n: int, probs) -> np.ndarray:
    """Largest-remainder allocation of n items to probabilities (sums exactly to n)."""
    probs = np.asarray(probs, dtype=float)
    raw = probs * n
    base = np.floor(raw).astype(int)
    rem = n - base.sum()
    order = np.argsort(-(raw - base), kind="stable")
    base[order[:rem]] += 1
    return base


def stratified_modes(n: int, probs, rng) -> np.ndarray:
    counts = stratified_counts(n, probs)
    modes = np.repeat(np.arange(len(probs)), counts)
    rng.shuffle(modes)
    return modes


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
@dataclass
class Conditions:
    subject: np.ndarray  # (n,3)
    phi0: np.ndarray  # (n,)
    r0: np.ndarray
    r1: np.ndarray
    h: np.ndarray

    def __len__(self):
        return len(self.phi0)


def sample_conditions(rng, n: int, cfg: Config = FROZEN, ood_position: bool = False) -> Conditions:
    c = cfg.cond
    if ood_position:
        rad = rng.uniform(*c.subject_xy_ood_annulus, size=n)
        ang = rng.uniform(0, 2 * np.pi, size=n)
        sxy = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=-1)
    else:
        sxy = rng.uniform(*c.subject_xy_train, size=(n, 2))
    sz = rng.uniform(*c.subject_z, size=(n, 1))
    return Conditions(
        subject=np.concatenate([sxy, sz], axis=-1),
        phi0=rng.uniform(*c.start_bearing, size=n),
        r0=rng.uniform(*c.radius, size=n),
        r1=rng.uniform(*c.radius, size=n),
        h=rng.uniform(*c.height, size=n),
    )


def endpoint_positions(s, phi0, r0, r1, h):
    p_start = s + np.array([r0 * np.cos(phi0), r0 * np.sin(phi0), h])
    phi_end = phi0 + np.pi / 2
    p_end = s + np.array([r1 * np.cos(phi_end), r1 * np.sin(phi_end), h])
    return p_start, p_end


# ---------------------------------------------------------------------------
# Trajectory primitives
# ---------------------------------------------------------------------------
def time_warp(u, w):
    return u + w * np.sin(2 * np.pi * u) / (2 * np.pi)


def e1_positions(s, phi0, r0, r1, h, mode: int, warp: float, bump: float, L: int, cfg: Config = FROZEN):
    """Positions (L,3) for one E1 path. mode: 0..3 in MODE_NAMES order."""
    ang_branch, rad_branch = divmod(int(mode), 2)  # 0 short / 1 long ; 0 near / 1 far
    dphi = cfg.modes.sweep_short if ang_branch == 0 else cfg.modes.sweep_long
    sgn = -1.0 if rad_branch == 0 else 1.0
    u = np.linspace(0.0, 1.0, L)
    tau = time_warp(u, warp)
    phi = phi0 + dphi * tau
    r = (r0 + (r1 - r0) * tau) * (1.0 + sgn * bump * np.sin(np.pi * tau))
    p = np.stack([s[0] + r * np.cos(phi), s[1] + r * np.sin(phi), np.full(L, s[2] + h)], axis=-1)
    p_start, p_end = endpoint_positions(s, phi0, r0, r1, h)
    p[0], p[-1] = p_start, p_end  # bit-exact shared endpoints
    return p


def offset_profile(u, peak_time):
    """Smooth single-peak profile, f(0)=f(1)=0, f(peak_time)=1, C1 at the peak."""
    g = np.where(u < peak_time, 0.5 * u / peak_time, 0.5 + 0.5 * (u - peak_time) / (1.0 - peak_time))
    f = np.sin(np.pi * g)
    f[0] = 0.0
    f[-1] = 0.0
    return f


def offset_rotations(L: int, axis_id: int, signed_amp: float, peak_time: float):
    if axis_id < 0:
        return np.broadcast_to(np.eye(3), (L, 3, 3)).copy()
    u = np.linspace(0.0, 1.0, L)
    f = offset_profile(u, peak_time)
    w = AXIS_VEC[axis_id][None, :] * (signed_amp * f)[:, None]
    return G.so3_exp(w)


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------
def _empty_record():
    return {k: [] for k in (
        "R", "p", "condition_id", "start_pose", "end_pose", "subject", "shot",
        "condition_params", "latent_mode", "latent_warp", "latent_bump",
        "orientation_offset_params", "family", "split",
    )}


def _append(rec, *, R, p, cid, subject, shot, cparams, mode, warp, bump, off, family, split):
    rec["R"].append(R)
    rec["p"].append(p)
    rec["condition_id"].append(cid)
    rec["start_pose"].append(G.make_pose(R[0], p[0]))
    rec["end_pose"].append(G.make_pose(R[-1], p[-1]))
    rec["subject"].append(subject)
    rec["shot"].append(shot)
    rec["condition_params"].append(cparams)
    rec["latent_mode"].append(mode)
    rec["latent_warp"].append(warp)
    rec["latent_bump"].append(bump)
    rec["orientation_offset_params"].append(off)
    rec["family"].append(family)
    rec["split"].append(split)


def _finalize(rec, meta: dict) -> dict:
    ds = {k: np.asarray(v) for k, v in rec.items()}
    ds["condition_id"] = ds["condition_id"].astype(str)
    ds["shot"] = ds["shot"].astype(str)
    ds["family"] = ds["family"].astype(str)
    ds["split"] = ds["split"].astype(str)
    ds["meta"] = meta
    return ds


def _e1_one(cond: Conditions, j: int, mode: int, rng_lat, family: str, cfg: Config):
    L = cfg.split.L
    lat = cfg.latent
    s = cond.subject[j]
    warp = rng_lat.uniform(*lat.time_warp)
    bump = rng_lat.uniform(*lat.radial_bump_abs)
    p = e1_positions(s, cond.phi0[j], cond.r0[j], cond.r1[j], cond.h[j], mode, warp, bump, L, cfg)
    R = G.look_at(p, s)
    off = np.array([AXIS_IDS["none"], 0.0, 0.0])
    if family == "E1b":
        axis_id = int(rng_lat.integers(0, 2))
        amp = np.deg2rad(rng_lat.uniform(*lat.offset_amp_deg)) * rng_lat.choice([-1.0, 1.0])
        tp = rng_lat.uniform(*lat.offset_peak_time)
        R = R @ offset_rotations(L, axis_id, amp, tp)
        off = np.array([axis_id, amp, tp])
    cparams = np.array([cond.phi0[j], cond.r0[j], cond.r1[j], cond.h[j]])
    return R, p, cparams, warp, bump, off


def _meta(cfg: Config, family: str, split: str, extra=None) -> dict:
    m = {
        "family": family,
        "split": split,
        "master_seed": cfg.seed,
        "L": cfg.split.L,
        "dt": 1.0 / (cfg.split.L - 1),
        "mode_names": list(MODE_NAMES),
        "declared_target_mode_probability": list(cfg.modes.joint()),
        "condition_params_fields": ["phi0", "r0", "r1", "h"],
        "orientation_offset_params_fields": ["axis_id(-1 none,0 pan,1 roll)", "signed_amp_rad", "peak_time"],
        "public_fields": ["start_pose", "end_pose", "subject", "shot"],
        "hidden_fields": ["latent_mode", "latent_warp", "latent_bump", "orientation_offset_params"],
    }
    if extra:
        m.update(extra)
    return m


# ---------------------------------------------------------------------------
# E1 splits
# ---------------------------------------------------------------------------
def generate_e1_train(family: str, n: int | None = None, cfg: Config = FROZEN) -> dict:
    assert family in ("E1a", "E1b")
    if n is None:
        n = cfg.split.n_train_e1a if family == "E1a" else cfg.split.n_train_e1b
    k = cfg.split.paths_per_train_condition
    n_cond = int(np.ceil(n / k))
    rc, rm, rl = make_rngs(cfg.seed, family, "train")
    cond = sample_conditions(rc, n_cond, cfg)
    modes = stratified_modes(n, cfg.modes.joint(), rm)  # split-level stratification
    rec = _empty_record()
    for i in range(n):
        j = i // k
        R, p, cp, w, b, off = _e1_one(cond, j, modes[i], rl, family, cfg)
        _append(rec, R=R, p=p, cid=f"{family}-train-c{j:05d}", subject=cond.subject[j], shot="orbit",
                cparams=cp, mode=modes[i], warp=w, bump=b, off=off, family=family, split="train")
    return _finalize(rec, _meta(cfg, family, "train", {
        "stratification": "split-level largest remainder, shuffled independently of conditions",
        "observed_mode_counts": np.bincount(modes, minlength=4).tolist(),
        "paths_per_condition": k,
    }))


def generate_e1_audit(family: str, n_cond: int | None = None, group: int | None = None, cfg: Config = FROZEN) -> dict:
    assert family in ("E1a", "E1b")
    if n_cond is None:
        n_cond = cfg.split.n_audit_conditions_e1a if family == "E1a" else cfg.split.n_audit_conditions_e1b
    group = group or cfg.split.audit_group_size
    rc, rm, rl = make_rngs(cfg.seed, family, "gt_audit")
    cond = sample_conditions(rc, n_cond, cfg)
    rec = _empty_record()
    for j in range(n_cond):
        modes = stratified_modes(group, cfg.modes.joint(), rm)  # exact per-condition counts
        for m in modes:
            R, p, cp, w, b, off = _e1_one(cond, j, m, rl, family, cfg)
            _append(rec, R=R, p=p, cid=f"{family}-audit-c{j:03d}", subject=cond.subject[j], shot="orbit",
                    cparams=cp, mode=m, warp=w, bump=b, off=off, family=family, split="gt_audit")
    return _finalize(rec, _meta(cfg, family, "gt_audit", {
        "stratification": "exact per-condition largest remainder",
        "group_size": group,
        "expected_counts_per_condition": stratified_counts(group, cfg.modes.joint()).tolist(),
    }))


def generate_e1_eval(family: str, panel: str = "eval_id", n_cond: int | None = None,
                     n_ref: int | None = None, cfg: Config = FROZEN) -> dict:
    """Held-out conditions + a small GT reference set per condition (distributional only)."""
    assert family in ("E1a", "E1b") and panel in ("eval_id", "eval_ood_position")
    ood = panel == "eval_ood_position"
    if n_cond is None:
        n_cond = cfg.split.n_eval_ood_conditions if ood else cfg.split.n_eval_id_conditions
    if n_ref is None:
        n_ref = cfg.split.eval_gt_ref_per_condition_e1b if family == "E1b" else cfg.split.eval_gt_ref_per_condition
    rc, rm, rl = make_rngs(cfg.seed, family, panel)
    cond = sample_conditions(rc, n_cond, cfg, ood_position=ood)
    rec = _empty_record()
    for j in range(n_cond):
        modes = stratified_modes(n_ref, cfg.modes.joint(), rm)
        for m in modes:
            R, p, cp, w, b, off = _e1_one(cond, j, m, rl, family, cfg)
            _append(rec, R=R, p=p, cid=f"{family}-{panel}-c{j:03d}", subject=cond.subject[j], shot="orbit",
                    cparams=cp, mode=m, warp=w, bump=b, off=off, family=family, split=panel)
    return _finalize(rec, _meta(cfg, family, panel, {
        "gt_reference_note": "distributional reference only; never pair with model samples by index",
        "gt_ref_per_condition": n_ref,
        "ood": "subject xy in annulus %s (train: square %s)" % (cfg.cond.subject_xy_ood_annulus, cfg.cond.subject_xy_train) if ood else None,
    }))


# ---------------------------------------------------------------------------
# E0 sanity and E2 stub
# ---------------------------------------------------------------------------
def generate_e0(n_per_kind: int | None = None, cfg: Config = FROZEN) -> dict:
    """Single-mode sanity shots: orbit (+90 deg, look-at), dolly (radial, look-at),
    tracking (lateral straight line, fixed orientation)."""
    n_per_kind = n_per_kind or cfg.split.n_e0_per_kind
    L = cfg.split.L
    rc, _, rl = make_rngs(cfg.seed, "E0", "sanity")
    rec = _empty_record()
    u = np.linspace(0, 1, L)
    idx = 0
    for kind in ("orbit", "dolly", "tracking"):
        cond = sample_conditions(rc, n_per_kind, cfg)
        for j in range(n_per_kind):
            s, phi0, r0, r1, h = cond.subject[j], cond.phi0[j], cond.r0[j], cond.r1[j], cond.h[j]
            if kind == "orbit":
                p = e1_positions(s, phi0, r0, r1, h, 0, 0.0, 0.0, L, cfg)
                R = G.look_at(p, s)
            elif kind == "dolly":
                rr = r0 + (0.5 * r0 - r0) * u  # dolly-in to half the distance
                p = s + np.stack([rr * np.cos(phi0), rr * np.sin(phi0), np.full(L, h)], -1)
                R = G.look_at(p, s)
            else:
                p0 = s + np.array([r0 * np.cos(phi0), r0 * np.sin(phi0), h])
                R0 = G.look_at(p0, s)
                lateral = R0[:, 0]  # camera right
                p = p0[None] + (u * 2.0)[:, None] * lateral[None]
                R = np.broadcast_to(R0, (L, 3, 3)).copy()
            _append(rec, R=R, p=p, cid=f"E0-{kind}-c{j:03d}", subject=s, shot=kind,
                    cparams=np.array([phi0, r0, r1, h]), mode=-1, warp=0.0, bump=0.0,
                    off=np.array([-1, 0.0, 0.0]), family="E0", split="sanity")
            idx += 1
    return _finalize(rec, _meta(cfg, "E0", "sanity", {"kinds": ["orbit", "dolly", "tracking"]}))


def generate_e2_composition_stub(n: int | None = None, cfg: Config = FROZEN) -> dict:
    """Minimal E2 OOD-composition stub: short CCW orbit + crane (height change).

    Exists so the E1 classifier scope guard can be tested. Not a finished E2 benchmark.
    """
    n = n or cfg.split.n_e2_stub
    L = cfg.split.L
    rc, _, rl = make_rngs(cfg.seed, "E2_composition", "stub")
    cond = sample_conditions(rc, n, cfg)
    rec = _empty_record()
    u = np.linspace(0, 1, L)
    for j in range(n):
        s = cond.subject[j]
        p = e1_positions(s, cond.phi0[j], cond.r0[j], cond.r1[j], cond.h[j], 0, 0.0, 0.0, L, cfg)
        dz = rl.uniform(1.0, 2.0) * rl.choice([-1.0, 1.0])
        p[:, 2] += dz * (3 * u**2 - 2 * u**3)  # smooth crane
        R = G.look_at(p, s)
        _append(rec, R=R, p=p, cid=f"E2c-stub-c{j:03d}", subject=s, shot="orbit_crane",
                cparams=np.array([cond.phi0[j], cond.r0[j], cond.r1[j], cond.h[j]]), mode=-1,
                warp=0.0, bump=0.0, off=np.array([-1, 0.0, 0.0]), family="E2_composition", split="stub")
    return _finalize(rec, _meta(cfg, "E2_composition", "stub", {"note": "scope-guard stub only"}))


# ---------------------------------------------------------------------------
# Public view & I/O
# ---------------------------------------------------------------------------
PUBLIC_KEYS = ("condition_id", "start_pose", "end_pose", "subject", "shot")


def public_conditions(ds: dict) -> dict:
    """One row per unique condition_id with ONLY public fields (what a model may see)."""
    _, first = np.unique(ds["condition_id"], return_index=True)
    first = np.sort(first)
    return {k: ds[k][first] for k in PUBLIC_KEYS}


def conditions_from_public(start_pose, end_pose, subject) -> Conditions:
    """Recover E1 condition scalars from PUBLIC fields only (start/end pose, subject)."""
    s = np.asarray(subject, dtype=float)
    p0, p1 = np.asarray(start_pose)[:3, 3], np.asarray(end_pose)[:3, 3]
    d0, d1 = p0 - s, p1 - s
    return Conditions(subject=s[None], phi0=np.array([np.arctan2(d0[1], d0[0])]),
                      r0=np.array([np.hypot(d0[0], d0[1])]), r1=np.array([np.hypot(d1[0], d1[1])]),
                      h=np.array([d0[2]]))


def oracle_sampler(family: str, cfg: Config = FROZEN):
    """A reference 'model' that draws fresh GT paths for a public condition.

    Modes are drawn i.i.d. from the declared p(mode|c). Used to unit-test the
    evaluation contract and as a ceiling in reports; it sees public fields only.
    """
    probs = np.asarray(cfg.modes.joint())

    def sample(row: dict, K: int, rng):
        cond = conditions_from_public(row["start_pose"], row["end_pose"], row["subject"])
        modes = rng.choice(4, size=K, p=probs)
        Rs, ps = [], []
        for m in modes:
            R, p, *_ = _e1_one(cond, 0, m, rng, family, cfg)
            Rs.append(R)
            ps.append(p)
        return np.stack(Rs), np.stack(ps)

    return sample


def save_dataset(ds: dict, path: str) -> None:
    arrays = {k: v for k, v in ds.items() if k != "meta"}
    arrays["meta_json"] = np.array(json.dumps(ds["meta"]))
    np.savez_compressed(path, **arrays)


def load_dataset(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        ds = {k: z[k] for k in z.files if k != "meta_json"}
        ds["meta"] = json.loads(str(z["meta_json"]))
    return ds


def build_all(out_dir: str, cfg: Config = FROZEN) -> dict:
    """Generate every Phase-1 split and write npz files + config_frozen.json."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    built = {
        "E0_sanity": generate_e0(cfg=cfg),
        "E1a_train": generate_e1_train("E1a", cfg=cfg),
        "E1b_train": generate_e1_train("E1b", cfg=cfg),
        "E1a_gt_audit": generate_e1_audit("E1a", cfg=cfg),
        "E1b_gt_audit": generate_e1_audit("E1b", cfg=cfg),
        "E1a_eval_id": generate_e1_eval("E1a", "eval_id", cfg=cfg),
        "E1b_eval_id": generate_e1_eval("E1b", "eval_id", cfg=cfg),
        "E1a_eval_ood_position": generate_e1_eval("E1a", "eval_ood_position", cfg=cfg),
        "E2_composition_stub": generate_e2_composition_stub(cfg=cfg),
    }
    for name, ds in built.items():
        save_dataset(ds, os.path.join(out_dir, f"{name}.npz"))
    with open(os.path.join(out_dir, "config_frozen.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    return built
