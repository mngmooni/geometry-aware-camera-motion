# Why did `rfm_so3` underperform? SO(3) transport diagnostics

Single training seed per model; no model was trained or retrained for this analysis.

## 1. Question

At the matched evaluation (K = 200, Euler NFE = 50), `rfm_so3` produced valid rotations but fit the
data worse than the Euclidean baselines: E1a TV5 0.074 vs 0.053 (`fm_rel_pose`); E1b orientation
excursion W1 4.82° vs 1.55°; E1b amplitude outside the declared support 14.6 % vs 5.5 %. Why?

## 2. Hypotheses

- **H1** The independent per-frame Haar prior makes intermediate rotation sequences temporally irregular.
- **H2** Framewise geodesic interpolation does not produce temporally smooth trajectories.
- **H3** The benchmark lacks the geometric difficulty that SO(3) transport is designed to handle; the 6D
  Euclidean baseline already covers its rotations well.
- **H4** Numerical integration error contributes, but may not explain everything.

## 3. Setup

- Targets: the first 1,000 trajectories of the frozen E1a and E1b training splits (each generated with
  10,000 paths, as in training; the split is shuffled independently of the hidden modes). They are encoded
  with `rfm_so3.encode`, giving the relative rotations of frames 1–31. The anchored frame 0 is not
  generated and is excluded from interior metrics.
- **Paths, all ending at the same targets, seed 0:**
  - RFM, independent Haar (the trained construction).
  - RFM, shared Haar: one Q per trajectory used for every frame, drawn independently of the targets.
    This is a counterfactual path, not a trained model.
  - The `fm_rel_pose` Euclidean path: linear in standardised 9-channel coordinates with the training
    normaliser, and 6D decoded by Gram–Schmidt. Its prior and path construction differ, so the comparison
    is not geometry-only.
- **Metrics** at t ∈ {0, .1, .25, .5, .75, .9, 1}:
  - δ: inter-frame geodesic angle (deg/frame).
  - a: ‖ω_{i+1} − ω_i‖ with ω_i = Log(R_iᵀR_{i+1}) (deg/frame²). This is a discrete body-frame variation
    proxy, not a covariant acceleration.
  - Distance to target.
- Transport targets ξ = Log(R0ᵀR1); GT rotation geometry; existing solver sweeps (Gate 5).
- Reproduce:
  `python scripts/diagnose_so3_transport.py --out outputs/diagnostics/so3_transport --sweep rfm_so3_E1b=<…>/solver_sweep_eval_id.json --sweep fm_rel_pose_E1b=<…> --sweep rfm_so3_E1a=<…>`
  (CPU, ~15 s). Tests: `pytest tests/test_so3_transport_diagnostics.py` (6 passed).

## 4. Findings

Numbers are for E1b; E1a is within a few degrees unless stated. Full tables are in `summary.json` and
`metrics.csv`.

| Quantity | RFM independent Haar | RFM shared Haar | Euclidean 6D path | GT |
|---|---|---|---|---|
| per-frame transport angle, median | 132° (‖ξ‖) | 131° | 76° (decoded prior → target; E1a 64°) | — |
| δ median at t = 0 / 0.5 / 0.9 (deg/frame) | 132 / 85 / 18 | 0 / 2.5 / 4.2 | 82 / 39 / 7.3 | 4.5 |
| a median at t = 0.5 / 0.9 (deg/frame²) | 144 / 30 | 0.2 / 0.3 | 67 / 9.7 | 0.33 |
| distance to target at t = 0.75 / 0.9 | 33° / 13° | 33° / 13° | 12° / 4.5° | 0 |
| ‖ξ‖ within 10° of π | 11.1 % | 10.6 % | n/a (target linear, continuous) | — |
| adjacent ‖ξ_{i+1} − ξ_i‖ > 90° | 89 % of pairs | 1.4 % of pairs (38 % of trajectories have ≥ 1 flip) | — | — |

- **Numerics are sound.** Exp(Log) round-trip error is ≤ 2e-15, including near π. Orthogonality and
  determinant errors are ≤ 4e-15, endpoints are exact, and all values are finite. The near-π share
  matches Haar theory, (δ + sin δ)/π = 11.08 % for δ = 10°.
- **GT geometry is benign.**
  - Inter-frame rotation has a median of 3.3° (E1a) and 4.5° (E1b), with a maximum of 19°. Angular
    velocity varies very little (median 0.14 and 0.33 deg/frame²).
  - Relative orientation reaches ~180° from the start in about 28 % of trajectories (the long routes).
    This is where a principal-log chart would struggle, but the 6D code is continuous there.
  - E1b's difficulty is a fine-scale distributional detail: excursions of 20–60° from look-at, median 39°.
- **Existing solver evidence** (same checkpoint, same conditions):
  - For `rfm_so3` E1b, going from Euler 20 to Euler 100 improves jerk (23.3 → 9.0 × GT), angular
    acceleration (9.1 → 3.7) and mode bias (×218 → ×38). Orientation W1 does **not** improve
    (4.75 → 5.11°), and out-of-support only moves from 17.6 % to 14.1 %.
  - At NFE 50, midpoint vs Euler gives W1 5.26 vs 4.82° and out-of-support 13.7 vs 14.6 %.
  - Under the same sweep `fm_rel_pose` reaches W1 1.19–1.33°.
  - Even at its best setting, `rfm_so3`'s mode bias (×16–20) stays well above `fm_rel_pose`'s (×0.8–4).

## 5. Evidence by hypothesis

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H1** independent prior → rough paths | **Supported as a property of the RFM path, but not sufficient to explain the gap.** | The RFM path is far rougher than GT until late flow time (δ 18 vs 4.5 deg/frame at t = 0.9), and a shared prior removes this completely. But the Euclidean baseline also uses a temporally independent prior, and its path is also rough (δ 82 → 7 deg/frame), yet that model fits well. Independence alone does not separate the two models. |
| **H2** framewise geodesics ≠ smooth trajectories | **Supported, with a correction.** | With a smooth prior the framewise geodesics stay smooth (shared Haar: δ ≤ GT for all t). Roughness comes from the prior, not from the geodesic construction. One genuine geodesic issue remains: ~11 % of targets lie within 10° of the cut locus, where ξ flips direction under tiny perturbations. With a shared prior, 38 % of trajectories still contain such a flip between adjacent frames. The Euclidean target has no such discontinuity. |
| **H3** benchmark not geometrically challenging | **Supported.** | GT rotations are small-step and smooth. The one SO(3) hazard present (principal angles near 180° on long routes) is a chart problem that the 6D baseline does not have. No failure mode of the baseline was observed that SO(3) transport would fix: the baseline's rotations are valid by construction, its orientation W1 is 1.2–1.5°, and its residual errors shrink with solver accuracy. |
| **H4** numerical integration | **Supported for smoothness and mode bias; rejected for orientation fidelity.** | Jerk, angular acceleration and mode bias improve strongly with NFE or midpoint. Orientation W1 and out-of-support do not, so the orientation-distribution error lies in the learned field or the transport problem, not the integrator. |

**Additional mechanism (observed, not proven causal): transport distance.** Under the Haar prior each
frame must travel a median 132°. For the decoded Euclidean path it is 64–76°, because standardisation
places the Gaussian prior near the data region. As a result the RFM states are still 13° from their
targets at t = 0.9, versus 4.5° for the Euclidean path. The 20–60° E1b excursions must therefore be
resolved by the vector field in a short, late segment of a long, rough transport. This is consistent with
the observed blurring of excursion amplitude: both too-small and too-large peaks.

## 6. Confounds and what remains unproven

- One training seed per model. The two methods differ in prior, state parameterisation, loss scale and
  integrator. The Euclidean comparison measures decoded states of a different path and does not isolate
  SO(3) geometry.
- No intervention was trained. We have not shown that a shared, concentrated or data-scaled SO(3) prior
  would train better. The shared-prior result is a path diagnostic only.
- The relative contributions of cut-locus discontinuity, transport distance and temporal independence to
  the learned field's error are not separated.
- Unexplained: `rfm_so3`'s translational jerk in E1b is 12× GT, versus 4.7× for `fm_rel_pose`, even
  though positions use the same Euclidean FM. A plausible but untested cause is coupling through the
  shared network, which sees noisy rotation states for most of the flow.

## 7. Conclusion

**Recommendation: D (mixed), with A and C as the leading components and B partial.**

- **A, refined.** The likely path mismatch is not temporal independence per se. It is a long transport
  from an uninformative prior (median 132° per frame), with discontinuous velocity targets near the cut
  locus (~11 % of training targets). This leaves states far from the data until late flow time.
- **C.** The benchmark's rotations are small-step and smooth, and the 6D baseline already handles its
  only chart hazard, so SO(3) transport had no failure mode to fix.
- **B, partial.** Integration error explains part of the smoothness and mode-bias gap, but not the
  orientation-distribution gap.

**Short version for the paper.** *With a Haar-uniform prior, each camera-frame rotation in `rfm_so3`
must be transported a median 132°, versus 64–76° for the decoded states of the standardised Euclidean
6D path. About 11 % of training targets lie within 10° of the cut locus, where the geodesic velocity
target is discontinuous. Intermediate states remain far from the data and temporally rough until late
flow time, so the fine 20–60° orientation excursions of E1b must be resolved in a short final segment.
Because the benchmark's camera rotations are small-step and smooth, and the 6D baseline already produces
valid, continuous rotations, the manifold formulation offered no structural advantage to offset these
costs. Higher integration accuracy reduced `rfm_so3`'s jerk and mode bias but not its orientation-
distribution error. These are path-level observations from a single training seed, not a demonstrated
causal account.*
