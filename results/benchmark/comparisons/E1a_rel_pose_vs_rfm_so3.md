# E1a: rfm_so3 vs rel_pose

Evaluators: a = m2, b = m2; K = 200, euler NFE 50, classifier e1-v1.1. **One training seed per model.** Paired 95% CIs are bootstrap over held-out conditions. `matched = no` rows must not be used for claims.

| panel | metric | rel_pose | rfm_so3 | b − a [95% CI] | oracle | matched | note |
|---|---|---|---|---|---|---|---|
| eval_id | per-condition tv5 (mean over 50) | 0.0526 | 0.0743 | +0.0217 [+0.0136, +0.0298] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_id | per-condition abs_err_p_long (mean over 50) | 0.0252 | 0.0588 | +0.0336 [+0.0238, +0.0435] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_id | per-condition abs_err_p_far (mean over 50) | 0.0349 | 0.0367 | +0.0018 [-0.006103, +0.0097] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_id | per-condition unassigned (mean over 50) | 0.0006 | 0.0019 | +0.0013 [+0.0004, +0.0022] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_id | per-condition e_p_median (mean over 50) | 0.006858 | 0.007805 | +0.0009476 [+0.0007771, +0.001098] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_id | per-condition e_R_deg_median (mean over 50) | 0.0603 | 0.1375 | +0.07724 [+0.07449, +0.08024] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_id | dispersion bias chi2 (excess ratio) | 7.442 | 71.16 | — | 1 | yes | p_a=0.00025, p_b=0.00025; recomputed from saved counts |
| eval_id | dispersion between chi2 (excess ratio) | 1.113 | 1.127 | — | 1 | yes | p_a=0.167, p_b=0.143; recomputed from saved counts |
| eval_id | TV5 mean | 0.0526 | 0.0743 | — | 0.0434 | yes |  |
| eval_id | endpoint e_p_mean | 0.007726 | 0.01103 | — | 0 | yes |  |
| eval_id | endpoint e_p_p90 | 0.01386 | 0.01654 | — | 0 | yes |  |
| eval_id | endpoint e_R_deg_mean | 0.06945 | 0.1864 | — | 0 | yes |  |
| eval_id | endpoint e_R_deg_p90 | 0.1255 | 0.2787 | — | 0 | yes |  |
| eval_id | smoothness trans_acc / GT | 0.9753 | 1.071 | — | 0.9703 | yes |  |
| eval_id | smoothness trans_jerk / GT | 3.249 | 3.673 | — | 0.9718 | yes |  |
| eval_id | smoothness ang_acc / GT | 1.076 | 1.988 | — | 1.001 | yes |  |
| eval_ood_position | per-condition tv5 (mean over 20) | 0.05525 | 0.07025 | +0.015 [-0.0015, +0.03025] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_ood_position | per-condition abs_err_p_long (mean over 20) | 0.02875 | 0.059 | +0.03025 [+0.01475, +0.04625] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_ood_position | per-condition abs_err_p_far (mean over 20) | 0.0275 | 0.0285 | +0.001 [-0.0145, +0.0165] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_ood_position | per-condition unassigned (mean over 20) | 0.001 | 0.001 | +0 [-0.00125, +0.00125] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_ood_position | per-condition e_p_median (mean over 20) | 0.006497 | 0.007678 | +0.001182 [+0.0007484, +0.001728] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_ood_position | per-condition e_R_deg_median (mean over 20) | 0.05755 | 0.1344 | +0.07682 [+0.0731, +0.08059] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_ood_position | dispersion bias chi2 (excess ratio) | 5.884 | 25.65 | — | 1 | yes | p_a=0.00075, p_b=0.00025; recomputed from saved counts |
| eval_ood_position | dispersion between chi2 (excess ratio) | 1.075 | 0.9696 | — | 1 | yes | p_a=0.318, p_b=0.545; recomputed from saved counts |
| eval_ood_position | TV5 mean | 0.05525 | 0.07025 | — | 0.0505 | yes |  |
| eval_ood_position | endpoint e_p_mean | 0.007427 | 0.01039 | — | 0 | yes |  |
| eval_ood_position | endpoint e_p_p90 | 0.0136 | 0.01631 | — | 0 | yes |  |
| eval_ood_position | endpoint e_R_deg_mean | 0.06642 | 0.1736 | — | 0 | yes |  |
| eval_ood_position | endpoint e_R_deg_p90 | 0.1214 | 0.2698 | — | 0 | yes |  |
| eval_ood_position | smoothness trans_acc / GT | 0.9664 | 1.062 | — | 0.9647 | yes |  |
| eval_ood_position | smoothness trans_jerk / GT | 3.207 | 3.692 | — | 1.002 | yes |  |
| eval_ood_position | smoothness ang_acc / GT | 1.111 | 2.073 | — | 1.034 | yes |  |
