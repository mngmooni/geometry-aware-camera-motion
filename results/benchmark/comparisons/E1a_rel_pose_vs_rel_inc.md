# E1a: rel_inc vs rel_pose

Evaluators: a = m2, b = m2; K = 200, euler NFE 50, classifier e1-v1.1. **One training seed per model.** Paired 95% CIs are bootstrap over held-out conditions. `matched = no` rows must not be used for claims.

| panel | metric | rel_pose | rel_inc | b − a [95% CI] | oracle | matched | note |
|---|---|---|---|---|---|---|---|
| eval_id | per-condition tv5 (mean over 50) | 0.0526 | 0.0504 | -0.0022 [-0.0095, +0.0054] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_id | per-condition abs_err_p_long (mean over 50) | 0.0252 | 0.0307 | +0.0055 [-0.0008, +0.0119] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_id | per-condition abs_err_p_far (mean over 50) | 0.0349 | 0.026 | -0.0089 [-0.017, -0.0013] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_id | per-condition unassigned (mean over 50) | 0.0006 | 0.0002 | -0.0004 [-0.001, +0.0002] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_id | per-condition e_p_median (mean over 50) | 0.006858 | 0.01913 | +0.01227 [+0.01168, +0.01288] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_id | per-condition e_R_deg_median (mean over 50) | 0.0603 | 0.2074 | +0.1471 [+0.14, +0.1545] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_id | dispersion bias chi2 (excess ratio) | 7.442 | 2.526 | — | 1 | yes | p_a=0.00025, p_b=0.06; recomputed from saved counts |
| eval_id | dispersion between chi2 (excess ratio) | 1.113 | 1.103 | — | 1 | yes | p_a=0.167, p_b=0.188; recomputed from saved counts |
| eval_id | TV5 mean | 0.0526 | 0.0504 | — | 0.0434 | yes |  |
| eval_id | endpoint e_p_mean | 0.007726 | 0.0257 | — | 0 | yes |  |
| eval_id | endpoint e_p_p90 | 0.01386 | 0.05098 | — | 0 | yes |  |
| eval_id | endpoint e_R_deg_mean | 0.06945 | 0.3129 | — | 0 | yes |  |
| eval_id | endpoint e_R_deg_p90 | 0.1255 | 0.6736 | — | 0 | yes |  |
| eval_id | smoothness trans_acc / GT | 0.9753 | 0.9501 | — | 0.9703 | yes |  |
| eval_id | smoothness trans_jerk / GT | 3.249 | 0.9592 | — | 0.9718 | yes |  |
| eval_id | smoothness ang_acc / GT | 1.076 | 0.9559 | — | 1.001 | yes |  |
| eval_ood_position | per-condition tv5 (mean over 20) | 0.05525 | 0.048 | -0.00725 [-0.0185, +0.00425] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_ood_position | per-condition abs_err_p_long (mean over 20) | 0.02875 | 0.02025 | -0.0085 [-0.01675, -0.00075] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_ood_position | per-condition abs_err_p_far (mean over 20) | 0.0275 | 0.02375 | -0.00375 [-0.01851, +0.0105] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_ood_position | per-condition unassigned (mean over 20) | 0.001 | 0.0005 | -0.0005 [-0.00125, +0] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_ood_position | per-condition e_p_median (mean over 20) | 0.006497 | 0.01765 | +0.01115 [+0.01018, +0.0122] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_ood_position | per-condition e_R_deg_median (mean over 20) | 0.05755 | 0.1962 | +0.1386 [+0.1282, +0.1496] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_ood_position | dispersion bias chi2 (excess ratio) | 5.884 | 2.218 | — | 1 | yes | p_a=0.00075, p_b=0.0887; recomputed from saved counts |
| eval_ood_position | dispersion between chi2 (excess ratio) | 1.075 | 1.043 | — | 1 | yes | p_a=0.318, p_b=0.383; recomputed from saved counts |
| eval_ood_position | TV5 mean | 0.05525 | 0.048 | — | 0.0505 | yes |  |
| eval_ood_position | endpoint e_p_mean | 0.007427 | 0.02412 | — | 0 | yes |  |
| eval_ood_position | endpoint e_p_p90 | 0.0136 | 0.04792 | — | 0 | yes |  |
| eval_ood_position | endpoint e_R_deg_mean | 0.06642 | 0.2967 | — | 0 | yes |  |
| eval_ood_position | endpoint e_R_deg_p90 | 0.1214 | 0.6566 | — | 0 | yes |  |
| eval_ood_position | smoothness trans_acc / GT | 0.9664 | 0.9561 | — | 0.9647 | yes |  |
| eval_ood_position | smoothness trans_jerk / GT | 3.207 | 0.9787 | — | 1.002 | yes |  |
| eval_ood_position | smoothness ang_acc / GT | 1.111 | 0.9795 | — | 1.034 | yes |  |
