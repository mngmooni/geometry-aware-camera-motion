# E1b: rel_inc vs rel_pose

Evaluators: a = m2, b = m2; K = 200, euler NFE 50, classifier e1-v1.1. **One training seed per model.** Paired 95% CIs are bootstrap over held-out conditions. `matched = no` rows must not be used for claims.

| panel | metric | rel_pose | rel_inc | b − a [95% CI] | oracle | matched | note |
|---|---|---|---|---|---|---|---|
| eval_id | per-condition tv5 (mean over 50) | 0.0524 | 0.0504 | -0.002 [-0.0108, +0.006902] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_id | per-condition abs_err_p_long (mean over 50) | 0.0234 | 0.0245 | +0.0011 [-0.0059, +0.008402] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_id | per-condition abs_err_p_far (mean over 50) | 0.0355 | 0.0327 | -0.0028 [-0.0126, +0.007] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_id | per-condition unassigned (mean over 50) | 0.0021 | 0.004 | +0.0019 [+0.0004, +0.0034] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_id | per-condition e_p_median (mean over 50) | 0.01388 | 0.07 | +0.05612 [+0.0539, +0.05847] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_id | per-condition e_R_deg_median (mean over 50) | 0.1419 | 0.8035 | +0.6617 [+0.647, +0.6767] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_id | dispersion bias chi2 (excess ratio) | 9.985 | 6.548 | — | 1 | yes | p_a=0.00025, p_b=0.0005; recomputed from saved counts |
| eval_id | dispersion between chi2 (excess ratio) | 0.9986 | 1.023 | — | 1 | yes | p_a=0.489, p_b=0.409; recomputed from saved counts |
| eval_id | TV5 mean | 0.0524 | 0.0504 | — | 0.0445 | yes |  |
| eval_id | endpoint e_p_mean | 0.0157 | 0.09111 | — | 0 | yes |  |
| eval_id | endpoint e_p_p90 | 0.02852 | 0.1775 | — | 0 | yes |  |
| eval_id | endpoint e_R_deg_mean | 0.1615 | 1.046 | — | 0 | yes |  |
| eval_id | endpoint e_R_deg_p90 | 0.2756 | 2.011 | — | 0 | yes |  |
| eval_id | smoothness trans_acc / GT | 1.043 | 0.9564 | — | 0.9963 | no | GT reference set differs (m1: 20 refs/cond, m2: 200) |
| eval_id | smoothness trans_jerk / GT | 4.737 | 0.9852 | — | 1.003 | no | GT reference set differs (m1: 20 refs/cond, m2: 200) |
| eval_id | smoothness ang_acc / GT | 1.03 | 0.9364 | — | 1.009 | no | GT reference set differs (m1: 20 refs/cond, m2: 200) |
| eval_id | E1b W1 pooled max_excursion_deg | 1.549 | 1.931 | — | 0.2527 | yes |  |
| eval_id | E1b W1 pooled peak_time | 0.02243 | 0.01335 | — | 0.001842 | yes |  |
| eval_id | E1b amplitude outside support | 0.0552 | 0.0527 | — | 0 | yes |  |
