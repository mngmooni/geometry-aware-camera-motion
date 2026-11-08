# E1b: rfm_so3 vs rel_pose

Evaluators: a = m2, b = m2; K = 200, euler NFE 50, classifier e1-v1.1. **One training seed per model.** Paired 95% CIs are bootstrap over held-out conditions. `matched = no` rows must not be used for claims.

| panel | metric | rel_pose | rfm_so3 | b − a [95% CI] | oracle | matched | note |
|---|---|---|---|---|---|---|---|
| eval_id | per-condition tv5 (mean over 50) | 0.0524 | 0.0738 | +0.0214 [+0.0122, +0.0305] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_id | per-condition abs_err_p_long (mean over 50) | 0.0234 | 0.0389 | +0.0155 [+0.0076, +0.0235] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_id | per-condition abs_err_p_far (mean over 50) | 0.0355 | 0.0572 | +0.0217 [+0.0109, +0.0324] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_id | per-condition unassigned (mean over 50) | 0.0021 | 0.0066 | +0.0045 [+0.002798, +0.0064] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_id | per-condition e_p_median (mean over 50) | 0.01388 | 0.01984 | +0.005961 [+0.005431, +0.006529] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_id | per-condition e_R_deg_median (mean over 50) | 0.1419 | 0.4965 | +0.3546 [+0.3377, +0.3721] | — | yes | paired by condition; identical eval seed -> identical base-noise seeds |
| eval_id | dispersion bias chi2 (excess ratio) | 9.985 | 63.4 | — | 1 | yes | p_a=0.00025, p_b=0.00025; recomputed from saved counts |
| eval_id | dispersion between chi2 (excess ratio) | 0.9986 | 1.243 | — | 1 | yes | p_a=0.489, p_b=0.0262; recomputed from saved counts |
| eval_id | TV5 mean | 0.0524 | 0.0738 | — | 0.0445 | yes |  |
| eval_id | endpoint e_p_mean | 0.0157 | 0.04884 | — | 0 | yes |  |
| eval_id | endpoint e_p_p90 | 0.02852 | 0.1155 | — | 0 | yes |  |
| eval_id | endpoint e_R_deg_mean | 0.1615 | 1.221 | — | 0 | yes |  |
| eval_id | endpoint e_R_deg_p90 | 0.2756 | 2.998 | — | 0 | yes |  |
| eval_id | smoothness trans_acc / GT | 1.043 | 2.342 | — | 0.9963 | no | GT reference set differs (m1: 20 refs/cond, m2: 200) |
| eval_id | smoothness trans_jerk / GT | 4.737 | 12.02 | — | 1.003 | no | GT reference set differs (m1: 20 refs/cond, m2: 200) |
| eval_id | smoothness ang_acc / GT | 1.03 | 4.908 | — | 1.009 | no | GT reference set differs (m1: 20 refs/cond, m2: 200) |
| eval_id | E1b W1 pooled max_excursion_deg | 1.549 | 4.819 | — | 0.2527 | yes |  |
| eval_id | E1b W1 pooled peak_time | 0.02243 | 0.01384 | — | 0.001842 | yes |  |
| eval_id | E1b amplitude outside support | 0.0552 | 0.1463 | — | 0 | yes |  |
