| model | data | panel | TV5 (oracle) ↓ | modes found | unassigned % | endpoint cm | endpoint ° | jerk / GT | E1b W1 amp ° (oracle) | E1b amp out % | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rel_pose | E1a | held-out (ID) | 0.053 (0.043) | 4.0/4 | 0.06 | 0.67 | 0.06 | 3.25 | — | — | main |
| rel_pose | E1a | OOD position | 0.055 (0.051) | 4.0/4 | 0.10 | 0.63 | 0.06 | 3.21 | — | — | main |
| rel_pose | E1b | held-out (ID) | 0.052 (0.045) | 4.0/4 | 0.21 | 1.34 | 0.14 | 4.74 | 1.55 (0.25) | 5.5 | main |
| rel_inc | E1a | held-out (ID) | 0.050 (0.043) | 4.0/4 | 0.02 | 1.89 | 0.21 | 0.96 | — | — | main |
| rel_inc | E1a | OOD position | 0.048 (0.051) | 4.0/4 | 0.05 | 1.73 | 0.19 | 0.98 | — | — | main |
| rel_inc | E1b | held-out (ID) | 0.050 (0.045) | 4.0/4 | 0.40 | 6.96 | 0.80 | 0.99 | 1.93 (0.25) | 5.3 | main |
| rfm_so3 | E1a | held-out (ID) | 0.074 (0.043) | 4.0/4 | 0.19 | 0.77 | 0.14 | 3.67 | — | — | main |
| rfm_so3 | E1a | OOD position | 0.070 (0.051) | 4.0/4 | 0.10 | 0.74 | 0.13 | 3.69 | — | — | main |
| rfm_so3 | E1b | held-out (ID) | 0.074 (0.045) | 4.0/4 | 0.66 | 1.97 | 0.50 | 12.02 | 4.82 (0.25) | 14.6 | main |

TV5: total-variation distance between the generated 5-category mode frequencies (4 routes + unassigned) and the declared p(route|shot); the oracle column is a perfect sampler (fresh ground-truth draws) at the same K, i.e. the finite-sample floor. Endpoint: median error of the generated last pose (soft-conditioned, never projected). Jerk / GT: median translational jerk relative to ground truth. One training seed per model; differences are exploratory unless paired CIs are reported.
