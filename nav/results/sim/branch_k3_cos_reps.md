# Step 4b: simulator branch ranking (k3_cos_reps)

7 scenes, 700 episodes: ARNzJeq3xxb (100), EU6Fwq7SyZv (100), JmbYfDe2QKZ (100), VLzqgDo317F (100), YmJkqBEsHnH (100), mJXqzFtmKg4 (100), yqstnuAEVhm (100)

Reference (geodesic to goal after the candidate, metres, mean over episodes):

| start | best | random | straight | Euclid heuristic (privileged) | frac. candidates colliding |
|---|---|---|---|---|---|
| 2.112 | 1.409 | 1.680 | 1.662 | 1.421 | 0.291 |

Models (mean over episodes per seed, then mean ± std over seeds):

| group | seeds | geo_pick | progress_vs_random | gap_closed | rho | pick_in_best10pct | beats_random | beats_straight | pick_collided |
|---|---|---|---|---|---|---|---|---|---|
| s4_disc_fs8_h3 | 3 | 1.685 ± 0.004 | -0.005 ± 0.004 | -0.129 ± 0.013 | 0.008 ± 0.011 | 0.175 ± 0.012 | 0.546 ± 0.003 | 0.343 ± 0.021 | 0.299 ± 0.003 |
| s4_xy_fs8_h3 | 3 | 1.691 ± 0.004 | -0.011 ± 0.004 | -0.116 ± 0.003 | 0.019 ± 0.007 | 0.160 ± 0.009 | 0.532 ± 0.005 | 0.390 ± 0.006 | 0.296 ± 0.005 |
| s4_xyt_fs8_h3 | 3 | 1.693 ± 0.003 | -0.013 ± 0.003 | -0.128 ± 0.025 | 0.011 ± 0.003 | 0.174 ± 0.003 | 0.523 ± 0.008 | 0.393 ± 0.014 | 0.298 ± 0.013 |
| s5inv_dbin_fs8_h3_w1 | 3 | 1.668 ± 0.000 | 0.011 ± 0.000 | 0.028 ± 0.013 | 0.108 ± 0.009 | 0.196 ± 0.016 | 0.573 ± 0.007 | 0.501 ± 0.016 | 0.286 ± 0.001 |
| s5inv_disc_fs8_h3_w1 | 3 | 1.682 ± 0.004 | -0.002 ± 0.004 | -0.051 ± 0.014 | 0.031 ± 0.014 | 0.168 ± 0.012 | 0.564 ± 0.010 | 0.345 ± 0.024 | 0.290 ± 0.001 |
| s5inv_xy_fs8_h3_w1 | 3 | 1.656 ± 0.008 | 0.024 ± 0.008 | 0.061 ± 0.031 | 0.154 ± 0.003 | 0.216 ± 0.006 | 0.600 ± 0.007 | 0.518 ± 0.008 | 0.269 ± 0.007 |
| s5inv_xyt_fs8_h3_w0.1 | 1 | 1.688 ± 0.000 | -0.008 ± 0.000 | -0.092 ± 0.000 | 0.048 ± 0.000 | 0.163 ± 0.000 | 0.534 ± 0.000 | 0.421 ± 0.000 | 0.304 ± 0.000 |
| s5inv_xyt_fs8_h3_w0.3 | 1 | 1.667 ± 0.000 | 0.012 ± 0.000 | 0.006 ± 0.000 | 0.121 ± 0.000 | 0.199 ± 0.000 | 0.561 ± 0.000 | 0.493 ± 0.000 | 0.280 ± 0.000 |
| s5inv_xyt_fs8_h3_w1 | 3 | 1.655 ± 0.004 | 0.025 ± 0.004 | 0.056 ± 0.031 | 0.171 ± 0.011 | 0.234 ± 0.001 | 0.608 ± 0.017 | 0.532 ± 0.017 | 0.265 ± 0.010 |
| s5inv_xyt_fs8_h3_w3 | 1 | 1.636 ± 0.000 | 0.044 ± 0.000 | 0.141 ± 0.000 | 0.200 ± 0.000 | 0.256 ± 0.000 | 0.656 ± 0.000 | 0.577 ± 0.000 | 0.271 ± 0.000 |

- progress_vs_random: random-pick geodesic minus the model-pick geodesic (m; > 0 = better than random)
- gap_closed: (random - pick) / (random - best); 0 = random, 1 = oracle
- rho: Spearman between the model score and the final geodesic over the candidates of an episode
