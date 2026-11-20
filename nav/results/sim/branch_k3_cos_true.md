# Step 4b: simulator branch ranking (k3_cos_true)

7 scenes, 700 episodes: ARNzJeq3xxb (100), EU6Fwq7SyZv (100), JmbYfDe2QKZ (100), VLzqgDo317F (100), YmJkqBEsHnH (100), mJXqzFtmKg4 (100), yqstnuAEVhm (100)

Reference (geodesic to goal after the candidate, metres, mean over episodes):

| start | best | random | straight | Euclid heuristic (privileged) | frac. candidates colliding |
|---|---|---|---|---|---|
| 2.112 | 1.409 | 1.680 | 1.662 | 1.421 | 0.291 |

Models (mean over episodes per seed, then mean ± std over seeds):

| group | seeds | geo_pick | progress_vs_random | gap_closed | rho | pick_in_best10pct | beats_random | beats_straight | pick_collided | rho_true_latent | geo_pick_true_latent | rho_pred_vs_true_latent | pred_true_cos_median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s4_xysc_fs8_h3 | 3 | 1.693 ± 0.003 | -0.014 ± 0.003 | -0.123 ± 0.009 | 0.010 ± 0.009 | 0.144 ± 0.013 | 0.513 ± 0.004 | 0.362 ± 0.011 | 0.298 ± 0.008 | 0.314 ± 0.002 | 1.586 ± 0.006 | 0.039 ± 0.003 | 0.298 ± 0.007 |
| s5inv_xysc_fs8_h3_w1 | 3 | 1.655 ± 0.006 | 0.024 ± 0.006 | 0.072 ± 0.021 | 0.167 ± 0.009 | 0.238 ± 0.007 | 0.613 ± 0.009 | 0.535 ± 0.012 | 0.265 ± 0.016 | 0.294 ± 0.006 | 1.575 ± 0.005 | 0.424 ± 0.026 | 0.296 ± 0.003 |

- progress_vs_random: random-pick geodesic minus the model-pick geodesic (m; > 0 = better than random)
- gap_closed: (random - pick) / (random - best); 0 = random, 1 = oracle
- rho: Spearman between the model score and the final geodesic over the candidates of an episode
