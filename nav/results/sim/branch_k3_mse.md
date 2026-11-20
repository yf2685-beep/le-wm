# Step 4b: simulator branch ranking (k3_mse)

7 scenes, 700 episodes: ARNzJeq3xxb (100), EU6Fwq7SyZv (100), JmbYfDe2QKZ (100), VLzqgDo317F (100), YmJkqBEsHnH (100), mJXqzFtmKg4 (100), yqstnuAEVhm (100)

Reference (geodesic to goal after the candidate, metres, mean over episodes):

| start | best | random | straight | Euclid heuristic (privileged) | frac. candidates colliding |
|---|---|---|---|---|---|
| 2.112 | 1.409 | 1.680 | 1.662 | 1.421 | 0.291 |

Models (mean over episodes per seed, then mean ± std over seeds):

| group | seeds | geo_pick | progress_vs_random | gap_closed | rho | pick_in_best10pct | beats_random | beats_straight | pick_collided |
|---|---|---|---|---|---|---|---|---|---|
| s4_xysc_fs8_h3 | 3 | 1.693 ± 0.003 | -0.014 ± 0.003 | -0.175 ± 0.005 | 0.007 ± 0.013 | 0.180 ± 0.001 | 0.509 ± 0.005 | 0.417 ± 0.010 | 0.298 ± 0.004 |
| s5inv_xysc_fs8_h3_w1 | 3 | 1.661 ± 0.009 | 0.019 ± 0.009 | 0.022 ± 0.030 | 0.171 ± 0.012 | 0.240 ± 0.007 | 0.597 ± 0.019 | 0.529 ± 0.019 | 0.266 ± 0.013 |

- progress_vs_random: random-pick geodesic minus the model-pick geodesic (m; > 0 = better than random)
- gap_closed: (random - pick) / (random - best); 0 = random, 1 = oracle
- rho: Spearman between the model score and the final geodesic over the candidates of an episode
