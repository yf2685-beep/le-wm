# Step 4b: simulator branch ranking (k2_cos)

7 scenes, 700 episodes: ARNzJeq3xxb (100), EU6Fwq7SyZv (100), JmbYfDe2QKZ (100), VLzqgDo317F (100), YmJkqBEsHnH (100), mJXqzFtmKg4 (100), yqstnuAEVhm (100)

Reference (geodesic to goal after the candidate, metres, mean over episodes):

| start | best | random | straight | Euclid heuristic (privileged) | frac. candidates colliding |
|---|---|---|---|---|---|
| 2.112 | 1.663 | 1.788 | 1.779 | 1.667 | 0.200 |

Models (mean over episodes per seed, then mean ± std over seeds):

| group | seeds | geo_pick | progress_vs_random | gap_closed | rho | pick_in_best10pct | beats_random | beats_straight | pick_collided |
|---|---|---|---|---|---|---|---|---|---|
| s4_xysc_fs8_h3 | 3 | 1.794 ± 0.001 | -0.006 ± 0.001 | -0.104 ± 0.008 | 0.005 ± 0.004 | 0.156 ± 0.015 | 0.543 ± 0.006 | 0.358 ± 0.013 | 0.206 ± 0.009 |
| s5inv_xysc_fs8_h3_w1 | 3 | 1.776 ± 0.004 | 0.012 ± 0.004 | 0.076 ± 0.028 | 0.169 ± 0.008 | 0.272 ± 0.014 | 0.610 ± 0.011 | 0.540 ± 0.013 | 0.187 ± 0.009 |

- progress_vs_random: random-pick geodesic minus the model-pick geodesic (m; > 0 = better than random)
- gap_closed: (random - pick) / (random - best); 0 = random, 1 = oracle
- rho: Spearman between the model score and the final geodesic over the candidates of an episode
