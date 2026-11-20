# Step 5: closed-loop LeWM-CEM navigation (7 val scenes: ARNzJeq3xxb, EU6Fwq7SyZv, JmbYfDe2QKZ, VLzqgDo317F, YmJkqBEsHnH, mJXqzFtmKg4, yqstnuAEVhm)

success: geodesic < 0.5 m reached at any time (oracle stop); spl: success weighted by path length; self_*: the model's own latent stop rule (tau calibrated on a train scene); false_stop: stopped > 0.5 m away; progress: start geodesic - min geodesic (m); collisions per episode; plan_ms per step.

## 1-2 m (140 episodes per policy)

| policy | n | success | spl | self_success | self_spl | false_stop | progress | final_geo | collisions | steps | plan_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s4_xysc_fs8_h3 | 3 | 0.417 ± 0.022 | 0.389 ± 0.020 | 0.102 ± 0.013 | 0.094 ± 0.016 | 0.286 ± 0.027 | 0.829 ± 0.027 | 2.104 ± 0.082 | 7.812 ± 0.291 | 17.538 ± 0.362 | 22.944 ± 1.483 |
| s5inv_xysc_fs8_h3_w1 | 3 | 0.579 ± 0.021 | 0.558 ± 0.018 | 0.248 ± 0.044 | 0.234 ± 0.041 | 0.298 ± 0.113 | 0.900 ± 0.006 | 1.608 ± 0.078 | 6.052 ± 0.421 | 13.898 ± 0.673 | 21.882 ± 0.006 |
| straight | 1 | 0.329 | 0.328 |  |  |  | 0.717 | 2.081 | 9.257 | 15.936 |  |
| euclid_greedy | 1 | 0.964 | 0.943 |  |  |  | 1.196 | 0.382 | 1.200 | 5.500 |  |
| geo_greedy | 1 | 0.986 | 0.962 |  |  |  | 1.209 | 0.372 | 0.621 | 5.093 |  |
| random | 1 | 0.279 | 0.277 |  |  |  | 0.717 | 2.166 | 9.707 | 16.871 |  |

## 2-4 m (140 episodes per policy)

| policy | n | success | spl | self_success | self_spl | false_stop | progress | final_geo | collisions | steps | plan_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s4_xysc_fs8_h3 | 3 | 0.140 ± 0.024 | 0.125 ± 0.019 | 0.021 ± 0.012 | 0.020 ± 0.011 | 0.236 ± 0.015 | 1.279 ± 0.045 | 3.402 ± 0.045 | 15.729 ± 0.705 | 29.679 ± 0.212 | 21.900 ± 0.008 |
| s5inv_xysc_fs8_h3_w1 | 3 | 0.262 ± 0.012 | 0.248 ± 0.009 | 0.102 ± 0.003 | 0.093 ± 0.005 | 0.264 ± 0.087 | 1.458 ± 0.037 | 2.848 ± 0.064 | 14.740 ± 0.646 | 26.919 ± 0.391 | 21.898 ± 0.007 |
| straight | 1 | 0.129 | 0.127 |  |  |  | 1.155 | 3.392 | 19.229 | 28.607 |  |
| euclid_greedy | 1 | 0.779 | 0.775 |  |  |  | 2.235 | 0.883 | 7.421 | 14.800 |  |
| geo_greedy | 1 | 0.957 | 0.925 |  |  |  | 2.599 | 0.449 | 1.964 | 11.407 |  |
| random | 1 | 0.150 | 0.146 |  |  |  | 1.243 | 3.431 | 18.207 | 28.586 |  |

## 4-8 m (140 episodes per policy)

| policy | n | success | spl | self_success | self_spl | false_stop | progress | final_geo | collisions | steps | plan_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s4_xysc_fs8_h3 | 3 | 0.074 ± 0.009 | 0.065 ± 0.007 | 0.019 ± 0.007 | 0.016 ± 0.006 | 0.124 ± 0.033 | 2.006 ± 0.125 | 6.045 ± 0.181 | 31.014 ± 0.862 | 50.390 ± 0.094 | 21.962 ± 0.010 |
| s5inv_xysc_fs8_h3_w1 | 3 | 0.121 ± 0.021 | 0.115 ± 0.019 | 0.036 ± 0.006 | 0.033 ± 0.006 | 0.181 ± 0.061 | 2.035 ± 0.100 | 5.763 ± 0.220 | 31.919 ± 0.876 | 48.624 ± 0.502 | 21.945 ± 0.001 |
| straight | 1 | 0.029 | 0.028 |  |  |  | 1.804 | 5.663 | 39.493 | 50.750 |  |
| euclid_greedy | 1 | 0.564 | 0.560 |  |  |  | 3.795 | 2.444 | 23.436 | 35.414 |  |
| geo_greedy | 1 | 0.950 | 0.940 |  |  |  | 5.379 | 0.658 | 3.329 | 21.914 |  |
| random | 1 | 0.071 | 0.065 |  |  |  | 2.200 | 5.804 | 35.671 | 50.129 |  |
