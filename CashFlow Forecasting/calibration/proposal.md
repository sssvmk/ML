# Layer 2 calibration: pilot results and PROPOSED parameters (for approval)

Source: `calibration/pilot_results.jsonl` -- 24 records. **Nothing here is approved**; `config.json -> admission.calibration` stays null until you fill it in.

Proposed conventions (need your approval): significance level alpha = 0.05; target power = 0.8; margin rule = 1.0 x the reference's own seed-to-seed SD of its gap to the oracle.


## deepstate/local_level_weekly  (8 seeds, score = crps, oracle mean 1.485)

Mean seconds per seed: custom 10s, reference 187s

- Reference gap to oracle: mean 0.173, seed-to-seed SD 0.189  ->  margin (1 SD) = 0.189
- Custom gap to oracle: mean 0.198, SD 0.153; paired difference (custom - reference): mean 0.025, SD 0.173
- Non-inferiority on this pilot: p = 0.015 -> non-inferior (upper bound of the difference 0.140 vs margin 0.189)
- Seeds needed for 80% power if the two are truly equal: 7 (conservative, using the upper 80% bound of the SD: 12)
- Alternative margins: 0.5xSD: margin 0.094, p 0.145, seeds 23; 1.0xSD: margin 0.189, p 0.015, seeds 7; 2.0xSD: margin 0.378, p 0.000, seeds 4
- 80% interval coverage, reference: 0.699 over 336 points; Kupiec p (as if independent) 0.000; design effect 1.9 -> effective points 180, p 0.001
- 80% interval coverage, custom: 0.765 over 336 points; Kupiec p (as if independent) 0.115; design effect 1.3 -> effective points 268, p 0.160
- Parameter recovery (one-step predictive std / truth, log): reference -0.052 (SD 0.077), custom +0.015 (SD 0.092); custom inside the reference's 95% interval: True

## deepvar/var1_corr0.0  (8 seeds, score = energy_score, oracle mean 52.170)

Mean seconds per seed: custom 2s, negative_control_diagonal 2s, reference 31s

- Reference gap to oracle: mean 4.248, seed-to-seed SD 7.125  ->  margin (1 SD) = 7.125
- Custom gap to oracle: mean 6.591, SD 7.394; paired difference (custom - reference): mean 2.342, SD 7.854
- Non-inferiority on this pilot: p = 0.064 -> NOT shown non-inferior (upper bound of the difference 7.603 vs margin 7.125)
- Seeds needed for 80% power if the two are truly equal: 10 (conservative, using the upper 80% bound of the SD: 16)
- Alternative margins: 0.5xSD: margin 3.563, p 0.337, seeds 32; 1.0xSD: margin 7.125, p 0.064, seeds 10; 2.0xSD: margin 14.250, p 0.002, seeds 4
- 80% interval coverage, reference: 0.726 over 168 points; Kupiec p (as if independent) 0.022; design effect 3.3 -> effective points 51, p 0.201
- 80% interval coverage, custom: 0.690 over 168 points; Kupiec p (as if independent) 0.001; design effect 2.8 -> effective points 60, p 0.033
- Cross-correlation recovery (true 0.0): reference +0.07, custom -0.10, diagonal control -0.01
- Negative control (full vs diagonal covariance): mean gain 4.73, p(full better) 0.226 -- expected: full covariance clearly better when corr > 0; no material difference when corr = 0

## deepvar/var1_corr0.6  (8 seeds, score = energy_score, oracle mean 35.660)

Mean seconds per seed: custom 5s, negative_control_diagonal 2s, reference 33s

- Reference gap to oracle: mean 7.308, seed-to-seed SD 11.184  ->  margin (1 SD) = 11.184
- Custom gap to oracle: mean 2.878, SD 5.705; paired difference (custom - reference): mean -4.430, SD 9.714
- Non-inferiority on this pilot: p = 0.001 -> non-inferior (upper bound of the difference 2.077 vs margin 11.184)
- Seeds needed for 80% power if the two are truly equal: 7 (conservative, using the upper 80% bound of the SD: 11)
- Alternative margins: 0.5xSD: margin 5.592, p 0.011, seeds 21; 1.0xSD: margin 11.184, p 0.001, seeds 7; 2.0xSD: margin 22.367, p 0.000, seeds 3
- 80% interval coverage, reference: 0.827 over 168 points; Kupiec p (as if independent) 0.366; design effect 1.5 -> effective points 110, p 0.467
- 80% interval coverage, custom: 0.774 over 168 points; Kupiec p (as if independent) 0.403; design effect 2.4 -> effective points 69, p 0.516
- Cross-correlation recovery (true 0.6): reference +0.59, custom +0.35, diagonal control -0.01
- Negative control (full vs diagonal covariance): mean gain 5.13, p(full better) 0.016 -- expected: full covariance clearly better when corr > 0; no material difference when corr = 0
