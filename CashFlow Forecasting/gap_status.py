"""
Status of every gap in the traceability matrix (G-01 .. G-42), with the tests that are the evidence. Hand-maintained on purpose: a status
is a judgement about what was and was not verified. tests/test_docs_status.py fails if an id is missing, a cited test file does not exist,
or GAP_STATUS.md / TRACEABILITY.md are stale, so this cannot silently rot.

Status vocabulary
  closed                     -- implemented and its acceptance checks pass
  closed_with_deviation      -- implemented; behaviour differs from the matrix text in a way stated in `note`
  built_not_admitted         -- custom model built and tested but NOT allowed to compete: G-42 Layer 2 has no approved calibration
  built_unavailable          -- adapter built and tested against a tiny stand-in checkpoint; real weights not supplied / never exercised
  partial                    -- part of the gap is closed; `note` says what is not
  open                       -- not done; `note` says what is needed
"""

T = "tests/"
GAPS = {
    "G-01": ("Model lifecycle: registry stages + production-monitoring rollback", "closed_with_deviation", [T + "test_lifecycle.py"],
             "Rollback is PROPOSED by comparing the live six-metric rank-sum of Production vs the prior version and DECIDED by a named human reviewer (your decision); the matrix said automatic. Monitoring and the in-sample train-metric check work per segment, so they do not run for a joint model (DeepVAR)."),
    "G-02": ("Per-trial MLflow tracking in HPO", "closed", [T + "test_g02_trial_tracking.py"], ""),
    "G-03": ("Cross-segment pooling pipeline", "closed", [T + "test_g03_pooling.py"], "One pool across all entities, currencies and directions, per-series scaling, no FX (your decisions). The full-pipeline test takes about 20 minutes on one core."),
    "G-04": ("Joint window + hyperparameter optimisation", "closed", [T + "test_gaps_nonneural.py"], ""),
    "G-05": ("Fold-level trial pruning", "closed", [T + "test_gaps_nonneural.py"], ""),
    "G-06": ("Seasonal / exogenous order terms searched", "closed", [T + "test_gaps_nonneural.py"], ""),
    "G-07": ("Required evaluation charts", "closed", [T + "test_gaps_nonneural.py"], ""),
    "G-08": ("Post-fit diagnostics", "closed", [T + "test_gaps_nonneural.py"], "Not available for iTransformer (the library gives no in-sample prediction for multivariate models), WaveNet, TimesFM-3 (zero-shot) or the joint DeepVAR."),
    "G-09": ("Early stopping for gradient-boosted trees", "closed", [T + "test_gaps_nonneural.py"], ""),
    "G-10": ("Train-vs-validation gap tracking", "closed", [T + "test_gaps_nonneural.py"], ""),
    "G-11": ("SVR compute_ceiling", "closed", [T + "test_gaps_nonneural.py"], "compute_ceiling=10000, ceiling_action=limit_window (your decision): trains SVR on the most recent 10,000 observations above the ceiling rather than refusing."),
    "G-12": ("Prophet is a stand-in", "closed", [T + "test_gaps_nonneural.py"], ""),
    "G-13": ("Synthetic adapter dataset labels", "closed", [T + "test_gaps_nonneural.py"], ""),
    "G-14": ("Provenance as a nested lineage object", "closed", [T + "test_gaps_nonneural.py"], "Flat columns are refused; upgrade_flat_lineage() migrates stored data."),
    "G-15": ("SVR / KNN eligibility flag vs PRD inconsistency", "closed", [T + "test_gaps_nonneural.py"], "The PRD row-level table governs (your decision): no necessary eligibility condition; required-observation floors and the implementation stay."),
    "G-20": ("RNN", "closed_with_deviation", [T + "test_neural_nf.py"], "NeuralForecast route: real nn.RNN core, but an MLP decoder emits all h outputs (the spec text says last hidden state -> linear head)."),
    "G-21": ("LSTM", "closed_with_deviation", [T + "test_neural_nf.py"], "Same decoder note as G-20."),
    "G-22": ("GRU", "closed_with_deviation", [T + "test_neural_nf.py"], "Same decoder note as G-20."),
    "G-23": ("TCN", "closed", [T + "test_tcn_g23.py"], "Custom PyTorch exactly per the G-23 spec (your decision): residual blocks, two weight-normed dilated causal convolutions each."),
    "G-24": ("WaveNet", "closed_with_deviation", [T + "test_wavenet_g24.py"], "GluonTS PyTorch route: the output head is categorical over quantised bins (not the continuous regression head in the spec) and there are no historical exogenous inputs."),
    "G-25": ("DeepAR", "closed", [T + "test_deepar_g25.py"], "Quantiles cover the first h steps; beyond h the point forecast rolls forward."),
    "G-26": ("DeepState", "built_not_admitted", [T + "test_deepstate_g26_g42.py"], "Native model with a differentiable Kalman filter; F and a are structural as in the paper (the matrix lists them as network outputs). Layer 1 and Layer 3 pass; Layer 2 needs an approved calibration."),
    "G-27": ("DeepVAR", "built_not_admitted", [T + "test_deepvar_g27.py", "calibration/pilot_results.jsonl"], "Native joint model served through infer_pool (your decision). Layer 1 and Layer 3 pass; Layer 2 blocked (see G-42). "
             "FLAGGED ACCURACY ISSUE (calibration pilot, 12 seeds each): accuracy depends on whether the pooled series are actually related. When truly correlated (corr=0.6 synthetic setting), DeepVAR beat the GluonTS reference (10% worse than the true-parameter oracle vs the reference's 23%, winning 8/12 seeds). When the series are independent (corr=0 setting), DeepVAR was the WORST of the three (17% worse than oracle vs the reference's 9%, winning only 4/12 seeds) -- consistent with a joint model finding spurious cross-series structure when none exists ('pairs-trading on unrelated stocks'). Mitigated structurally today: a segment only gets DeepVAR as its Production model if it wins that segment's backtest ranking against every single-series alternative, so this cannot force a worse forecast onto a segment where independent forecasting wins. Not mitigated: the model still pays an accuracy cost internally on unrelated pairs inside a shared pool. Options identified, none implemented: shrink the learned low-rank covariance factor toward zero during training (Ledoit-Wolf-style), curate pools to only series with a plausible real relationship instead of one pool for every entity x currency x direction (would partly revisit the G-03 pooling decision), or post-hoc threshold unstable correlations. Detection: DeepVARModule.cross_series_correlation() exposes the model's believed correlation per pair for inspection; the per-segment backtest already scores DeepVAR against single-series candidates in comparable units; calibration.sim.sim_var1(corr=0.0) is a repeatable synthetic regression test for this specific failure mode."),
    "G-28": ("TFT", "closed", [T + "test_future_known_g28_g29.py"], "Future-known covariates inferred from structure (your decision); eligibility derived from the data. Point-in-time correctness of such series is the Data Module's responsibility."),
    "G-29": ("TiDE", "closed_with_deviation", [T + "test_future_known_g28_g29.py"], "Works around an upstream defect: NeuralForecast's TiDE applies LayerNorm to a one-value output, which disconnects the whole encoder/decoder branch. Only that final LayerNorm is switched off, in the adapter; the vendored library is untouched."),
    "G-30": ("PatchTSMixer", "dropped", [T + "test_gaps_nonneural.py"], "Removed entirely by your decision, not merely left unconfigured: algorithms/tsmixer.py and its test are deleted, the catalog has no #30 entry, and prebuilt_models.patchtsmixer no longer exists in config.json. Roster is now 37 (was 38); part of G-40."),
    "G-31": ("TimesNet", "closed", [T + "test_neural_nf_family.py"], ""),
    "G-32": ("iTransformer", "closed", [T + "test_itransformer_g32.py"], "Exogenous series enter as variate tokens; no residual diagnostics."),
    "G-33": ("Informer", "closed", [T + "test_neural_nf_family.py"], ""),
    "G-34": ("Autoformer", "closed_with_deviation", [T + "test_neural_nf_family.py"], "Architecture (progressive decomposition, FFT-based Auto-Correlation) verified and real. Accepted as final (your decision): the acceptance item 'trend and seasonal branches separately retrievable' is not implemented -- the library does not expose them, and no fix is planned."),
    "G-35": ("FEDformer", "closed", [T + "test_neural_nf_family.py"], ""),
    "G-36": ("ETSformer", "closed", [T + "test_etsformer_g36.py"], "Official Salesforce code vendored unmodified at a pinned commit, BSD-3 notice retained, OSS scan in vendor/etsformer/OSS_SCAN.md. The archived status of the upstream repo was not re-verified (API rate-limited)."),
    "G-37": ("N-BEATS", "closed", [T + "test_neural_nf_family.py"], "No exogenous inputs (library limitation)."),
    "G-38": ("N-HiTS", "closed", [T + "test_neural_nf_family.py"], ""),
    "G-39": ("TimesFM-3 (licence-gated)", "closed", [T + "test_timesfm3_g39.py"], "Closure bar set by your decision: code initiates the real timesfm3 model, takes input data, and produces inference measurable against the six metrics -- demonstrated end to end (tested with a tiny stand-in checkpoint; the real IBM/Google weights were never loaded, and loading them is out of scope for this closure). Zero-shot; eligibility is context length + horizon. Weights only from config prebuilt_models.timesfm3.path, non-commercial guard enforced, RESTRICTED TO DEMO/SKILLS USE ONLY (allowed_environments=['demo']). Registered as an extension outside the PRD manifest."),
    "G-40": ("PRD and roster count change", "closed_with_deviation", [T + "test_timesfm3_g39.py", T + "test_gaps_nonneural.py"], "Confirmed by you: the roster is 37 mandatory algorithms (PatchTSMixer #30 dropped) plus TimesFM-3 as an optional, demo-only #39 outside the manifest -- not 38. Code and config already match this (catalog.py asserts len==37, config.json has no patchtsmixer entry, TimesFM-3 lives in a separate EXTENSION_CATALOG). Deviation from the matrix's own remedy: it called for the PRD OWNER to edit the PRD document's five '38' references (Sections 1, 3.2, the eligibility summary, 5, Conclusion) before code changed; no PRD document exists in this working environment (only the traceability matrix and the code were provided), so your confirmation here stands in place of that document edit. If the actual PRD file becomes available, those five references still need updating to match."),
    "G-41": ("Adapter layer and framework integration", "closed", [T + "test_neural_nf.py", T + "test_neural_nf_family.py"], "Vendored NeuralForecast 3.2.2 with a documented 3-file patch (strategy S2); imports without ray or optuna."),
    "G-42": ("Admission validation for custom DeepState / DeepVAR", "partial", [T + "test_deepstate_g26_g42.py", T + "test_deepvar_g27.py", T + "test_calibration_g42.py"],
             "Layers 1 and 3 pass for both models; Layer 2 is implemented (calibration approved: alpha 0.05, 12 seeds, margin = 1x reference SD) but its coverage check uses a naive per-point Kupiec test that ignores within-seed correlation, overstating significance -- a corrected (design-effect) version exists in calibration/analyze.py but is not yet wired into admission.py. Neither model is admitted; DeepVAR's correlated setting would likely pass coverage once fixed."),
}
STATUSES = {"closed", "closed_with_deviation", "built_not_admitted", "built_unavailable", "partial", "open", "dropped"}
