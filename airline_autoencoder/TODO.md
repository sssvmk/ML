# TODO: what only you can do

## What was run and what was not
- Run: full pipeline on synthetic data (6,000 rows, 2-3 autoencoders, 10-13 classifier candidates incl. LightGBM heads, tiny search budgets); 22 pytest tests; offline agent; AG2 tool loop against a mock LLM server.
- Not run: the real 699,635-row dataset; a GPU; the agent against a real LLM; Docker build; CI; MLflow against a shared server; the promotion gate (not implemented in this project; see below).

## Blocking
- [ ] Run on the real data and read `eda/data_profile.md`, `splits/population_report.json`, each `autoencoders/<kind>/sanity_checks.json`
  - Why: all numbers so far are from synthetic data with tiny budgets.
  - Where: `python -m aeclf.run_pipeline --data <file>` (try `data.sample_rows=50000` first for timing)
  - Done when: no unexplained leakage/shift warning, sanity checks pass or are understood.
- [ ] Confirm the primary metric and set a target
  - Why: ROC-AUC assumed; `metric.target_value` unset, so the winner is only "best available".
  - Where: `config/default.yaml` -> `metric`
  - Done when: target set and false-positive vs false-negative cost agreed (drives `threshold_rule`).
- [ ] Provide the sample output file
  - Why: none attached; `id,satisfaction` TRUE/FALSE assumed.
  - Where: `config/default.yaml` -> `output`
  - Done when: `predict.py` output matches the sample column for column.

## Needed for production
- [ ] Raise search budgets for the real run (defaults: 12 autoencoder trials, 10 per classifier, 100k-row search subsample) and use a GPU if available.
- [ ] Review reconstruction fidelity per column; if ratings are poorly rebuilt, try larger latent or `encoding.ordinal_as: numeric`.
- [ ] Shared MLflow server (`tracking.uri`); promotion gate with human sign-off (the earlier project has `promote.py`; port it once the metric/target are fixed).
- [ ] Pin exact library versions from `run_context.json`; build and load-test the Dockerfile; drift thresholds and retraining owner.
- [ ] LLM gateway settings (`LLM_*`, corporate CA via `SSL_CERT_FILE`) and a review of what run data may be sent to the model.

## Deferred decisions
- [ ] LightGBM heads use a trees x leaves proxy for 'trainable params' in the tie-break; if you want a different complexity rule, change `selection.tie_breakers`.
- [ ] Fine-tuning the encoder against LightGBM (end-to-end) is not supported; the encoder is frozen for the LightGBM heads.
- [ ] Whether the classifier should also see raw features (it currently sees only the latent code, as specified); a hybrid could score higher but is no longer a pure encoder-based classifier.
- [ ] Semi-supervised variant: also pretrain on unlabelled validation/test features (off by default to keep evaluation clean).
- [ ] Probability recalibration if scores are used as probabilities (ECE is reported).

## Governance
- [ ] Complete TODO markers in `reports/MODEL_CARD.md`; agree fairness thresholds for slice gaps.

## Assumption log (confirm or remove)
- Metric ROC-AUC; split 70/15/15 seed 42 stratified on target x Class x Type of Travel; ratings 0-5 treated as classes (0 = not applicable); `id` carries no signal; no passenger/flight/date column.
- Overfit gap is a soft rule; VAE beta capped at 0.1; control uses the architecture of the first successful autoencoder kind.
