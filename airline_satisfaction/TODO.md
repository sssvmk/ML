# TODO: what only you can do

## What was run and what was not
- Run: full pipeline on synthetic data (8,000 rows, 5 algorithms, 3 search trials each); 20 pytest tests; offline agent; AG2 tool-calling loop against a mock LLM server; promotion gate dry run.
- Not run: the real 699,635-row dataset; the PyTorch MLP (torch could not be installed in the authoring sandbox); the agent against a real LLM; Docker build; CI; MLflow against a shared server.

## Blocking (before trusting any result)
- [ ] Confirm the primary metric and set a target
  - Why: ROC-AUC was assumed; without `metric.target_value` the winner is only "best available".
  - Where: `config/default.yaml` -> `metric`
  - Done when: target set, `promotion.min_primary_metric` aligned, and the cost of false positives vs false negatives is agreed (it drives `threshold_rule`).
- [ ] Run on the real data and read `eda/data_profile.md`, `splits/population_report.json`, every `assumption_checks.json`
  - Why: all numbers so far come from synthetic data.
  - Where: `python -m airsat.run_pipeline --data <file>`
  - Done when: no unexplained leakage warning, split reported representative, adversarial AUC near 0.5.
- [ ] Run `pytest tests/test_mlp.py` and a real MLP run with torch installed
  - Why: the MLP code has never been executed.
  - Where: `algorithms/pytorch_residual_mlp.py`
  - Done when: tests pass and the module's status is `ok` in `selection.json`.
- [ ] Provide the sample output file
  - Why: none was attached; `id,satisfaction` with TRUE/FALSE is assumed.
  - Where: `config/default.yaml` -> `output`
  - Done when: `predict.py` output matches the sample column-for-column.

## Needed for production
- [ ] Shared MLflow tracking server and registry (`tracking.uri`); currently a local SQLite file per run.
- [ ] Serving constraints (latency, throughput, hardware) and a load test; pin exact library versions from `run_context.json`.
- [ ] Drift thresholds and a retraining trigger and owner (PSI scaffold exists in `monitoring.py`).
- [ ] LLM gateway settings (`LLM_*`, CA bundle if TLS is inspected) and a review of what run data may be sent to the model.
- [ ] Check bundle size for the random forest on the full data; reduce `max_depth`/`n_estimators` if too large.

## Deferred decisions
- [ ] Search budget: defaults are 25 trials/algorithm on a 150k-row subsample; raise `search.n_trials` if curves show headroom.
- [ ] Whether to report test metrics for all models (`selection.report_test_for_all`); default keeps the test set for the winner only.
- [ ] Probability recalibration (isotonic/Platt) if scores are used as probabilities; ECE is reported.

## Governance and handoff
- [ ] Complete the TODO markers in `reports/MODEL_CARD.md` (intended use, data provenance); agree fairness thresholds for the slice gaps.
- [ ] Named approver runs `python -m airsat.promote --run <run> --approver "<name>"`; keep `approvals/` records.

## Assumption log (confirm or remove)
- Metric ROC-AUC; split 70/15/15 seed 42 stratified on target x Class x Type of Travel; duplicates grouped.
- Rating 0 = "not applicable"; `id` carries no signal; no passenger/flight/date column, so repeat passengers and time drift cannot be controlled.
- Overfit gap is a soft rule; CV stability uses a 100k-row subsample with 3 folds.
