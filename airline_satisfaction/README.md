# airsat: airline passenger satisfaction, experiment-driven ML pipeline

Config-driven pipeline that explores the data, builds representative splits, trains six algorithms under one identical
protocol, declares a winner with an auditable decision trace, evaluates it once on a held-out test set, packages it for
inference and explains everything through an AG2 LLM agent.

> Status: verified end to end on SYNTHETIC data with the same schema (20 tests pass). It has NOT been run on the real
> 699,635-row file, and the PyTorch MLP module has not been executed in the authoring sandbox (torch unavailable there).
> See `TODO.md` before trusting any number.

## Quick start
```bash
pip install -e ".[all]"                       # or: pip install -r requirements.txt && pip install -e . --no-deps
python -m airsat.run_pipeline --data airline.csv --explain offline          # train, select, test, package, explain
python -m airsat.predict --input unseen.csv --model runs/<run_id>/champion --output predictions.csv --drift-report
python -m airsat.agent.ag2_explainer --run runs/<run_id> --mode llm         # LLM explanations (needs LLM_* env vars)
python -m airsat.agent.ag2_explainer --run runs/<run_id> --mode llm --interactive
AIRSAT_MODEL_DIR=runs/<run_id>/champion python -m airsat.serving.app       # HTTP service on :8000
python -m airsat.promote --run runs/<run_id> --approver "Name"              # gate: candidate -> champion alias
bash scripts/run_example.sh                                                 # synthetic demo
```
Everything is controlled by `config/default.yaml`; override with `--set search.n_trials=40`. Useful overrides:
`data.sample_rows=50000` for quick iteration, `--algorithms lightgbm_gradient_boosted_trees ...` for a subset.

LLM endpoint (any OpenAI-compatible server or gateway): `LLM_MODEL`, `LLM_API_KEY`, optional `LLM_BASE_URL`,
`LLM_DEFAULT_HEADERS` (JSON). Behind a TLS-inspecting proxy also set `SSL_CERT_FILE`. Without credentials the agent
falls back to deterministic offline explanations built from the same facts.

## Experiment design
1. **Goal and metric.** Primary metric ROC-AUC (threshold-free, classes roughly balanced, so ranking quality is what we
   compare). Reported alongside: PR-AUC, log loss, Brier, expected calibration error, accuracy, balanced accuracy,
   precision, recall, F1, MCC. The decision threshold is chosen on validation only (`metric.threshold_rule`).
   The business target is `metric.target_value` (unset: to be confirmed).
2. **EDA and tests** (`eda.py`). Missingness, duplicates, rating ranges and "0 = not applicable" shares, skew/outliers,
   normality, Mann-Whitney and single-feature AUC (leak scan), Cramér's V and chi-square, mutual information, Spearman
   collinearity pairs, id-vs-target check. Effect sizes accompany p-values because n is large.
3. **Variable types** (`schema.py`): non-numeric -> nominal; integer-valued with <=10 levels -> ordinal (the 0-5 ratings);
   other numeric -> continuous; `id` and the target are excluded from features.
4. **Splits** (`splitting.py`): 70/15/15, stratified on target x Class x Type of Travel; identical rows never straddle
   splits; split before any learned preprocessing. Representativeness is judged by KS statistic, Cramér's V, prevalence
   difference (with sample-size-aware limits) and adversarial validation (AUC ~0.5 is good).
5. **Features** (`features.py`): rating aggregates computed on non-zero ratings, count of zero/low/high ratings,
   group means (digital, comfort, service, logistics), delay transforms, recovered delay, on-time flag, log distance,
   distance/age bands, class ordinal, business x loyalty and digital x business interactions. Fitted on train only.
6. **Per-algorithm protocol** (`algorithms/base.py`, identical for all): encode/scale -> assumption checks ->
   Optuna TPE search (Bayesian) on a training subsample, defaults enqueued as trial 0 -> final fit on full train
   (boosters/MLP early-stop on a 10% hold-out of TRAIN, so validation stays untouched) -> metrics, ROC/PR, calibration,
   confusion, learning curve, permutation importance, slice metrics -> K-fold stability check with preprocessing refit
   inside each fold -> inference bundle -> MLflow logging.
7. **Winner** (`analysis/select_winner.py`): all models scored on the same validation rows; 95% bootstrap CIs and paired
   bootstrap differences against the best; a model within `min_practical_delta` or whose difference CI contains 0 is a
   tie; ties are broken by (no large train-validation gap, simplicity, latency, Brier). Unstable models (CV std high and
   above 2x the median) are excluded. McNemar is reported as supporting evidence. The test set is then evaluated once,
   for the winner only; `test_set_used.lock` blocks a second evaluation.

## Algorithms, losses, hyper-parameters

| Module | Loss / objective | Searched hyper-parameters | Key assumptions checked |
|---|---|---|---|
| `majority_class_baseline` | none (constant prior) | none | class balance, sample size |
| `logistic_regression_elasticnet` | regularised cross-entropy, (1/C)[a L1 + (1-a)/2 L2] | C, l1_ratio, class_weight | linear log-odds (Box-Tidwell), VIF/collinearity, separation, events per variable, independence (Durbin-Watson) |
| `random_forest_bagged_trees` | Gini/entropy per split, mean leaf frequency | trees, depth, min leaf, max features, bootstrap fraction, criterion | leakage, duplicates, shift, range coverage |
| `xgboost_gradient_boosted_trees` | logistic loss + gamma T + lambda/alpha leaf penalties | eta, depth, min child weight, subsample, colsample, lambda, alpha, gamma (trees by early stopping) | leakage, shift, labels |
| `lightgbm_gradient_boosted_trees` | logistic loss + L1/L2 + min split gain | eta, leaves, min child samples, subsample, colsample, lambda, alpha, min split gain | same as XGBoost |
| `pytorch_residual_mlp` | BCE on logits + AdamW weight decay | width, blocks, dropout, lr, weight decay, batch size (median pruner on epoch val loss) | scaled inputs, params/sample ratio, initial loss ~ ln 2, can overfit tiny batch |

Shared checks everywhere: finite inputs, class balance, duplicates, single-feature leakage, train-vs-validation shift.
Why six: a trivial floor, a linear model, bagging, two boosting variants with different tree growth, and a deep model, so
the experiment shows whether non-linearity and interactions matter at all.

## Outputs (`runs/<run_id>/`)
`eda/`, `splits/`, `algorithms/<name>/{result.json, assumption_checks.json, search_trials.csv, training_curve.csv, plots/, bundle/}`,
`selection.json`, `model_comparison.md`, `reports/{experiment_report.md, MODEL_CARD.md, model_comparison.png, roc_overlay.png, agent_explanations/}`,
`test_evaluation.json`, `champion/` (inference bundle), `registry.json`, `experiment_log.jsonl`, `mlflow.db`, `config_used.yaml`.

## Inference
The bundle takes RAW rows and returns probabilities: it validates the schema, engineers features, encodes and predicts, so
training and serving cannot drift. `predict.py` writes `id,satisfaction` (TRUE/FALSE by default; `output.*` in the config
changes column names, label style and optional probability). `--drift-report` computes PSI against the training reference.
Bundles are pickles: serve with the library versions in `run_context.json`.

## Tests
`pytest` runs a tiny end-to-end pipeline, split/feature/metric unit tests, the HTTP service and the agent loop (mock LLM).
`tests/test_mlp.py` runs only when torch is installed.

---

## Airline Satisfaction Competition Overview

### What is it?
The **AirSat** (Airline Passenger Satisfaction) project is an experiment-driven Machine Learning pipeline. The primary objective of this project is to accurately predict whether a passenger will be satisfied with their flight experience based on a variety of passenger, flight, and survey data points. It is set up as a binary classification problem (target: `satisfaction` = TRUE/FALSE), with ROC-AUC serving as the primary evaluation metric to assess the ranking quality of the predictions. The project emphasizes rigorous ML practices, including automated exploratory data analysis (EDA), statistical assumption checks, hyperparameter search, auditable model selection, drift monitoring, and explainability via LLM agents.

### The Dataset
The pipeline is designed for a large tabular dataset containing approximately 699,635 rows (though verified heavily on synthetic data mimicking the exact same schema). The dataset encompasses several features:
* **Demographics:** `Age`, `Gender`
* **Travel Details:** `Customer Type` (Loyal/Disloyal), `Type of Travel` (Business/Personal), `Class` (Business, Eco, Eco Plus), `Flight Distance`
* **Service Ratings:** A suite of 13 attributes rated from 1 to 5 (with 0 meaning 'not applicable' or missing). These include: `Inflight wifi service`, `Departure/Arrival time convenient`, `Ease of Online booking`, `Gate location`, `Food and drink`, `Online boarding`, `Seat comfort`, `Inflight entertainment`, `On-board service`, `Leg room service`, `Baggage handling`, `Checkin service`, `Cleanliness`
* **Delays:** `Departure Delay in Minutes`, `Arrival Delay in Minutes`
* **Target Variable:** `satisfaction` (Boolean: TRUE if satisfied, FALSE if neutral or dissatisfied).

---

## Codebase Script Directory

### Core Pipeline & Utilities (`src/airsat/`)
* **`__init__.py`**: Package initialization.
* **`assumptions.py`**: Performs statistical and data assumption checks tailored to different model types (e.g., checking linearity for logistic regression, or input scaling for neural nets).
* **`bundle.py`**: Defines `ModelBundle`, a unified object that packages validation, feature engineering, encoding, the trained model, and decision thresholds together. This guarantees that training and serving code cannot drift apart.
* **`config.py`**: Manages YAML-driven configuration loading (`config/default.yaml`) with support for command-line overrides.
* **`eda.py`**: Executes automated Exploratory Data Analysis (EDA) and statistical testing (e.g., missingness, duplication, leakage, collinearity) to drive downstream modeling choices.
* **`features.py`**: Houses the stateless feature engineering logic (learned on TRAIN only), creating rating aggregates, handling weak-link metrics, transforming delays, and mapping segment interactions.
* **`io_utils.py`**: Helpers for safely reading tabular data (CSV/Parquet) and coercing the target column into standard boolean representations.
* **`metrics.py`**: Defines shared evaluation metrics (e.g., expected calibration error, fast weighted AUC) and calibration helpers utilized by all algorithms.
* **`monitoring.py`**: Implements data drift monitoring, calculating Population Stability Index (PSI) to track feature distribution shifts over time against a training reference.
* **`plots.py`**: Headless-safe plotting utilities that generate and save key diagnostic charts (ROC, PR, Calibration, Confusion Matrix) directly to disk.
* **`predict.py`**: CLI application for running batch inference on unseen tabular datasets using the packaged champion model bundle.
* **`preprocessing.py`**: Assembles scikit-learn standard preprocessing pipelines (like one-hot/ordinal encoding, imputation, and scaling) matched to specific algorithm families.
* **`promote.py`**: Implements a strict promotion gate that checks if a candidate model clears baseline margins, slice-gap constraints, and performance thresholds before being tagged as 'champion'.
* **`registry.py`**: Best-effort MLflow model packaging and registry module to manage deployment aliases safely without blocking local execution.
* **`reporting.py`**: Generates human-readable Markdown artifacts based purely on measured metrics, such as the overall experiment report and the `MODEL_CARD.md`.
* **`run_pipeline.py`**: The main entry point script. It takes input data, triggers the entire suite of algorithm experiments, selects a winner, packages the champion, and outputs comprehensive reports.
* **`schema.py`**: Infers and enforces variable types (continuous, ordinal, nominal) via declarative rules or config overrides.
* **`splitting.py`**: Handles robust dataset splitting (train/validation/test). It stratifies by a composite key (Target x Class x Travel Type) and keeps identical rows grouped to prevent leakage into evaluation sets.
* **`synthetic.py`**: Generates a highly realistic synthetic dataset with matching schema and correlation structure, utilized for smoke tests, CI environments, and end-to-end demonstrations.
* **`tracking.py`**: Provides standardized MLOps experiment tracking via MLflow, concurrently mirrored to a robust, local JSONL log.
* **`utils.py`**: Shared general-purpose helpers spanning random seeding, hashing, JSON encoding, timing, and system context recording.
* **`validation.py`**: Handles strict input schema validation ensuring required columns and data types are respected for both batch CLI and real-time HTTP inference.

### LLM Explainer Agents (`src/airsat/agent/`)
* **`__init__.py`**: Package initialization.
* **`ag2_explainer.py`**: Implements an AG2 multi-agent workflow (Interpreter and Auditor) designed to transparently explain the pipeline's decisions and model results to non-experts.
* **`tools.py`**: Provides the read-only, strict-access toolsets that agents use to query actual experiment facts to prevent hallucinations.

### Algorithms (`src/airsat/algorithms/`)
* **`__init__.py`**: Package initialization.
* **`base.py`**: Defines `AlgorithmModule`, the foundational protocol template forcing every candidate model to standardise hyperparameter searching, loss functions, assumption checks, and fitting strategies for a fair comparison.
* **`lightgbm_gradient_boosted_trees.py`**: Configures LightGBM (histogram-based boosting with leaf-wise growth optimization).
* **`logistic_regression_elasticnet.py`**: Configures Logistic Regression featuring an elastic-net penalty; serves as an interpretable linear contender.
* **`majority_class_baseline.py`**: A non-learning dummy classifier serving as an absolute metric floor reference.
* **`pytorch_residual_mlp.py`**: Configures a PyTorch-based Residual Multi-Layer Perceptron representing the deep-learning tabular candidate.
* **`random_forest_bagged_trees.py`**: Configures a Random Forest classifier consisting of bagged, de-correlated decision trees.
* **`xgboost_gradient_boosted_trees.py`**: Configures XGBoost, implementing second-order gradient boosted decision trees.

### Analysis & Selection (`src/airsat/analysis/`)
* **`__init__.py`**: Package initialization.
* **`final_evaluation.py`**: Executes the single, one-shot evaluation of the winning model against the untouched holdout test set to get an unbiased performance metric.
* **`select_winner.py`**: Implements the rigorous experiment analysis workflow to automatically declare a winner. It calculates bootstrapped confidence intervals and evaluates model stability (CV variance) to create an auditable decision trail.

### Real-time Serving (`src/airsat/serving/`)
* **`__init__.py`**: Package initialization.
* **`app.py`**: A FastAPI application establishing a real-time HTTP inference service. Provides endpoints for liveness checks (`/health`), detailed model insights (`/model-info`), and online scoring (`/predict`).

### Shell Scripts (`scripts/`)
* **`run_example.sh`**: A comprehensive end-to-end bash script that generates synthetic data, executes the full modeling pipeline, scores an unseen dataset, and produces a drift report as a quickstart demo.

### Test Suite (`tests/`)
* **`conftest.py`**: Pytest configuration file establishing crucial shared test fixtures (like injecting small synthetic datasets into tests).
* **`mock_llm_server.py`**: A lightweight, standalone HTTP server mocking OpenAI-compatible responses, enabling agent tool-calling tests without real network/API costs.
* **`test_metrics_selection.py`**: Unit tests verifying custom metric calculations (e.g., AUC handling of ties) and the automated model selection/tie-breaking logic.
* **`test_mlp.py`**: Conditional unit tests validating the PyTorch Residual MLP architecture (skipped automatically in environments missing torch).
* **`test_pipeline_end_to_end.py`**: High-level integration tests that run the entire pipeline fast-path to ensure training, model selection, saving, and inference steps cohesively execute.
* **`test_schema_split_features.py`**: Unit tests targeting the data-prep phases: asserting correct schema inference, ensuring no leakage during dataset splitting, and verifying feature engineering transforms.
* **`test_service_and_agent.py`**: Verifies the FastAPI inference service endpoints and checks the local execution of the agent loop mechanisms.
