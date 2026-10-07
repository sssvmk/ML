# aeclf: encoder-decoder (autoencoder) pretraining + satisfaction classifier

Separate project from `airline_satisfaction_ml`. Here the central idea is representation learning:

1. build an **encoder-decoder** that reproduces each passenger record (all 22 columns, mixed types);
2. take the **encoder**, add a **classification layer**, and train it to predict `satisfaction`;
3. run hyper-parameter search, log every experiment, compare variants, declare a winner, package it for inference,
   and explain everything with an AG2 LLM agent.

> Status: verified end to end on SYNTHETIC data with the same schema (20 tests pass). It has NOT been run on the real
> 699,635-row file. Read `TODO.md` before trusting any number.

## Quick start
```bash
pip install -r requirements.txt && pip install -e . --no-deps      # or: uv sync --extra agent --extra serve --extra dev
python -m aeclf.run_pipeline --data airline.csv --explain offline
python -m aeclf.predict --input unseen.csv --model runs/<run_id>/champion --output predictions.csv --drift-report --extras
python -m aeclf.agent.ag2_explainer --run runs/<run_id> --mode llm            # needs LLM_MODEL / LLM_API_KEY [/ LLM_BASE_URL]
AECLF_MODEL_DIR=runs/<run_id>/champion python -m aeclf.serving.app          # /predict /embed /reconstruct
bash scripts/run_example.sh                                                   # synthetic demo
```
Overrides: `--set autoencoder.search.n_trials=20`, `--set data.sample_rows=50000`, `--set metric.primary=accuracy`.
First try `--set data.sample_rows=50000 --set autoencoder.search.n_trials=3 --set classifier.search.n_trials=3` to check timing.

## Experiment design
**Stage 0, data** (`eda.py`, `schema.py`, `splitting.py`): EDA with effect sizes next to p-values; variable roles inferred
(4 continuous, 13 integer ratings = ordinal, 4 nominal; `id` and the target excluded); 70/15/15 split stratified on
target x Class x Type of Travel, duplicates never straddle splits, representativeness checked (KS, Cramér's V, prevalence,
adversarial validation).

**Encoding** (`encoding.py`, fitted on train only): continuous columns standardised (log1p first for skewed non-negative
ones such as delays); nominal columns AND the 0-5 ratings become integer classes with a reserved *unknown/missing* class.
Ratings are classes because 0 means "not applicable", so a numeric average would be meaningless (`encoding.ordinal_as: numeric` overrides).

**Stage 1, encoder-decoder** (`models.py`): per-column embeddings + continuous columns -> MLP trunk -> latent vector;
decoder = MLP trunk -> one regression head for all continuous columns + one softmax head per categorical column.

| Variant | What differs | Loss |
|---|---|---|
| `ae` | plain autoencoder | `MSE(continuous) + w_cat * mean CE(categorical)` |
| `dae` | input corrupted (Gaussian noise, random "unknown" masking), CLEAN row is the target | same |
| `vae` | stochastic latent q(z\|x) | same + `beta * KL(q \|\| N(0,I))`, KL warm-up, beta capped to [1e-3, 0.1] |

Unsupervised: the target is never used. Early stopping on validation reconstruction loss.

**Stage 2, classifier** (`models.py`, `training.py`): latent mean -> head (linear or small MLP) -> one logit, loss =
binary cross-entropy (BCEWithLogits). Modes: `frozen` (encoder fixed, a probe of representation quality), `finetune`
(encoder trained at `lr * enc_lr_mult`), and the **`scratch` control**: same network from random weights.
Early stopping on validation log loss.

**LightGBM on the encoder** (`lgbm_head.py`): the frozen pretrained encoder produces latent codes and LightGBM is trained on them.
`lgbm_latent` = latent code only (still a pure encoder-based classifier); `lgbm_hybrid` = latent code + raw encoded columns
(an upper bound on what the encoder can add). Trees early-stop on a 10% hold-out of TRAIN; Optuna tunes learning rate, leaves,
min child samples, subsampling, column sampling and L1/L2 penalties. Their "trainable params" is a trees x leaves proxy
(not comparable one-to-one with neural weights), used only as a tie-breaker.

**Controls and floor**: majority-class baseline (AUC 0.5) and two no-encoder controls: the scratch neural network (control for
the neural heads) and LightGBM on the raw encoded columns (`lgbm_raw_control`, control for the LightGBM heads). Without the
controls you cannot tell whether the autoencoder helped at all, so they are part of the experiment, not extras.

**Hyper-parameter search** (`experiments.py`): Optuna TPE (Bayesian) with a median pruner on per-epoch validation loss;
defaults are enqueued as trial 0. Stage 1 tunes width, depth, latent size, dropout, lr, weight decay, batch size, `w_cat`,
(`noise_sigma`, `mask_p` for DAE; `beta` for VAE) on reconstruction loss. Stage 2 tunes head size/dropout, lr, weight decay,
batch size (and `enc_lr_mult` for fine-tuning) on validation ROC-AUC. Searches run on a training subsample (`train_subsample`).
Caveat: VAE reconstruction-only tuning is biased toward small beta, hence the cap and the collapse check.

**Metrics.** Autoencoder: weighted reconstruction loss; per continuous column R², RMSE/MAE in ORIGINAL units vs the mean
predictor; per categorical column accuracy vs the majority level (ratings also MAE in levels); exact categorical row-match
rate; latent diagnostics (active dimensions, PCA plot coloured by satisfaction). Classifier: ROC-AUC (primary, selection),
PR-AUC, log loss, Brier, ECE, accuracy, balanced accuracy, precision, recall, F1, MCC; decision threshold chosen on validation only.

**Sanity checks** (stored per autoencoder in `sanity_checks.json`): finite inputs, category codes in range, ability to overfit
64 rows (bug detector, blocking), reconstruction beats mean/majority predictors, no latent collapse.

**Data collected for analysing the experiments**: `experiment_log.jsonl` (every param/metric event) and an MLflow store
(`mlflow.db`, one nested run per autoencoder, trial and candidate); per run `search_trials.csv`, `training_curve.csv`,
`result.json`, plots, `val_predictions.npz`, `label_efficiency.csv`, `config_used.yaml`, data hash, git commit, library versions.

**Winner** (`selection.py`): all models scored on the same validation rows; 95% bootstrap CIs; paired bootstrap against the
best; a model within `min_practical_delta` or whose difference CI contains 0 ties; ties broken by (no large train-validation
gap -> fewest trainable parameters -> lowest latency). Reported separately: pretraining benefit (each AE model vs the
scratch control) and the label-efficiency study (AUC vs fraction of labels). The test set is evaluated once, for the winner
only (`test_set_used.lock`). If the control wins, the report says autoencoder pretraining did not help.

## Inference
`ModelBundle` takes RAW rows (schema validation, encoding, network) and offers `predict_proba`, `embed` (latent code of the
pretrained encoder) and `reconstruct` (decoded rows + per-row error, usable as an anomaly score). Stored as a directory
(`encoder.joblib`, `weights.pt` loadable with `weights_only=True`, `meta.json`), not a pickled model.
`predict.py` writes `id,satisfaction` (TRUE/FALSE default; see `output:` in the config), plus optional drift report
(PSI vs training) and embeddings. HTTP service: `/predict`, `/embed`, `/reconstruct`, `/health`, `/model-info`.

## Layout
`src/aeclf/` encoding, models, training, experiments, selection, final_evaluation, bundle, predict, serving/, agent/ (AG2),
registry (MLflow), reporting. Outputs under `runs/<run_id>/`: `autoencoders/<kind>/`, `candidates/<name>/`, `selection.json`,
`model_comparison.md`, `reports/` (experiment report, model card, plots, agent_explanations/), `test_evaluation.json`, `champion/`.

## Tests
`pytest` (22 tests): encoder round trips, unknown categories, model variants, the autoencoder learning, a tiny end-to-end
run, frozen-encoder weights identical to pretrained, test-set lock, unseen-file inference format, HTTP service, AG2 tool loop
against a mock LLM server.
