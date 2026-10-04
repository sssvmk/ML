# TODO: what only you can do

The code, tests and MLflow wiring are in place. The items below need your data, your infrastructure or your sign-off. Until the **Blocking** items are closed, treat this as production-minded scaffolding, not a production-ready system.

## What was run and what was not
- Run (in the build sandbox, CPU): 104 pytest tests and a 59-check end-to-end smoke test (`tests/smoke_test.py`). They run the whole pipeline (split audit, Optuna search with nested MLflow runs, final training, test evaluation, registration, champion/challenger gate, reload of the registered model in a fresh process) on **synthetic data**: shaped like MNIST and California Housing, and, for the CSV mode, like Santander (ID + binary target + 200 numeric columns) and Zillow (two files joined on a key, many missing values, code and text columns, dates, an all-missing column), including a pre-split test file, a Parquet source, and a plug-in loaded from a Python file (`examples/sqlite_source.py`, reading a SQLite table).
- Also run: one full CSV pipeline run on a synthetic 200,000 x 200 file (Santander size): 78 s, about 2.0 GB peak memory, on 1 CPU / 3 GB RAM. `mlflow models serve --env-manager local` served a CSV-trained model and scored JSON rows (integers, booleans, nulls) correctly.
- Not run: **anything on the real MNIST and California Housing data.** The sandbox blocks both downloads, so `_load_mnist` (torchvision), `_load_housing` (scikit-learn), `infer.py --test-index` and `--demo` are untested against the real sources. No real accuracy / AUC / MSE has been measured. The numbers in the smoke-test logs come from synthetic data and mean nothing.
- Not built: drift monitoring, a production serving deployment, a load test, multi-seed repeats, time-ordered / grouped splits, PR-AUC.

## Blocking (do before trusting any result)
- [ ] Run the pipeline once on the real data on your machine
  - Why: the real loaders and real-data behaviour are unverified
  - Where: `python run.py --data mnist --out results/mnist` and `python run.py --data california --out results/california`
  - Done when: both finish, `data/audit.json` shows `ok: true`, and the model card lists sizes 55,000 / 5,000 / 10,000 (MNIST) and 12,384 / 4,128 / 4,128 (Housing)
- [ ] Set the acceptance targets
  - Why: the target comes from the problem, not the data; none is set, so the gate only checks verification and validation loss
  - Where: `--min-auc` (classification) and `--max-mse` (regression); also decide the metric is right for your use
  - Done when: the targets are written down with their reasoning and passed on every run (or put in your CI/job config)
- [ ] Review the audit warnings on the real data
  - Why: identical rows shared between splits (possible leakage) are reported as warnings, not failures
  - Where: MLflow artifact `data/audit.json`, field `warnings`
  - Done when: each warning is explained or the data is fixed

- [ ] Run the CSV mode on your real file(s)
  - Why: the CSV code is only tested on synthetic data; real files bring surprises (odd dtypes, encodings, huge columns)
  - Where: `python run.py --data FILE --task ... --target ... --out ...` (examples in README)
  - Done when: the printed task, split sizes, audit warnings and the preprocessing notes (dropped / one-hot / date columns) all match what you expect
- [ ] Choose the split to match how the model will be used (time and grouped data)
  - Why: the pipeline splits at random (or uses the files you give it). If rows are related by time (Zillow transactions) or by customer, a random split leaks and the test numbers are optimistic
  - Where: split by date / group yourself, then pass `--data train.csv ... --source-opt test_csv=later_period.csv` (Parquet: `test_file=`)
  - Done when: the test file is later in time (or has different groups) than the training file
- [ ] Check the target makes sense for what you want to learn
  - Why: the Zillow Prize target `logerror` is the error of Zillow's own estimate and is mostly noise; expect an R2 near 0 and a model that barely beats predicting the mean (the paired MSE difference vs the linear reference shows this honestly)
  - Where: `--task`, the target column, the reference-model lines in the run output
  - Done when: you know what a useful result would look like before you start tuning

- [ ] Try your real data source through the plug-in interface
  - Why: only a SQLite example and file sources were exercised; a real warehouse brings credentials, large reads and types the example never sees
  - Where: subclass `TableSource` (see `examples/sqlite_source.py`, README "Data sources are plug-ins")
  - Done when: `python run.py --data your_source.py:YourSource ...` finishes and the data notes / audit look right, including the memory footprint on your largest table (the whole table is held in RAM)

## Needed for production
- [ ] Provision a shared MLflow tracking server and model registry
  - Why: the default store is a SQLite file inside each result folder (`<out>/mlflow.db`): fine for one laptop, but every folder has its own separate registry, which is not what a team or a serving setup needs
  - Where: set `MLFLOW_TRACKING_URI` (and credentials) in the environment, or pass `--tracking-uri`
  - Done when: runs and registered models from two different machines appear in one place
- [ ] Hook the tests into CI
  - Why: `.github/workflows/ci.yml` is provided but has never run on your repository
  - Where: your repository settings
  - Done when: a pull request runs `pytest` and `tests/smoke_test.py` green
- [ ] Decide how the model is served and test it under load (and note the dtype strictness)
  - Why: `mlflow models serve` worked in a local test, but there is no deployment and no load test; separately, `mlflow.pyfunc ... .predict(df)` rejects whole-number columns held as int64 (e.g. straight from `pd.read_csv`) because MLflow's schema enforcement allows only small ints for a `double` input. `infer.py`, `Predictor.tabular_frame` and the REST server accept them
  - Where: `serving_model.py`, signature creation in `run.py` (a fix would loosen the logged input schema); README "Scoring new rows"
  - Done when: your real callers score successfully through the path you choose, and latency / throughput meet your requirement
  - Why: the registered model is a self-contained MLflow pyfunc (preprocessing included), but no deployment exists
  - Where: e.g. `mlflow models serve -m models:/MLP-housing@champion`
  - Done when: latency and throughput meet your requirement on target hardware
- [ ] Add drift monitoring and a retraining trigger
  - Why: nothing watches live inputs; `infer.py` only warns when a single housing input is far outside the training range
  - Where: new code, using the training statistics stored in the model bundle (`preproc`)
  - Done when: input and prediction drift thresholds are set and alert someone
- [ ] Pin dependencies for your environment
  - Why: `requirements.txt` gives minimum versions (tested: torch 2.14.1, mlflow 3.16.1, optuna 5.0.0, scikit-learn 1.8.0, pandas 3.0.2, Python 3.12); the model's `pip_requirements` pins torch / numpy / pandas / mlflow at the versions used for training
  - Where: `requirements.txt`, lock file of your choice
  - Done when: a clean environment reproduces the run

## Deferred decisions
- [ ] Search budget and objective
  - Why: defaults are 30 trials, a 20k-row subset for MNIST search, and best validation loss as the objective (not AUC)
  - Where: `--trials`, `--train-subset`, `tuning.py`
  - Done when: you have compared a larger budget against the default using `search_improvement_vs_default` in MLflow
- [ ] Multi-seed repeats
  - Why: the reported SE / CI covers test-set size only, not run-to-run training variation
  - Where: repeat `run.py --trials 0 --params best_params.json --seed N` for several seeds
  - Done when: you have the spread of test metrics across seeds
- [ ] A classification baseline
  - Why: housing compares against linear regression (paired MSE difference); MNIST has no reference model
  - Where: `mlp_core._load_mnist` (`info["baseline"]`)
  - Done when: a simple model (e.g. logistic regression) is evaluated on the same split and reported
- [ ] Promotion margin
  - Why: a challenger must beat the champion's validation loss by 0.1% (relative) to be promoted; that is a judgement call
  - Where: `--promotion-margin`
  - Done when: the margin matches how much improvement matters to you

## Governance and handoff
- [ ] Review and sign off the model card
  - Why: it is generated; "Human approval: not recorded"
  - Where: MLflow artifact `MODEL_CARD.md` (also `<out>/MODEL_CARD.md`)
  - Done when: a named owner approves it and the approval is recorded where your process requires
- [ ] Name a model owner, a retraining trigger and a rollback plan
  - Why: the registry alias `champion` can be moved back to an earlier version, but nobody is assigned to do it
  - Where: your runbook
  - Done when: written down
- [ ] Fairness / robustness review if the model affects people
  - Why: none was done; the model card says so
  - Where: not covered by this repository
  - Done when: assessed or explicitly judged not applicable

## Assumption log (each should be confirmed or removed)
- Headline metrics: ROC-AUC (macro one-vs-rest) with bootstrap 95% CI for classification; MSE ± SE for regression (as requested).
- The standard error of MSE is std(squared error) / sqrt(n) on the test set; RMSE SE uses the delta method. The SE for AUC is a bootstrap (200 resamples by default).
- Model selection uses validation loss, so the test set is touched once per run. An optional acceptance target is checked on test metrics; repeatedly re-running until a target passes would make the test result optimistic.
- MNIST keeps its official test set; the 5,000-row validation set is carved from the official train set (stratified by digit, seed 42).
- California Housing is split 60/20/20 at random (seed 42); scalers are fit on the training rows only.
- Hyperparameter search on MNIST uses the first 20,000 training rows to save time; the final training uses all 55,000.
- Defaults chosen without asking: SGD with momentum as trial 0, early-stopping patience, search space ranges (see `tuning.suggest_config`).
- The MLflow store is created inside the result folder (`<out>/mlflow.db`, artifacts in `<out>/mlartifacts`) unless `--tracking-uri` or `MLFLOW_TRACKING_URI` is given. Running again into the same folder adds model versions to that folder's registry; a different folder has its own.
- Data sources: `--task` is required for your own data (`auto` is allowed and guesses). The built-in sources fix their task and target. A plug-in for non-tabular data would also need an inference adapter (the packaged model knows three input types: `mnist`, `housing`, `tabular`).
- CSV mode: the task is auto-detected (text target, or integer target with at most 20 distinct values, means classification). Preprocessing choices (median imputation + missing indicators, top-20 one-hot levels, ID-like text dropped) are defaults, not tuned.
- Binary classification uses two softmax outputs, so the reported macro one-vs-rest AUC equals the usual binary AUC.
- Memory: the whole table is held in RAM (about 7x the CSV size in the 200k x 200 measurement); a join file is read in chunks.
