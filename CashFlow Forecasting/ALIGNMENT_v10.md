# Alignment to STC_prd_v10.md

> **Status note (2026-09-28).** This file records ONE earlier alignment pass and is kept as history. Its "Still open" section and its
> statements that the neural family is a simplified stand-in predate the gap-closing work and are **superseded by `GAP_STATUS.md`**
> (per-gap status with evidence) and the regenerated `TRACEABILITY.md`. Where they disagree, those two files are current.


This documents what changed in this pass to bring the implementation in
line with PRD v10, and — as importantly — what is still open. Written
in the same spirit as TRACEABILITY.md: generated from and cross-checked
against the running code, not aspirational.

## What changed

### §3.4 Six evaluation metrics
`algorithms/base.py`'s `evaluate()` previously computed a 4-metric
subset (MAE, RMSE, MAPE, bias) with a docstring flagging the full
six-metric set as future scope. It now computes all six: **sMAPE,
WAPE, MAE, RMSE, MASE, Bias %**. MASE is scaled by the in-sample
seasonal-naive(seasonal_period) error of *that fold's own training
window* (Hyndman & Koehler scaling), with `seasonal_period` defaulting
to 7 to match the Seasonal-Naive-7 baseline — so `MASE < 1` means
exactly "beat the baseline," which §3.5 step 1 relies on directly.

### §3.2 Required-observations table, per algorithm
Every one of the 38 modules now implements `required_observations()`
with the actual formula from the PRD v10 table (not a generic
`min_observations` config floor). Along the way this fixed several
places where the *floor* term was missing from the original code (e.g.
ARIMAX and SARIMAX's `max(50, …)` / `max(100, …)` floors were dropped
in the original eligibility checks — the formula ran unfloored).

The `has_eligibility_condition` count was already correct in the
existing code (15 of 38, matching PRD v10's row-level table exactly —
WaveNet/DeepState/TiDE inherit `True` from TCN/DeepAR/TFT rather than
restating it, which is why a naive `grep` undercounts it). Only the
stale comment ("10 of 38") needed fixing.

### §5.1 Ten module interfaces
The PRD table lists ten: `describe_contract, validate, check_eligibility,
train, save, load, infer, diagnose, evaluate, log`. The code had nine —
`validate()` didn't exist as a per-module callable, only as the
Orchestrator-level `contract.validate_contract()`. Added `validate()`
to `AlgorithmModule`, delegating to that same shared implementation
(there is one universal-validation rule set, not 38 copies of it).

### §3.5 Elimination & Ranking (previously a single-metric threshold)
`orchestrator.py`'s `_select_winner` previously eliminated on one
configured metric (`elimination_metric: "mae"`) against the baseline's
own MAE. Rewritten as a new shared module, `ranking.py`, implementing
the five steps as specified:
1. Eliminate on MASE ≥ 1.
2. Eliminate on `|Bias %| > bias_threshold` (now a real, configurable
   `config.json` field — previously didn't exist at all).
3. Rank the six metrics per survivor (bias ranked by absolute value).
4. Combine ranks by sum; lowest wins.
5. **Consistency check** (`orchestrator.Orchestrator._select_winner`):
   the ranked winner must have fold-level MASE < 1 in at least
   `consistency_min_pass_rate` of its own backtest folds, or the next
   ranked candidate is tried instead. This cascades to the fallback if
   nothing passes.

### §3.5 Baseline challengers
Seasonal-Naive-7 remains the sole elimination gate. `Naive-1`
(`algorithms/naive_1.py`, new) and `Seasonal-Naive-30` are now
computed alongside every run and logged for comparison, as the PRD
specifies — previously neither existed.

### §3.2 Per-algorithm searchable rolling window + held-out test set
Previously one fixed `window=20` was applied to every algorithm
uniformly — directly contradicting §3.2's "not a single fixed value
applied uniformly across all 38 algorithms." The Orchestrator now:
- computes each candidate's window floor from its own
  `required_observations()`, and searches a small set of window sizes
  between that floor and `min(available history, backtest_years_of_history × 365)`;
- reserves a configurable `holdout_periods` tail of each segment's
  history, untouched by window search, hyperparameter search, or
  algorithm selection, and reports the winner's performance on it
  separately from the metrics used for selection;
- refits the winner once on all pre-holdout history (to score the
  holdout) and again on the full history including the holdout (the
  artifact actually registered for production inference).

Step size between windows (`backtest_step`) is now a `config.json`
field rather than hardcoded, per §3.2's "step size between windows
remains configurable."

### §3.3 Hyperparameter search — now covers all 38 algorithms
Every one of the 38 modules now declares a real
`hyperparameter_search_space()`: the statistical/ARIMA-like family
(ARIMAX, SARIMAX, VARMAX, Dynamic Regression) searches `p`/`d`/`q`
(and VARMAX's `p`/`q`) directly — their `__init__` methods were
extended to build `order` from separately-sampled `p`/`d`/`q` when
present, falling back to the old single `order` hyperparameter
otherwise, so existing `config.json` entries are unaffected. State
Space and Structural TS search their `level`/`seasonal_period` choice;
Prophet-like searches its harmonic counts; Linear/Polynomial
Regression search `n_lags`/`degree`. All 19 deep-learning-family
stand-ins get a shared space (`n_lags`, `max_iter`,
`hidden_layer_sizes`) from `_neural_template.py`'s base class in one
place, with TCN/WaveNet overriding it to search `kernel_size`/`levels`
instead (inherited by WaveNet automatically). Combined with the 10
already covered in the previous pass (Ridge/Lasso/ElasticNet/
RandomForest/ExtraTrees/XGBoost/LightGBM/CatBoost/SVR/KNN), that's
**38 of 38** with a declared, functioning search space — verified by
instantiating every catalog entry and confirming a non-empty space,
and by end-to-end search runs on ARIMAX and TCN that visibly improve
MASE over the `config.json` defaults.

### Bug found and fixed: HPO silently discarding its own results
`search.py` originally reused `ranking.eliminate_and_rank()` — the
*final-selection* function that eliminates any candidate with MASE ≥ 1
— to compare an algorithm's own hyperparameter trials against each
other. When every trial for an algorithm happened to score MASE ≥ 1 on
a segment (a real, not-hypothetical case: ARIMAX/SARIMAX/State-Space
without a seasonal term against a strongly seasonal series), *all*
trials got eliminated, `winner` came back `None`, and the search
silently fell back to the un-searched `config.json` defaults — as if
no search had run at all, with no error or warning. Fixed by adding an
`eliminate_on_mase` flag to `ranking.eliminate_and_rank()`: final
algorithm selection still eliminates on MASE ≥ 1 (§3.5 step 1,
unchanged, confirmed by a direct unit check), but `search.py`'s
internal trial comparisons now rank-only (§3.3.6's rank-sum, without
the baseline-elimination step, since that comparison is picking the
*least-bad* configuration for an algorithm the Orchestrator will
independently screen against the baseline later, not screening
against the baseline itself). Confirmed by rerunning the ARIMAX search
that previously returned the untouched default and now correctly
returns its best-found `{p, d, q}`.

### Bug found and fixed: shared-RNG race condition under parallel HPO
`search.py` originally seeded and drew from Python's *global* `random`
module. Because multiple candidates' hyperparameter searches run
concurrently (`parallel.py`, §3.7 — a thread pool at minimum, Ray if
available), several threads were drawing from the same shared
generator at once: the sequence of draws any one candidate's search
actually got depended on other candidates' thread-scheduling timing,
so the *same* configured `seed` did not reproduce the *same* search
trajectory from run to run — a correctness gap against §3.3.10's
"search-space version" / experiment-tracking reproducibility intent.
Fixed by giving each `random_then_adaptive_search()` call its own
`random.Random(seed)` instance rather than touching the global module.
Verified by running `full_train` twice, independently, with the same
`hyperparameter_search.seed` and a multi-candidate concurrent config,
and confirming both runs picked the identical winning algorithm and
identical hyperparameters.

### §3.7 Ray-with-fallback parallel execution
New `parallel.py`: tries `ray` first, and only falls back to a thread
pool if Ray can't be imported or initialized — matching the flow
diagram exactly, not just "there is a thread pool." Neither `ray` nor
its absence changes any eligibility/backtest/metric/selection rule
(§3.7 point 10); it only changes how candidates are dispatched. `ray`
is not installed in this sandbox, so every test run here exercised the
fallback path automatically, which is itself a demonstration of §3.7
point 5 ("shall not require Ray as a mandatory dependency").

### §4.4 MLflow
New `mlflow_logging.py`: wraps each training/inference run in an
MLflow run (tags: segment, process, algorithm, rule_version; params:
hyperparameters; metrics: the six-metric set, plus `holdout_`-prefixed
holdout metrics) when `config.json`'s `mlflow.enabled` is true and
`mlflow` is importable — else it's a silent no-op. The existing
per-segment/process/algorithm/window JSON log (`AlgorithmModule.log()`)
is unchanged and keeps running unconditionally, per §4.4's own framing
of it as "a human-readable supplement, not the primary tracking
mechanism." `mlflow` is not installed in this sandbox either, so this
too only ran its fallback path here.

### `config.json`
Added: `bias_threshold`, `consistency_min_pass_rate`, `holdout_periods`,
`window_search_enabled`, `backtest_step`, `seasonal_period`,
`hyperparameter_search.{enabled,budget,seed}`, `mlflow.{enabled,
tracking_uri,experiment_name}`, `challengers.{naive_1,seasonal_naive_30}`.
Removed: `backtest_window` (replaced by per-algorithm window search) and
`elimination_metric` (replaced by the fixed five-step §3.5 procedure,
which isn't a single configurable metric anymore).

## Still open

- **Neural-family architectures are a stand-in, not a faithful port.**
  All 19 "deep learning family" algorithms (#20–38) share one
  simplified MLP backbone (`_neural_template.py`) rather than 19
  distinct published architectures. This was already true and already
  documented before this pass; it's unchanged and is the single
  largest remaining gap against §5's "no algorithm's implementation is
  satisfied by a shared or simplified model class standing in for its
  specified architecture... mandatory, not an open design question."
  Closing it means an actual DL framework (PyTorch, or GluonTS for the
  DeepAR-family) and per-architecture implementations — out of scope
  for what this pass could responsibly attempt.
- **Cross-segment pooling** (DeepAR/DeepState/DeepVAR's multivariate
  target, TFT/TiDE's future-known exogenous feed) is still not wired
  up; those modules still report themselves ineligible in a
  single-segment harness, exactly as before. Confirmed still accurate
  against the demo (all report `ineligible: insufficient history` or
  their own pooling-not-wired-up reason).
- **Hyperparameter search and window search are not jointly
  optimized** — HPO runs once at the largest candidate window, then
  window selection reuses whatever hyperparameters that produced. A
  fully faithful reading of §3.2 would search both jointly per
  algorithm; this is a documented simplification.
- **The seasonal/exogenous order terms (SARIMAX's `P/D/Q/m`, Dynamic
  Regression's seasonal component) are held fixed from `config.json`
  while `p/d/q` are searched** — searching the full order space is a
  larger combinatorial problem than this pass's random+narrow strategy
  was built for; documented rather than silently done partially.
- **Trial pruning (§3.3.5) is trial-level, not fold-level.** A pruned
  trial still runs every backtest fold before being compared and
  discarded; true early-stopping mid-fold (à la ASHA) isn't
  implemented.
- **SAP connectivity, MLflow, and Ray remain unavailable in this
  sandbox** (no network egress to an MLflow tracking server or a Ray
  cluster, no SAP RFC/HANA access) — all three integration points are
  real, tested code paths, but only their fallback branches have
  actually executed here. `sap_stub.py` was already documented as
  field-mapping-only, unchanged by this pass.
- **Full-38-algorithm runs at the shipped `backtest_step: 4` are slow
  on a single machine** (minutes, not seconds, when window search and
  HPO are both on) — expected, and exactly what §3.7's Ray-cluster
  preference is for in production. `config.full_fast.json`-style
  configs (larger `backtest_step`, `window_search_enabled: false`) are
  useful for quick local iteration; the shipped `config.json` keeps
  the more faithful (slower) defaults.

## What was verified to actually run, end-to-end, in this pass

- `algorithms/base.py`'s new `evaluate()` against real fold data (six
  metrics, no NaNs/crashes) — caught and fixed a `pd.option_context`
  call using a pandas option (`mode.use_inf_as_na`) removed in the
  pandas version installed here, which had silently zeroed out *every*
  candidate's backtest before the fix (all 38 were reaching
  `eliminated_error` instead of being ranked).
- A 6-candidate subset (Ridge, Lasso, ARIMAX, RandomForest, XGBoost,
  Linear Regression) through `full_train` → `weekly_revalidate` →
  `daily_infer`, with and without `hyperparameter_search.enabled`,
  producing a real winner (XGBoost / Lasso depending on run),
  sensible elimination/ranking/consistency logs, and non-degenerate
  holdout metrics.
- All 38 candidates through `full_train` at a coarser `backtest_step`
  for tractable runtime: 15 correctly reported ineligible (the
  pooling/future-exog/1,000-observation-floor algorithms, exactly the
  ones expected to be ineligible at this demo's ~900-day history), 12
  eliminated on MASE ≥ 1, 0 eliminated on bias, and a clean six-metric
  ranked list for the rest.
- `parallel.py` and `mlflow_logging.py`'s fallback paths (`ray` and
  `mlflow` are both absent from this sandbox, so every run here
  exercised — and confirmed — the "unavailable" branch of each).
- `search.py` in isolation (Ridge, 8-trial budget): produces a
  materially better hyperparameter configuration than the
  `config.json` default, confirming the ranking-based selection picks
  the lowest-MASE trial rather than the last one tried.
- Every one of the 38 catalog entries instantiated and its
  `hyperparameter_search_space()` confirmed non-empty (all 38 now
  declare one).
- ARIMAX's and TCN's declared search spaces end-to-end through
  `search.py`, including catching and fixing the "all trials MASE ≥ 1"
  discard bug above (ARIMAX) and confirming real MASE improvement
  across narrowed trials (TCN: 1.65 → 0.87).
- Reproducibility: `full_train` run twice, independently, with a fixed
  `hyperparameter_search.seed` and four concurrently-searched
  candidates (Ridge/Lasso/XGBoost + baseline) — identical winner and
  identical hyperparameters both times, confirming the local-RNG fix.
- A 9-candidate mixed run (statistical + tree + neural-stand-in +
  state-space families together, HPO on) through all three cadences
  with no crashes and a plausible elimination/ranking outcome
  (Seasonal models without a seasonal term correctly eliminated on
  MASE against a strongly seasonal series; Lasso/TCN ranked at the
  top).
