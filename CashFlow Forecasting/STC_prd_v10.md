# Cash Flow Forecasting PRD

## 1. Overview
Builds a cash flow forecast from aggregated vendor invoices, vendor
payments through bank, customer invoices, and customer payments received.
Forecasting runs per entity × currency × cash-flow-direction (AR/AP)
segment across 38 candidate algorithms, ranked via rolling backtest on
six metrics. Input is pluggable behind a canonical schema; a model
registry decouples training, weekly revalidation, and daily inference.
Everything is configurable; several business and architecture decisions
remain open.

Even though forecasting runs per entity × currency × cash-flow-direction
segment, there is no restriction on the number of trained model instances
per algorithm. If an algorithm supports pooling, a single model instance
may cover multiple entities, currencies, and cash-flow directions at
once.



## 2. Input Datasets
The logic is broken down into **four distinct datasets**, aggregated by:
1. **Date** (Net Due Date, Cleared Date, or Bank Value Date)
2. **Company Code** (`BUKRS`)
3. **Company Currency** (Local Currency)

*("Vendor payment received in bank" is treated as "Vendor payments made
from bank" — bank outflows — the AP side of cash leaving the accounts.)*

**Modeling grain:** every segment is one (Company Code × Currency) pair —
confirmed during review. Statistical tests, algorithm eligibility,
backtesting, and the model registry all operate at this grain, subject to
the pooling note above.

---


### 2.1 Dataset 1: Customer Invoices (Expected & Cleared) — AR subledger

**Source Tables:** `BSAD` (Cleared), `BSID` (Open)
* **Indicator:** `SHKZG` = 'S' (Debit)
* **Amount Field:** `DMBTR` (Amount in Local Currency)
* **Currency Field:** `HWAER` (local currency)

```sql
-- ACTUALS (Historical Cleared Customer Invoices)
SELECT
    'AR_Actual_Cleared' AS Status,
    BUKRS               AS Company_Code,
    HWAER               AS Company_Currency,
    AUGDT               AS Cash_Flow_Date,
    SUM(DMBTR)          AS Amount_Local_Currency
FROM BSAD
WHERE SHKZG = 'S'
  AND AUGDT IS NOT NULL
GROUP BY BUKRS, HWAER, AUGDT

UNION ALL

-- FORECAST (Expected Unpaid Customer Invoices)
SELECT
    'AR_Expected_Unpaid' AS Status,
    BUKRS                AS Company_Code,
    HWAER                AS Company_Currency,
    CASE
        WHEN ZBD3T > 0 THEN (ZFBDT + ZBD3T)
        WHEN ZBD2T > 0 THEN (ZFBDT + ZBD2T)
        ELSE (ZFBDT + ZBD1T)
    END                  AS Cash_Flow_Date,
    SUM(DMBTR)           AS Amount_Local_Currency
FROM BSID
WHERE SHKZG = 'S'
GROUP BY BUKRS, HWAER,
    CASE
        WHEN ZBD3T > 0 THEN (ZFBDT + ZBD3T)
        WHEN ZBD2T > 0 THEN (ZFBDT + ZBD2T)
        ELSE (ZFBDT + ZBD1T)
    END;
```

**Resolved:** `NETDT` is not populated at this landscape — net due date
is recomputed via the `ZFBDT + ZBD1T/2T/3T` cascade above. Kept swappable
via config (`due_date.method: zbd_cascade | netdt_field`).

### 2.2 Dataset 2: Customer Payments Received in Bank (Bank Inflows)

**Source Tables:** `FEBKO` & `FEBEP` (EBS) only. In-House Cash (`BKKIT`)
is explicitly NOT implemented at this landscape.

* **EBS Indicator:** `FEBEP-EPVOZ` = '+' (Incoming cash)
* **Currency field:** `WAERS` on `FEBKO` — `FEBKO` has no `HWAER` field,
  so `WAERS` (document/transaction currency) is the only currency
  available on the bank side; not necessarily identical to `HWAER` on the
  AR/AP side, and this has not been validated.

```sql
SELECT
    'Bank_Inflow_EBS'   AS Status,
    A.BUKRS             AS Company_Code,
    A.WAERS             AS Company_Currency,
    B.VALUT             AS Cash_Flow_Date,
    SUM(B.KWERT)        AS Amount_Local_Currency
FROM FEBKO A
JOIN FEBEP B ON A.KUKEY = B.KUKEY
WHERE B.EPVOZ = '+'
GROUP BY A.BUKRS, A.WAERS, B.VALUT;
```

Bank statement postings are final cash events — no forecast/actual
duality, so this dataset does not pass through the item-level ledger
described in Section 4.1.

### 2.3 Dataset 3: Vendor Invoices (Expected & Cleared) — AP subledger

**Source Tables:** `BSAK` (Cleared), `BSIK` (Open)
* **Indicator:** `SHKZG` = 'H' (Credit)
* **Amount Field:** `DMBTR`
* **Currency Field:** `HWAER`
* **Forecast Filter:** Exclude payment blocks (`ZLSPR` != '')

```sql
SELECT
    'AP_Actual_Cleared' AS Status,
    BUKRS               AS Company_Code,
    HWAER               AS Company_Currency,
    AUGDT               AS Cash_Flow_Date,
    SUM(DMBTR)          AS Amount_Local_Currency
FROM BSAK
WHERE SHKZG = 'H'
  AND AUGDT IS NOT NULL
GROUP BY BUKRS, HWAER, AUGDT

UNION ALL

SELECT
    'AP_Expected_Unpaid' AS Status,
    BUKRS                AS Company_Code,
    HWAER                AS Company_Currency,
    CASE
        WHEN ZBD3T > 0 THEN (ZFBDT + ZBD3T)
        WHEN ZBD2T > 0 THEN (ZFBDT + ZBD2T)
        ELSE (ZFBDT + ZBD1T)
    END                  AS Cash_Flow_Date,
    SUM(DMBTR)           AS Amount_Local_Currency
FROM BSIK
WHERE SHKZG = 'H'
  AND ZLSPR = ''
GROUP BY BUKRS, HWAER,
    CASE
        WHEN ZBD3T > 0 THEN (ZFBDT + ZBD3T)
        WHEN ZBD2T > 0 THEN (ZFBDT + ZBD2T)
        ELSE (ZFBDT + ZBD1T)
    END;
```

### 2.4 Dataset 4: Vendor Payments Made from Bank (Bank Outflows)

Same resolution as Dataset 2: EBS only, `WAERS` as currency, no IHC.

```sql
SELECT
    'Bank_Outflow_EBS'  AS Status,
    A.BUKRS             AS Company_Code,
    A.WAERS             AS Company_Currency,
    B.VALUT             AS Cash_Flow_Date,
    SUM(B.KWERT)        AS Amount_Local_Currency
FROM FEBKO A
JOIN FEBEP B ON A.KUKEY = B.KUKEY
WHERE B.EPVOZ = '-'
GROUP BY A.BUKRS, A.WAERS, B.VALUT;
```

---

## 3. Forecasting & Algorithm Selection Methodology

### 3.1 Variable Assignment & Data Testing
* **AR Forecasting Process:** Endogenous = Customer Payments Received in
  Bank (Dataset 2). Exogenous = Customer Invoices (Dataset 1).
* **AP Forecasting Process:** Endogenous = Vendor Payments Made from
  Bank (Dataset 4). Exogenous = Vendor Invoices (Dataset 3).

### 3.2 Execution & Rolling Backtesting

Qualification runs in two steps before training, followed by training and
a separate model-comparison stage after:

1. **Universal pre-fit data validation** — applied identically to all 38
   algorithms: target exists and is numeric; timestamps valid, correctly
   ordered, and de-duplicated; missingness checked; sufficient temporal
   coverage; no future-data leakage; exogenous variables aligned to
   timestamp; currency/entity consistency; sufficient observations.
2. **Algorithm-specific necessary-and-sufficient eligibility** — a
   A genuine pre-fit eligibility gate exists only where violating the condition makes the algorithm mechanically impossible to fit, prevents the required input structure from being constructed, or makes its core estimation invalid (see table below). Required-observation thresholds are also hard pre-fit gates: if the available observations do not satisfy the algorithm-specific minimum, that algorithm does not proceed to training for that run.

   Where no genuine pre-fit eligibility condition exists, the algorithm proceeds to training once its required inputs and observations are available; no additional eligibility gate should be invented.

   Diagnostic statistics such as ADF, KPSS, ACF, PACF, CCF, Granger, STL, CUSUM, Chow, VIF, correlation, variance, trend/seasonality strength, and cointegration are valid and useful, but they generally inform model configuration and feature preparation (for example, differencing order, seasonal period, lag inclusion, feature selection, or levels versus differences). They should not independently gate algorithm eligibility unless the resulting condition creates a genuine mechanical or mathematical impossibility for the specified model.

   Residual-based diagnostics such as residual autocorrelation, calibration, and Ljung-Box, together with the six Section 3.4 metrics (sMAPE, WAPE, MAE, RMSE, MASE, and Bias %), require a fitted model and therefore belong to the subsequent model evaluation and comparison stage, not the pre-fit eligibility stage.

   If the eligibility check disqualifies an algorithm for a given segment, the system updates that algorithm's configuration to reflect its
   applicability with respect to that data. Validation results, evaluation
   outputs, and visualizations are logged to a config-defined folder for
   each algorithm — mandatory, for human review.

**Per-algorithm necessary-and-sufficient eligibility condition and
required observations:**

| # | Algorithm | Pre-fit eligibility condition | Required observations |
|---|---|---|---|
| 1 | ARIMAX | Data must support the specified differencing/model specification; stationarity diagnostics (ADF/KPSS) are used to determine/configure `d`, not as an independent eligibility gate | N ≥ max(50, 10×(p+q+k+1)) |
| 2 | SARIMAX | Data must support the specified seasonal/differencing/model specification; ADF/KPSS and seasonal diagnostics inform configuration, not eligibility | N ≥ max(100, 10×(p+q+P+Q+k+1)) |
| 3 | VARMAX | Multivariate target required; model must be estimable for the specified dimensions/lags. Stationarity/cointegration diagnostics inform levels vs. differences, not automatic eligibility | N ≥ max(100, 10×m²×p) |
| 4 | Dynamic Regression | Target, exogenous variables and specified ARMA-error structure must be constructible; stationarity diagnostics inform differencing/configuration | N ≥ max(50, 10×(k+p+q+1)) |
| 5 | State Space Models | None beyond valid model specification and constructible inputs; state dimension is a configuration choice | N ≥ max(50, 10×state_dimension) |
| 6 | Structural Time Series | None beyond valid model specification and constructible inputs; component selection is a configuration choice | N ≥ max(100, 10×component_count) |
| 7 | Prophet | None beyond valid timestamp/target data and sufficient observations | N ≥ 50 |
| 8 | Linear Regression | Design matrix must be constructible and estimable; rank/collinearity checks are implementation diagnostics | N ≥ max(50, 10×(k+1)) |
| 9 | Polynomial Regression | Expanded design matrix must be constructible and estimable; rank/collinearity checks are implementation diagnostics | N ≥ 10×effective_feature_count |
| 10 | Ridge | None beyond valid feature/target data; regularization handles rank deficiency | N ≥ max(50, k) |
| 11 | Lasso | None beyond valid feature/target data | N ≥ max(50, 0.5×k) |
| 12 | ElasticNet | None beyond valid feature/target data | N ≥ max(50, k) |
| 13 | Random Forest | None beyond valid feature/target data | N ≥ 5×2^max_depth |
| 14 | Extra Trees | None beyond valid feature/target data | N ≥ 5×2^max_depth |
| 15 | XGBoost | None beyond valid feature/target data and constructible feature matrix | N ≥ max(200, 20×k) |
| 16 | LightGBM | None beyond valid feature/target data and constructible feature matrix | N ≥ max(500, 20×k) |
| 17 | CatBoost | None beyond valid feature/target data and constructible feature matrix | N ≥ max(200, 20×k) |
| 18 | SVR | None; feature scaling is preprocessing/configuration, not an eligibility gate | 100 ≤ N ≤ compute_ceiling |
| 19 | KNN Regression | None; trend/extrapolation behavior is a modeling consideration, not an eligibility gate | N ≥ max(200, 20×dims×k_neighbors) |
| 20 | RNN | None beyond constructible sequence windows | N ≥ lookback+horizon+500 |
| 21 | LSTM | None beyond constructible sequence windows | N ≥ lookback+horizon+500 |
| 22 | GRU | None beyond constructible sequence windows | N ≥ lookback+horizon+500 |
| 23 | TCN | History must be sufficient to construct the required receptive-field input | N ≥ receptive_field+horizon+500 |
| 24 | WaveNet | History must be sufficient to construct the required receptive-field input | N ≥ max(1,000, receptive_field+horizon+500) |
| 25 | DeepAR | Multiple series required if pooling is intrinsic to the implementation | Pooled N ≥ 1,000 |
| 26 | DeepState | Multiple series required if pooling is intrinsic to the implementation | Pooled N ≥ 1,000 |
| 27 | DeepVAR | Multivariate target required; endogenous dimensions ≥ 2 | Pooled N ≥ 1,000 |
| 28 | TFT | Required input features/covariates must be available; future-known covariates are required only when specified by the model configuration | N ≥ max(1,000, context+horizon+500) |
| 29 | TiDE | Required input features/covariates must be available for the configured forecast setup | N ≥ max(1,000, context+horizon+500) |
| 30 | TSMixer | None beyond constructible multivariate/time-series input | N ≥ max(1,000, context+horizon+500) |
| 31 | TimesNet | None beyond constructible time-series input | N ≥ max(1,000, context+horizon+500) |
| 32 | iTransformer | None beyond constructible multivariate/time-series input | N ≥ max(1,000, context+horizon+500) |
| 33 | Informer | None beyond constructible sequence input | N ≥ max(1,000, context+horizon+500) |
| 34 | Autoformer | None beyond constructible sequence input | N ≥ max(1,000, context+horizon+500) |
| 35 | FEDformer | None beyond constructible sequence input | N ≥ max(1,000, context+horizon+500) |
| 36 | ETSformer | None beyond constructible sequence input | N ≥ max(1,000, context+horizon+500) |
| 37 | N-BEATS | None beyond constructible sequence windows | N ≥ lookback+horizon+500 |
| 38 | N-HiTS | None beyond constructible sequence windows | N ≥ lookback+horizon+500 |
| 39 |TimesFM 3.0| None beyond constructible sequence windows | N ≥ lookback+horizon+500 |
`k, p, q, P, Q, m, lookback, context, receptive_field, state_dimension,
component_count` are per-algorithm configuration values, not fixed
constants. Where an algorithm's required observations are given as a
formula alongside a fixed floor (`max(...)`), the algorithm needs
whichever value is higher. **15 of 38 algorithms (#1–4, 8–9, 18, 19,
23–29) have a real eligibility condition; the remaining 23 have none —
they are gated only by required observations.**

**Diagnostics that inform configuration, not eligibility** (computed
pre-fit, never gate an algorithm):

| Diagnostic | Applies to (algorithm #s) | Drives |
|---|---|---|
| ADF, KPSS | 1, 2, 3, 4, 6 | Differencing order `d` (and seasonal `D` for #2) |
| ACF, PACF | 1, 2, 4 | AR order `p`, MA order `q` |
| CCF | 1, 2, 4, 8 | Which exogenous lag(s) to include |
| Granger causality | 1, 2, 3, 4, 8 | Whether to include the exogenous variable at all |
| STL (trend/seasonality strength) | 2, 6, 7, 31, 37, 38 | Seasonal order, structural components, seasonality config, period selection |
| CUSUM, Chow | 1, 2, 4, 6 | Whether to truncate/segment the training window |
| VIF, correlation | 4, 8, 10, 11, 12 | Which features to drop or regularization strength |
| Feature variance | 8, 9, 10, 11, 12 | Whether to prune a near-constant feature |
| Cointegration | 3, 27 | Whether to model in levels vs. differences |

Post-fit diagnostics (residual autocorrelation, calibration, Ljung-Box)
and the six Section 3.4 metrics require a fitted model and belong to
model comparison, not eligibility.

**Pipeline flow:** 38 Algorithms → Universal Data Validation → Algorithm
Eligibility Check → Training → Rolling Backtest → Metrics → Post-fit
Diagnostics (where applicable) → Model Comparison → Selected Model.

Use **rolling backtesting** so that every algorithm predicts the same
historical periods that are already known.

**Rolling window parameters:** the training window length is a
per-algorithm searchable hyperparameter, sized to meet or exceed that
algorithm's own required-observations threshold (table above), up to
the full four years of available history — not a single fixed value
applied uniformly across all 38 algorithms. The forward prediction
horizon defaults to 4 days. Step size between windows remains
configurable.

**Hyperparameter tuning:** each algorithm's hyperparameters are tuned
via a configurable search procedure (grid, random, or Bayesian) prior to
final training, scored against the same rolling-backtest scheme used for
algorithm selection. A held-out period at the end of each segment's
history, untouched by hyperparameter search or algorithm selection, is
reserved to report final performance. Gradient-boosted tree algorithms
(XGBoost, LightGBM, CatBoost) and the deep-learning-family architectures
use early stopping against a validation split during training. The gap
between training-set and validation-set performance is tracked per
algorithm and rolling window as an overfitting signal.

### 3.3 Hyperparameter Search and Optimization

The system shall perform algorithm-specific hyperparameter optimization for each eligible forecasting algorithm before final model comparison. Hyperparameter optimization shall identify the best configuration for an individual algorithm; it shall not be used to determine which algorithm is selected for production.

#### 3.3.1 Search-Space Definition

For each algorithm, the system shall maintain an algorithm-specific hyperparameter search space containing:

- Tunable hyperparameters relevant to the algorithm.
- Valid ranges or distributions for each hyperparameter.
- Conditional dependencies between hyperparameters where applicable.
- Algorithm-specific constraints required to produce a valid model.
- Data-dependent limits based on available observations, feature count, forecast horizon, lookback/context length, and model complexity.

The system shall not apply a common hyperparameter search space across all algorithms.

#### 3.3.2 Initial Search

The system shall perform an initial randomized search over the valid hyperparameter space to explore a broad range of configurations.

The initial search shall:

1. Generate valid hyperparameter configurations from the algorithm-specific search space.
2. Reject configurations that violate algorithm or data constraints before model fitting.
3. Train each valid configuration using the defined rolling-backtesting procedure.
4. Calculate the six evaluation metrics defined in Section 3.4:
   - sMAPE
   - WAPE
   - MAE
   - RMSE
   - MASE
   - Bias %
5. Record the configuration, fold-level results, aggregate metrics, training status, and execution metadata.

#### 3.3.3 Adaptive Optimization

Following the initial exploration, the system shall use an adaptive/Bayesian optimization strategy to identify promising hyperparameter regions and iteratively generate subsequent configurations.

The optimizer shall use the results of previously evaluated configurations to select subsequent configurations rather than exhaustively evaluating every possible combination.

The optimization process shall continue until the configured search budget or stopping criterion is reached.

#### 3.3.4 Rolling Backtesting During Hyperparameter Optimization

Hyperparameter configurations shall be evaluated using the same rolling-origin backtesting methodology defined for model evaluation.

For each hyperparameter configuration:

1. Train the algorithm independently on each applicable backtesting fold.
2. Generate predictions for the corresponding validation horizon.
3. Calculate the six defined evaluation metrics.
4. Aggregate the fold-level results.
5. Apply the configured hyperparameter-selection logic.

Hyperparameter optimization shall not use random train/test splitting for time-series data.

#### 3.3.5 Trial Pruning

The system shall support early termination of hyperparameter trials that are unlikely to produce a competitive configuration.

A trial may be pruned when its intermediate backtesting performance is demonstrably inferior to the configured optimization threshold or to the performance of previously evaluated configurations.

For algorithms that support iterative training, including gradient-boosting and deep-learning algorithms, the system shall additionally support training-level early stopping where applicable.

Pruned trials shall be recorded with their status and termination reason.

#### 3.3.6 Hyperparameter Selection

The best hyperparameter configuration for an algorithm shall be selected using the same evaluation principles defined for model comparison, rather than optimizing a single metric such as MAE alone.

The selection process shall consider:

- MASE-based elimination rules.
- Bias-threshold elimination rules.
- The six-metric rank-sum defined in Section 3.5.

The selected configuration shall be the best eligible configuration for that algorithm according to the defined selection rules.

#### 3.3.7 Search Budget

The hyperparameter search budget shall be configurable and may vary by algorithm family.

The configuration shall support:

- Maximum number of hyperparameter trials.
- Maximum training time.
- Maximum number of backtesting folds evaluated per trial, where applicable.
- Early-stopping criteria.
- Trial-pruning criteria.

More computationally expensive algorithms may use a smaller search budget, while less expensive algorithms may use a larger search budget.

#### 3.3.8 Data-Dependent Search Space

The system shall dynamically constrain the hyperparameter search space based on the characteristics of the available dataset.

The search-space generation shall consider, at minimum:

- Number of observations.
- Number of target and exogenous variables.
- Forecast horizon.
- Available historical context.
- Number of features.
- Algorithm-specific minimum observation requirements.
- Computational constraints.

The system shall not generate configurations that require more historical observations than are available for the applicable training/backtesting configuration.

#### 3.3.9 Separation of Hyperparameter Optimization and Algorithm Selection

Hyperparameter optimization shall operate independently for each eligible algorithm.

The process shall therefore be:

1. Determine algorithm eligibility using the pre-fit eligibility rules.
2. Determine whether the algorithm satisfies its required-observation threshold.
3. Generate the algorithm-specific hyperparameter search space.
4. Optimize the hyperparameters using rolling backtesting.
5. Select the best configuration for that algorithm.
6. Evaluate the selected configuration on the held-out test set.
7. Compare the resulting algorithms using the final algorithm-selection rules defined in Section 3.5.

Hyperparameter optimization shall answer:

> "What is the best configuration for this algorithm?"

The final model-comparison process shall answer:

> "Which eligible algorithm should be selected based on the defined evaluation criteria?"

The hyperparameter optimization process shall not introduce an additional algorithm-ranking or selection criterion outside the rules defined in Section 3.5.

#### 3.3.10 Experiment Tracking

Every hyperparameter trial shall be tracked in MLflow, including:

- Algorithm name and version.
- Hyperparameter configuration.
- Search-space version.
- Backtesting configuration.
- Fold-level metrics.
- Aggregate metrics.
- Trial status.
- Pruning or early-stopping reason, where applicable.
- Training duration.
- Dataset/version identifier.
- Code/version identifier.
- Final selected configuration.

The selected hyperparameter configuration shall be registered as part of the model lifecycle defined in the MLflow model-management requirements.

### 3.4 Evaluation Metrics
For every algorithm in each rolling period, calculate: sMAPE, WAPE, MAE,
RMSE, MASE, Bias %. Combine results across all periods into an overall
performance profile per algorithm.

### 3.5 Elimination & Ranking
1. **Eliminate against Baseline (MASE):** eliminate algorithms that do
   not beat the baseline. If MASE is 1 or greater, the algorithm is not
   better than the chosen naive/seasonal-naive benchmark.
2. **Eliminate Unacceptable Bias:** eliminate algorithms with
   unacceptable bias against the configured bias threshold.
3. **Rank Remaining Algorithms:** lowest sMAPE/WAPE/MAE/RMSE/MASE gets
   the best rank; bias closest to zero gets the best rank.
4. **Combine Ranks:** the algorithm with the lowest combined rank across
   all six metrics becomes the statistical winner.
5. **Consistency Check:** confirm the winner performs consistently across
   rolling periods, not just well in a few and poorly in others.

**Baseline:** default is Seasonal-Naive-7, with Naive-1 and
Seasonal-Naive-30 run alongside as challengers. **Bias threshold:** fully
configurable, no hardcoded business/company value.

### 3.6 Logging and Final Output
For each process, log the results of all steps — including pre-fit test
results per algorithm, in a config-defined folder — and formally declare
the winning algorithm.

If no algorithm survives elimination, a rolling mean or rolling median of
the eligible candidate forecasts is used as the winner.

The winner is the algorithm (or fallback rolling mean/median) that
consistently produces the best combination of accuracy and low bias
across historical rolling forecast periods, after eliminating algorithms
that fail the baseline or business-bias requirements.

### 3.7 Parallel Training of Algorithms

The system shall train eligible forecasting algorithms in parallel to minimize overall training and evaluation execution time.

1. After completion of the pre-fit eligibility and required-observation checks, all eligible algorithms shall be submitted as independent training tasks for parallel execution.
2. **Ray shall be the default and preferred parallel-processing framework** for distributing and executing algorithm-training tasks across available compute resources.
3. If Ray is unavailable, cannot be initialized, or is not supported in the execution environment, the system shall automatically fall back to the **best available native or platform-supported parallel-processing mechanism**.
4. The fallback mechanism may include the execution environment's native multiprocessing, multithreading, distributed-computing, or other supported parallel-execution capability.
5. The system shall not require Ray as a mandatory dependency for model training.
6. The parallel execution framework shall dynamically manage available CPU, GPU, memory, and configured concurrency limits.
7. Each eligible algorithm shall execute as an independent training task with its own algorithm configuration, hyperparameters, training data, and rolling-backtesting process.
8. Failure of one algorithm-training task shall not prevent other eligible algorithms from continuing their training and evaluation.
9. Results from all completed algorithm-training tasks shall be consolidated before proceeding to final algorithm comparison and selection.
10. The use of Ray or any fallback parallel-processing mechanism shall not change the algorithm eligibility rules, hyperparameter-search methodology, rolling-backtesting methodology, six evaluation metrics, or final algorithm-selection rules.

**Execution priority:**

Ray  
↓  
If Ray unavailable → best available platform/native parallel processing  
↓  
Train eligible algorithms concurrently  
↓  
Consolidate results  
↓  
Final algorithm comparison and selection

---



---

## 4. Architecture — Data Module, Algorithm Module & the Contract Between Them

The pipeline is two independently owned modules. The canonical contract
in 4.2 is the *only* interface between them — neither module is allowed
any other point of contact.

### 4.1 Module boundary

**Data Module** (owns Section 2's four datasets and their source
adapters): SAP field mapping (`BUKRS`, `HWAER`/`WAERS`, `SHKZG`,
`DMBTR`/`KWERT`, the `ZFBDT+ZBDnT` cascade), currency/date resolution,
item-level ledger dedup for open→cleared transitions (`BUKRS+BELNR
+GJAHR+BUZEI`). Source adapters — SAP, a CSV test fixture, a synthetic
generator, a future ERP — each own their own field mappings and
transformation rules, which may change over time without the Algorithm
Module noticing. Raw source connectivity (e.g. SAP RFC/HANA extraction)
is out of scope for this system — a source module is expected to supply
data already conforming to the contract. The Data Module has no
knowledge of algorithm classes, hyperparameters, or eligibility
conditions.

**Algorithm Module** (owns Section 3 methodology and the Section 5
per-algorithm contract): the 38 algorithm implementations, universal
pre-fit validation, eligibility checks, hyperparameter search, training,
rolling backtest, metrics, model registry. It consumes only contract
rows (4.2) — it has no knowledge of `BUKRS`, EBS vs. IHC, SAP table
names, or any source-specific field.

### 4.2 The contract

A contract validator is the single gate every adapter's output must
clear before any algorithm sees it; a row that fails validation blocks
that segment/batch rather than reaching the Algorithm Module in a
malformed state. This validator, and the schema it enforces, is the only
place Data Module logic and Algorithm Module logic touch.

| Field | Type | Description | Produced by | Consumed by |
|---|---|---|---|---|
| `segment_id` | string | `company_code + currency + direction`, derived key | Data Module | Algorithm Module (grouping, backtesting) |
| `company_code` | string | `BUKRS` | Data Module | Both (partitioning) |
| `currency` | string | `HWAER` (AR/AP) or `WAERS` (bank) | Data Module | Algorithm Module |
| `date` | date | Resolved `Cash_Flow_Date` | Data Module | Algorithm Module (endog/exog alignment) |
| `series_role` | enum {endogenous, exogenous} | Per §3.1 variable assignment | Data Module | Algorithm Module |
| `dataset` | enum {AR_Actual_Cleared, AR_Expected_Unpaid, Bank_Inflow_EBS, AP_Actual_Cleared, AP_Expected_Unpaid, Bank_Outflow_EBS} | Which of the four sources | Data Module | Algorithm Module |
| `value` | numeric | `Amount_Local_Currency` | Data Module | Algorithm Module |
| `lineage` | object {source_adapter, rule_version, extraction_batch_id} | Provenance | Data Module | Logging only — no algorithm branches on lineage |

Guarantees run both directions:
- **Data → Algorithm:** rows arriving at the Algorithm Module are already
  deduplicated (item-level ledger, §4.1), correctly signed and
  aggregated, currency-labeled, and date-resolved. The Algorithm Module
  never re-derives `Cash_Flow_Date` or re-applies `SHKZG` sign logic
  itself.
- **Algorithm → Data:** no algorithm ever reads or depends on Data Module
  internals (SAP table/field names). A new adapter, or a schema change
  inside an existing one, requires zero downstream changes as long as
  its output clears the validator.

Each registry entry records the `rule_version` its model was trained
under. A `rule_version` change for a segment — for example, a corrected
currency mapping in the Data Module — triggers retraining for that
segment at the next scheduled cadence, rather than continuing to serve a
model fit under superseded transformation rules (see Section 4.4).

### 4.3 Orchestration cadences

Three cadences share one model registry:
- **Full historical training** — one-time (or per a configured cadence),
  all segments, all history. Seeds the registry.
- **Weekly revalidation** — re-runs the backtest/ranking pipeline on the
  latest data and updates the registry only if a new winner is found.
- **Daily inference (6am)** — reads the registry only, fits the active
  model(s) on current history, forecasts the next horizon.

The registry stores either a single named algorithm, or — when the
fallback rolling mean/median is used — its combine method plus component
list.

### 4.4 Model Lifecycle Management

MLflow is the system of record for the full model lifecycle: experiment
tracking, model packaging, the model registry, and production
monitoring. Every training run — full training, weekly revalidation, and
each hyperparameter-search trial — is logged as an MLflow run, tagged
with segment, process, algorithm, hyperparameters, `rule_version`, and
the resulting metrics. The per-segment/process/algorithm/window folder
output described in Section 5 (item 8) remains available as a
human-readable supplement, not the primary tracking mechanism.

The model registry described in Section 4.3 is backed by MLflow's Model
Registry: each segment's active model is a registered model version,
transitioned between stages (Staging, Production, Archived) as training,
weekly revalidation, and rollback dictate.

Production forecast accuracy is monitored against the backtest metrics
recorded at training time. If a model promoted by weekly revalidation
underperforms its backtest expectation by more than a configurable
threshold once live, it is rolled back to the previously active
registered version, and the segment is flagged for investigation at the
next revalidation cycle.

---

## 5. Algorithm Module Contract

Each of the 38 algorithms is implemented as its own independent,
self-contained module, realizing the algorithm's own method as specified
in Section 3.2 — no algorithm's implementation is satisfied by a shared
or simplified model class standing in for its specified architecture.
This is a mandatory requirement, not an open design question. Each
module owns:

1. **Input data contract** — module should publish its data expectation. generally accepts an endogenous series and one or more
   exogenous series along date etc in the canonical shape (Section 4). if requires additional addional columns or features, module should publish its data contract. 
2. **Universal validation + eligibility check** — runs the universal
   pre-fit checks (3.2) and, if this algorithm has a genuine
   necessary-and-sufficient eligibility condition (15 of 38 do; the other
   23 have none), evaluates it. An algorithm with no eligibility
   condition always proceeds to training.
3. **Train** — fits the model with a defined, inspectable set of
   hyperparameters, selected via the hyperparameter-search procedure in
   Section 3.2.
4. **Post-fit diagnostics (model-comparison stage)** — where meaningful
   for its model class, checks the fitted model's residuals; not every
   algorithm requires this, and it never feeds back into eligibility.
5. **Save** — persists the trained model artifact and hyperparameters
   used.
6. **Load / inference interface** — reconstitutes a saved model and
   produces a forecast for a requested horizon, independent of training.
7. **Evaluation** — computes the six Section 3.4 metrics and produces
   required charts.
8. **Logging** — writes its own validation results, eligibility outcome,
   post-fit diagnostics (where applicable), evaluation metrics, charts,
   and model/hyperparameter artifacts as an MLflow run (Section 4.4),
   keyed by segment, process, algorithm, and rolling window. Mandatory,
   for human review.

Each module implements its own logic independently — statistical test
code is not assumed to be drawn from a shared central library by default;
each of the 38 modules stands on its own. This is a mandatory
requirement, not an open design question.

### 5.1 External interface surface

The eight responsibilities above are exposed as separate, independently
callable interfaces to the orchestrator — not steps bundled inside a
single `train()` call. The orchestrator can invoke any one of these on
its own (e.g. an eligibility read without training, or `evaluate()` on
an existing backtest window without re-running `infer()`):

| Interface | Called by | Input | Output |
|---|---|---|---|
| `describe_contract()` | Orchestrator, setup | — | This algorithm's data expectations (endogenous/exogenous shape, any extra required columns beyond Section 4's canonical fields) |
| `validate()` | Orchestrator, pre-fit stage | Canonical contract rows for the segment | Pass/fail against the universal checks (§3.2 step 1) |
| `check_eligibility()` | Orchestrator, pre-fit stage | Canonical contract rows for the segment | Eligible/ineligible + reason (meaningful for 15 of 38; the other 23 always return eligible) |
| `train()` | Orchestrator, full training / weekly revalidation | Canonical contract rows, hyperparameters | Trained model artifact |
| `save()` | Orchestrator | Trained model artifact | Persisted artifact + hyperparameters, addressable by segment/process/algorithm/window |
| `load()` | Orchestrator, daily inference | Segment/process/algorithm/window key | Reconstituted model, independent of training |
| `infer()` | Orchestrator, daily inference | Loaded model + latest canonical contract rows (endogenous history through current date, plus any exogenous — carried-forward or known-future) + `horizon` (number of future periods requested) | Forecast for exactly `horizon` periods, indexed by date |
| `diagnose()` | Orchestrator, model-comparison stage (where meaningful) | Fitted model residuals | Post-fit diagnostic report; never feeds back into eligibility |
| `evaluate()` | Orchestrator, model-comparison stage | Forecast vs. actuals | Six §3.4 metrics + required charts |
| `log()` | Orchestrator, every stage | That stage's outputs | Logged as an MLflow run (Section 4.4), keyed by segment/process/algorithm/window |

`infer()` is stateless with respect to data: `load()` only reconstitutes
the fitted model/parameters, it does not carry data forward from
training. Every call to `infer()` takes the current canonical contract
rows and an explicit `horizon` as arguments — an arbitrary number of
future periods, chosen by the caller — so the same loaded model can be
asked for a forecast of any length, independent of the backtest horizon,
without retraining or reloading.

The business-facing declaration in Sections 3.5–3.6 uses "winner"; within
this technical pipeline and module contract, "Selected Model" (or
"Champion Model") is the equivalent term, kept distinct for architecture
neutrality.

---

## Conclusion
The winner is declared per segment (Company Code × Currency ×
cash-flow-direction) and process (AR/AP), among the 38-algorithm roster
(Section 3.2), through the elimination and ranking rules in Section 3.5
and the consistency check across rolling windows. If no algorithm
survives elimination, a rolling mean or rolling median of the eligible
candidates is used instead. Roughly a third of the 38 have a pipeline
architecture precondition that must be resolved before they
can be fairly evaluated. Every threshold, window parameter, algorithm
roster entry, artifact location, and orchestration schedule is externally
configurable.
