# Boosting Methods Study (ESLII Chapter 10)

This repository contains a comprehensive pipeline for studying, tuning, and comparing various boosting algorithms as discussed in Chapter 10 of *Elements of Statistical Learning II* (ESLII). It implements an end-to-end machine learning orchestrator for evaluating multiple tree-based and boosting approaches on real-world and synthetic datasets.

## 1. Overall Project Functionality and Purpose
The project serves as an empirical study and replication of ESLII's boosting methods. It provides a robust, parallelized pipeline (`run_all.py`) that executes:
1. Exploratory Data Analysis (EDA).
2. Advanced feature engineering and data splitting (temporal for regression, stratified for classification).
3. Leakage-free variable ranking within Cross-Validation (CV) folds.
4. Distributed hyper-parameter tuning of boosting models using a 10-fold CV with a one-standard-error (One-SE) rule.
5. Final model training on the full training set, predicting on validation/test sets, and performing rigorous statistical comparisons to crown a "winner" among the evaluated methods.

It supports two distinct tasks:
* **Regression**: Predicting `logerror`.
* **Classification**: Binary classification on anonymous features.

## 2. Algorithms Implemented
The pipeline integrates multiple boosting implementations via Scikit-Learn, LightGBM, and XGBoost (located in `methods/`):
* **AdaBoost.M1** (`adaboost.py`): Classification only. Uses discrete SAMME with trees of depth $d$ as the weak learner.
* **Boosted Trees** (`boosted_trees.py`): Uses Scikit-Learn's GradientBoosting with no shrinkage (learning rate = 1), tuning the number of max leaf nodes ($J$).
* **Forward Stagewise Additive Modelling** (`forward_stagewise.py`): Uses Scikit-Learn's GradientBoosting with no shrinkage and **stumps** (depth=1) as the basis function (least-squares boosting for regression).
* **Gradient Boosting Machine (GBM)** (`gbm.py`): Uses Scikit-Learn's GradientBoosting with shrinkage ($\nu = 0.1$) and trees fit to the negative gradient. 
* **Shrinkage and Subsampling** (`shrinkage_subsampling.py`): Stochastic gradient boosting using Scikit-Learn, evaluating various shrinkage rates ($\nu \in (0, 1]$) and subsampling fractions ($\eta$).
* **LightGBM** (`lightgbm_boost.py`): Uses the `lightgbm` package. Implements gradient boosting on histogram-binned features with leaf-wise tree growth and $L2$ regularization on leaf values.
* **XGBoost** (`xgboost_boost.py`): Uses the `xgboost` package. Implements regularized gradient boosting optimized with a second-order Taylor expansion (histogram tree method).

## 3. Exploratory Data Analysis (EDA)
Comprehensive EDA is automatically performed and saved to `<results_dir>/eda/` before modeling:
* **Classification (`eda_clf.py`)**: Computes summary statistics strictly on the 80% train split to prevent data leakage. Analyzes feature skewness, excess kurtosis, percentage of unique values, and outlier percentages (using robust Z-scores > 4). Evaluates class balance and uses the Kolmogorov-Smirnov (KS) test to find the strongest single features separating the positive and negative classes. Also checks pairwise correlations.
* **Regression (`eda_reg.py`)**: Evaluates the `logerror` distribution to establish clip bounds (1st/99th percentiles) and IQR fences. Investigates target drift over time/splits (monthly aggregations). Computes missingness percentages, numeric feature skewness, Spearman correlations with the target, and identifies highly collinear pairs (Pearson > 0.9).

## 4. Feature Engineering
Feature engineering pipelines are isolated to the training split to prevent leakage, outputting engineered arrays for the models:
* **Classification (`sfeatures.py`, `prep_clf.py`)**: Stratified 80/10/10 split on the Target. Creates a pool of 600 candidate variables from the original 200:
  * **200 Lean features**: Winsorized at the 0.1/99.9 percentiles, rank-to-Gaussian transformed (using the train ECDF), and standardized.
  * **200 Frequency features**: Leave-one-out log counts (`log(1 + count)`) for train rows to prevent target leakage.
  * **200 Weight-of-Evidence (WoE) features**: Uses 20 train-quantile bins to calculate smoothed log-odds ratios, utilizing 5-fold out-of-fold calculations for the training set.
* **Regression (`zfeatures.py`, `prep_reg.py`)**: Temporal split (Train: <= Feb 2017, Val: Mar-Jul 2017, Test: >= Aug 2017).
  * Zero-fills specific count/area columns where NaN implies absence.
  * Creates missing indicator columns for features with > 5% missingness.
  * Generates derived ratios (e.g., age, tax per sqft, structure ratio) and lat/lon quadratics.
  * Applies `log1p` to strongly right-skewed non-negative features.
  * Median imputation and winsorizing at the 0.5/99.5 percentiles.
  * One-hot encodes categorical features (pooling rare levels into "other").
  * **Target Encoding**: `prep_reg.py` adds smoothed mean target encoding for 6 high-cardinality location/zoning codes (calculated out-of-fold on the training set). 
  * The fitting target is clipped at the 1st and 99th percentiles (evaluation uses the raw target).

## 5. Loss Functions
Each algorithm employs specific loss functions depending on the task:
* **AdaBoost**: Exponential loss ($\exp(-y f(x))$).
* **Boosted Trees**: Squared error (Regression); Binomial deviance or Exponential loss (Classification).
* **Forward Stagewise**: Squared error (Regression); Binomial deviance or Exponential loss (Classification).
* **GBM**: Squared error, Absolute error, or Huber loss (Regression); Binomial deviance (Classification).
* **Shrinkage/Subsampling**: Squared error (Regression); Binomial deviance (Classification).
* **LightGBM**: L2 (squared error), L1 (absolute), or Huber (Regression); Binary logistic loss (Classification).
* **XGBoost**: Squared error (`reg:squarederror` for Regression); Logistic loss (`binary:logistic` for Classification).

## 6. Evaluation Metrics
Evaluations are strictly calculated on held-out test predictions (`common.py`):
* **Regression**: Mean Squared Error (MSE) is the primary metric. Additional metrics include Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and MSE relative to a predict-the-mean baseline.
* **Classification**: Area Under the ROC Curve (AUC) is the primary winner metric. Also computes Log-Loss, classification Error (using a Youden threshold tuned on validation), Brier score, Expected Calibration Error (ECE), Sensitivity, Specificity, and DeLong standard errors.

## 7. Results Tracking and Generation
Results are generated in `results_clf/` or `results_reg/` depending on the task:
* **Global Overviews**: Generates a `comparison.csv` with all method metrics, a `run_all.log` tracking execution, and `winner.json`/`winner.txt` declaring the best model.
* **Global Plots**: ROC curves, Test MSE comparison, Error and Log-loss dot-plots.
* **Per-Method Results**: Each algorithm gets a subfolder containing:
  * Raw NumPy predictions (`test_pred.npy`, `val_pred.npy`).
  * Detailed metrics (`summary.json`, `test_metrics.json`, `validation_metrics.json`).
  * Diagnostic plots (`cv_curve.png`, `partial_dependence.png`, `variable_importance.png`, `variable_selection_curve.png`).
  * Extracted feature importance tables.

## 8. Cross-Method Evaluation Methodology
The evaluation methodology is designed for rigorous statistical comparison (`compare.py`):
* **Tuning**: A grid search over hyperparameters is evaluated using 10-fold CV on a tuning subsample. The optimal model is selected using a **One-Standard-Error (One-SE) rule** calculated on paired fold differences, penalizing unnecessary complexity.
* **Final Fit**: The chosen configuration is trained on all available training rows and evaluated on the full validation and test sets.
* **Statistical Comparison**: The absolute best method (lowest test MSE or highest test AUC) is crowned the "Winner".
  * **Regression**: The Diebold-Mariano test (time-ordered rows, Newey-West variance) and Paired t-tests are used to compare the winner's per-point squared errors against all other methods. 
  * **Classification**: DeLong's test (for AUC), McNemar's test (for error), and Paired t-tests (for log-loss) are used.
  * All p-values are adjusted for multiple comparisons using the Holm step-down method to identify methods that are "not significantly worse" than the winner.

## 9. Datasets Used
The project defaults to evaluating on two specific datasets:
1. **Santander (`santander_train.csv`)**: Used for the classification task. Contains an anonymized set of 200 numeric features and a binary `Target`.
2. **Zillow (`zillow.csv`)**: Used for the regression task. Predicts the `logerror` of real estate Zestimates using historical housing features and transaction dates.
3. **Synthetic Datasets**: For testing and CI environments, scripts in `tests/` (`make_synthetic_santander.py` and `make_synthetic_zillow.py`) generate synthetic datasets that perfectly mimic the schemas and data types of the real datasets to validate the pipeline's correctness.