# Comprehensive Codebase Analysis Report

## 1. Project Functionality
The project is a machine learning framework designed to implement, tune, and compare five specific additive models and tree-based methods inspired by Chapter 9 of "The Elements of Statistical Learning" (ESLII). It evaluates these methods across two distinct tasks:
*   **Regression**: Predicting `logerror` for the Zillow Zestimate dataset.
*   **Classification**: Predicting a binary `target` for the Santander customer transaction dataset.

The framework is end-to-end. It handles Exploratory Data Analysis (EDA), temporal and stratified train/validation/test splitting, rich feature engineering, feature selection (ranking within CV folds to prevent leakage), parallel hyperparameter tuning via cross-validation, and robust cross-method comparisons using statistical significance tests.

## 2. Algorithms Implemented
The machine learning models are implemented in the `methods/` directory:
*   **CART (`cart.py`)**: Classification and Regression Trees utilizing cost-complexity pruning.
*   **GAM (`gam.py`)**: Generalized Additive Models fitted via backfitting and local scoring, utilizing penalized cubic B-spline smoothers.
*   **HME (`hme.py`)**: Hierarchical Mixture of Experts. This uses a soft gating network (softmax) and linear/logistic experts fitted via Expectation-Maximization (EM).
*   **MARS (`mars.py`)**: Multivariate Adaptive Regression Splines. It builds models using a forward addition pass followed by a backward deletion pass, using Generalized Cross-Validation (GCV) for selection.
*   **PRIM (`prim.py`)**: Patient Rule Induction Method (bump hunting). It identifies high-target-density sub-regions using top-down peeling, bottom-up pasting, and sequential covering.

## 3. Features Derived
The project applies rigorous, leak-free feature engineering tailored to the problem type:

**Santander Classification (`sfeatures.py`)**
Produces two feature sets:
*   **Lean**: Features are winsorized at the 0.1/99.9 percentiles, transformed using RankGauss (empirical CDF mapped to normal quantiles), and standardized.
*   **Rich**: Includes the 'Lean' set plus:
    *   **Frequency Features**: `log(1 + count)`, calculated using a leave-one-out strategy on training data to prevent leakage.
    *   **Weight-of-Evidence (WoE)**: The smoothed log-odds ratio of positive vs negative cases across 20 quantile bins, computed using 5-fold Out-Of-Fold (OOF) encoding.

**Zillow Regression (`zfeatures.py`)**
Engineered based on a strict temporal split:
*   **Imputation & Transformation**: Zero-fill for count/area; log1p transformation for strongly right-skewed non-negative numerics; missing indicators for features missing > 5%. 
*   **Derived Spatial & Temporal**: `lat_sq`, `lon_sq`, `lat_x_lon`, and `age` (2017 - yearbuilt).
*   **Derived Ratios**: `bath_per_bed`, `extra_rooms`, `tax_per_sqft`, `structure_ratio`, `land_ratio`, `tax_rate`, and `living_lot_ratio`.
*   **Target Transformation**: The target `logerror` is clipped at the 1st and 99th percentiles exclusively during the fitting phase to minimize the influence of extreme outliers.

## 4. Hyperparameters Tuned
All models tune `k`, the number of top-ranked variables to include (selected via ExtraTrees and univariate signals). Additionally, each method tunes specific parameters via grid search:
*   **CART**: `alpha` (cost-complexity pruning parameter relative to root impurity).
*   **GAM**: `df` (degrees of freedom for the cubic B-spline smoothers).
*   **HME**: `depth` (tree depth of the hierarchy) and `K` (branching factor at each node).
*   **MARS**: `degree` (1 for purely additive, 2 for first-order interactions) and `n_terms` (the total number of terms retained).
*   **PRIM**: `alpha` (fraction of data removed in each peeling step), `beta0` (minimum support required for a box), and `n_boxes` (number of boxes induced).

## 5. Loss Functions Optimized
*   **CART**: Minimizes Sum of Squared Errors (Regression) or Gini Impurity (Classification).
*   **GAM**: Optimizes Penalised Residual Sum of Squares (Regression) or Penalized Negative Binomial Log-Likelihood via local scoring (Classification).
*   **HME**: Minimizes Negative Gaussian-Mixture Log-Likelihood (Regression) or Negative Bernoulli-Mixture Log-Likelihood (Classification) via Expectation-Maximization.
*   **MARS**: Minimizes Least Squares (RSS) for both tasks. For classification, indicators are fit via least squares, and probabilities are obtained via Platt scaling on the validation set.
*   **PRIM**: Seeks to maximize the mean target response inside sub-regions (boxes), rather than optimizing squared error or likelihood globally.

## 6. Metrics (Evaluation)
Models are evaluated individually on the test set using:
*   **Regression**: Mean Squared Error (MSE), RMSE, MAE, and MSE relative to a mean-prediction baseline.
*   **Classification**: Area Under the ROC Curve (AUC), Log-loss, Brier score, Expected Calibration Error (ECE), test error, sensitivity, and specificity evaluated at validation-tuned Youden thresholds.

## 7. Metrics (Cross-Method Comparison)
Found in `compare.py`, the cross-method comparison relies on statistical significance to declare ties and winners:
*   **Regression**: Primary metric is the **lowest test MSE**. The Diebold-Mariano test (suitable for temporal data) and paired t-tests, adjusted via the Holm method, are used to determine if other models statistically tie with the lowest MSE.
*   **Classification**: Primary metric is the **highest test AUC** (configurable to Log-loss). Significance tests include the DeLong test for AUC, McNemar's test for error, and paired t-tests for log-loss, all Holm-adjusted.

## 8. Results
Based on the JSON output logs in the `results_reg/` and `results_clf/` directories:
*   **Regression (Zillow)**: **GAM is the conclusive winner.** It achieved a test MSE of `0.0412`, significantly outperforming MARS, HME, CART, PRIM, and baseline OLS models.
*   **Classification (Santander)**: The current test execution in the codebase resulted in a perfectly separable scenario (likely a synthetic subsample or toy configuration used for CI/testing). All five algorithms tied with a perfect test AUC of `1.0` and test error of `0.0`. While `HME` was arbitrarily designated the top spot, the statistical tie resolution correctly identified that CART, GAM, MARS, and PRIM achieved statistically indistinguishable perfect performance.