# ESLII Ch. 16 - Ensemble Learning: ISLE, RuleFit, and Boosting as a Regularization Path

## 1. Project Overview
This project comprehensively implements and evaluates ensemble learning techniques as described in Chapter 16 of "The Elements of Statistical Learning" (ESLII). The focus is on analyzing the properties, sparsity, and predictive performance of post-processed tree and rule ensembles. 

The evaluation is conducted on two distinct machine learning tasks:
*   **Regression**: Zillow Zestimate dataset (predicting `logerror`).
*   **Classification**: Santander Customer Transaction Prediction dataset (binary classification).

The pipeline encompasses exploratory data analysis (EDA), extensive feature engineering, rigorous cross-validation (with a paired one-standard-error rule), parallelized model training, and robust statistical comparisons (Diebold-Mariano, DeLong, McNemar).

## 2. Algorithms Used
The project implements five distinct algorithms (found in `methods/`):

*   **`isle_gbm` (Importance Sampled Learning Ensemble)**: Implements ESLII Alg. 16.2. 
    *   *Stage 1*: Generates a dictionary of trees using Gradient Boosting with subsampling (`eta`), shrinkage (`nu`), and `J`-leaf trees.
    *   *Stage 2*: Applies L1-regularized post-processing (Lasso for regression, L1-logistic for classification) to select a sparse subset of trees from the dictionary.
*   **`isle_rf` (ISLE with Random Forest)**: Similar to `isle_gbm` but generates the initial tree dictionary using a Random Forest (shallow trees grown on subsamples with random variable subsets) before applying L1 post-processing.
*   **`rulefit` (Rule Ensemble)**: Implements ESLII 16.3.2. Every node of every tree in a boosted ensemble is extracted as a distinct rule. Optionally, winsorized linear terms are added. Lasso/L1-logistic post-processing is then used to find a sparse linear combination of these rules and variables.
*   **`fs_path` (Forward Stagewise Path View)**: Represents boosting as an epsilon-forward-stagewise path. It uses shrinkage boosting with a small learning rate. The optimal stopping point is chosen via CV (one-SE rule). It compares the L1 arc length of the boosting path to an exact lasso path on the same tree dictionary.
*   **`gbm_reference`**: The unprocessed, full gradient boosted ensemble (with `nu=0.1`). It serves as the baseline to evaluate the sparsity and performance gains (or losses) introduced by the ISLE and RuleFit post-processing techniques.

## 3. Loss Functions
*   **Regression**: Squared Error loss is used both for the initial boosting stages and the Lasso post-processing. The fit target is the `logerror` clipped at the 1st and 99th percentiles (to handle extreme outliers), though all evaluations are done on the raw `logerror`.
*   **Classification**: Log-loss (Binomial Deviance) is the primary loss function. The L1-logistic path is utilized during post-processing for classification algorithms.

## 4. Metrics Used for Evaluation
*   **Regression Metrics**: Mean Squared Error (MSE) - primary CV selection metric, Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Relative MSE versus a predict-the-mean baseline.
*   **Classification Metrics**: Area Under the ROC Curve (AUC) - primary CV selection metric, Error Rate (at Youden's J threshold), Log-loss, Brier score, and Expected Calibration Error (ECE).
*   **Statistical Tests**: 
    *   *Regression*: Diebold-Mariano test (with Harvey correction) and paired t-tests.
    *   *Classification*: DeLong's test (for AUC), McNemar's test (for error), and paired t-tests (for log-loss). 
    *   Holm's method is used to adjust p-values for multiple comparisons.

## 5. Datasets
*   **Classification (Santander)**: A binary classification task with 200 anonymized numerical features. The split is a stratified 80/10/10 (Train/Val/Test). A synthetic generator (`make_synthetic_santander.py`) is provided for smoke-testing, which simulates normally distributed and shifted variables.
*   **Regression (Zillow)**: Predicting the `logerror` of property Zestimates based on housing attributes (location, rooms, tax info, etc.). The split is temporal: Train (<= Feb 2017), Validation (Mar-Jul 2017), and Test (>= Aug 2017). A synthetic generator (`make_synthetic_zillow.py`) simulates realistic properties of real-estate data.

## 6. Feature Engineering
Feature engineering is strictly fitted on the Train split to avoid data leakage.
*   **Classification (`prep_clf.py`, `sfeatures.py`)**: Generates an expanded pool of 600 candidate variables.
    *   *Lean*: RankGauss transformation (winsorized at 0.1/99.9 percentiles mapped to a normal distribution).
    *   *Frequency*: Leave-one-out frequency encoding (log of counts).
    *   *Weight-of-Evidence (WoE)*: 20-bin out-of-fold smoothed log-odds ratio.
    *   *Leakage Guard*: Excludes any feature with a univariate train AUC >= 0.90.
*   **Regression (`prep_reg.py`, `zfeatures.py`)**: 
    *   Zero-fills "none means absent" columns (e.g., pools, garages).
    *   Creates missing-indicator columns.
    *   Computes derived features: property age, lat/lon quadratics, tax-to-sqft ratio, and land/structure ratios.
    *   Applies `log1p` transformation to heavily skewed positive numerics.
    *   One-hot encoding for low-cardinality categorical variables.
    *   Out-of-fold Target Encoding (smoothed mean of clipped target) for high-cardinality categorical codes (zip codes, city, zoning).

## 7. Results and Findings
Based on the comprehensive grid search and testing:

*   **Classification (Santander)**:
    *   **Winner**: `fs_path` achieved the highest test AUC (~0.8734) and lowest log-loss (~0.2265).
    *   `fs_path` retained all 600 dictionary elements, indicating the dataset requires dense combinations of variables.
    *   Sparse post-processed methods like `rulefit`, `isle_rf`, and `isle_gbm` performed significantly worse. `rulefit` struggled extensively (AUC ~0.50), effectively collapsing to predict the majority class (resulting in artificially "lowest error" but terrible AUC).
*   **Regression (Zillow)**:
    *   **Winner**: `fs_path` achieved the lowest test MSE (~0.04124).
    *   **Not Significantly Worse**: `isle_rf` and `gbm_reference` performed statistically on par with the winner.
    *   **Sparsity Insight**: `isle_rf` emerged as an incredibly efficient model. It matched the predictive power of the winner while being the sparsest model, retaining only **51 out of 150 dictionary elements**. 

## 8. Conclusion
The forward stagewise path (`fs_path`) consistently demonstrated the best predictive performance across both regression and classification, proving that shrinkage boosting is a highly effective, robust regularization technique. 

However, the behavior of sparse dictionary methods heavily depends on the nature of the data:
*   In the Zillow **regression** task, **ISLE with a Random Forest dictionary (`isle_rf`)** was highly successful. By post-processing the forest, it stripped away redundant trees, resulting in a highly sparse (51 trees) and interpretable model without sacrificing any accuracy compared to dense boosting. 
*   In the Santander **classification** task, the signal is notoriously distributed among many variables. Sparse methods (`rulefit`, `isle_gbm`) failed to capture the complexity, heavily underperforming compared to dense ensembles (`fs_path`, `gbm_reference`).

Overall, post-processing tree dictionaries (ISLE) can offer massive gains in model sparsity and execution efficiency (as seen in Regression), provided the underlying signal can be concisely captured. 

## Quickstart
```bash
pip install -r requirements.txt
python run_all.py --task regression     --data zillow.csv          --results results_reg
python run_all.py --task classification --data santander_train.csv --results results_clf
```
*Tip: Reduce runtime using `--n-configs-isle`, `--tune-rows`, and `--workers` parameters.*