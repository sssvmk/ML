# ESLII Ch. 12 - SVMs and Flexible Discriminants: Regression (Zillow) and Classification (Santander)

## 1. Project Overview
This project is an implementation of **Chapter 12 of The Elements of Statistical Learning (ESLII)**, focusing on Support Vector Machines (SVMs) and Flexible Discriminants. It benchmarks these methods on two distinct machine learning tasks:
- **Regression**: Predicting log-error using the Zillow Zestimate dataset.
- **Classification**: Predicting customer transactions using the Santander dataset.

The pipeline automates exploratory data analysis (EDA), feature engineering, hyper-parameter tuning (with internal cross-validation feature selection), and statistical cross-method comparison.

### Installation & Usage

    pip install -r requirements.txt
    python run_all.py --task regression     --data zillow.csv          --results results_reg
    python run_all.py --task classification --data santander_train.csv --results results_clf

Per task workflow: 
EDA -> split + FULL feature engineering -> variable ranking inside every CV fold -> the task's methods IN PARALLEL (live progress) -> comparison + winner.

Verify installation: `python tests/test_correctness.py`
Run one method alone: `python methods/pda.py --prepared results_clf/prepared --results results_clf`

---

## 2. Datasets Used

### Regression: Zillow Zestimate
- **Goal**: Predict the `logerror` of Zestimates.
- **Split**: Temporal split based on `transactiondate`. 
  - Train: `<= 2017-02-28`
  - Validation: `2017-03-01` to `2017-07-31`
  - Test: `>= 2017-08-01`

### Classification: Santander Customer Transaction
- **Goal**: Binary classification predicting customer transaction probabilities.
- **Split**: Stratified 80/10/10 split on the Target to maintain identical class ratios across train, val, and test sets.

---

## 3. Feature Engineering

### Regression (`zfeatures.py`)
- **Derived Features**: Added interaction terms (lat/lon quadratic, age, bath_per_bed, extra_rooms, tax ratios).
- **Zero-Fills**: Areas and counts where 'none means absent' are zero-filled.
- **Missing Indicators**: Created for columns with >5% missing data.
- **Transformations**: `log1p` on right-skewed non-negative numerics. Median imputation and winsorizing (0.5/99.5 percentiles) learned on Train.
- **Categoricals**: One-hot encoding (rare levels mapped to 'other').
- **Scaling**: All features are standardized. Target for fitting is clipped at 1st/99th percentiles (evaluation uses raw logerror).

### Classification (`sfeatures.py`)
- **Target Leakage Guard**: Features with univariate train AUC >= 0.9 are excluded.
- **Lean Set (200 cols)**: Winsorized (0.1/99.9 percentiles) -> Rank-to-Gaussian mapping (preserves the Gaussian assumption of discriminant methods) -> Standardized.
- **Rich Set (600 cols)**: Includes the Lean set plus:
  - **Frequency Features (200 cols)**: Log counts of raw values (leave-one-out for train rows).
  - **Weight of Evidence (WoE) Features (200 cols)**: 20-bin smoothed log-odds ratio of positives vs. negatives. Computed using 5-fold out-of-fold for train rows to prevent leakage.

---

## 4. Methods & 5. Loss Functions

Variable selection (`k` top-ranked variables) is treated as a hyperparameter and tuned alongside model parameters via random-search CV.

### Classification Methods
- **Support Vector Classifier (SVC)**: `scikit-learn` LinearSVC. 
  - *Loss*: Hinge loss + L2 penalty: `1/2 ||β||^2 + C * sum( [1 - y_i f(x_i)]_+ )`
- **Kernel SVM (SVM)**: `scikit-learn` SVC with RBF and Polynomial kernels.
  - *Loss*: Hinge loss in the kernel feature space (penalized RKHS).
- **Flexible Discriminant Analysis (FDA)**: Non-parametric regression of the class-indicator matrix on a Spline or Nystroem-RBF basis + Ridge, followed by LDA on the fitted values.
  - *Loss*: Optimal scoring: `min_θ,β sum( (θ(g_i) - h(x_i)'β)^2 )`
- **Penalized Discriminant Analysis (PDA)**: Ridge regression (`Ω = I`) of class-indicators on the features, followed by LDA on the fitted values.
  - *Loss*: Penalized optimal scoring: `min sum( (θ(g_i) - x_i'β)^2 ) + λ * β'Ωβ`
- **Mixture Discriminant Analysis (MDA)**: Custom EM loop where each class is a mixture of Gaussian subclasses with a common pooled covariance. Initialized with KMeans.
  - *Loss*: Negative mixture log-likelihood.

### Regression Methods
- **SVR Linear, Poly, RBF**: `scikit-learn` LinearSVR and SVR.
  - *Loss*: ε-insensitive loss + L2 penalty: `1/2 ||β||^2 + C * sum( max(0, |y_i - f(x_i)| - ε) )`. Target is standardized internally to make C and ε scale-free.

### Baselines
- **Classification**: Base-rate (predicting the train mean / majority class).
- **Regression**: Predict-the-train-mean, and Ridge regression on all engineered variables.

---

## 6. Evaluation Metrics

### Classification
- **Primary**: Test AUC (+- DeLong SE).
- **Secondary**: 
  - Test Error (+- SE) computed at a Youden's J threshold tuned on the validation set.
  - Test Log-Loss (+- SE). Note: SVC and SVM output raw margins; probabilities for log-loss are obtained via Platt scaling fitted on the validation set.

### Regression
- **Primary**: Test MSE (+- SE) evaluated on the raw `logerror`.
- **Secondary**: Test MAE, RMSE, test ε-insensitive loss, and the fraction of support vectors.

---

## 7. Cross-Method Comparison

Rigorous statistical testing is performed on the common held-out test set, and p-values are adjusted using the **Holm correction** for multiple comparisons against the winner.
- **Regression**: 
  - **Diebold-Mariano test**: Assesses per-point squared errors (respects time-ordered rows with a Newey-West long-run variance).
  - **Paired t-test**: Assesses per-point squared errors.
- **Classification**: 
  - **DeLong test**: Compares AUCs.
  - **McNemar test**: Compares classification error.
  - **Paired t-test**: Compares per-point log-loss.

---

## 8. Results & Winning Method

### Classification Winner: `svc`
- **Test AUC**: 0.88544 ± 0.00393
- **Test Error**: 0.17990 ± 0.00272
- **Test Log-Loss**: 0.21381 ± 0.00403
- **Variables Used (k)**: 600
- **Ties**: `svm` (Kernel SVM) tied with the winner (not significantly worse, Holm-adjusted DeLong p > 0.05). `pda` achieved the lowest absolute error (0.17575) but slightly lower AUC.

### Regression Winner: `svr_linear`
- **Test MSE**: 0.041273 ± 0.003111
- **Test MAE**: 0.07380
- **Variables Used (k)**: 60
- **Ties**: `svr_poly` (Polynomial SVR) tied with the winner (Holm-adjusted DM p > 0.05). The linear SVR significantly beat the predict-the-mean baseline.

---

## 9. Conclusion
The results indicate that despite the theoretical power of highly flexible methods (Kernel SVMs, MDA, FDA), simpler linear architectures (`svc` and `svr_linear`) with strong, regularized penalties proved to be the winners. Given the rich feature engineering (WoE, RankGauss, missing indicators, interaction terms), linear methods were sufficient to capture the signal while remaining robust against overfitting, outperforming baselines and matching or exceeding complex kernel models.