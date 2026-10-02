# ESLII Ch. 6 kernel smoothing methods - regression (Zillow) and classification (Santander)

    pip install -r requirements.txt
    python run_all.py --task regression     --data zillow.csv          --results results_reg
    python run_all.py --task classification --data santander_train.csv --results results_clf

Per task: EDA -> split + features -> 5 methods IN PARALLEL (live progress) -> comparison + winner (winner.txt / winner.json / comparison*.png).

regression     (temporal split; fit target = clipped logerror; metrics on raw logerror): nw_regression, local_polynomial (1-D, chosen predictor),
               structured_local_regression (varying-coefficient, top-6 predictors), local_likelihood_regression (Gaussian), rbf_network (top-6 predictors)
               winner = lowest test MSE +- SE; paired Diebold-Mariano / t tests, Holm-adjusted
classification (stratified 80/10/10): nw_classification, local_logistic, kernel_density_classifier (top-8 features),
               naive_bayes, gaussian_mixture_classifier (all 200 features)
               winner = lowest test log-loss +- SE; error +- SE and AUC +- SE reported; paired t / McNemar / DeLong, Holm-adjusted

Compute: kernel methods are memory-based. Smoothing parameters are tuned by 10-fold CV on a 30k-row subsample (--tune-rows); the final model uses ALL training rows and is
evaluated on the full validation and test sets. Spans are fractions of the reference set, so they transfer. Slowest: local_logistic and kernel_density_classifier.
Other flags: --methods ..., --workers N, --cv-folds, --one-se paired|plain, --predictor, --n-top-features, --nb-compare-rows (0 = skip), --skip-eda, --reuse-prepared.
One method alone: python methods/local_logistic.py --prepared results_clf/prepared --results results_clf
Verify numerics: python tests/test_correctness.py     Synthetic data: tests/make_synthetic_zillow.py, tests/make_synthetic_santander.py

## Codebase Analysis

### 1. Algorithms Used
The repository implements various kernel smoothing and local regression methods, separated into classification (Santander dataset) and regression (Zillow dataset) tasks:

**Classification Methods:**
1.  **Gaussian Mixture Classifier** (`gaussian_mixture_classifier.py`)
2.  **Kernel Density Classifier** (`kernel_density_classifier.py`)
3.  **Local Logistic** (`local_logistic.py`)
4.  **Naive Bayes** (`naive_bayes.py`)
5.  **Nadaraya-Watson Classification** (`nw_classification.py`)

**Regression Methods:**
1.  **Local Likelihood Regression** (`local_likelihood_regression.py`)
2.  **Local Polynomial** (`local_polynomial.py`)
3.  **Nadaraya-Watson Regression** (`nw_regression.py`)
4.  **RBF Network** (`rbf_network.py`)
5.  **Structured Local Regression** (`structured_local_regression.py`)

---

### 2. Loss Functions and Evaluation Metrics

#### Classification Methods
*   **Gaussian Mixture Classifier:**
    *   **Loss/Fit:** Per-class negative log-likelihood of a diagonal-covariance Gaussian mixture (fitted via EM).
    *   **Metrics:** Held-out log-likelihood per class, BIC/AIC, test error $\pm$ SE, log-loss, AUC, calibration. Model complexity $M$ (components) is chosen by CV log-loss (one-SE rule).
*   **Kernel Density Classifier:**
    *   **Loss/Fit:** No single loss. It utilizes a Gaussian product-free isotropic kernel density estimation (KDE) for each class. Bandwidth $h$ is chosen by CV log-loss.
    *   **Metrics:** Test error $\pm$ SE, log-loss, AUC, calibration, held-out density log-likelihood per class, CV error.
*   **Local Logistic:**
    *   **Loss/Fit:** Kernel-weighted negative binomial log-likelihood, maximized at every query point (using batched IRLS with ridge-penalized slopes). Uses a tri-cube kernel with k-NN bandwidth.
    *   **Metrics:** CV deviance (log-loss) vs. span, test error $\pm$ SE, AUC, calibration, pointwise standard errors of the fitted logit.
*   **Naive Bayes:**
    *   **Loss/Fit:** Class-conditional log-likelihood with independent features (each marginal feature distribution is a 1-D Gaussian kernel density).
    *   **Metrics:** Test error $\pm$ SE, log-loss, AUC, calibration, and an explicit comparison with logistic regression and GAM.
*   **Nadaraya-Watson Classification:**
    *   **Loss/Fit:** Kernel-weighted squared error on the class indicator. Fits a locally constant logit using a tri-cube kernel and k-NN bandwidth.
    *   **Metrics:** Test error $\pm$ SE (at a validation-tuned Youden threshold), log-loss, Brier score, AUC, calibration, CV error/log-loss/AUC vs. span.

#### Regression Methods
*   **Local Likelihood Regression:**
    *   **Loss/Fit:** Local negative log-likelihood for the Gaussian case (which is equivalent to local regression).
    *   **Metrics:** Test MSE $\pm$ SE, 10-fold CV MSE over (span, degree), LOO-CV, GCV, Cp, pointwise SE bands. Checks max absolute difference against local polynomial fit.
*   **Local Polynomial:**
    *   **Loss/Fit:** Kernel-weighted squared error to fit a local polynomial of degree $d$ (1, 2, or 3).
    *   **Metrics:** Test MSE $\pm$ SE, 10-fold CV MSE over (span, degree), LOO-CV, GCV, Cp, pointwise SE bands, boundary error compared to a locally constant fit.
*   **Nadaraya-Watson Regression:**
    *   **Loss/Fit:** Kernel-weighted squared error with a locally constant fit (degree 0).
    *   **Metrics:** Test MSE $\pm$ SE, 10-fold CV MSE vs. span/df, LOO-CV, GCV, $C_\lambda$, pointwise SE bands.
*   **RBF Network:**
    *   **Loss/Fit:** Least squares on radial basis functions. Centers are chosen via k-means (unsupervised), followed by a plain least squares fit for coefficients.
    *   **Metrics:** Test MSE $\pm$ SE, 10-fold CV MSE vs. number of basis functions and width. Also performs a non-convex check (optimizing centers, widths, and weights jointly via L-BFGS).
*   **Structured Local Regression:**
    *   **Loss/Fit:** Varying-coefficient model where the kernel acts only on a single conditioning variable $z$, fitting locally linear coefficients for the other predictors.
    *   **Metrics:** Test MSE $\pm$ SE, 10-fold CV MSE over (span, $M$, coefficient degree), GCV, LOO-CV.

---

### 3. Cross-Method Comparison
The orchestrator `run_all.py` coordinates the evaluation pipeline: EDA, feature prep, parallel method training, and finally comparison.
The actual comparison is implemented in `compare.py`:

*   **Regression Comparison:**
    *   The "winner" is determined by the lowest **test MSE of the raw logerror**.
    *   Statistical significance against the winner is tested using paired tests on per-point squared errors: **Diebold-Mariano test** and **paired t-test** (both Holm-adjusted for multiple comparisons).
    *   Baselines: A model predicting the training mean, and a simple linear regression on the best chosen predictor.
*   **Classification Comparison:**
    *   The "winner" is determined by the lowest **test log-loss**.
    *   Statistical significance against the winner is evaluated using: paired t-test on per-point log-loss, **McNemar's test** for classification error, and **DeLong's test** for AUC. P-values are Holm-adjusted.
    *   Baseline: A base-rate model predicting the training class proportions.

---

### 4. Feature Derivation and Preparation
Data preparation is handled by task-specific scripts (`prep_clf.py` and `prep_reg.py`):

*   **Classification (`prep_clf.py`):**
    *   **Split:** 80/10/10 stratified split based on the target class.
    *   **Transformations:** Missing values are median-imputed. All 200 continuous features are winsorized (0.1/99.9 percentiles), transformed to a Gaussian distribution based on the training ECDF (rank-gauss), and then standardized (z-scored). These 200 features (`x200`) are used by *Naive Bayes* and *Gaussian Mixture Classifier*.
    *   **Feature Selection:** To avoid the curse of dimensionality for memory/compute-heavy kernel methods (*Nadaraya-Watson, Local Logistic, Kernel Density*), the top $k$ features (default 8, `x8`) are selected based on their univariate signal on the training set (highest $|AUC - 0.5|$).
*   **Regression (`prep_reg.py`):**
    *   **Split:** Temporal split using `transactiondate`. Train $\le$ Feb 2017, Validation Mar-Jul 2017, Test $\ge$ Aug 2017.
    *   **Target:** `logerror`. For model fitting, it is clipped at the 1st and 99th training percentiles, but all evaluation metrics are calculated on the raw, unclipped `logerror`.
    *   **Feature Scoring/Selection:** Candidate numerical features are scored using a 5-fold CV (time-blocked) evaluating a 20-bin step function on the train set.
        *   The single best predictor (`x1`) is chosen for 1-D methods (*N-W Regression, Local Polynomial, Local Likelihood*).
        *   The top $m$ features (`xm`, default 6) are chosen for multi-dimensional methods (*RBF Network, Structured Local Regression*).
    *   **Transformations:** Features are winsorized (0.5/99.5 percentiles). If a feature's training skewness is $> 2$ (and it is non-negative with $>20$ unique values), a `log1p` transform is applied. Finally, they are median-imputed and z-scored.