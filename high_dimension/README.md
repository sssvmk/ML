# High-Dimensional Data Problems ($p \gg N$)

This project implements and empirically compares the statistical learning methods described in **Chapter 18 of 'The Elements of Statistical Learning' (ESLII)**. It focuses on high-dimensional problems where the number of predictors heavily outnumbers the number of samples ($p \gg N$). 

The scope covers two main tasks:
- **Regression**: Predicting a continuous outcome on the Riboflavin dataset (4,088 genes, 71 samples).
- **Classification**: Predicting tumor classes on the SRBCT dataset (2,308 genes, 83 samples).

Models are rigorously tuned via nested cross-validation and evaluated using conservative statistical testing to account for the small sample sizes inherent in this type of data.

---

## Algorithms Implemented

The `methods/` directory implements a variety of high-dimensional techniques.

### Regression (Riboflavin)
- **Lasso** (`reg_lasso.py`): Provides L1 shrinkage for continuous outcomes, yielding sparse models by selecting only the most relevant genes.
- **Fused Lasso** (`reg_fused_lasso.py`): Applies an L1 penalty to both the coefficients and the differences of adjacent coefficients. This encourages 'blocks' of correlated genes to be selected together.
- **Kernel Ridge** (`reg_kernel_ridge.py`): Performs a non-linear mapping (RBF or Linear kernel) using L2 regularization. It uses all genes without feature selection.
- **Supervised PCA** (`reg_supervised_pca.py`): Filters genes based on a univariate association with the target, applies PCA on the survivors, and regresses on the top principal components.
- **Baselines**: Standard Ridge Regression (`reg_ridge.py`) and a Mean predictor (`reg_mean.py`).

### Classification (SRBCT)
- **Nearest Shrunken Centroids** (`clf_nsc.py`): A lasso-penalized naive-Bayes model. Shrinks class centroids towards the overall centroid to remove irrelevant genes, yielding a sparse classifier.
- **L1 Logistic Regression** (`clf_l1_logreg.py`): Multinomial logistic regression with an L1 penalty (lasso) for sparse feature selection.
- **Linear SVC** (`clf_linear_svc.py`): Finds the maximal margin hyperplane in high dimensions using L2 regularization.
- **Diagonal LDA** (`clf_diag_lda.py`): The naive-Bayes / independence rule with a common diagonal covariance (uses all genes).
- **Regularized LDA** (`clf_reg_lda.py`): Shrinks the within-class covariance toward a scaled identity matrix to maintain invertibility when $p \gg N$.
- **Regularized Logistic Regression** (`clf_reg_logreg.py`): Standard multinomial logistic regression with an L2 penalty (ridge).
- **Supervised PCA** (`clf_supervised_pca.py`): Similar to its regression counterpart, but applies logistic regression on the PCA components of pre-filtered genes.
- **Baseline**: Majority class predictor (`clf_majority.py`).

---

## Loss Functions

The algorithms optimize the following loss functions:
- **Lasso / Fused Lasso**: Squared error with L1 penalties. (Fused lasso adds $\lambda_2 \sum |b_{j+1} - b_j|$).
- **Kernel Ridge / Ridge**: Squared error with L2 penalty (in kernel space for Kernel Ridge).
- **Supervised PCA**: Squared error (Regression) or Multinomial log-likelihood (Classification) on the reduced principal components.
- **Nearest Shrunken Centroids**: Squared standardized distance to the centroid (with log-prior offsets).
- **L1 / L2 Logistic Regression**: Negative multinomial log-likelihood with L1 or L2 penalties.
- **Linear SVC**: One-vs-rest hinge loss with L2 penalty: $\sum [1 - y_i f(x_i)]_+ + \frac{1}{2C}||b||^2$.
- **Diagonal / Regularized LDA**: Gaussian log-likelihood.

---

## Evaluation Metrics

Model evaluation is handled in `hd_runner.py` and calculated via `hd_lib.py`:

- **Regression**: Mean Squared Error (MSE) is the primary metric. Secondary metrics include the Standard Error of MSE (SE), RMSE, MAE, and $R^2$ vs train mean.
- **Classification**: Error rate (accuracy) is the primary metric. Secondary metrics include Log-loss and Macro One-vs-Rest AUC.
- **Statistical Significance**: Given the small test set sizes, the project uses Diebold-Mariano tests (regression), exact McNemar tests (classification), and paired t-tests (log-loss/squared errors) to determine if a model truly outperforms another. All p-values use the Holm adjustment for multiple testing.

---

## Feature Engineering and Derived Features

Feature transformations are implemented as scikit-learn Transformers in `hd_lib.py`. To prevent data leakage, they are strictly refitted inside every CV fold:

- **Variance Filter** (`VarianceThreshold(1e-10)`): Drops completely constant genes.
- **Winsorization** (`Winsorizer`): Clips extremely high or low gene expressions (e.g., at the 1% and 99% quantiles) to stabilize scale estimates against outliers.
- **Standardization** (`StandardScaler`): Centers variables to zero mean and unit variance.
- **Within-Class Scaling** (`WithinClassScaler`): Used in Diagonal LDA to center data by the overall mean and divide by the pooled within-class standard deviation.
- **Gene Ordering** (`GeneOrderer`): Computes a hierarchical clustering leaf order based on correlation. This ensures that neighboring genes are structurally similar, which is critical for the Fused Lasso penalty to be meaningful on microarray data.

---

## Analysis Results & Conclusions

### Regression (Riboflavin)
The test set is extremely small (12 samples), leading to high variance and low statistical power. 
- **Winner**: The `fused_lasso` model performed best, achieving a test MSE of 0.219 while selecting 169 genes.
- **Statistical Nuance**: Due to the low sample power, models like `kernel_ridge`, `supervised_pca`, and `lasso` were deemed *not significantly worse* than the winner. Furthermore, due to standard error overlap, the winning model was not significantly better than the basic ridge regression baseline.

### Classification (SRBCT)
The test set contains 20 samples.
- **Winner**: The `reg_lda` model was the winner, achieving a perfect test error rate of 0.0 (0 misclassifications) and a log-loss of 0.0, utilizing all 2,308 genes.
- **Statistical Nuance**: The classification task on this dataset is quite separable. Practically all other models (`l1_logreg`, `supervised_pca`, `nsc`, `reg_logreg`, `linear_svc`, `diag_lda`) also performed flawlessly or close to it, and were statistically not significantly worse. 
- **Efficiency**: Among the top-performing perfect models, `supervised_pca` was the most sparse and efficient, requiring only 50 selected genes.
