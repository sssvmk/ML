# Santander Linear Classification Study - Codebase Analysis

Based on a thorough investigation of the codebase, here is a comprehensive report on the Santander Linear Classification study:

### 1. Overall Functionality & Flow
The codebase serves as a robust, parallelized evaluation pipeline designed to benchmark various linear classification methods on the Santander customer transaction prediction dataset. 
* **Execution Flow (`run_all.py`)**: The entry point loads the dataset and splits it into a stratified 80/10/10 (Train/Validation/Test) split.
* **Feature Engineering (`features.py`)**: It applies two distinct feature engineering pipelines ("lean" and "rich").
* **Training & Evaluation**: It concurrently trains, validates, and tests 9 different classification methods across both feature sets. The models use stratified 10-fold Cross-Validation (CV) on the training set for hyperparameter tuning (using the One-Standard-Error rule). The validation set is used exclusively to tune the decision threshold (using Youden's J statistic). Final unbiased metrics are reported purely on the held-out test set.

### 2. Methods & Loss Functions
The `methods/` directory implements 9 distinct approaches. Here are the methods and the specific loss/objective functions they optimize:
* **l1_logistic**: L1-regularized logistic regression. 
  * *Objective*: Minimizes the negative binomial log-likelihood (deviance) + an L1 penalty ($\lambda ||\beta||_1$). Solved via a proximal Newton / Iteratively Reweighted Least Squares (IRLS) path.
* **logistic**: Standard unpenalized logistic regression. 
  * *Objective*: Minimizes the negative binomial log-likelihood. It uses damped IRLS to handle near-separation issues in the data.
* **lda** (Linear Discriminant Analysis): 
  * *Objective*: Maximizes the Gaussian class-conditional log-likelihood assuming a shared (pooled) covariance matrix across classes.
* **qda** (Quadratic Discriminant Analysis): 
  * *Objective*: Maximizes the Gaussian class-conditional log-likelihood assuming distinct, class-specific covariance matrices.
* **rda** (Regularized Discriminant Analysis): 
  * *Objective*: Gaussian log-likelihood with shrunken covariance matrices. It introduces an $\alpha$ parameter to interpolate between LDA and QDA covariances, and a $\gamma$ parameter to shrink towards a scaled identity matrix.
* **reduced_rank_lda**: Canonical discriminant analysis.
  * *Objective*: Maximizes the ratio of between-class variance to within-class variance ($a^T B a$ subject to $a^T W a = 1$). For a 2-class problem like Santander, this is mathematically equivalent to standard LDA.
* **indicator_regression**: Ordinary Least Squares (OLS) regression acting on a binary target. 
  * *Objective*: Minimizes the residual sum of squares (RSS), $||Y - X\beta||^2$, where targets are coded as $0$ and $1$.
* **separating_hyperplane**: Soft-margin Support Vector Machine (SVM). 
  * *Objective*: Minimizes $\frac{1}{2}||\beta||^2 + C \sum \text{hinge\_loss}$. A very large $C$ is used because the Santander data is notoriously non-linearly separable.
* **perceptron**: Rosenblatt's perceptron algorithm. 
  * *Objective*: Minimizes $-\sum y_i f(x_i)$ iteratively over misclassified points using stochastic gradient descent. (Note: Because the data is non-separable, standard perceptron theoretically does not converge to a 0-error state).

### 3. Derived Features
The pipeline creates two rigorous feature sets based strictly on the training split to prevent data leakage:
* **Lean (200 columns)**: Raw features are winsorized at the 0.1 and 99.9 percentiles. Then, a "Rank-to-Gaussian" transformation is applied (the empirical CDF is mapped to normal quantiles). Finally, features are standardized.
* **Rich (600 columns)**: This set includes the 200 "Lean" features, plus:
  * **200 Frequency Features**: Computed using leave-one-out frequency encoding smoothed with $\log(1+\text{count})$.
  * **200 Weight-of-Evidence (WoE) Features**: Smooths the log-odds of positive outcomes across 20 quantile bins. Crucially, during training, a 5-fold Out-Of-Fold (OOF) encoding technique is used to prevent the target from leaking into the training features.

### 4. Evaluation Metrics
Models are thoroughly evaluated on the held-out test set in `common.py`. The metrics include:
* **AUC (Area Under the ROC Curve)**: The primary performance metric. Evaluated alongside DeLong variance to get standard errors.
* **Test Error (Misclassification Rate)**: Computed at the optimal decision threshold. The threshold is not fixed at 0.5; it is dynamically chosen on the validation set by maximizing Youden's J statistic (True Positive Rate - False Positive Rate).
* **Probabilistic Metrics**: Log-loss (deviance), Brier score, and Expected Calibration Error (ECE) are collected for methods capable of outputting probabilities.

### 5. Cross-Method Comparison
The comparative methodology (`compare.py`) is highly rigorous:
* **Identical Rows**: Results are strictly compared using identical predictions on the exact same test rows.
* **Winner Selection**: The overall winner is determined by the highest AUC.
* **Statistical Significance**: To test if other models are "statistically indistinguishable" from the winner, paired tests are performed:
  * **DeLong's test** for AUC equivalence.
  * **McNemar's exact test** for Misclassification Error equivalence.
* **P-value adjustment**: P-values are strictly adjusted for multiple comparisons against the winning model using the Holm step-down procedure.

### 6. Results Analysis
Based on the data aggregated in the `results/` directory (e.g., `winner.json`, `comparison.csv`):
* **The Winner**: **`l1_logistic` trained on the `rich` feature set** is the undisputed winner. It achieved an AUC of `~0.8878` and a Test Error of `16.61%`. Statistically, it outperformed all other methods significantly (Holm-adjusted DeLong p-value < 0.05 against the closest competitor).
* **Impact of Feature Sets**: 
  * The "rich" feature set provided a dramatic improvement for almost all models. For instance, `l1_logistic` gained roughly `+0.035` AUC by using the rich features.
  * *Exception*: **QDA** performed notably worse (`-0.010` AUC) on the rich set. Given it estimates distinct covariance matrices for each class, jumping from 200 to 600 dimensions triggers the curse of dimensionality, heavily penalizing QDA due to parameter explosion.
* **Method Equivalence**: On the rich dataset, standard LDA, Reduced-Rank LDA, Indicator Regression, and RDA (which cross-validated to select $\alpha=0$ (LDA) and $\gamma=1$ (no identity shrinkage)) all converged to virtually identical performance (AUC `~0.8865`).
* **The Power of Regularization**: `l1_logistic` soundly beat unpenalized `logistic` regression on the rich set. This proves that as the feature space grew to 600 dimensions (due to the WoE and Frequency augmentations), the capacity for feature selection and coefficient shrinkage provided by L1 regularization was crucial to prevent overfitting and achieve the winning result.