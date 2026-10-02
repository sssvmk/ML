# Basis Expansions and Regularization Analysis

This repository compares various basis expansion and regularization methods (based on *The Elements of Statistical Learning*, Chapter 5) on a temporal split of the Zillow dataset. The objective is to evaluate 1D and 2D smoothers using different penalty structures and basis functions.

## 1. Algorithms in `methods/`

The codebase implements eight core modeling approaches:

1. **`b_splines.py` (B-Splines)**
   Fits a cubic regression spline using a B-spline basis. It spans the exact same function space as the truncated-power cubic regression spline. The B-spline formulation is strictly used to improve numerical conditioning via a local, partition-of-unity basis. 

2. **`natural_cubic_splines.py` (Natural Cubic Splines)**
   A regression spline constrained to be linear beyond the boundary knots (the minimum and maximum data values). This restriction reduces variance at the boundaries at the cost of some bias. Knots are placed at data quantiles.

3. **`regression_splines.py` (Piecewise Polynomials / Regression Splines)**
   Fits piecewise polynomials using a truncated power basis. The basis can model piecewise constant (degree 0), continuous linear (degree 1), or cubic (degree 3) functions. Knots are defined at quantiles of the input data.

4. **`smoothing_splines.py` (Smoothing Splines)**
   A penalized regression approach using a cubic B-spline basis. Unlike regression splines, knots are placed densely at predictor quantiles, and complexity is constrained by an explicit penalty on the integrated squared second derivative ($\int f''(t)^2 dt$).

5. **`nonparametric_logistic.py` (Nonparametric Logistic Regression)**
   Tailored for a binary classification task (`large_miss` indicator). Fits a logistic regression model where the logit is modeled by a penalized cubic B-spline basis. It is trained via Penalized Iteratively Reweighted Least Squares (IRLS).

6. **`thin_plate_splines.py` (Thin-Plate Splines)**
   A 2D spatial smoothing method applied to geographic coordinates (latitude and longitude). It uses a reduced-rank radial basis function ($\eta(r) = r^2 \log r$) with $K$-means centroids acting as knots. It includes an unpenalized planar component, heavily penalized otherwise based on 2D curvature.

7. **`rkhs.py` (Reproducing Kernel Hilbert Space / Kernel Ridge)**
   A 1D Gaussian-kernel ridge regression. To avoid the $O(N^3)$ computational cost of full kernel inversion, it utilizes the Nystrom method, restricting the expansion to a subset of "landmark" points placed at predictor quantiles.

8. **`wavelet_smoothing.py` (Wavelet Smoothing)**
   Treats the data as a uniformly spaced signal by binning the predictor into $N=2^k$ bins and taking logerror means. It applies a Discrete Wavelet Transform (DWT), regularizes via soft-thresholding of detail coefficients (using `SureShrink` or CV), and rebuilds the signal with IDWT.

---

## 2. Loss Functions and Evaluation Metrics

The codebase defines a unified cross-validation and evaluation protocol (`common.py`), though objectives differ by method.

### Training and Loss Functions
- **Unpenalized Least Squares**: `b_splines`, `natural_cubic_splines`, and `regression_splines` optimize the residual sum of squares (RSS). Models are fit securely via QR decomposition (`ls_fit` in `common.py`).
- **Penalized Least Squares (Ridge/Kernel)**: `smoothing_splines`, `thin_plate_splines`, and `rkhs` optimize a penalized RSS where the penalty corresponds to basis smoothness or RKHS norm.
- **Penalized Maximum Likelihood**: `nonparametric_logistic` minimizes the penalized negative log-likelihood (cross-entropy).
- **L1 Penalized Wavelets**: `wavelet_smoothing` minimizes least squares subject to an $L_1$ penalty on the wavelet detail coefficients.

### Validation Protocol
All hyperparameters (e.g., number of knots, basis degree, or the regularization $\lambda$ mapped to effective degrees of freedom) are chosen using **10-fold Cross-Validation** exclusively on the training set. Folds are **time-blocked** to prevent temporal leakage. 

Model selection strictly follows the **One-Standard-Error (1-SE) Rule**. The "paired" 1-SE rule isolates model performance from shared fold-level variance by calculating the standard error of fold-wise differences relative to the CV-minimum model.

### Test Metrics
Models are retrained on train+validation sets using the CV-selected hyperparameter and evaluated on the test set:
- **Regression Tasks**: 
  - Test MSE on the raw target (`logerror`).
  - Standard Error of the Test MSE.
  - RMSE, MAE.
  - Relative MSE against a baseline predicting the training mean.
  - Thin-plate splines also compute **Moran's I** on test residuals to measure remaining spatial autocorrelation.
- **Logistic Task**:
  - Test Log-Loss against a base-rate (null) model.
  - AUC, Brier Score, and Expected Calibration Error (ECE).
  - Error rate at an optimal Youden threshold tuned on validation data.

---

## 3. Cross-Method Comparison (`compare.py` & `run_all.py`)

The orchestration script `run_all.py` manages multiprocessing, running EDA, data preparation, and training all methods simultaneously. 

Once training finishes, `compare.py` compiles results to find the global winner:
1. **Baselines**: Predict-the-mean, and a simple linear regression on the chosen predictor.
2. **Winner Selection**: The regression smoother with the lowest Test MSE is crowned the "winner".
3. **Statistical Significance**: A **Diebold-Mariano (DM) test** (using Newey-West long-run variance to handle time-series autocorrelations) and a paired t-test are calculated between the squared errors of the winner and every other method.
4. **Holm Adjustment**: P-values are adjusted using Holm's method for multiple comparisons to declare which methods are "not significantly worse than the winner".
5. **Output**: A comprehensive DataFrame (`comparison.csv`) is generated alongside plots overlaying fitted curves (`comparison_fitted_curves.png`) and summarizing Test MSEs with error bars.

---

## 4. Feature and Basis Function Derivation

The transformation of raw features into smooth basis functions is handled between `prepare.py` and `bases.py`.

### Feature Preparation (`prepare.py`)
- **1D Predictor Selection**: 1-D methods accept a single continuous predictor. If set to `auto`, candidates (like `taxvaluedollarcnt`, `yearbuilt`) are evaluated on the train set via a 5-fold CV of a 20-bin step function. The candidate with the best raw-logerror MSE is selected.
- **Transformation**: The chosen predictor is winsorized (0.5 to 99.5 percentiles), log1p-transformed (if heavily right-skewed), median-imputed, and standardized (z-scored).
- **2D Coordinates**: `latitude` and `longitude` are scaled by $10^6$ and z-scored to form the inputs for the thin-plate splines.

### Basis Generations (`bases.py`)
- **Truncated Power Basis**: Computes polynomial terms $[1, x, \dots, x^d]$ alongside rectified terms $(x - \xi_k)_+^d$ at knot boundaries.
- **Natural Cubic Basis**: Forms basis functions $N_k$ representing cubic polynomials that are forced to be linear beyond the outer boundary knots ($d_k - d_{K-1}$).
- **B-Splines (`bs_design` / `bs_penalty`)**: Wraps `scipy.interpolate.BSpline`. Calculates the exact, continuous integrated-second-derivative matrix $\Omega$ iteratively across knot intervals using Gauss-Legendre quadrature.
- **Thin-Plate Splines (`tps_setup` / `tps_design`)**: Calculates a pairwise distance matrix between coordinates and k-means chosen knots to evaluate $\eta(|x - \xi_j|)$. An orthogonal projection $Z$ (via QR decomposition) separates the penalized radial expansions from the unpenalized linear plane.
- **Effective DF ($\lambda$) Mapping**: Using the Demmler-Reinsch orthogonalization of the penalty matrices, `df_to_lambda` traces target degrees of freedom back to the precise Ridge multiplier $\lambda$ needed to achieve them.