# Project Overview
This project is an empirical machine learning study focused on prototype methods and nearest neighbors, based on Chapter 13 of *Elements of Statistical Learning (ESLII)*. It orchestrates a 5-step pipeline via `run_all.py`:
1. **Exploratory Data Analysis (EDA)**
2. **Data Split & Full Feature Engineering**: Generating a large pool of candidate variables.
3. **Variable Ranking**: Conducted inside cross-validation folds to prevent data leakage.
4. **Parallel Training & Tuning**: Methods undergo random-search CV to optimize hyper-parameters and select the best top-k variables.
5. **Cross-Method Comparison**: Evaluating models on a common held-out test set and declaring a winner.

# Algorithms Used
Implemented in the `methods/` directory:
*   **DANN (Discriminant Adaptive Nearest-Neighbour)**: A classification method that uses a local metric built from within-class and between-class scatter matrices, shrinking neighborhoods along discriminative directions.
*   **GMM (Gaussian Mixture Classifier)**: Fits a Gaussian Mixture Model per class, computing posteriors using Bayes' rule and training class proportions.
*   **KMeans Prototype Classifier**: Runs K-means separately per class. Points are classified based on their distance to the nearest class prototype.
*   **KNN (k-Nearest Neighbors)**: Standard k-NN classifier utilizing distance-weighted majority voting based on Euclidean distance.
*   **LVQ (Learning Vector Quantization - LVQ1)**: An online prototype algorithm initialized from K-means. Prototypes are iteratively attracted to same-class points and repelled by different-class points.
*   **Tangent Distance**: An extension of KNN providing metric invariance to small transformations. On tabular data (where transformations are unknown), it correctly reduces to Euclidean k-NN.
*   **KNN Regression**: Uniform-kernel Nadaraya-Watson smoother predicting the weighted mean of `y` over `k` nearest points (variants include plain/distance-weighted and Euclidean/Manhattan metrics).

# Loss Functions Used
*   **DANN / KNN**: Implicit 0-1 loss for classification via majority voting.
*   **GMM**: Negative mixture log-likelihood per class (fitted via Expectation-Maximization).
*   **KMeans**: Within-cluster sum of squares (minimized independently per class).
*   **LVQ**: No fixed global optimization criterion; uses Kohonen's LVQ1 online heuristic updates.
*   **KNN Regression**: Squared error with a locally constant fit.

# Metrics Used for Evaluation
*   **Classification**: Test AUC (the primary winning criterion), Error, Log-Loss, Brier score, Expected Calibration Error (ECE), Sensitivity, and Specificity. CV focuses on AUC.
*   **Regression**: Test MSE (the primary winning criterion), RMSE, MAE, relative MSE vs. a predict-the-mean baseline, and leave-one-out CV MSE vs `k`.

# Feature Engineering & Derived Features
*   **Classification (Santander)**: A stratified 80/10/10 split generating ~600 candidate variables. Techniques include:
    *   Rank-gaussed features (winsorized, mapped via ECDF to normal quantiles).
    *   Leave-one-out frequency features.
    *   Weight-of-evidence features (binned quantiles, out-of-fold for training).
*   **Regression (Zillow)**: Temporal split with extensive engineering:
    *   Missing indicators, zero-filling.
    *   Derived domain ratios (e.g., age, tax/sqft, structure & land ratio).
    *   Lat/lon quadratics, log1p of skewed columns, and one-hot encoding of cardinality ids.
    *   Target encoding (smoothed out-of-fold mean of clipped target) for high-cardinality geographic and zoning codes.

# Cross-Method Comparison Methods
*   **Classification**: Pairwise comparisons against the winner use the **DeLong test** for AUC, **McNemar's test** for classification error, and **paired t-tests** on per-point log-loss. All p-values are **Holm-adjusted**. Baseline is a base-rate model.
*   **Regression**: Evaluated using the **Diebold-Mariano test** (suitable for time-ordered rows) on raw target MSE and **paired t-tests** on per-point squared errors, both Holm-adjusted. Baselines include predict-the-mean and Ridge regression.

# Results
*   **Classification (`results_clf`)**: The winner is **LVQ** (Highest AUC: ~0.7534, Lowest Log-Loss: ~0.2841). It statistically tied with KMeans and GMM (DeLong p > 0.05). GMM achieved the lowest raw classification error (0.2807).
*   **Regression (`results_reg`)**: The winner is **knn_reg_distance_euclidean** (Lowest Test MSE: 0.041244). All other tested k-NN variants (uniform/distance, Euclidean/Manhattan) tied statistically with the winner, and all significantly outperformed the predict-the-mean baseline.

# Datasets
*   **Classification**: Santander dataset (`santander_train.csv`).
*   **Regression**: Zillow dataset (`zillow.csv`).
*   *(Synthetic versions can be generated via scripts in the `tests/` directory).*

# Conclusion
The experiments demonstrate the strong viability of prototype-based and nearest-neighbor methods when coupled with rigorous feature engineering and selection.
For classification, prototype methods (**LVQ** and **KMeans**) slightly edged out traditional approaches, proving highly effective at navigating the massive 600-variable engineered space to maximize AUC.
For regression, **Euclidean distance-weighted k-NN** emerged victorious, although performance differences between k-NN variants were statistically insignificant. The heavy reliance on target encoding and derived spatial/tax ratios was critical in making the neighborhood space meaningful enough to easily defeat standard baselines.
