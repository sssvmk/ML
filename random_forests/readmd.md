# Codebase Investigation Report

### 1. Codebase Overview
This codebase is a highly parallelized, end-to-end machine learning experimentation pipeline designed to replicate and expand upon the tree-based ensemble methods described in Chapter 15 of "The Elements of Statistical Learning" (ESLII). The architecture follows a strict 5-step orchestration process managed by `run_all.py`:
1. **Exploratory Data Analysis (EDA):** Initial profiling of the datasets.
2. **Data Preparation:** Splitting and comprehensive feature engineering.
3. **Variable Ranking:** Feature selection within cross-validation (CV) folds to prevent target leakage.
4. **Model Training:** Parallel training and hyperparameter tuning of different tree ensemble models via random search.
5. **Comparison:** Rigorous statistical comparison of the models on a held-out test set to declare a statistically sound "winner".

### 2. Business Functionality
The codebase provides automated solutions for two common, large-scale tabular data business problems:
*   **Real Estate Valuation (Regression):** Predicting the monetary error (the difference between an algorithmic price estimate and the actual sale price) of real estate properties. This helps refine pricing models, reduce financial risk, and offer more accurate property valuations.
*   **Customer Satisfaction Prediction (Classification):** Identifying whether a customer will be satisfied or dissatisfied based on anonymized historical transaction or interaction metrics. This capability allows a business to proactively address unhappy clients, improve retention, and optimize customer service resources.

### 3. Algorithm Explanations
The pipeline leverages the `scikit-learn` library to implement four specific tree-based ensemble algorithms (found in `methods/rf_entry.py`). Here is how they work in layman's terms:
*   **Random Forest (`rf_tuned` & `rf_defaults`):** Imagine a diverse council of decision-makers. A Random Forest builds hundreds of distinct decision trees by giving each tree a slightly different, random subset of the training data. Furthermore, when each tree is deciding how to make a split (a decision rule), it is only allowed to look at a random subset of the available clues (features). The final prediction is the average (for regression) or the majority vote (for classification) of all the trees. The codebase tests a theoretically default version (`rf_defaults`) and a mathematically tuned version (`rf_tuned`).
*   **Extra Trees (`extratrees` - Extremely Randomized Trees):** This is similar to a Random Forest but introduces even more randomness to prevent memorizing the data (overfitting). Instead of finding the absolute best mathematical threshold to split the data at each decision point, Extra Trees picks random thresholds for the chosen features and selects the best among those random splits. It also uses the entire dataset for every tree rather than a sample, making it faster to train.
*   **Bagging (`bagging` - Bootstrap Aggregating):** This is essentially the predecessor to the Random Forest. It builds many decision trees using random subsets of the training data, but unlike Random Forests, every tree is allowed to look at *all* available clues (features) at every decision point. Because the trees have access to all the same features, they tend to be more similar to each other than the trees in a Random Forest.

### 4. Loss Functions
To build the individual decision trees, the models must measure how "good" a potential data split is. The codebase utilizes standard tree-building loss functions:
*   **Regression (Zillow):** Uses **Squared Error** (Residual Sum of Squares or RSS). The tree tries to find splits that minimize the variance (the squared differences) between the actual target values and the predicted values in the resulting data branches.
*   **Classification (Santander):** Uses **Gini Impurity** for splitting. Gini impurity measures how often a randomly chosen element from the dataset would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset. The tree seeks splits that create the most "pure" groups (e.g., a group containing only satisfied customers).

### 5. Evaluation Metrics
To determine which fully-trained ensemble model performs best, the `compare.py` script calculates several metrics on a held-out test set:
*   **Regression Metrics:**
    *   **Test MSE (Mean Squared Error):** The primary metric used to determine the winner. It measures the average of the squares of the errors (the difference between estimated and actual values).
    *   **Test MAE (Mean Absolute Error):** Measures the average magnitude of the errors without considering their direction, providing a more interpretable scale.
    *   **Test RMSE (Root Mean Squared Error):** The square root of the MSE, bringing the error back to the original unit of the target variable.
*   **Classification Metrics:**
    *   **Test AUC (Area Under the ROC Curve):** The primary metric to declare the winner. It measures the model's ability to distinguish between the two classes across all possible decision thresholds. An AUC of 1.0 is perfect, while 0.5 is random guessing.
    *   **Test Error Rate:** The percentage of incorrect predictions at a specifically tuned probability threshold.
    *   **Test Log-loss (Cross-Entropy Loss):** Measures the performance of the classification model where the prediction input is a probability value between 0 and 1. It heavily penalizes confident but incorrect predictions.
    *   **Brier Score:** Essentially the mean squared error applied to predicted probabilities.

### 6. Datasets Used
The pipeline processes two distinct datasets (with synthetic generators provided in the `tests/` directory for pipeline verification):
*   **Zillow Dataset (Regression):** Contains real estate data with spatial, structural, and tax-related characteristics (e.g., square footage, latitude/longitude, room counts). The goal is to predict the `logerror`. The data is processed temporally (time-blocked cross-validation) due to the nature of real estate transactions. The feature engineering pipeline creates over 170 candidate variables.
*   **Santander Dataset (Classification):** Contains 200 anonymized numeric features and a binary `Target` variable representing customer satisfaction. The data uses a stratified split to maintain class balance. The feature engineering pipeline expands this into hundreds of candidate variables using advanced statistical techniques like rank-gaussing and weight-of-evidence.

### 7. Cross-Algorithm Evaluation
The codebase does not just pick the model with the best raw score; it uses rigorous statistical tests in `compare.py` to ensure the "winner" is genuinely better and not just benefiting from random chance:
*   **For Regression:** The model with the lowest Test MSE is the provisional winner. The script then applies the **Diebold-Mariano test** (suitable for time-ordered data like Zillow) and **paired t-tests** to compare the squared errors of the winner against every other model.
*   **For Classification:** The model with the highest Test AUC is the provisional winner. The script uses **DeLong's test** to compare AUCs, **McNemar's test** to compare error rates, and paired t-tests for log-loss.
*   In both cases, **Holm corrections** are applied to the p-values to adjust for multiple comparisons. The final output lists the absolute winner, any models that are "not significantly worse" than the winner, and plots comparing the models against simple baselines.

### 8. Conclusion
This codebase is a highly robust, production-grade benchmarking framework for tabular data. By implementing exhaustive feature engineering, automated hyperparameter random search, and strict statistical validation techniques, it ensures that model selection is driven by mathematical significance rather than marginal metric variations. Its parallelized architecture makes it capable of scaling to large datasets, while the synthetic data generators ensure the complex pipeline can be tested efficiently.