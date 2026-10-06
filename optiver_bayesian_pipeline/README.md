# Optiver Realized Volatility Prediction 📈

## Project Overview
This project implements an industry-grade **Bayesian Dynamic & Hierarchical Model** to predict short-term volatility in financial markets. Built around the context of the Optiver Kaggle competition, the goal is to forecast the realized volatility of hundreds of stocks across 10-minute windows using high-frequency tick-level data.

Due to the extreme noise and heavy-tailed nature of financial microstructure data, this pipeline avoids standard deterministic ML models (like XGBoost or simple Neural Networks). Instead, it leverages Probabilistic Programming via **PyMC** to natively model uncertainty, market regimes, and cross-stock correlations. 

The entire training and tuning lifecycle is orchestrated via **MLflow**.

---

## Datasets Used
The pipeline ingests high-frequency tick data provided in three core files:

1. **Order Book Data (`book_train.csv`)**: 1-second resolution snapshots of the top two price levels of the limit order book. Includes `bid_price`, `ask_price`, `bid_size`, and `ask_size` to capture passive market interest.
2. **Trade Data (`trade_train.csv`)**: Records of actual completed transactions (executions) occurring within the same 10-minute windows. Includes execution `price`, `size` (volume), and `order_count`.
3. **Labels (`train.csv`)**: The ground truth target mapping each `(stock_id, time_id)` pair to its true future 10-minute realized volatility.

*Note: A preprocessing script (`create_mini_dataset.py`) is included to generate 30% strictly chronological slices of this massive dataset. This prevents `joblib` memmap errors and RAM overflow during local development.*

---

## Algorithm Architecture
The core model is built using **PyMC** and optimized using **Optuna**. It incorporates highly advanced Bayesian techniques to handle financial data:

1. **Latent Market Factors ($k$):** Replaces naive stock/time intercepts with a matrix factorization approach. Stocks have $k$ loadings and times have $k$ factors. Their dot product dynamically captures complex, market-wide volatility clusters (e.g., sector-wide shocks).
2. **Gaussian Random Walk ($\delta$):** Time-series trends are modeled chronologically using a Random Walk.
3. **Horseshoe Shrinkage Prior ($\tau$):** An aggressive, heavy-tailed prior is applied to the 40+ engineered microstructure features. It acts as an intelligent Bayesian Lasso, violently shrinking useless noise features to exactly zero while leaving strong, true signals unpenalized.
4. **Student-$t$ Observation Noise ($\nu$):** Replaces standard Gaussian error assumptions with a Student-$t$ distribution, making the model highly robust to extreme market outliers.

---

## Hyperparameter Definition & Tuning Technique

The project uses **Optuna** to execute a Bayesian optimization search over the following highly complex mathematical priors:

### The Hyperparameters
*   **Observation Noise ($\nu \in [3, 8]$):** The degrees of freedom for the Student-$t$ likelihood. Lower values allow the model to ignore massive outliers (fat tails) caused by bid-ask bounces.
*   **System State Variance ($\beta \in [10^{-5}, 10^{-2}]$):** The scale parameter of the Inverse-Gamma distribution. It dictates the flexibility of the baseline volatility. Higher values allow the model to react violently to market shocks; lower values enforce smoother, long-term forecasts.
*   **Global Shrinkage ($\tau \in [10^{-4}, 10^{-1}]$):** The global scale of the Horseshoe prior. It determines how aggressively the model prunes the ~50 engineered order book features to prevent overfitting.
*   **Discount Factor ($\delta \in [0.90, 0.995]$):** Used to determine the variance of the Gaussian Random Walk ($\sigma = \frac{1-\delta}{\delta}$). It controls the sequential memory weight of the Dynamic Linear Model (how much history matters).
*   **Latent Factors ($k \in \{2, 3, 5\}$):** The dimensionality of the cross-stock correlation space.

### The Tuning Technique (Strict Chronological Splitting)
To prevent "look-ahead bias" (where a model cheats by learning from future market data to predict the past), the pipeline utilizes a strict temporal splitting technique rather than random K-Fold cross-validation:

1.  **The Vault (15%):** The final 15% of chronological `time_id`s are completely sliced off and hidden before tuning begins.
2.  **Optuna Objective (85%):** During each Optuna trial, the remaining 85% of data is split *chronologically again* into an 80% Train and 20% Validation set. Optuna attempts to minimize the Validation score.
3.  **Final Evaluation:** The best hyperparameters found by Optuna are used to train a final model on the full 85%, which is then evaluated on the hidden 15% Test Vault to provide the final, unbiased metric.

---

## Loss Function (Optimization Objective)
Because this is a Bayesian model trained via **Automatic Differentiation Variational Inference (ADVI)**, the mathematical loss function is not a standard MSE. 

The optimizer minimizes the negative **Evidence Lower Bound (ELBO)**. 
*   **The Likelihood component:** Maximizes the log-likelihood of the observed data assuming a Student-$t$ distribution.
*   **The Kullback-Leibler (KL) Divergence component:** Acts as mathematical regularization, ensuring the learned posterior distributions do not stray too far from our structural priors (like the Horseshoe shrinkage).

---

## Evaluation Metric
While the model optimizes the ELBO internally, the business evaluation metric strictly adheres to the Optiver competition standard: **Root Mean Square Percentage Error (RMSPE)**.

$$ \text{RMSPE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left(\frac{y\_true_i - y\_pred_i}{y\_true_i}\right)^2} $$

**Why RMSPE?** Volatility targets vary wildly. RMSPE is scale-independent; a 10% forecasting error on a highly volatile stock is penalized exactly the same as a 10% error on a very stable stock.

---

## Getting Started

1. **Environment Setup (using `uv`):**
   ```bash
   uv venv
   .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```
2. **Handle C++ Compiler Requirements:**
   If you do not have a C++ compiler (`g++`) installed on Windows, PyMC will default to the slower Python backend. To run the model without a compiler, execute:
   ```cmd
   set PYTENSOR_FLAGS=mode=FAST_RUN,cxx=
   uv run optiver_bayesian_pipeline\optiver_bayesian_pipeline.py
   ```
3. **View Tracking UI:**
   The project utilizes **MLflow** for experiment tracking. View trial runs, metrics, and parameters by running:
   ```bash
   mlflow ui
   ```
