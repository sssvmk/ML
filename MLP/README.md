# MLP Pipeline Project Overview

This project implements an end-to-end Machine Learning pipeline for training Multi-Layer Perceptron (MLP) models on both image data (e.g., MNIST) and tabular data (e.g., California Housing, Santander, Zillow). The framework automates data ingestion, dynamic preprocessing, hyperparameter tuning, model training with early stopping, evaluation, and logging to an MLflow model registry.

## MLP Algorithm & Architecture

The core model is a flexible Multi-Layer Perceptron (MLP) defined in `mlp_core.py`.
- **Architecture**: It supports a configurable number of fully connected (Linear) hidden layers with optional tapering (where layer width can halve sequentially). It applies Dropout regularization and ReLU (Rectified Linear Unit) activation functions between layers.
- **Activation Functions**: ReLU is utilized exclusively for hidden layers. The output layer applies a linear transformation without activation (the loss function—e.g., CrossEntropy for classification or MSE for regression—handles the corresponding logits or direct values during training).
- **Initialization**: Hidden layers are initialized using He (Kaiming) normal initialization, combined with a small positive bias (0.01) to prevent dead ReLUs early in training. The output layer uses Xavier (Glorot) uniform initialization.
- **Optimizers**: The framework supports both Minibatch Stochastic Gradient Descent (`sgd`) with momentum, and `adamw`. It pairs the optimizer with an `ExponentialLR` learning rate scheduler and applies weight decay for regularization.
- **Training Algorithm**: The training loop relies on standard forward and backward passes to minimize the specified loss using gradients, iterating through mini-batches of data, and validating against a holdout validation set with an early stopping mechanism.

## Hyperparameter Tuning

Hyperparameter tuning is orchestrated via the `tuning.py` module, powered by **Optuna**:
- **Search Strategy**: It utilizes a `TPESampler` (Tree-structured Parzen Estimator) for Bayesian optimization and a `MedianPruner` to halt unpromising trials early.
- **Search Space**: The parameter space is dynamic based on dataset profiles (e.g., "mnist" vs "tabular"). It searches over:
  - **Network Topology**: Number of hidden layers (1-4) and base layer widths (e.g., 32-1024), plus a boolean flag to taper widths.
  - **Optimization**: Optimizer type (`sgd` or `adamw`), Learning rate (`lr`), Batch size (e.g., 64-1024), Momentum (for SGD), Weight decay, and Dropout rate.

## End-to-End Pipeline

The pipeline is primarily driven by `run.py` and modularized data sources (`datasources.py`):
1. **Data Ingestion**: A pluggable DataSource system (`datasources.py`) resolves raw tabular files (CSV/Parquet via `tabular.py`) or built-in datasets (MNIST, California Housing).
2. **Preprocessing**: Tabular data undergoes dynamic preprocessing (imputation, scaling, one-hot encoding for categoricals). The preprocessor is fit strictly on the training split to prevent data leakage and is subsequently packaged inside the final PyTorch model via MLflow's `pyfunc`.
3. **Tuning**: A nested MLflow run is created to execute the Optuna hyperparameter search.
4. **Training**: The best configuration is trained from scratch with early stopping on the validation set.
5. **Evaluation**: The model is evaluated precisely once on the untouched test set. Metrics (e.g., MSE, Accuracy, ROC-AUC) are calculated, and diagnostic plots are generated.
6. **Tracking & Registry**: Results, hyperparameters, metrics, and plots are tracked via MLflow. If the model beats the previous "champion" model by a specified margin, it is automatically promoted in the registry, and a detailed `MODEL_CARD.md` is generated.

## Datasets

The pipeline has been executed on four primary datasets:
1. **MNIST** (`mnist`): 28x28 grayscale handwritten digits classification dataset. Pre-split: 55k Train, 5k Val, 10k Test.
2. **Santander** (`santander`): Tabular binary classification dataset. Randomly split: 120k Train, 40k Val, 40k Test.
3. **California Housing** (`housing_csv`): Tabular regression dataset predicting house values.
4. **Zillow** (`zillow`): Tabular regression dataset. Randomly split: 100k Train, 33k Val, 33k Test.

## Results & Plots

Based on the test set evaluations logged in the respective `results.json` files, the finalized models achieved the following metrics:
- **MNIST**: Accuracy = 98.39%
- **Santander**: Accuracy = 91.13%
- **Zillow**: MSE = 0.02705
- **California Housing**: MSE = 2,888,527,319.20 

The pipeline automatically generates evaluation and diagnostic plots saved as MLflow artifacts. (Note: The exact UUID hashes in the paths vary by run, but they follow this structure):

### Classification Plots (MNIST & Santander)
- **Tuning History**: `results/mnist/mlartifacts/<run_id>/artifacts/search/tuning.png`
- **Training Curves**: `results/mnist/mlartifacts/<run_id>/artifacts/curves.png`
- **ROC Curves**: `results/mnist/mlartifacts/<run_id>/artifacts/evaluation/roc_curves.png`
- **Confusion Matrix**: `results/mnist/mlartifacts/<run_id>/artifacts/evaluation/confusion_matrix.png`

### Regression Plots (Zillow & Housing)
- **Tuning History**: `results/zillow/mlartifacts/<run_id>/artifacts/search/tuning.png`
- **Training Curves**: `results/zillow/mlartifacts/<run_id>/artifacts/curves.png`
- **Regression Diagnostics**: `results/zillow/mlartifacts/<run_id>/artifacts/evaluation/regression_diagnostics.png`

## Conclusion

The project demonstrates a robust, production-ready ML architecture capable of seamlessly switching between deep learning for image classification and robust structured tabular regression/classification. The Optuna-based hyperparameter optimization combined with automated MLflow tracking and rigorous, leakage-free data splits results in highly performant and reproducible MLP models.