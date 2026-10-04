# Model card: MLP-housing_data

**Task:** regression   **Model:** fully connected network (ReLU hidden units, 795,137 parameters)
**Hidden layers:** [512, 512, 512, 512]   **Optimizer:** adamw (lr 0.0010113, weight decay 0.00084, dropout 0.05)

## Data and splits
- Source: csv:housing.csv
- NOT pre-split: a single table of 20,640 rows was split 12,384 train / 4,128 validation / 4,128 test (random, seed 42). Preprocessing (imputation, scaling, one-hot) was fit on the training rows only.
- Sizes: {'train': 12384, 'val': 4128, 'test': 4128}
- Data notes (joins, dropped rows or columns, auto-detected task):
  - none
- Data audit: passed; warnings:
- none
- Split/data fingerprint: `84285f39ee5f4f4b`

## Test performance (test set used once, after model selection)
- MSE: 2888527319.2007 +- 133139672.9542 (SE)   [units: (median_house_value)^2; 95% CI 2627573560.2104 - 3149481078.1909]
- RMSE: 53745.0213 +- 1238.6233 (SE)   (same units as median_house_value)
- MAE: 35844.4587 +- 623.3682 (SE)   R2: 0.7796
- MSE difference vs linear regression (paired): -2023319229.1433 +- 123435007.6944 (SE) -> model better
- linear-regression reference MSE: 4911846389.8369

Standard errors and intervals reflect the finite size of the test set only; they do not include
variation between training runs with different seeds.

Validation (selection) metrics at the chosen epoch: {'val_loss': 0.2158, 'val_rmse': 53461.3876, 'val_mae': 35974.6328, 'val_r2': 0.7919}

## Training
- Hyperparameter search: Optuna TPE + MedianPruner, 30 trials (14 pruned); best validation loss 0.2109 vs 0.2448 for the default config
- Selection criterion: lowest validation loss; early stopping on validation loss (best epoch 35 of 43).
- Diagnosis: no large train/validation gap (val/train loss ratio 1.25; heuristic).

## Intended use and limitations
Trained on csv:housing.csv (target: median_house_value). Valid only for data that looks like the training file. Inputs far outside the training range are flagged but still scored. Not evaluated for robustness or fairness.

## Governance and lineage
- Acceptance target: no acceptance target set
- Registry decision: promoted: first version, no champion yet
- MLflow: experiment `MLP-housing_data`, parent run `3decbefcc2164903945e13c9cd511ec6`, final run `31a29440f19f451abc60d48d41076c3c`
- Code version (git): 498126c; torch 2.11.0+cpu, mlflow 3.16.1, python 3.13.3
- Human approval: **not recorded** (see TODO.md)
