# Model card: MLP-zillow_full

**Task:** regression   **Model:** fully connected network (ReLU hidden units, 55,169 parameters)
**Hidden layers:** [128, 128, 128]   **Optimizer:** sgd (lr 0.01066, weight decay 0.0016, dropout 0.15000000000000002)

## Data and splits
- Source: csv:zillow_full.csv
- NOT pre-split: a single table of 167,888 rows was split 100,732 train / 33,578 validation / 33,578 test (random, seed 42). Preprocessing (imputation, scaling, one-hot) was fit on the training rows only.
- Sizes: {'train': 100732, 'val': 33578, 'test': 33578}
- Data notes (joins, dropped rows or columns, auto-detected task):
  - none
- Data audit: passed; warnings:
- 76 identical feature rows appear in both 'train' and 'val' (possible leakage or genuine duplicates)
- 61 identical feature rows appear in both 'train' and 'test' (possible leakage or genuine duplicates)
- 16 identical feature rows appear in both 'val' and 'test' (possible leakage or genuine duplicates)
- Split/data fingerprint: `cf3f0bd79352d71c`

## Test performance (test set used once, after model selection)
- MSE: 0.0271 +- 0.0015 (SE)   [units: (logerror)^2; 95% CI 0.0242 - 0.0300]
- RMSE: 0.1645 +- 0.0045 (SE)   (same units as logerror)
- MAE: 0.0691 +- 0.0008 (SE)   R2: -0.0038
- MSE difference vs linear regression (paired): +0.0003 +- 0.0001 (SE) -> model worse
- linear-regression reference MSE: 0.0268

Standard errors and intervals reflect the finite size of the test set only; they do not include
variation between training runs with different seeds.

Validation (selection) metrics at the chosen epoch: {'val_loss': 1.1434, 'val_rmse': 0.1746, 'val_mae': 0.0701, 'val_r2': -0.0029}

## Training
- Hyperparameter search: Optuna TPE + MedianPruner, 30 trials (21 pruned); best validation loss 1.1351 vs 1.1371 for the default config
- Selection criterion: lowest validation loss; early stopping on validation loss (best epoch 1 of 2).
- Diagnosis: no large train/validation gap (val/train loss ratio 1.07; heuristic).

## Intended use and limitations
Trained on csv:zillow_full.csv (target: logerror). Valid only for data that looks like the training file. Inputs far outside the training range are flagged but still scored. Not evaluated for robustness or fairness.

## Governance and lineage
- Acceptance target: no acceptance target set
- Registry decision: promoted: first version, no champion yet
- MLflow: experiment `MLP-zillow_full`, parent run `b5528c08f91947f992dc3743b60e4d46`, final run `9e1119471c494c0dbbfc0b0e1c30e03c`
- Code version (git): 498126c; torch 2.11.0+cpu, mlflow 3.16.1, python 3.13.3
- Human approval: **not recorded** (see TODO.md)
