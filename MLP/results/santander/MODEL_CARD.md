# Model card: MLP-santander_train

**Task:** classification   **Model:** fully connected network (ReLU hidden units, 6,994 parameters)
**Hidden layers:** [32, 16]   **Optimizer:** adamw (lr 0.0046039, weight decay 1.5e-05, dropout 0.30000000000000004)

## Data and splits
- Source: csv:santander_train.csv
- NOT pre-split: a single table of 200,000 rows was split 120,000 train / 40,000 validation / 40,000 test (stratified by class, random, seed 42). Preprocessing (imputation, scaling, one-hot) was fit on the training rows only.
- Sizes: {'train': 120000, 'val': 40000, 'test': 40000}
- Data notes (joins, dropped rows or columns, auto-detected task):
  - none
- Data audit: passed; warnings:
- none
- Split/data fingerprint: `d1cc0417a33ff3a5`

## Test performance (test set used once, after model selection)
- ROC-AUC (macro one-vs-rest): 0.8520   [95% bootstrap CI 0.8464 - 0.8579, SE 0.0029]
- accuracy: 0.9113 +- 0.0014 (SE)
- log-loss: 0.2386 +- 0.0028 (SE)
- macro-F1: 0.6434
- reference (not tuned): logistic-regression AUC 0.8544, majority-class accuracy 0.8995

Standard errors and intervals reflect the finite size of the test set only; they do not include
variation between training runs with different seeds.

Validation (selection) metrics at the chosen epoch: {'val_loss': 0.2404, 'val_accuracy': 0.912, 'val_auc': 0.8474}

## Training
- Hyperparameter search: Optuna TPE + MedianPruner, 30 trials (17 pruned); best validation loss 0.2404 vs 0.2466 for the default config
- Selection criterion: lowest validation loss; early stopping on validation loss (best epoch 5 of 13).
- Diagnosis: no large train/validation gap (val/train loss ratio 1.00; heuristic).

## Intended use and limitations
Trained on csv:santander_train.csv (target: target). Valid only for data that looks like the training file. Inputs far outside the training range are flagged but still scored. Not evaluated for robustness or fairness.

## Governance and lineage
- Acceptance target: no acceptance target set
- Registry decision: promoted: first version, no champion yet
- MLflow: experiment `MLP-santander_train`, parent run `ef5f6b081f3e40a2a39d3cbb70df8aa0`, final run `1c79d39f54fd477c8f00c8574b613aec`
- Code version (git): 498126c; torch 2.11.0+cpu, mlflow 3.16.1, python 3.13.3
- Human approval: **not recorded** (see TODO.md)
