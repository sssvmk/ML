# Model card: MLP-mnist

**Task:** classification   **Model:** fully connected network (ReLU hidden units, 669,706 parameters)
**Hidden layers:** [512, 512]   **Optimizer:** sgd (lr 0.15255, weight decay 0.002, dropout 0.2)

## Data and splits
- Source: torchvision.datasets.MNIST
- PRE-SPLIT by the dataset: official 60,000 train / 10,000 test (test set kept untouched). No validation set ships with MNIST, so 5,000 rows were carved out of the official train set (stratified by digit, seed 42); 55,000 rows remain for training.
- Sizes: {'train': 55000, 'val': 5000, 'test': 10000}
- Data notes (joins, dropped rows or columns, auto-detected task):
  - none
- Data audit: passed; warnings:
- none
- Split/data fingerprint: `c79f63bf7151b0d1`

## Test performance (test set used once, after model selection)
- ROC-AUC (macro one-vs-rest): 0.9998   [95% bootstrap CI 0.9997 - 0.9999, SE 0.0000]
- accuracy: 0.9839 +- 0.0013 (SE)
- log-loss: 0.0541 +- 0.0032 (SE)
- macro-F1: 0.9838

Standard errors and intervals reflect the finite size of the test set only; they do not include
variation between training runs with different seeds.

Validation (selection) metrics at the chosen epoch: {'val_loss': 0.0615, 'val_accuracy': 0.9828, 'val_auc': 0.9996}

## Training
- Hyperparameter search: Optuna TPE + MedianPruner, 30 trials (13 pruned); best validation loss 0.0958 vs 0.1184 for the default config
- Selection criterion: lowest validation loss; early stopping on validation loss (best epoch 26 of 30).
- Diagnosis: possible overfitting (val/train loss > 1.3) (val/train loss ratio 1.80; heuristic).

## Intended use and limitations
Trained on centred 28x28 handwritten digits (white on black). Expect lower accuracy on photos, other scripts, off-centre or noisy digits. Not evaluated for robustness or fairness.

## Governance and lineage
- Acceptance target: no acceptance target set
- Registry decision: promoted: val loss 0.0615 beats champion v1 0.0815 by >= 0.1%
- MLflow: experiment `MLP-mnist`, parent run `6a91565f7194435781955f1002790098`, final run `d763ac229f1b4b38b43ec443b881f083`
- Code version (git): 498126c; torch 2.11.0+cpu, mlflow 3.16.1, python 3.13.3
- Human approval: **not recorded** (see TODO.md)
