# Model card: MNIST digit classifier (training technique: Batch normalization)

*Generated from logged results. Items marked TODO need a human owner.*

## Model
- Fully connected network trained with the technique below (L1 + L2 penalties on all methods).
- Selected technique: **Batch normalization** (book section 8.7.1), selected by validation accuracy among the methods evaluated on the full network and full data.
- Hyperparameters (tuned with Optuna on the validation set): `{'lr': 0.0730953983591291, 'l2': 5.589524205217922e-07, 'l1': 1.146210740342504e-06}`

## Intended use
- TODO: owner to complete. This pipeline is a study of optimization techniques on MNIST; it is not validated for any production decision.

## Data
- MNIST, source `mnist`; the dataset ships pre-split into train/test only, so a stratified validation set was carved from train.
- Sizes: {'train': 49999, 'val': 10001, 'test': 10000}; data hash `1e35582d42a90940`.
- Audit: duplicates train/val=0, train/test=0, val/test=0.

## Metrics (test set, evaluated once for this model)
- Accuracy: 0.9824 ± 0.0013 (SE)
- AUC (macro one-vs-rest): 0.99981, 95% bootstrap CI [0.9997669013635972, 0.9998667563868587]
- Cross-entropy: 0.0628

## Limitations
- Handwritten digits only; no evidence about other image types or distribution shift.
- Slice, calibration and fairness analyses were not performed. TODO: owner to decide whether they are needed.

## Monitoring plan
- TODO: owner to define input-drift and performance-decay thresholds.

## Version
- Code commit: `498126c`; torch 2.11.0+cpu; mlflow 3.16.1.
