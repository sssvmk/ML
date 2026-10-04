# Model card: MNIST digit classifier (Dataset augmentation)

*Generated from logged results. Items marked TODO need a human owner.*

## Model
- Fully connected ReLU network trained with SGD + momentum and exponential learning-rate decay.
- Selected regularization method: **Dataset augmentation** (book section 7.4), selected by validation accuracy among full-data methods.
- Hyperparameters (tuned with Optuna on the validation set): `{'lr': 0.019733115965076906, 'max_shift': 1.6319239585843495, 'max_rot': 13.681695777998716, 'max_scale': 0.004530608029935657}`

## Intended use
- TODO: owner to complete. This pipeline is a study of regularization methods on MNIST; it is not validated for any production decision.

## Data
- MNIST, source `mnist`; the dataset ships pre-split into train/test only, so a stratified validation set was carved from train.
- Sizes: {'train': 49999, 'val': 10001, 'test': 10000}; data hash `1e35582d42a90940`.
- Audit: duplicates train/val=0, train/test=0, val/test=0.

## Metrics (test set, evaluated once for this model)
- Accuracy: 0.9874 ± 0.0011 (SE)
- AUC (macro one-vs-rest): 0.99988, 95% bootstrap CI [0.9998359978862659, 0.9999198770869059]
- Cross-entropy: 0.0393
- FGSM (eps=0.1) accuracy on validation: 0.110

## Limitations
- Handwritten digits only; no evidence about other image types or distribution shift.
- Slice, calibration and fairness analyses were not performed. TODO: owner to decide whether they are needed.

## Monitoring plan
- TODO: owner to define input-drift and performance-decay thresholds.

## Version
- Code commit: `498126c`; torch 2.11.0+cpu; mlflow 3.16.1.
