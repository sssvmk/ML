# Model card: <arch>-<dataset> (e.g. resnet20-cifar10; version: fill from registry)

## Intended use
- Purpose: TODO: owner to complete (this repository trains CIFAR image classifiers as a learning/benchmark exercise)
- Out-of-scope uses: TODO: owner to complete. Known limit: trained on 32x32 CIFAR-10 or CIFAR-100 images with fixed classes;
  arbitrary photos, other resolutions' fine detail and unseen classes are out of distribution.

## Data
- Source and period: CIFAR-10 / CIFAR-100 (Krizhevsky, 2009), as shipped by torchvision
- Size, split scheme, data hash: 50k train (10% stratified validation) / 10k test; hash in the MLflow `data_version` tag
- Known gaps or biases: TODO: owner to complete

## Training procedure
- Model family and key hyperparameters: ResNet or VGG (see `config/train.yaml`, logged as MLflow params)
- Seeds, library versions, MLflow run id, git commit: logged as MLflow tags and `config/versions.json`

## Performance
- Primary metric and target: top-1 accuracy, target from config `metric.target_value` (ASSUMED, not confirmed)
- Test-set result (evaluated once): TODO: fill from MLflow `test_top1` after the full run
- Slice metrics: per-class test accuracy in `test/test_per_class.csv`. No other slices measured.
- Calibration: not measured

## Limitations and failure modes
- See `worst_errors/` artifacts of the run (most confidently wrong validation images). TODO: owner to review.

## Ethical and fairness considerations
- Measured: per-class accuracy only
- Not measured: anything about people or sensitive attributes
- Judgment and thresholds: TODO: owner to complete

## Monitoring and retraining
- Drift and performance monitors: `src/monitor.py` (input channel statistics, prediction-confidence PSI); thresholds in
  `config/serve.yaml` are placeholders
- Retraining trigger and owner: TODO: owner to complete

## Approval
- Approver, date, gate results: TODO: owner to complete (written to `approvals/` by `python run.py promote`)
