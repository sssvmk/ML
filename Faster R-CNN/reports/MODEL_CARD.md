# Model card: Faster R-CNN on Pascal VOC 2007 (<arch>-voc2007; version: fill from registry)

## Intended use
- Purpose: TODO: owner to complete (this repository fine-tunes a Faster R-CNN detector on Pascal VOC 2007 as a
  learning/benchmark exercise)
- Out-of-scope uses: TODO: owner to complete. Known limit: detects only the 20 VOC classes (aeroplane ... tvmonitor)
  in everyday photographs; anything else (other classes, aerial/medical/industrial imagery) is out of distribution.

## Data
- Source and period: Pascal VOC 2007 (Everingham et al.), trainval (5011 images) and test (4952 images)
- Split: 10% of trainval held out for validation and early stopping (seeded); test untouched until `test`;
  data version hash logged as the MLflow tag `data_version`
- Handling of `difficult` objects: used for training, ignored in evaluation (official VOC protocol)
- Known gaps or biases: TODO: owner to complete (VOC is Flickr photos from 2007; person-heavy; limited class set)

## Training procedure
- Model family and key hyperparameters: see `config/*.yaml`, logged as MLflow params. Pretraining: see `model.pretrained`
  (default `coco`: COCO detector weights with a new 21-way box predictor).
- Seeds, library versions, MLflow run id, git commit: MLflow tags and `config/versions.json`
- Regularisation used in this run: see params (`data.aug`, `model.box_head_dropout`, `training.weight_decay`,
  `training.ema.*`, `training.early_stopping.*`)

## Performance
- Primary metric and target: VOC2007 11-point mAP@0.5 (`map50_voc07`), target from `metric.target_value` (ASSUMED)
- Test-set result (evaluated once): TODO: fill from MLflow `test_*` metrics after the full run
- Also logged: mAP@0.5 (all-point), mAP@0.75, COCO-style mAP@[.5:.95], mAR@100, per-class AP@0.5, error breakdown
- Calibration of detection scores: not measured

## Limitations and failure modes
- See `qualitative_val.png`, `val_error_breakdown.json` and per-class AP of the run. TODO: owner to review.

## Ethical and fairness considerations
- Measured: per-class AP only. Not measured: anything about people or sensitive attributes (the `person` class is
  detected, but no demographic analysis was done). TODO: owner to complete.

## Monitoring and retraining
- Drift monitor: `src/monitor.py` (input channel means, detection-score PSI, detections per image, class mix);
  thresholds in `config/serve.yaml` are placeholders
- Retraining trigger and owner: TODO: owner to complete

## Approval
- Approver, date, gate results: TODO: owner to complete (written to `approvals/` by `python run.py promote`)
