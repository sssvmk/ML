# Faster R-CNN on Pascal VOC 2007: training, MLflow tracking, registry, inference

R-CNN (2014) ran a CNN on ~2000 region proposals per image; Fast R-CNN shared one feature map; **Faster R-CNN** adds a
learned Region Proposal Network. This project fine-tunes torchvision's Faster R-CNN, the practical descendant, because
the original R-CNN is far too slow to train or serve on a laptop.

```
pip install -r requirements-base.txt               # install torch + torchvision first (CPU laptop: see below)
python run.py prepare-data                         # VOC2007 into data/ (see data/README.md if the proxy blocks it)
python run.py benchmark --set training.device=cpu  # seconds per iteration, minutes per epoch, ms per image
python run.py train --set training.device=cpu training.amp=false data.num_workers=2
python run.py train --config config/regularized.yaml --set training.device=cpu training.amp=false
mlflow ui --backend-store-uri sqlite:///mlflow.db  # http://127.0.0.1:5000
python run.py test --run-id <RUN_ID>               # test set, once (a second attempt is refused)
python run.py refit --run-id <RUN_ID>              # optional: retrain on ALL trainval for the best epoch count
python run.py register --run-id <RUN_ID>           # package, register, reload in a fresh process, alias `candidate`
python run.py promote --version 1 --approver "<name>"   # gates, then alias `champion`
python run.py predict --model-uri models:/<registered-name>@champion --image photo.jpg
uvicorn src.api:app --port 8000                    # POST /predict (multipart image), GET /health
```
Pass the same `--config` (and `--set` overrides that change the model) to `test`, `register`, `promote`.
Registry name is `<arch>-<dataset>`; override with `REGISTERED_MODEL_NAME`.
CPU laptop: with torch 2.11.0+cpu install torchvision 0.26.0 (matching pair). Use `--set training.device=cpu training.amp=false`.

## Architecture & Pipeline Overview

### Overall Architecture
The core architecture is based on Faster R-CNN with a Feature Pyramid Network (FPN). The default model is `fasterrcnn_mobilenet_v3_large_320_fpn`. Other supported architectures include `fasterrcnn_mobilenet_v3_large_fpn`, `fasterrcnn_resnet50_fpn`, and `fasterrcnn_resnet50_fpn_v2`. The model relies on COCO-pretrained weights as a starting point, which are then fine-tuned for a customized number of classes (20 for VOC + 1 for background).

### Specific Layers Used
*   **Backbone:** MobileNetV3-Large or ResNet50. The number of trainable backbone layers is configurable (default is 3, to prevent overfitting and speed up training).
*   **FPN (Feature Pyramid Network):** Used across all supported architectures to extract multi-scale features.
*   **RPN (Region Proposal Network):** Inherited directly from `torchvision`. Images are resized to a random `min_size` (e.g., 320) during training.
*   **RoI Heads:** The box predictor is replaced with a `FastRCNNPredictor`. It uses Detectron-style initialization: tiny normal distributions (std=0.01 for classification, std=0.001 for bounding boxes) with zeroed biases. An optional dropout layer can be applied to the box head representation.

### Activation Layers
The model relies on the default activation layers embedded within the `torchvision` backbones and heads (e.g., `Hardswish` and `ReLU` for MobileNetV3, `ReLU` for ResNet50).

### Normalization Layers (Norms)
The architecture uses `nn.BatchNorm2d`. However, during training, a custom `set_loss_mode()` function is used which forces `BatchNorm2d` (and `Dropout`) layers into `eval()` mode. This freezes the batch statistics (no stat updates) which is standard practice when fine-tuning detection models on small batch sizes.

### Optimizer Details
*   **Algorithm:** SGD with momentum (0.9) is the default; `AdamW` is also supported.
*   **Parameter Grouping:** Weight decay (0.0001) is applied *only* to weights. Biases and normalization parameters are excluded from decay.
*   **Gradient Clipping:** Gradients are clipped at a max norm of 10.0 to prevent rare exploding box-regression gradients.
*   **Backbone LR Multiplier:** Supports discriminative learning rates, allowing the pretrained backbone to be trained with a lower multiplier (defaults to 1.0).

### Learning Rate Schedule
*   **Base LR:** 0.005 (derived from the linear-scaling rule: 0.02 for batch size 16 -> 0.005 for batch size 4).
*   **Warmup:** Linear warmup for the first 500 iterations (starting at a factor of 0.001).
*   **Main Schedule:** After warmup, defaults to a `cosine` annealing schedule down to a minimum ratio of 0.01. Alternative supported schedules are `multistep` (milestones at 67% and 89% with gamma 0.1) and `plateau` (patience 2, gamma 0.1).

### Datasets and Data Pipeline
*   **Dataset:** Pascal VOC 2007 (20 foreground classes + background).
*   **Splits:** 10% of the `trainval` dataset is deterministically carved out for validation. The `test` set is left untouched for final evaluation.
*   **Dataloader:** Batch size of 4, with 4 workers. A custom collate function is used to handle variable-length target dictionaries.
*   **Augmentation (torchvision v2):** By default, uses a `standard` pipeline (Random Photometric Distort 50%, Random Horizontal Flip 50%, and Sanitize Bounding Boxes). A `strong` SSD-style recipe is available which adds Random Zoom Out and Random IoU Crop. Data is represented in float32 XYXY format.

### Overall Execution Pipeline
*   **Pre-flight:** Starts with a deterministic sanity check to evaluate initial classifier loss and verify that the model can perfectly overfit on a tiny batch.
*   **Training Loop:** Iterates over the training dataloader using AMP (FP16 Autocast). Aggregates four losses (`loss_classifier`, `loss_box_reg`, `loss_objectness`, `loss_rpn_box_reg`). Updates the Exponential Moving Average (EMA) model at every step.
*   **Validation:** At the end of each epoch, computes validation loss and evaluation metrics (mAP, mAP50) on the val subset using `DetectionEvaluator`.
*   **Early Stopping & Checkpointing:** Stops training if the target metric does not improve for a set number of epochs (default 3). Saves `best.pt` and `last.pt`.
*   **Logging & Diagnostics:** Tracks metrics, configs, and models in MLflow. Generates a comprehensive diagnosis post-training, including per-class error breakdowns, qualitative grid images, and training curve plots to offer "next steps" advice.

## Choices and why
| Area | Choice | Why |
|---|---|---|
| Pretraining (Ch. 8.7.4) | COCO detector, new 21-way box predictor (`model.pretrained`: coco / imagenet / none) | A few thousand images cannot train a detector from scratch; COCO contains all 20 VOC classes, so results are not comparable with the 2007-era papers |
| Frozen early layers + frozen BatchNorm (8.7.1) | `trainable_backbone_layers=3`; torchvision freezes BN when pretrained | Batches of 4 give unreliable BN statistics; fewer trainable parameters overfit less |
| Weight decay (7.1) | 1e-4 on weights only, none on biases/norm | Standard for SGD detectors; 5e-4 in `regularized.yaml` |
| Dataset augmentation (7.4) | `data.aug`: none / flip / standard (+photometric) / strong (+zoom-out, IoU-crop), multi-scale `min_size` | Usually the strongest anti-overfitting lever for detection; boxes follow the image and degenerate boxes are dropped |
| Dropout (7.12) | Off by default; `model.box_head_dropout=0.2` option | FPN detectors rely on pretraining, augmentation and decay; the book notes dropout costs capacity and helps less with little data, so it is a tested option, not a default |
| Early stopping (7.8) | Stop on validation mAP, patience 3, min_delta 0.002; best epoch saved | Cheapest regulariser; picks the training length automatically |
| Early-stopping refit (Alg. 7.2) | `python run.py refit`: retrain on train+val for best_epoch+1 epochs | Uses the validation images for learning once the length is known |
| Optimiser (8.3) | SGD + momentum 0.9 (AdamW selectable) | The reference recipe for CNN R-CNN family; Adam-type optimisers are mostly used for transformer detectors |
| Adaptive learning rate (8.5) | Linear warmup, then cosine (multistep and validation-driven plateau selectable); `backbone_lr_mult` for a lower backbone LR | Schedules, not per-parameter adaptivity, are the norm here; plateau reacts to the validation metric |
| Gradient clipping | max-norm 10 | Guards against rare exploding box-regression gradients |
| Initialisation (8.4) | Pretrained weights; new predictor N(0, 0.01) / N(0, 0.001), zero bias | Starts the classifier near uniform: initial loss ~ ln(K+1), checked before training |
| Polyak averaging (8.7.3) | EMA of weights (decay 0.999 with ramp); EMA weights are evaluated and saved | Smooths noisy SGD steps; a cheap form of ensembling |
| Loss | Faster R-CNN multi-task: RPN objectness (BCE) + RPN box (smooth L1) + RoI class (cross-entropy) + RoI box (smooth L1) | Logged separately every epoch, so you can see which part overfits or stalls |
| Metrics (Ch. 11) | Primary: VOC07 11-point mAP@0.5. Also mAP@0.5 (all-point), mAP@0.75, COCO-style mAP@[.5:.95], mAR@100, per-class AP, error breakdown | Accuracy is meaningless for detection; difficult objects are ignored per the VOC protocol |
Everything above is a reasoned default, not a result measured here. Change one lever at a time and compare in MLflow.

## Preventing over- and underfitting
- Each epoch logs train loss, validation loss (BN and dropout frozen), validation mAP and a train-subset mAP (`eval_train_subset`).
- After training, `scripts/diagnose_curves.py` classifies the curves (healthy / overfitting / underfitting / unstable /
  not_learning / still_improving); `next_steps.json` lists concrete levers for the verdict (e.g. `data.aug=strong`,
  `box_head_dropout=0.2`, more frozen layers for overfitting; more trainable layers, higher resolution, longer training for underfitting).
- Pre-training sanity checks (`sanity.json`): initial classifier loss ~ ln(K+1), and a 4-image batch must be memorised.
- Early stopping + best-epoch checkpointing + EMA limit the damage of late overfitting.
- `config/regularized.yaml` bundles the overfitting levers as a hypothesis to compare against the baseline.

## What a run produces
MLflow params (whole config), per-epoch metrics, tags (git commit, data version, device), `metrics.csv`, `curves.png`,
`diagnosis.json`, `next_steps.json`, `sanity.json`, `qualitative_val.png` (green = truth, red = prediction),
`val_per_class_ap50.json`, `val_error_breakdown.json` (correct / duplicate / localization / confusion / background / missed),
`reference_stats.json` (drift monitoring), `checkpoints/best.pt`. Local copies in `reports/runs/<run_id>/`.

## Model contract
MLflow pyfunc: input DataFrame column `image_b64` (base64 PNG/JPEG, any size); output column `detections` (JSON:
`width`, `height`, `detections[{box [x1,y1,x2,y2] in original pixels, score, label_id, label}]`); optional param `score_floor`.
`Predictor` / the API filter by `score_threshold` (default 0.5) and `max_detections`.

## Promotion gates
Test set used exactly once; target met (or a written reason); load-and-predict verified in a fresh process; sanity
checks passed (a refit run inherits them); not worse than the champion on validation (a refit run uses the validation
metric of its source run); model card exists; approver named. Each attempt is written to `approvals/`.

## Tests
`pytest -q` (about 1.5 minutes on CPU): metric maths, VOC parsing on a fake tree, augmentation validity, schedulers,
early stopping, EMA, offline pretrained-weight paths, and the full train -> test -> register -> promote -> predict ->
refit -> resume flow on synthetic data. No downloads.
