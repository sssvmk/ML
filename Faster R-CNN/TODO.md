# TODO: what only you can do

## What was run and what was not
- Run (sandbox, 1 CPU core, synthetic data and a fake VOC2007 tree): 35 tests pass. They cover metric maths against
  hand-computed cases, VOC parsing, augmentation validity, schedulers, early stopping, EMA, the offline COCO / ImageNet
  weight-loading paths (locally built state dicts), and the full flow train -> test (second look refused) -> register ->
  fresh-process reload -> promote -> predict -> refit -> resume. `ruff check` passes.
- Measured in the sandbox only (1 CPU core, batch 4, random images, weights not loaded): MobileNetV3-320 FPN 1.76 s per
  training iteration (about 33 min per epoch, 6.6 h for 12 epochs), 88 ms per image inference; MobileNetV3 FPN at 480 px
  2.9 s per iteration, 403 ms per image. Your 8-core laptop should be faster: run `python run.py benchmark`.
- NOT run: the real Pascal VOC 2007 download and any real training (the sandbox cannot reach the dataset host), the real
  COCO / ImageNet weight download (`model.pretrained=coco` is the default), GPU or fp16 paths, Docker, GitHub Actions,
  the FastAPI service against a real registered model (stub predictor only). No mAP has been measured on real data.

## Blocking (do before trusting any result)
- [ ] Download VOC2007: `python run.py prepare-data` (manual URLs in data/README.md if the proxy blocks it)
  - Done when: data/VOCdevkit/VOC2007 exists
- [ ] Confirm the model-weights download works on your network (torchvision fetches COCO weights from download.pytorch.org
  on first use). If blocked, use `--set model.pretrained=imagenet` (also downloads) or download the weight file manually
  and place it in the torch hub cache, or accept `model.pretrained=none` (will train much worse)
- [ ] Run `python run.py benchmark` and decide epochs / resolution from the projected hours
  - Where: config/train.yaml `training.max_epochs`, `model.min_size`
- [ ] Confirm the target: assumed VOC07 mAP@0.5 of 0.60 (config `metric.target_value`)
  - Why: Checkpoint 1 was not answered; the promotion gate uses it
- [ ] Run the baseline, read `diagnosis.json` / `next_steps.json`, then `test` exactly once for the final candidate

## Needed for production
- [ ] Shared MLflow tracking server and registry (currently a local SQLite file): config/env.yaml or MLFLOW_TRACKING_URI
- [ ] Choose serving mode, latency/throughput budget and hardware (config/serve.yaml is placeholders; MobileNet-320 is fast, bigger models are not)
- [ ] Schedule `src/monitor.py` (library only) and set drift thresholds; decide how ground-truth labels would come back
- [ ] Build and test the Docker image; run CI once on your runner
- [ ] If you train on a GPU, check `training.amp=true` and the whole pipeline once on that machine

## Deferred decisions
- [ ] Compare baseline vs `config/regularized.yaml` (and single levers) over several seeds before believing small differences
- [ ] Larger backbones (`fasterrcnn_resnet50_fpn[_v2]`), higher resolution, GIoU box loss, mixup/mosaic augmentation: not implemented or run

## Governance and handoff
- [ ] Complete reports/MODEL_CARD.md (intended use, limitations, fairness section) and name an approver
- [ ] Name a retraining owner and trigger; rollback = move alias `champion` back to the previous version

## Assumption log
- Faster R-CNN (not the original R-CNN) because the original is impractical to train and serve on a laptop
- Metric VOC07 11-point mAP@0.5, target 0.60 (not confirmed); early stopping and model selection use the same metric
- Split: 10% of VOC2007 trainval held out (seed 42); test (4952) used once; `difficult` objects trained on, ignored in evaluation
- Baseline: MobileNetV3-Large 320 FPN, COCO-pretrained, SGD lr 0.005 batch 4, cosine, EMA, standard augmentation, 12 epochs
- Dropout off by default; box-head dropout, strong augmentation and discriminative LR are options (regularized.yaml)
