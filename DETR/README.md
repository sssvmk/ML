# DETR (Carion et al., ECCV 2020, arXiv 2005.12872) with an ImageNet-pretrained ResNet-50

Two inputs only: **`PATH`** (everything is stored there) and **where to train** (`cpu` or `gpu`).

| Deliverable | How to run |
|---|---|
| Single file `dist/detr_pipeline.py` (keep this file name) | `python detr_pipeline.py PATH [cpu\|gpu]` (asks when the device is omitted) |
| Package | `python run_pipeline.py PATH [cpu\|gpu]`; single steps: `python run.py {prepare-data\|benchmark\|train\|test\|package\|predict} --dest PATH [--stage voc\|coco]` |

The single file is generated from the package by `python tools/build_script.py` (same code, same tests).

## The sequence
1. **Load datasets:** VOC2012 trainval (10% held out for validation) and VOC2007 test. (`--stage coco` in the package CLI uses COCO train2017, the paper's dataset.)
2. **Train DETR in a single phase**, from the ImageNet-pretrained torchvision ResNet-50 (stage 1 of any pipeline is the pretrained weights, cached under `PATH`), with early stopping.
3. **Test once** on VOC2007 test; refuses a second look at the test set.
4. **Save** `PATH/models/detr_<dataset>/`: `checkpoint.pt` and a self-contained MLflow pyfunc (also logged in the run), reloaded and checked in a fresh process.
Interrupted runs resume when you run again with the same `PATH` (`pipeline_state.json`). Step 1 prints a time estimate.

## What follows the paper (verified by tests where noted)
| Item | Here |
|---|---|
| Architecture | ResNet-50 (stem and layer1 frozen, frozen BatchNorm) -> 1x1 conv to d = 256 -> 6 encoder + 6 decoder layers, 8 heads, FFN 2048, dropout 0.1, 100 object queries, sine positional encodings added at every attention layer, shared decoder LayerNorm, class head (K + no-object) and 3-layer box MLP. **41.5M parameters (41.3M trainable): matches the paper's 41.3M** (tested) |
| Loss | Hungarian matching with costs 1 / 5 / 2 (class / L1 / GIoU); loss = CE (no-object weight 0.1) + 5 L1 + 2 GIoU, normalised by the number of boxes; auxiliary losses after every decoder layer. Uniform logits give exactly ln(K+1) and perfect predictions give 0 (tested) |
| Optimiser | AdamW, transformer LR 1e-4, backbone 1e-5, weight decay 1e-4, gradient clipping 0.1, Xavier initialisation of the transformer; one LR step-down x0.1 (paper ablation schedule: 300 epochs, drop after 200; COCO stage 500 epochs, drop after 400) |
| Augmentation | horizontal flip; with p 0.5 a random resize (shorter side 480..800, max 1333), otherwise resize 400/500/600 -> random crop 384..600 -> random resize (official recipe) |
| Inference | one detection per query, class = best real class (the paper's "second-highest class" override for empty slots), no NMS |
| Hungarian solver | scipy if it imports, a tested numpy solver otherwise (a broken scipy install, like the one on your cluster, would otherwise stop training) |

**Deviations:** batch size 16 instead of 64 (one GPU); bf16 on the A100; early stopping, best-checkpoint selection and sanity checks are added; EMA is optional (off).

## Overfitting
Dropout 0.1, weight decay, auxiliary losses, the paper's crop and scale augmentation, frozen BN, a 10x lower backbone LR, and validation-based early stopping: when the validation mAP stops improving the run reloads the best weights and takes the next (lower) LR step early; a plateau in the last phase stops training. The best-validation epoch is the one tested and saved. The train-minus-validation mAP gap is logged at every evaluation (warning above 0.20), sanity checks run before training (initial class loss near ln(K+1), a tiny batch must be fitted), and `next_steps.json` lists levers for overfitting or underfitting. This reduces the risk; it cannot guarantee it.

## Honest expectations
DETR converges slowly (the paper trains 300 to 500 epochs on COCO's 118k images) and is known to struggle on small datasets from an ImageNet-only backbone (the DETR authors reported success with 10-15k images and none on fewer). VOC2012 has about 11.5k trainval images, so expect a long run and results below the paper's COCO numbers; if the metric stays near zero for many epochs, `next_steps.json` suggests more data (add VOC2007 trainval, or the COCO stage), a longer schedule, or fine-tuning a COCO-trained DETR instead.

## Where things live under PATH
`data/`, `cache/` (torch weights, matplotlib, CUDA), `tmp/`, `runs/<run_id>/` (metrics.csv, plots, diagnosis, checkpoints), `checkpoints/<stage>_best.pt`, `models/`, `mlflow/` (SQLite store off Databricks; the Databricks workspace is used on Databricks), `pipeline_state.json`, `pipeline_summary.json`.

## Tests
`pytest -q`: 71 tests, no downloads: parameter count, shapes, masks, frozen layers, init, Hungarian solver vs scipy and the broken-scipy fallback, matcher and loss maths, post-processing, augmentation, padded batches, VOC/COCO parsing and splits (no test leakage), downloader, LR phases, early stopping, resume, EMA, test-once, packaging with a fresh-process reload, and the pipeline end to end on a synthetic dataset (skip on rerun, resume after a simulated failure, device questions, the single-file script).
