# SSD (Liu et al., ECCV 2016) with a ResNet-50 base network: one sequential pipeline

Two inputs only: **`PATH`** (everything is stored there) and **where to train** (`cpu`, `gpu` or `combination`).

| Deliverable | How to run |
|---|---|
| Databricks notebook `notebooks/ssd_pipeline_databricks.ipynb` (or the `.py` source-format file) | import it, fill the widgets `path` and `device`, **Run all** |
| Plain Python, one file `dist/ssd_pipeline.py` | `python ssd_pipeline.py PATH [cpu\|gpu\|combination]` (asks when the device is omitted) |
| Plain Python, package | `python run_pipeline.py PATH [cpu\|gpu\|combination]` |

Both are generated from the package `ssd_voc/` by `python tools/build_notebook.py`, so they run the same code (and the same tests).

## The sequence (no flags)
1. **Load datasets:** VOC2012 trainval (10% held out for validation), VOC2007 test, COCO.
2. **Detection training** on VOC2012 -> **test** once on VOC2007 test -> **save** `PATH/models/01_detection_voc`.
3. **Optional second round:** train on COCO -> fine-tune on VOC2012 -> test -> **save the final model** `PATH/models/02_final_second_round`.
4. **Longer schedules** (zoom-out augmentation, twice the iterations) **starting from the saved final model**, with the respective datasets: COCO, then VOC2012 -> test -> save `PATH/models/03_final_longer_schedule`.
5. Summary table (`PATH/pipeline_summary.json`).

Stage 1 of the paper (ImageNet pretraining) is replaced by torchvision's pretrained ResNet-50.
Each saved model folder holds `checkpoint.pt`, `torchscript.pt`, `priors.npy`, `meta.json`, `ssd_pyfunc.py`; it is also logged to MLflow as a self-contained pyfunc and checked to reload and predict identically in a fresh process.

## Device modes
| Mode | Meaning |
|---|---|
| `cpu` | everything on the CPU |
| `gpu` | the model trains on the GPU; CPU workers load and augment images; all datasets are prepared before training; fails clearly if no GPU is visible |
| `combination` | as `gpu`, plus the CPU works in parallel: the COCO download for steps 3-4 runs in a background thread while the GPU trains step 2 |

PyTorch cannot split one training run across CPU and GPU, so `combination` means overlapping CPU work (data download) with GPU training, not a hybrid model.

## Resume and time
- Run again with the same `PATH` after an interruption: finished steps are skipped, an interrupted training resumes from its last checkpoint (`PATH/pipeline_state.json`).
- Step 1 measures the machine and prints an upper-bound estimate. The full paper schedules are long (VOC 80k iterations, COCO 240k, each doubled in step 4); early stopping usually ends steps sooner. On a CPU the full pipeline is impractical.

## Overfitting
Every one of the five trainings uses the paper's augmentation, weight decay on weights only, frozen BatchNorm and early layers, EMA weights, and validation-based early stopping with LR phases (plateau -> reload best weights -> next lower LR; plateau in the last phase -> stop). The best-validation checkpoint is the one tested and saved. The train-minus-validation mAP gap is logged each evaluation, with a warning above 0.20, and sanity checks run before each training. This reduces the risk; it cannot guarantee that no model overfits.
The test set is used once per saved model (three models). Choose the final model by **validation** mAP.

## Where things live under PATH
`data/`, `cache/` (torch weights, matplotlib, CUDA), `tmp/`, `runs/<run_id>/` (metrics.csv, plots, diagnosis, checkpoints), `checkpoints/<stage>_best.pt`, `models/`, `mlflow/` (SQLite store off Databricks), `pipeline_state.json`, `pipeline_summary.json`.
MLflow: off Databricks the SQLite store is under PATH. On Databricks the workspace is used (experiment `/Shared/ssd-pipeline`, artifact location inside PATH).

## Other files
`run.py` runs single stages (`python run.py train --dest PATH --stage voc ...`); `README` of that interface is in `run.py`'s docstring. Details of the SSD implementation (box counts, loss, initialisation, deviations from the paper) are in `TODO.md` and the module docstrings.

## Tests
`pytest -q`: about 100 tests, no downloads: paper box counts (8732 / 24564), matching and loss maths, model shapes and freezing, VOC/COCO parsing and splits, augmentation, downloader, LR phases, early stopping, stage-level training/resume/fine-tuning, and the pipeline end to end on a synthetic dataset (order, init chain, three tests, three saved models, resume, skip, device modes, background download, the generated notebook run cell by cell with a fake `dbutils`, and the single-file script).
