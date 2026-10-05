# TODO: what only you can do

## Update: sequential pipeline (notebook + single-file Python)
- Interpretations I made (tell me if you meant otherwise): (1) `combination` = GPU trains while the CPU downloads the later datasets in the background (PyTorch cannot split one training run across CPU and GPU); (2) the longer-schedule step starts from the SAVED final model of the second round and trains COCO (long) then VOC2012 (long), re-initialising the class heads when the dataset changes (20 vs 80 classes); (3) "test and save" happens after each VOC-based model (detection, second round, longer schedule); the COCO steps have no test set.
- Verified here: 100 tests on CPU with a tiny ResNet-18 SSD and a synthetic dataset (order of steps, init chain, one test per saved model, resume of an interrupted step, skip on rerun, device modes, background download, the generated notebook run cell by cell, the single-file script). Not verified: real VOC/COCO downloads, the ImageNet ResNet-50 weight download, the A100 path, Databricks workspace tracking, speed, mAP.
- Expect hours to days on the A100 for the whole sequence (step 1 prints an estimate). On a CPU it is impractical.


## What was run and what was not
- Run (sandbox: 1 CPU core, no GPU, synthetic data and fake VOC/COCO trees): 75 tests pass; the training loop was shown to learn (synthetic validation
  mAP rose from ~0 to 0.94-1.0 within 100-200 iterations); the notebook runs every step end to end.
- NOT run: real VOC2012 / VOC2007 / COCO downloads and training; the ImageNet ResNet-50 weight download (default `model.pretrained=imagenet`);
  anything on the A100 (bf16, channels_last, cudnn.benchmark); Databricks tracking (`mlflow.backend=databricks`) and Unity Catalog registration;
  reading data from a Volume under 20 workers; the speed numbers (run `benchmark` on your cluster). No mAP exists for real data.
- Known unknowns: iterations per second on your cluster, whether the SSD300-ResNet50 reaches the assumed target, whether VOC2012-only training
  is enough (the paper trained 07++12), whether 20 workers keep the A100 busy.

## Do first
- [ ] Use the Machine Learning GPU runtime (the screenshot showed "16.4 LTS" without ML). The notebook's first cell checks torch, torchvision, mlflow.
- [ ] Create the volume and pass `dest`, e.g. `/Volumes/main/default/ssd` (`CREATE VOLUME main.default.ssd`).
- [ ] `benchmark`: read `bottleneck` (model vs data loading) and `est_hours_if_run_to_the_end`; adjust `data.num_workers` (24 vCPUs).
- [ ] Confirm or change the target: `metric.target_value` 0.65 is ASSUMED.
- [ ] Run `prepare_data` once (VOC2012 ~2 GB, VOC2007 test ~1 GB; COCO train ~18 GB for stages 3/4) and check the proxy allows the hosts.
- [ ] Run stage `voc`, read `diagnosis.json`/`next_steps.json`/`qualitative_val.png`, then `test` ONCE for the final candidate.

## Decisions for you
- [ ] Storage speed: training reads many small JPEGs from `dest`. If `dest` is a Volume and the data loader is the bottleneck, point `dest` at faster
  storage (you cannot write `/local_disk0` today, so ask for access or use a bigger page cache).
- [ ] MLflow on Databricks: `mlflow.backend=databricks` needs `mlflow.experiment_name` as an absolute workspace path; registering needs a Unity
  Catalog name `catalog.schema.model` (`mlflow.registered_model_name`). Keep `sqlite` if everything must stay in `dest`.
- [ ] Paper fidelity vs data: add `"2007_trainval"` to `data.voc_train_sets` for the paper's 07+12 style training data.
- [ ] `torch.jit.trace` is deprecated in recent PyTorch; the registered pyfunc uses it today. Move to `torch.export` when you upgrade.

## Not built
- FastAPI service, drift monitor and promotion gates from the earlier R-CNN project (the registered pyfunc and `alias candidate` are here).
- SSD with VGG16, hyper-parameter search, multi-seed comparison (compare stages over several seeds before trusting small differences).
- GPU augmentation (DALI): only needed if `benchmark` shows data loading as the bottleneck.

## Assumptions
- ResNet-50 (ImageNet V1 weights) as the base network; input 300; stage 1 replaced by pretrained weights.
- Validation = 10% of VOC2012 trainval; final test = VOC2007 test (disjoint from VOC2012 trainval); COCO stage validates on 2,000 val2017 images.
- Primary metric all-point AP@0.5; early stopping and model selection use it; the test set is used once per run.
