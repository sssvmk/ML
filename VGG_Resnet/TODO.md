# TODO: what only you can do

## Update: ResNet-20/32/56 + CIFAR-10 added
- Run: 31 tests pass (ResNet parameter counts 0.27M / 0.46M / 0.85M match the paper; depth 6n+2 checked; fake CIFAR-10
  in torchvision's on-disk format loads; train -> test -> register -> reload -> promote -> predict passes for both a
  VGG and a ResNet-20 on synthetic data; CLI run for ResNet-20 on synthetic data).
- Measured in the sandbox (1 CPU core, batch 128, fp32): resnet20 0.73 s/step, 4.3 min/epoch; resnet32 1.23 s/step,
  7.2 min/epoch; resnet56 2.24 s/step, 13.1 min/epoch. Your 8-core CPU should be faster; run `python run.py benchmark`.
- Not run: real CIFAR-10 download and training (sandbox cannot reach the dataset host), so no accuracy has been measured.
- [ ] Confirm the target: assumed top-1 accuracy 0.88 for ~40 epochs (the paper reports roughly 91-93% with a much longer schedule)
  - Where: config/*.yaml `metric.target_value`
  - Done when: you edit or confirm it after the first real run
- [ ] Real CIFAR-10 run for ResNet-20, then 32 and 56, compared in MLflow; test set once per final candidate
- [ ] ResNet-20/32/56 differ only in depth: compare them across several seeds before calling one better

## What was run and what was not
- Run (sandbox, 1 CPU core, synthetic data, mlflow-skinny 3.16.1): 13 tests pass, including the end-to-end smoke test
  train -> sanity checks -> one-time test eval (second attempt refused) -> register -> reload in a fresh process ->
  promotion gates -> predict. CLI commands train/test/register/predict/benchmark were run once each. Training loop
  learned an easy synthetic task to 100% validation accuracy. `ruff check` passes.
- Measured in the sandbox only (1 CPU core, batch 128): VGG16 full 2.66 s/step (15.6 min/epoch), VGG16 width 0.5
  0.74 s/step (4.3 min/epoch), VGG11 width 0.5 0.40 s/step (2.3 min/epoch).
- Added for your laptop: `gpu-check`, ROCm Windows setup script, GPU-aware device selection. Only the CPU-only path was run (18 tests pass); no AMD GPU exists in the sandbox, so the ROCm path, GPU-vs-CPU correctness check and fp16 path are untested.
- Not run: real CIFAR-100 training (the sandbox cannot reach the dataset host), any GPU / mixed-precision path, Docker
  build, GitHub Actions, the FastAPI service against a real registered model (tested with a stub predictor only), the
  full `mlflow` package (only `mlflow-skinny` was installed).

## Blocking (do before trusting any result)
- [ ] Confirm the metric and target (assumed: top-1 accuracy, 0.65 for a ~40-epoch laptop budget)
  - Why: Checkpoint 1 was not answered; the promotion gate uses this number
  - Where: config/train.yaml `metric`
  - Done when: you edit or confirm `target_value`
- [ ] Run `python run.py benchmark` on your laptop and decide arch/width/epochs from the projected hours
  - Why: sandbox timings are not your hardware
  - Where: config/train.yaml `model`, `training.max_epochs`
  - Done when: projected full-run time fits your budget
- [ ] Run the full training once on real CIFAR-100, then `python run.py test --run-id <id>` exactly once
  - Why: no real-data result exists yet
  - Where: `python run.py train`
  - Done when: `best_val_top1`, `test_top1` and the diagnosis verdict are in MLflow

- [ ] Get the 780M working: follow README section "Using the AMD Radeon 780M", then run `python run.py gpu-check`
  - Why: AMD lists gfx1103 (Radeon 780M) for ROCm 10.0.0 PyTorch on Windows 11 25H2, but your exact CPU (7840U) is not named in the lists I read and nothing was tested on your hardware
  - Where: scripts/setup_windows_rocm.ps1, reports/gpu_check.json
  - Done when: gpu-check shows gpu_available true, correctness passed, and a recommended override with a speedup over CPU
- [ ] If this is an IT-managed laptop, get approval for the driver install and for turning off Application Guard / Smart App Control
  - Why: AMD's Windows prerequisites require both
  - Done when: IT approves or you choose the CPU path

## Needed for production
- [ ] Shared MLflow tracking server and registry (currently a local SQLite file)
  - Where: config/env.yaml or MLFLOW_TRACKING_URI plus credentials in environment variables
  - Done when: `register` and `promote` work against the shared server
- [ ] Choose serving mode, latency/throughput budget and hardware (Checkpoint 4 not asked)
  - Where: config/serve.yaml, Dockerfile
  - Done when: budget measured against the running service on target hardware
- [ ] Set drift thresholds and schedule `src/monitor.py` (library only, not scheduled); decide how labels return
  - Where: config/serve.yaml `monitoring`
- [ ] Build and test the Docker image; run CI once on your runner
- [ ] Verify the GPU/AMP path on a CUDA (or ROCm) machine if you use one

## Deferred decisions
- [ ] Hyperparameter search (lr first), deeper arch (vgg19), stronger augmentation; none was run
- [ ] Repeat the final config over several seeds before trusting small differences between runs

## Governance and handoff
- [ ] Complete reports/MODEL_CARD.md (intended use, limitations, fairness section) and name an approver
- [ ] Name a retraining owner and trigger; rollback = move alias `champion` back (see README)

## Assumption log (each should be confirmed or removed)
- Metric top-1 accuracy, target 0.65 (not confirmed)
- Split: stratified 90/10 of the 50k train set, seed 42; official 10k test set held out
- Baseline: VGG16 + BatchNorm, 512-unit head for 32x32 inputs, SGD+Nesterov, cosine schedule with warmup, 40 epochs
- Tracking: local SQLite file
