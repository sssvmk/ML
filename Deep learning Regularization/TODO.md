# TODO: what only you can do

## What was run and what was not
- **Run (in a CPU-only sandbox):** 72 unit/integration tests pass; all 19 methods run through the orchestrator on
  *synthetic* MNIST-shaped data (2 epochs, 2 trials); a 25-epoch sanity run of 9 methods on synthetic data behaved as
  the book predicts; the registry reload-in-a-fresh-process check passed; the inference CLI ran on a saved bundle.
- **Not run:** anything on **real MNIST** (the sandbox gets HTTP 403 from the MNIST host), full-budget tuning and
  training, GPU, `mlflow models serve`, `docker build`, the CI workflow, `--promote`. **No MNIST result in this
  project has been produced yet.**
- Because of that, call this *production-minded scaffolding*, not "production ready", until the blocking items close.

## Blocking (do before trusting any result)
- [ ] Run `python run.py` on real MNIST
  - Why: every number so far is from synthetic data or a smoke budget
  - Where: `run.py`, `config/train.yaml` (use `data.source=npz:...` if torchvision is blocked)
  - Done when: `results/comparison.md` exists and the baseline test accuracy is plausible for this MLP (about 98%)
- [ ] Confirm the metric and target
  - Why: assumed accuracy + macro AUC, tuning on validation accuracy, no numeric target (`promotion.min_accuracy: null`)
  - Where: `tune.metric`, `promotion.min_accuracy`
  - Done when: you have set (or consciously left empty) the target
- [ ] Confirm the reduced-data regimes (500 examples; 1,000 labels) and that they are excluded from champion selection
  - Where: `method_settings.under_constrained.n_train`, `method_settings.semi_supervised.n_labeled`
  - Done when: you have agreed or changed them
- [ ] Re-run with several seeds before ranking close methods
  - Why: SE reflects test sampling only; one seed cannot separate methods that differ by hundredths of a point
  - Where: `--set seed=N`
  - Done when: per-method mean ± std over at least 3 seeds exists

## Needed for production
- [ ] Choose serving mode, latency/throughput budget and hardware (checkpoint not asked); only a CLI, a pyfunc model and `mlflow models serve` are provided
- [ ] Provision a shared MLflow tracking server and registry (the default is a local SQLite file) and set `tracking.uri`
- [ ] Drift and performance monitoring and a retraining trigger are **not implemented**
- [ ] Run `docker build` and the CI workflow once; neither has been executed

## Deferred decisions
- [ ] Search budget (`tune.n_trials: 12`, `tune.epochs: 5`) and final `train.epochs: 20` are defaults, not tuned choices
- [ ] Whether to replace the local-PCA tangents with a contractive-autoencoder manifold tangent classifier

## Governance and handoff
- [ ] Review `results/MODEL_CARD.md` (intended use and monitoring sections are marked TODO)
- [ ] Name an approver and use `--promote --approver "Name"`; no approval has been recorded

## Assumption log
- Seed 42; He-normal init; SGD momentum 0.9; lr decay 0.95/epoch; hidden 256-128; validation 10,000 from train
- FGSM ε = 0.1 is reported for every method on the validation set; it is a diagnostic, not a tuning target
