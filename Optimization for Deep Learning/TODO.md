# TODO: what only you can do

## What was run and what was not
- **Run (CPU-only sandbox):** 190 unit/integration tests pass; all 40 methods ran through the orchestrator on *synthetic*
  MNIST-shaped data (2 epochs, 2 trials) including MLflow, registry, and a fresh-process reload check; the inference CLI
  ran on a saved bundle; optimizers match `torch.optim`/the book's formulas; line search satisfies the strong-Wolfe
  conditions; BFGS/L-BFGS/CG solve Rosenbrock; Newton is exact on a quadratic.
- **Not run:** anything on **real MNIST** (the sandbox gets HTTP 403 from the MNIST host), full-budget tuning and
  training, GPU, `mlflow models serve`, `docker build`, the CI workflow, `--promote`. **No MNIST result in this project
  has been produced yet.** Newton-CG cost on real MNIST (curvature batch 5,000 of 50,000) is estimated, not measured.
- Call this *production-minded scaffolding*, not "production ready", until the blocking items close.

## Blocking (do before trusting any result)
- [ ] Run `python run.py` on real MNIST
  - Where: `run.py`, `config/train.yaml` (use `data.source=npz:...` if torchvision is blocked)
  - Done when: `results/comparison.md` exists and the baseline test accuracy is plausible for this MLP (about 98%)
- [ ] Confirm the metric, budgets and target
  - Why: assumed accuracy + macro AUC; tuning on validation accuracy; 20 epochs vs 100 full-batch iterations; no numeric target
  - Where: `tune.metric`, `train.epochs`, `full_batch.iterations`, `promotion.min_accuracy`
  - Done when: set, or consciously left as is
- [ ] Re-run with several seeds before ranking close methods (`--set seed=N`)
  - Done when: per-method mean ± std over at least 3 seeds exists
- [ ] Check wall-clock time of the heavy methods on real MNIST (`newton`, `conjugate_gradients`, `design_skip`, `design_aux_heads`)
  - Done when: you have decided whether to reduce `full_batch.iterations` or `tune.n_trials`

## Needed for production
- [ ] Choose serving mode, latency budget and hardware (not asked); only a CLI, a pyfunc model and `mlflow models serve` are provided
- [ ] Provision a shared MLflow tracking server and registry (default is a local SQLite file) and set `tracking.uri`
- [ ] Drift and performance monitoring and a retraining trigger are **not implemented**
- [ ] Run `docker build` and the CI workflow once; neither has been executed

## Deferred decisions
- [ ] Search budget (`tune.n_trials: 12`, `tune.epochs: 5`) is a default, not a tuned choice
- [ ] Whether to replace the sub-sampled-curvature Newton with a Gauss-Newton or K-FAC variant
- [ ] Whether the pretraining modules should keep the same total epoch budget (current) or get extra epochs

## Governance and handoff
- [ ] Review `results/MODEL_CARD.md` (intended use and monitoring sections are marked TODO)
- [ ] Name an approver and use `--promote --approver "Name"`; no approval has been recorded

## Assumption log
- Seed 42; He-normal init; SGD momentum 0.9; lr decay 0.95/epoch; hidden 256-128; validation 10,000 from train
- L1 smoothed with eps 1e-3; l2 in [1e-8,1e-2], l1 in [1e-9,1e-4], tuned per method
- BFGS on a 784-8-10 network; Newton curvature batch 5,000; early stop of full-batch methods at max|g| <= 1e-5
