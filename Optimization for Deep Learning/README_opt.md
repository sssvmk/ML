# MNIST optimization study (Deep Learning book, chapter 8)

One module per technique, one orchestrator, hyperparameter tuning inside the pipeline, everything tracked in
MLflow, plus an inference system. Every method trains a ReLU MLP with **L1 + L2 penalties** (strengths tuned).

```
python run.py                       # all 40 methods on real MNIST (torchvision downloads it)
python run.py --fast                # 2 epochs, 2 trials, 5 full-batch iterations per method
python run.py --smoke               # synthetic MNIST-shaped data: checks the plumbing, says nothing about MNIST
python run.py --methods adam,lbfgs,init_glorot
mlflow ui --backend-store-uri sqlite:///results/mlflow.db
python -m pytest -q                 # 190 tests
```

## Data and splits
MNIST **is** pre-split by its authors into 60,000 train / 10,000 test images, with **no validation set**. The pipeline
carves a stratified 10,000-image validation set out of the training set (50,000 / 10,000 / 10,000), audits sizes, class
balance and exact duplicate images across splits, and logs the data hash. If torchvision cannot download MNIST, use
`--set data.source=npz:/path/mnist.npz` (keys `x_train,y_train,x_test,y_test`).

## Model and training
Fully connected ReLU network (784-256-128-10), `nn.Linear` layers, He-normal weights and zero biases unless a method
changes that. Baseline optimizer: SGD + momentum 0.9 with exponential learning-rate decay. Loss: cross-entropy + L1 + L2
(|w| is smoothed as sqrt(w²+ε²) so full-batch line searches and curvature stay valid; biases are never penalised).
Metrics for every method: **accuracy ± SE and macro one-vs-rest AUC** (bootstrap 95% CI), hard-label cross-entropy, plus
data passes and initial-state diagnostics. Per-epoch loss plots per method and a grid of all curves.

## The 40 methods (`optpipe/methods/`, one file each)
| Group | Modules |
|---|---|
| reference | `baseline` |
| update rules (8.1.3, 8.3) | `batch_gd` `sgd` `momentum` `nesterov` `lr_decay` |
| adaptive rates (8.5) | `adagrad` `rmsprop` `rmsprop_nesterov` `adam` |
| second order (8.6) | `newton` `conjugate_gradients` `bfgs` `lbfgs` |
| initialization (8.4) | `init_random` `init_fixed_scale` `init_glorot` `init_orthogonal` `init_random_walk` `init_sparse` `init_scale_search` · `bias_zero` `bias_output_marginal` `bias_relu_positive` `bias_gate` `variance_precision` · `init_unsupervised_pretrain` `init_supervised_pretrain` |
| gradient clipping (8.2.4) | `clip_value` `clip_norm` `clip_backprop` |
| meta-algorithms (8.7) | `batch_norm` `coordinate_descent` `polyak_averaging` `greedy_pretraining` `design_activations` `design_skip` `design_aux_heads` `continuation` `curriculum` |

First-order optimizers live in `optpipe/optim/` and follow the book's algorithm boxes (tests check them against
`torch.optim` or the formulas written out). Full-batch machinery: `optpipe/fullbatch.py`, `optpipe/linesearch.py`.

## Protocol decisions you should know about
* **Budgets differ by method family.** Minibatch methods get `train.epochs` (20) passes over the data. Full-batch methods
  (batch GD, CG, BFGS, L-BFGS, Newton) get `full_batch.iterations` (100) iterations; each iteration costs one or more
  passes, so every result reports **data passes**. Compare methods by passes as well as by accuracy.
* **Update-rule studies isolate the rule:** `sgd`, `momentum`, `nesterov`, `adagrad`, `rmsprop*`, `adam` use a constant
  learning rate; `lr_decay` is plain SGD plus a schedule. All other methods use the baseline optimizer.
* **Pretraining epochs are taken out of the budget** (`init_*_pretrain`, `greedy_pretraining`), so compute is equal.
  Their plotted curves show the fine-tuning phase only.
* **Newton** is truncated, damped Newton (Hessian-free: CG on exact Hessian-vector products from a curvature sub-sample);
  exact Newton needs an n×n Hessian (~220 GB here). **BFGS** stores a dense n×n matrix, so it runs on a **tiny
  784-8-10 network**; it is labelled as a reduced regime and excluded from best-method selection.
* **variance_precision** trains a Gaussian-output model (squared error + learned precision), not cross-entropy.
* **design_skip / design_aux_heads** use deep networks (3-10 layers) and each reports a no-skip / no-aux ablation on validation.
* **Tuning is part of `run.py`.** One Optuna study per method, scored on validation accuracy; the default configuration
  is always trial 0; every trial is a nested MLflow run. The test set is never read during tuning (a test asserts this).
* **Test set is evaluated once per method** (tagged `test_set_evaluations=1`); the best method is chosen on
  **validation** accuracy, never test.
* **Single seed.** The reported SE is test-set sampling error, not seed variance. Differences of a few hundredths of a
  point between methods are not evidence; repeat with `--set seed=N`.

## Outputs (`--out`, default `results/`)
`comparison.md|csv|png`, `all_loss_curves.png`, `MODEL_CARD.md`, `data_audit.json`, `results.json`, `registry.json`,
`mlflow.db`, and per method `methods/<name>/{loss_curve.png, history.csv, trials.csv, best_hp.json, result.json, bundle/}`.

## MLflow and registry
Runs are nested: orchestrator > method > trials. The best method evaluated on the full network and full data is logged as
a pyfunc model with signature and input example, registered as `MNIST-Opt` with alias `candidate`, and reloaded in a
**fresh process** to confirm predictions match. `champion` is set only with `--promote --approver "Name"` and only if
the reload check, the optional `promotion.min_accuracy` target and the beat-the-champion check pass.

## Inference
```
python infer.py --model results/methods/adam/bundle --image digit.png
python infer.py --model results/methods/lbfgs/bundle --npy batch.npy          # float [n,784] in [0,1]
python infer.py --model results/methods/adam/bundle --test-index 0 1 2        # needs the MNIST data
python infer.py --model "models:/MNIST-Opt@candidate" --tracking-uri sqlite:///results/mlflow.db --image d.png
```
`optpipe/bundle.py` (`Predictor`) needs only `bundle.json` + `model.pt`; auxiliary heads are discarded before saving.

## Layout
`run.py` · `infer.py` · `config/train.yaml` · `optpipe/{orchestrator,tuning,trainer,fullbatch,linesearch,core,tracking,data,model,evaluate,reporting,bundle,serving_model,transforms}.py` ·
`optpipe/optim/` · `optpipe/methods/` · `tests/` · `Dockerfile` · `.github/workflows/ci.yml` · `TODO.md`
